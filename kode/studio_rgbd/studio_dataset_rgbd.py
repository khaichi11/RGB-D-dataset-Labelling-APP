"""ZenExo Studio: rekam RGB-D D435 -> tinjau -> potong virtual -> ekspor -> label.

Prinsip keselamatan data
------------------------
`source/raw.bag` adalah rekaman primer dan TIDAK pernah diubah/dihapus oleh
aplikasi. Potongan video adalah ``edit/rentang.json`` (rentang indeks frame),
sedangkan frame RGB-D dan label adalah data turunan. Dengan demikian depth
scale, intrinsics, stream IR, timestamp, dan rekaman asli tetap ada untuk
kalibrasi ulang di kemudian hari.

Jalankan dari folder ``kode``:
    source .venv/bin/activate
    python -m studio_rgbd.studio_dataset_rgbd --preset jangan
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as e:
    sys.exit("Dependensi Studio belum tersedia (" + str(e) + ").\n"
             "Jalankan dari folder kode dengan: source .venv/bin/activate\n"
             "lalu: python -m studio_rgbd.studio_dataset_rgbd --preset jangan")

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 belum tersedia. Jalankan dari .venv proyek.")

# Mendukung dua cara aman menjalankan Studio: sebagai modul (disarankan) dan
# langsung dari berkas ini. Yang kedua berguna saat pengguna double-click atau
# menjalankan path lengkap; relative import biasa akan gagal pada cara itu.
if __package__:
    from .kamera_rgbd import (KameraRGBD, cari_rekaman, nama_aman, nama_rekaman,
                              tulis_json)
    from .resolusi_kamera import PRESET_PILIHAN
    from .geometri import Z_MAX, Z_MIN
    from .pengukuran_objek import ukur
    from .segmentasi_otomatis import usulkan as usulkan_segmentasi
    from .segmentasi_yolo_depth import usulkan as usulkan_yolo_depth, verifikasi_depth
    from .segmentasi_sam2 import rapikan_kelompok as rapikan_sam2
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from studio_rgbd.kamera_rgbd import (KameraRGBD, cari_rekaman, nama_aman, nama_rekaman,
                                         tulis_json)
    from studio_rgbd.resolusi_kamera import PRESET_PILIHAN
    from studio_rgbd.geometri import Z_MAX, Z_MIN
    from studio_rgbd.pengukuran_objek import ukur
    from studio_rgbd.segmentasi_otomatis import usulkan as usulkan_segmentasi
    from studio_rgbd.segmentasi_yolo_depth import usulkan as usulkan_yolo_depth, verifikasi_depth
    from studio_rgbd.segmentasi_sam2 import rapikan_kelompok as rapikan_sam2


BG = "#F3EEE7"
PANEL = "#FFFDFC"
INK = "#382D29"
MUTED = "#786962"
LINE = "#DED2C8"
ACCENT = "#754C3B"
ACCENT_SOFT = "#E9D8CC"
GREEN = "#6B8B63"
BLUE = "#407EA3"
RED = "#AA5A55"
KATEGORI = ("batu", "tangga_naik", "ramp_naik")
# Dua cara mengukur. Yang pertama adalah metode yang dipakai skripsi: RANSAC
# memasang bidang acuan, tinggi = persentil-95 jarak tegak lurus titik objek
# ke bidang itu - sama persis dengan pengukuran_objek.ukur() yang dipakai tab
# Label. Yang kedua sekadar jarak lurus antar dua titik, berguna untuk
# memeriksa lebar tapakan atau jarak ke objek, bukan untuk tinggi.
CARA_BIDANG = "Tinggi ke bidang acuan (RANSAC)"
CARA_JARAK = "Jarak lurus 2 titik"
# Kelas SEMANTIK, bukan kategori adegan. Nilai 0 dan 1 sengaja dikunci agar
# sama persis dengan dataset/dataset_tangga_seg/tangga_seg.yaml yang sudah
# berisi 2.832 gambar - kalau digeser, seluruh label lama itu jadi salah arti.
# Kelas baru ditambahkan di belakang, tidak menimpa yang sudah ada.
KELAS_YOLO = {"tapakan": 0, "bidang_tegak": 1, "batu": 2, "ramp": 3}
# Poligon biru selalu permukaan datar (tapakan). Poligon merah artinya
# bergantung kategori adegan yang sedang direkam.
KELAS_MODE = {
    "tangga_naik": {"acuan": "tapakan", "objek": "bidang_tegak"},
    "batu":        {"acuan": "tapakan", "objek": "batu"},
    "ramp_naik":   {"acuan": "tapakan", "objek": "ramp"},
}


def baca_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default.copy()
    return json.loads(path.read_text(encoding="utf-8"))


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _shade(hexwarna: str, faktor: float) -> str:
    """Gelapkan/terangkan warna #RRGGBB (dipakai untuk keadaan hover tombol)."""
    h = hexwarna.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"#{min(255, int(r * faktor)):02x}{min(255, int(g * faktor)):02x}{min(255, int(b * faktor)):02x}"


class PembacaBag:
    """Membaca .bag secara offline; tidak menyentuh rekaman asli."""

    def __init__(self, bag: Path):
        self.bag = bag

    def iter_frame(self):
        pipe = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, str(self.bag), repeat_playback=False)
        profile = pipe.start(cfg)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align = rs.align(rs.stream.color)
        # Playback kerap mengirim ulang frame terakhir yang masih tersangkut di
        # dekoder SEBELUM benar-benar mulai dari awal. Terukur pada rekaman
        # 105802: nomor frame keluar sebagai 7,7,7,7,0,1,2,... Empat frame
        # pertama itu salinan basi. Kalau dibiarkan, indeks 0-3 pada
        # frame_index.csv menunjuk frame yang sama, potongan yang dimulai dari
        # 0 menghasilkan frame kembar di ekspor, dan label pertama menempel
        # pada citra yang salah. Urutannya deterministik (dua pembacaan
        # menghasilkan urutan identik), jadi aman dibuang di sini.
        tunda, mulai, terakhir = [], False, [-1]
        try:
            while True:
                try:
                    native = pipe.wait_for_frames(5000)
                except RuntimeError:
                    break
                color = native.get_color_frame()
                depth = native.get_depth_frame()
                if not color or not depth:
                    continue
                aligned_set = align.process(native)
                aligned = aligned_set.get_depth_frame()
                if not aligned:
                    continue
                paket = (native, color, depth, aligned, profile)
                nomor = int(depth.get_frame_number())
                if mulai and nomor == terakhir[0]:
                    # Perekam sesekali menulis frame yang sama dua kali (terukur
                    # pada 105802: frame 287 kembar, frame 288 hilang). Untuk
                    # dataset ini berarti dua sampel latih yang identik, jadi
                    # salinannya dibuang.
                    continue
                if not mulai:
                    if tunda and nomor == tunda[-1][0]:
                        continue                    # salinan basi
                    if tunda and nomor < tunda[-1][0]:
                        tunda.clear()               # ketemu awal sebenarnya
                    tunda.append((nomor, paket))
                    if len(tunda) < 2:
                        continue                    # tunggu bukti menaik
                    mulai = True
                    for nm, p0 in tunda:
                        terakhir[0] = nm
                        yield p0
                    tunda.clear()
                    continue
                terakhir[0] = nomor
                yield paket
            for _, p0 in tunda:                     # rekaman sangat pendek
                yield p0
        finally:
            pipe.stop()

    def indeks(self) -> list[dict]:
        hasil: list[dict] = []
        for i, (_, color, depth, _, _) in enumerate(self.iter_frame()):
            hasil.append({"i": i, "frame": int(depth.get_frame_number()),
                          "timestamp_ms": float(depth.get_timestamp()),
                          "lebar": color.get_width(), "tinggi": color.get_height()})
        return hasil

    def fps_terukur(self, maks: int = 150) -> float | None:
        """FPS rekaman dari timestamp depth, TANPA align maupun dekode citra.

        Rekaman D435 jarang tepat pada fps yang diminta (terukur 29,9x fps)
        dan kadang menjatuhkan frame, jadi nilai inilah - bukan args.fps -
        yang layak dipakai sebagai laju preview.mp4. Hanya membaca timestamp
        beberapa puluh frame pertama, jadi jauh lebih cepat daripada pass
        dekode penuh.
        """
        ts: list[float] = []
        pipe = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, str(self.bag), repeat_playback=False)
        try:
            profile = pipe.start(cfg)
            profile.get_device().as_playback().set_real_time(False)
            while len(ts) < maks:
                try:
                    native = pipe.wait_for_frames(5000)
                except RuntimeError:
                    break
                depth = native.get_depth_frame()
                if not depth:
                    continue
                t = float(depth.get_timestamp())
                if ts and t <= ts[-1]:
                    continue                    # salinan basi dekoder
                ts.append(t)
        finally:
            pipe.stop()
        if len(ts) > 2 and ts[-1] > ts[0]:
            return (len(ts) - 1) / ((ts[-1] - ts[0]) / 1000.0)
        return None


class TombolRounded(tk.Canvas):
    """Tombol datar dengan radius kecil; elegan tanpa tampak seperti pil besar."""

    def __init__(self, master, text, command, color=ACCENT, fg="white"):
        self._siap = False
        # Lebar dasar cukup untuk kontrol yang di-pack berdampingan.
        # width=1 membuat tombol Tinjau tampak hilang seperti garis.
        super().__init__(master, width=104, height=32, bg=master.cget("bg"),
                         highlightthickness=0, bd=0, cursor="hand2")
        self.command, self.color, self.fg, self.text = command, color, fg, text
        self.enabled = True
        self._hover = False
        self._siap = True
        self._ukuran = (104, 32)
        self.bind("<Configure>", self._ubah_ukuran)
        self.bind("<Button-1>", self._klik)
        self.bind("<Enter>", self._masuk)
        self.bind("<Leave>", self._keluar)

    def _masuk(self, _event):
        self._hover = True
        super().configure(cursor="hand2" if self.enabled else "")
        self._gambar()

    def _keluar(self, _event):
        self._hover = False
        self._gambar()

    def _ubah_ukuran(self, event):
        self._ukuran = (event.width, event.height)
        self._gambar()

    def _gambar(self):
        self.delete("all")
        # Jangan pakai polygon smooth: pada Canvas yang di-resize oleh grid,
        # spline dapat memakai lebar lama dan menyisakan bagian putih di kanan.
        w, h = self._ukuran
        w, h, r = max(1, w), max(1, h), 7
        warna = self.color if self.enabled else "#D8D0CB"
        if self._hover and self.enabled:
            warna = _shade(warna, 0.88)
        garis = "#D8CAC1" if self.enabled and self.color == "#E8DDD5" else warna
        self.create_rectangle(r, 0, w-r, h, fill=warna, outline="")
        self.create_rectangle(0, r, w, h-r, fill=warna, outline="")
        for x, y, mulai in ((0, 0, 90), (w-2*r, 0, 0), (w-2*r, h-2*r, 270), (0, h-2*r, 180)):
            self.create_arc(x, y, x+2*r, y+2*r, start=mulai, extent=90,
                            style="pieslice", fill=warna, outline=warna)
        # Garis sangat tipis hanya pada tombol sekunder agar tetap terpisah
        # dari panel tanpa memberi kesan kotak berat.
        self.create_rectangle(r, 0, w-r, h-1, outline=garis)
        self.create_text(w / 2, h / 2, text=self.text, fill=self.fg if self.enabled else MUTED,
                         font=("Segoe UI", 9, "bold"))

    def _klik(self, _event):
        if self.enabled:
            self.command()

    def configure(self, cnf=None, **kw):
        # Dukungan atribut yang sudah dipakai oleh tombol lama.
        if not self._siap:
            return super().configure(cnf, **kw)
        if "text" in kw:
            self.text = kw.pop("text")
        if "bg" in kw:
            self.color = kw.pop("bg")
        if "state" in kw:
            self.enabled = kw.pop("state") != "disabled"
        result = super().configure(cnf, **kw)
        self._gambar()
        return result


class KanvasLabel(tk.Canvas):
    """Editor poligon responsif: tambah, pindah titik, zoom di posisi kursor."""

    def __init__(self, master, on_change, on_active=None, **kw):
        super().__init__(master, bg="#241F1D", highlightthickness=0, **kw)
        self.on_change = on_change
        self.on_active = on_active
        self.rgb: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self.depth_alpha = 0.0
        self.mode = "objek"
        # Tiap kelas menampung BANYAK poligon: satu anak tangga = satu instance.
        # Dulu hanya satu poligon per kelas, jadi tangga dengan lima anak tangga
        # mustahil dilabeli sesuai dataset lama yang rata-rata 4,75 instance.
        self.poligon = {"objek": [], "acuan": []}
        # Mask aktif tidak selalu yang terakhir: pengguna dapat memilih mask
        # nomor 2 lalu menambah/menghapus titik hanya pada mask tersebut.
        self.aktif_indeks = {"objek": None, "acuan": None}
        self._drag = None
        self._drag_titik = None
        self._render_terjadwal = False
        self._photo = None
        self._photo_key = None
        self._tampil_cache = None
        self._tampil_key = None
        self.bind("<Button-1>", self.tambah)
        self.bind("<B1-Motion>", self.geser_titik)
        self.bind("<ButtonRelease-1>", self.selesai_geser_titik)
        self.bind("<Button-3>", self.hapus_titik_dipilih)
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<Button-4>", lambda e: self.zoom_langkah(e, 1.15))
        self.bind("<Button-5>", lambda e: self.zoom_langkah(e, 1 / 1.15))
        self.bind("<ButtonPress-2>", self.pan_mulai)
        self.bind("<B2-Motion>", self.pan)
        self.bind("<Configure>", lambda e: self.fit_bila_baru())

    def set_frame(self, rgb: np.ndarray, depth: np.ndarray):
        self.rgb, self.depth = rgb, depth
        self.poligon = {"objek": [], "acuan": []}
        self.aktif_indeks = {"objek": None, "acuan": None}
        self._photo = self._photo_key = self._tampil_cache = self._tampil_key = None
        self.after(20, self.fit)

    def _beritahu_aktif(self, nama: str, indeks: int | None):
        if self.on_active is not None:
            self.on_active(nama, indeks)

    def pilih_mask(self, nama: str, nomor: int) -> bool:
        """Pilih instance bernomor 1-based untuk diedit, tanpa memindah urutan."""
        indeks = nomor - 1
        if indeks < 0 or indeks >= len(self.poligon[nama]):
            return False
        self.mode = nama
        self.aktif_indeks[nama] = indeks
        self._beritahu_aktif(nama, indeks)
        self.render()
        return True

    def poligon_baru(self, nama: str | None = None):
        """Mulai instance baru tanpa menghapus mask yang sudah ada."""
        nama = nama or self.mode
        if not self.poligon[nama] or self.poligon[nama][-1]:
            self.poligon[nama].append([])
        self.aktif_indeks[nama] = len(self.poligon[nama]) - 1
        self._beritahu_aktif(nama, self.aktif_indeks[nama])
        self.render(); self.on_change()

    def hapus_mask_aktif(self):
        """Hapus instance aktif mode aktif, bukan seluruh kelas mask."""
        daftar = self.poligon[self.mode]
        if daftar:
            indeks = self.aktif_indeks.get(self.mode)
            indeks = len(daftar) - 1 if indeks is None else min(indeks, len(daftar) - 1)
            daftar.pop(indeks)
            self.aktif_indeks[self.mode] = min(indeks, len(daftar) - 1) if daftar else None
            self._beritahu_aktif(self.mode, self.aktif_indeks[self.mode])
            self.render(); self.on_change()

    def _aktif(self, nama: str) -> list:
        if not self.poligon[nama]:
            self.poligon[nama].append([])
        indeks = self.aktif_indeks.get(nama)
        if indeks is None or indeks >= len(self.poligon[nama]):
            indeks = len(self.poligon[nama]) - 1
            self.aktif_indeks[nama] = indeks
            self._beritahu_aktif(nama, indeks)
        return self.poligon[nama][indeks]

    def fit_bila_baru(self):
        if self.rgb is not None and not any(self.poligon["objek"]) and not any(self.poligon["acuan"]):
            self.fit()

    def fit(self):
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        self.scale = max(0.05, min((self.winfo_width() - 28) / w, (self.winfo_height() - 28) / h))
        self.ox = (self.winfo_width() - w * self.scale) / 2
        self.oy = (self.winfo_height() - h * self.scale) / 2
        self.render()

    def gambar_tampil(self) -> np.ndarray:
        assert self.rgb is not None
        key = (id(self.rgb), id(self.depth), round(self.depth_alpha, 3))
        if self._tampil_key == key and self._tampil_cache is not None:
            return self._tampil_cache
        out = self.rgb.copy()
        if self.depth is not None and self.depth_alpha > 0:
            d = self.depth.astype(np.float32)
            valid = d > 0
            if valid.any():
                lo, hi = np.percentile(d[valid], (3, 97))
                vis = np.clip((d - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
                warna = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
                warna = cv2.cvtColor(warna, cv2.COLOR_BGR2RGB)
                out = cv2.addWeighted(out, 1 - self.depth_alpha, warna, self.depth_alpha, 0)
        self._tampil_key, self._tampil_cache = key, out
        return out

    def render(self):
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        key = (id(self.rgb), id(self.depth), round(self.depth_alpha, 3), size)
        # Drag titik bisa memanggil render puluhan kali/detik. Gambar dasar
        # cukup dibuat sekali; yang berubah hanya garis poligon di atasnya.
        if self._photo_key != key or self._photo is None:
            self._photo = ImageTk.PhotoImage(Image.fromarray(self.gambar_tampil()).resize(size, Image.LANCZOS))
            self._photo_key = key
        self.delete("all")
        self.create_image(self.ox, self.oy, anchor="nw", image=self._photo)
        # merah = sisi tinggi (riser), biru = permukaan datar (tapakan).
        # Warna ini bukan hiasan: yang biru dipakai RANSAC sebagai BIDANG ACUAN,
        # yang merah sebagai objek yang diukur tingginya terhadap bidang itu.
        specs = (("objek", "#FF5A5A", "#FFD9D9"), ("acuan", "#5AA9FF", "#D7E9FF"))
        for nama, garis, titik in specs:
            for idx, poly in enumerate(self.poligon[nama], start=1):
                pts = [(x * self.scale + self.ox, y * self.scale + self.oy) for x, y in poly]
                aktif = idx - 1 == self.aktif_indeks.get(nama)
                if len(pts) >= 3:
                    self.create_polygon(pts, fill=garis, outline="#FFFFFF" if aktif else garis,
                                        stipple="gray50", width=3 if aktif else 2)
                    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
                    self.create_text(cx, cy, text=str(idx), fill="white",
                                     font=("Segoe UI", 11, "bold"))
                elif len(pts) >= 2:
                    self.create_line(pts, fill=garis, width=2)
                for x, y in pts:
                    self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=titik, outline=garis)
        if self._drag_titik is not None:
            self._gambar_lup(self._drag_titik[3], self._drag_titik[4])

    def render_nanti(self):
        """Koaleskan redraw saat roda/drag; UI tidak mengantre ratusan resize."""
        if not self._render_terjadwal:
            self._render_terjadwal = True
            self.after_idle(self._render_tertunda)

    def _render_tertunda(self):
        self._render_terjadwal = False
        self.render()

    def _titik_dekat(self, x, y, batas=11):
        terbaik = None
        for nama, daftar in self.poligon.items():
            for ip, poly in enumerate(daftar):
                for it, (px, py) in enumerate(poly):
                    d = (px * self.scale + self.ox - x) ** 2 + (py * self.scale + self.oy - y) ** 2
                    if d <= batas ** 2 and (terbaik is None or d < terbaik[0]):
                        terbaik = (d, nama, ip, it)
        return None if terbaik is None else terbaik[1:]

    def _gambar_lup(self, x, y):
        """Lup kecil pada titik aktif agar batas masking presisi saat zoom."""
        if self.rgb is None:
            return
        gx, gy = self.canvas_ke_gambar(x, y) or (0, 0)
        r, faktor = 32, 3
        h, w = self.rgb.shape[:2]
        x0, y0, x1, y1 = max(0, int(gx-r)), max(0, int(gy-r)), min(w, int(gx+r)), min(h, int(gy+r))
        potong = self.gambar_tampil()[y0:y1, x0:x1]
        if potong.size == 0:
            return
        potong = cv2.resize(potong, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_NEAREST)
        foto = ImageTk.PhotoImage(Image.fromarray(potong))
        self._lup_photo = foto
        lx, ly = min(self.winfo_width()-potong.shape[1]-8, x+18), max(8, y-potong.shape[0]-18)
        self.create_rectangle(lx-3, ly-3, lx+potong.shape[1]+3, ly+potong.shape[0]+3, fill="#FFFDFC", outline="#FFD400", width=2)
        self.create_image(lx, ly, anchor="nw", image=foto)
        self.create_line(lx+potong.shape[1]/2-8, ly+potong.shape[0]/2, lx+potong.shape[1]/2+8, ly+potong.shape[0]/2, fill="#FFD400", width=2)
        self.create_line(lx+potong.shape[1]/2, ly+potong.shape[0]/2-8, lx+potong.shape[1]/2, ly+potong.shape[0]/2+8, fill="#FFD400", width=2)

    def canvas_ke_gambar(self, x, y):
        if self.rgb is None:
            return None
        gx, gy = (x - self.ox) / self.scale, (y - self.oy) / self.scale
        h, w = self.rgb.shape[:2]
        return (gx, gy) if 0 <= gx < w and 0 <= gy < h else None

    def tambah(self, e):
        dekat = self._titik_dekat(e.x, e.y)
        if dekat is not None:
            nama, ip, it = dekat
            self.mode = nama; self.aktif_indeks[nama] = ip
            self._beritahu_aktif(nama, ip)
            self._drag_titik = (nama, ip, it, e.x, e.y)
            self.render()
            return
        p = self.canvas_ke_gambar(e.x, e.y)
        if p:
            self._aktif(self.mode).append(p)
            self.render(); self.on_change()

    def geser_titik(self, e):
        if self._drag_titik is None:
            return
        nama, ip, it, _, _ = self._drag_titik
        p = self.canvas_ke_gambar(e.x, e.y)
        if p:
            self.poligon[nama][ip][it] = p
            self._drag_titik = (nama, ip, it, e.x, e.y)
            self.render_nanti()

    def selesai_geser_titik(self, _e=None):
        if self._drag_titik is not None:
            self._drag_titik = None
            self.render(); self.on_change()

    def undo(self, _e=None):
        daftar = self.poligon[self.mode]
        if not daftar:
            return
        indeks = self.aktif_indeks.get(self.mode)
        indeks = len(daftar) - 1 if indeks is None else min(indeks, len(daftar) - 1)
        if daftar[indeks]:
            daftar[indeks].pop()
        if not daftar[indeks] and len(daftar) > 1:
            daftar.pop(indeks)
            indeks = min(indeks, len(daftar) - 1)
        self.aktif_indeks[self.mode] = indeks if daftar else None
        self._beritahu_aktif(self.mode, self.aktif_indeks[self.mode])
        self.render(); self.on_change()

    def hapus_titik_dipilih(self, e):
        """Klik kanan hanya menghapus vertex yang dipilih, tidak yang terakhir."""
        dekat = self._titik_dekat(e.x, e.y, batas=14)
        if dekat is None:
            return
        nama, ip, it = dekat
        poly = self.poligon[nama][ip]
        poly.pop(it)
        # Instance kosong tidak perlu dibiarkan sebagai mask semu.
        if not poly:
            self.poligon[nama].pop(ip)
        daftar = self.poligon[nama]
        self.aktif_indeks[nama] = min(ip, len(daftar) - 1) if daftar else None
        self._beritahu_aktif(nama, self.aktif_indeks[nama])
        self.render(); self.on_change()

    def bersihkan(self, nama):
        self.poligon[nama] = []
        self.render(); self.on_change()

    def zoom(self, e):
        self.zoom_langkah(e, 1.15 if e.delta > 0 else 1 / 1.15)

    def zoom_langkah(self, e, faktor):
        if self.rgb is None:
            return
        sebelumnya = self.canvas_ke_gambar(e.x, e.y)
        self.scale = max(0.08, min(8.0, self.scale * faktor))
        if sebelumnya:
            self.ox, self.oy = e.x - sebelumnya[0] * self.scale, e.y - sebelumnya[1] * self.scale
        self.render_nanti()

    def pan_mulai(self, e): self._drag = (e.x, e.y, self.ox, self.oy)
    def pan(self, e):
        if self._drag:
            x, y, ox, oy = self._drag
            self.ox, self.oy = ox + e.x - x, oy + e.y - y
            self.render_nanti()


class Studio(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.root_data = Path(args.keluar).expanduser().resolve()
        self.cam = KameraRGBD(args.lebar, args.tinggi, args.fps, args.preset, args.batas_frame)
        self.q: queue.Queue[tuple[str, object]] = queue.Queue()
        self.sedang_rekam = False
        self.preview_diminta = False
        self.preview_info: dict | None = None
        self.preset_pilih = StringVar(value="(kamera belum menyala)")
        self.kategori_sesi = StringVar(value="(belum ada rekaman dipilih)")
        self.kecepatan = StringVar(value="1x")
        self.posisi = IntVar(value=0)
        self.mode_ukur = BooleanVar(value=False)
        self.cara_ukur = StringVar(value=CARA_BIDANG)
        self.hasil_ukur = StringVar(value="")
        self.titik_ukur: list = []          # klik dalam koordinat citra
        self.garis_ukur: list = []          # (p1, p2, teks)
        self._depth_frame_cache = (None, None)
        self._video_cap = None
        self._n_frame = 0
        self._fps_video = 30.0
        self._skala_tampil = 1.0
        self._bingkai_kini = None
        self._bingkai_siap = None
        self._kunci_video = threading.Lock()
        self._kunci_bingkai = threading.Lock()
        self.kategori_baru = StringVar(value=KATEGORI[0])
        self.HZ_RENDER = max(1, min(30, getattr(args, "fps_preview", 20)))
        self.sesi: Path | None = None
        self.indeks: list[dict] = []
        self.frame_paths: list[Path] = []
        self.label_path: Path | None = None
        self.label_info: dict | None = None
        self.preview_photo = None
        self._preview_w = 480               # lebar render preview; mengikuti widget
        self.title("ZenExo Studio — Rekam, Tinjau, Ekspor, Label")
        self.geometry("1500x940"); self.minsize(1180, 760); self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self.tutup)
        self._gaya()
        self.root_data.mkdir(parents=True, exist_ok=True)
        self.path_preferensi = self.root_data / ".studio_preferences.json"
        preferensi = baca_json(self.path_preferensi, {})
        self.kategori = StringVar(value=preferensi.get("kategori", "batu"))
        self.split = StringVar(value=preferensi.get("split", "train"))
        self.kode = StringVar(value=preferensi.get("kode_adegan", ""))
        self.status = StringVar(value="Menyalakan D435…")
        self.durasi = StringVar(value="Belum ada rekaman dipilih.")
        # Jarak antar frame sumber. Nilai 10 pada rekaman ~30 FPS berarti
        # sekitar 3 frame/detik, sehingga dataset tidak dipenuhi frame kembar.
        self.langkah_ekspor = IntVar(value=10)
        self.awal = IntVar(value=0); self.akhir = IntVar(value=0)
        self.ekspor_awal = IntVar(value=0); self.ekspor_akhir = IntVar(value=0)
        self.rentang_manual = BooleanVar(value=False)
        self.filter_sesi = StringVar(value="tangga_naik")
        self.nomor_mask = IntVar(value=1)
        self._autosave_setelah = None
        self.depth_alpha = DoubleVar(value=0.28)
        self.mode_label = StringVar(value="objek")
        self.ukur_status = StringVar(value="Tandai sisi tinggi/riser (merah) dan permukaan datar/tapakan (biru).")
        self.tampil_sampah = BooleanVar(value=False)
        self.tampil_sampah_frame = BooleanVar(value=False)
        self._siapkan_root(); self._buat_ui()
        self.tabs.bind("<<NotebookTabChanged>>", self.ganti_tab)
        self.after(30, self._poll)
        self.after(120, self.muat_daftar)   # daftar tangga langsung terlihat saat aplikasi dibuka
        self.status.set("Siap untuk labeling. Kamera dinyalakan hanya saat Preview Kamera atau Mulai rekam ditekan.")

    # ----- struktur data -----
    def _siapkan_root(self):
        self.root_data.mkdir(parents=True, exist_ok=True)
        for k in KATEGORI:
            (self.root_data / "rekaman" / k).mkdir(parents=True, exist_ok=True)
        (self.root_data / "tempat_sampah").mkdir(exist_ok=True)
        p = self.root_data / "README_STUDIO.json"
        if not p.exists():
            tulis_json(p, {"versi": 1, "prinsip": "source/raw.bag tidak pernah dimodifikasi",
                           "struktur": {"rekaman/<kategori>/<sesi>/source/raw.bag": "stream primer D435 RGB-D+IR",
                                        "edit/rentang.json": "potongan virtual, non-destruktif",
                                        "exports/<nama>/frames": "frame RGB-D berpasangan untuk dataset",
                                        "labels": "label YOLO segmentation dan pengukuran"},
                           "kelas_yolo": KELAS_YOLO})

    def _state(self, sesi: Path) -> dict:
        return baca_json(sesi / "state.json", {"di_sampah": False})

    def _tulis_state(self, sesi: Path, state: dict): tulis_json(sesi / "state.json", state)
    def bag(self, sesi: Path) -> Path:
        """Rekaman primer sesi ini.

        Sesi lama berisi raw.bag, sesi baru raw.db3 - pyrealsense2 2.56 ke atas
        hanya mau merekam ke SQLite. Keduanya harus tetap bisa dibuka, jadi yang
        dicari adalah berkas yang benar-benar ada; kalau belum ada satu pun,
        dikembalikan nama untuk versi pyrealsense2 yang terpasang sekarang.
        """
        src = sesi / "source"
        return cari_rekaman(src) or (src / nama_rekaman())

    def daftar_sesi(self) -> list[Path]:
        hasil = []
        for k in KATEGORI:
            hasil.extend(sorted((self.root_data / "rekaman" / k).glob("*"), reverse=True))
        return [p for p in hasil if p.is_dir() and self.bag(p).exists()]

    # ----- UI -----
    def _gaya(self):
        """Gaya desktop yang ringan: warna lembut, garis tipis, radius kecil."""
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(2, 0, 2, 0))
        s.configure("TNotebook.Tab", background="#EEE7E2", foreground=MUTED,
                    padding=(22, 11), font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        s.configure("TCombobox", fieldbackground="#FFFDFC", background="#FFFDFC",
                    foreground=INK, padding=5, bordercolor="#DED7D2", lightcolor="#DED7D2", darkcolor="#DED7D2")

    def card(self, parent, judul):
        b = tk.Frame(parent, bg=PANEL, highlightbackground="#E6DED8", highlightthickness=1, bd=0)
        tk.Label(b, text=judul.upper(), bg=PANEL, fg=ACCENT, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=18, pady=(13, 2))
        garis = tk.Frame(b, bg="#F0EAE6", height=1); garis.pack(fill="x", padx=18, pady=(0, 11))
        isi = tk.Frame(b, bg=PANEL); isi.pack(fill="both", expand=True, padx=18, pady=(0, 15))
        return b, isi

    def tombol(self, parent, text, cmd, color=ACCENT, fg="white"):
        return TombolRounded(parent, text, cmd, color, fg)

    def tombol_ringkas(self, parent, text, cmd, color=ACCENT, fg="white", width=88):
        """Versi pendek untuk toolbar/panel samping yang ruangnya terbatas."""
        btn = self.tombol(parent, text, cmd, color, fg)
        btn.configure(width=width, height=28)
        return btn

    def _buat_ui(self):
        top = tk.Frame(self, bg=BG); top.pack(fill="x", padx=24, pady=(16, 8))
        brand = tk.Frame(top, bg=BG); brand.pack(side="left")
        tk.Label(brand, text="ZenExo Studio", bg=BG, fg=INK, font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(brand, text="Rekam RGB-D mentah • sortir • ekspor • label • ukur", bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(1, 0))
        badge = tk.Label(top, text="D435  •  RGB + DEPTH", bg=ACCENT_SOFT, fg=ACCENT,
                         font=("Segoe UI", 9, "bold"), padx=12, pady=7)
        badge.pack(side="right", padx=(12, 0))
        # Baris status pindah ke kaki jendela: pesan progres yang panjang
        # (ekspor/preview) tidak lagi berjejal di kanan atas.
        footer = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        footer.pack(side="bottom", fill="x")
        tk.Label(footer, text="●", bg=PANEL, fg=GREEN, font=("Segoe UI", 9)).pack(side="left", padx=(14, 6))
        tk.Label(footer, textvariable=self.status, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10), anchor="w").pack(side="left", fill="x", expand=True, pady=7, padx=(0, 12))
        self.tabs = ttk.Notebook(self); self.tabs.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        self.tab_rekam = tk.Frame(self.tabs, bg=BG); self.tab_tinjau = tk.Frame(self.tabs, bg=BG)
        self.tab_ekspor = tk.Frame(self.tabs, bg=BG); self.tab_label = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(self.tab_rekam, text="  1. Rekam  "); self.tabs.add(self.tab_tinjau, text="  2. Tinjau & Potong  ")
        self.tabs.add(self.tab_ekspor, text="  3. Ekspor Frame  "); self.tabs.add(self.tab_label, text="  4. Label & Ukur  ")
        self.ui_rekam(); self.ui_tinjau(); self.ui_ekspor(); self.ui_label()

    def ui_rekam(self):
        f = tk.Frame(self.tab_rekam, bg=BG); f.pack(fill="both", expand=True, padx=24, pady=24)
        kiri, kanan = tk.Frame(f, bg=BG), tk.Frame(f, bg=BG); kiri.pack(side="left", fill="both", expand=True, padx=(0, 12)); kanan.pack(side="left", fill="both", expand=True)
        b, i = self.card(kiri, "Rekam satu adegan") ; b.pack(fill="x")
        tk.Label(i, text="Pilih objek yang direkam", bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(i, textvariable=self.kategori, values=KATEGORI, state="readonly", width=24).grid(row=0, column=1, sticky="ew", padx=10)
        tk.Label(i, text="Kode adegan (opsional)", bg=PANEL, fg=MUTED).grid(row=1, column=0, sticky="w", pady=4)
        tk.Entry(i, textvariable=self.kode, bg="#FFF9F4", fg=INK, relief="flat").grid(row=1, column=1, sticky="ew", padx=10)
        tk.Label(i, text="Split dataset", bg=PANEL, fg=MUTED).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(i, textvariable=self.split, values=("train", "val", "test"), state="readonly", width=14).grid(row=2, column=1, sticky="w", padx=10)
        tk.Label(i, text="Preset depth", bg=PANEL, fg=MUTED).grid(row=3, column=0, sticky="w", pady=4)
        self.cb_preset = ttk.Combobox(i, textvariable=self.preset_pilih, values=(),
                                      state="disabled", width=24)
        self.cb_preset.grid(row=3, column=1, sticky="ew", padx=10)
        self.cb_preset.bind("<<ComboboxSelected>>", self._ganti_preset)
        i.columnconfigure(1, weight=1)
        self.btn_rekam = self.tombol(i, "● Mulai rekam", self.toggle_rekam, RED)
        self.btn_rekam.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(16, 4))
        self.btn_preview_kamera = self.tombol(i, "Aktifkan preview kamera", self.toggle_preview_kamera, "#E8DDD5", INK)
        self.btn_preview_kamera.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        tk.Label(i, text="Kode adegan kosong = nama otomatis. Jika Anda mengisi misalnya Taman, nilai terakhir akan diingat sampai diganti.", bg=PANEL, fg=MUTED, justify="left", wraplength=490).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        tk.Label(i, text="Cukup tekan Mulai rekam, ambil berbagai sudut, lalu tekan Selesai rekaman. Semua stream mentah RGB, Z16 depth, IR, timestamp, intrinsics, dan extrinsics tersimpan dalam raw.bag.", bg=PANEL, fg=MUTED, justify="left", wraplength=490).grid(row=7, column=0, columnspan=2, sticky="w", pady=(5, 0))
        b, i = self.card(kanan, "Preview kamera dan data") ; b.pack(fill="both", expand=True)
        # Preview diutamakan: ia yang paling sering dilihat, jadi ditaruh
        # paling atas dan diberi seluruh sisa ruang kartu.
        previews = tk.Frame(i, bg=PANEL); previews.pack(fill="both", expand=True)
        previews.columnconfigure((0, 1), weight=1); previews.rowconfigure(1, weight=1)
        tk.Label(previews, text="RGB", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(previews, text="DEPTH (warna = jarak)", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.label_live = tk.Label(previews, text="Menunggu RGB…", bg="#27201E", fg="white")
        self.label_depth = tk.Label(previews, text="Menunggu depth…", bg="#27201E", fg="white")
        self.label_live.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(4, 0))
        self.label_depth.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(4, 0))
        # Ukuran render mengikuti lebar widget: diatur di thread Tk (aman),
        # dibaca thread render sebagai int biasa.
        self.label_live.bind("<Configure>", self._atur_ukuran_preview)
        tk.Label(i, text="• Rekaman asli tidak pernah dipotong atau dihapus.  • Potong hanya membuat rentang virtual.\n• Frame hasil ekspor membawa RGB, depth native Z16, depth selaras RGB, IR, timestamp, dan metadata kamera.\n• Jika kalibrasi berubah, ekspor dapat dibuat ulang dari raw.bag yang sama.", bg=PANEL, fg=MUTED, justify="left", wraplength=560, font=("Segoe UI", 9)).pack(anchor="nw", pady=(10, 0))

    def ui_tinjau(self):
        f = tk.Frame(self.tab_tinjau, bg=BG); f.pack(fill="both", expand=True, padx=18, pady=18)
        left = tk.Frame(f, bg=BG, width=320); left.pack(side="left", fill="y", padx=(0, 12)); left.pack_propagate(False)
        right = tk.Frame(f, bg=BG); right.pack(side="left", fill="both", expand=True)
        b, i = self.card(left, "Rekaman") ; b.pack(fill="both", expand=True)
        tk.Label(i, text="Kategori", bg=PANEL, fg=MUTED).pack(anchor="w")
        pilih_kategori = ttk.Combobox(i, textvariable=self.filter_sesi, values=KATEGORI,
                                      state="readonly")
        pilih_kategori.pack(fill="x", pady=(2, 8))
        pilih_kategori.bind("<<ComboboxSelected>>", lambda _e: self.muat_daftar())
        self.list_sesi = tk.Listbox(i, bg="#FFF9F4", fg=INK, relief="flat", selectbackground=ACCENT_SOFT, activestyle="none", height=22)
        self.list_sesi.pack(fill="both", expand=True); self.list_sesi.bind("<<ListboxSelect>>", lambda e: self.pilih_sesi())
        self.tombol(i, "Muat ulang daftar", self.muat_daftar, "#E8DDD5", INK).pack(fill="x", pady=(10, 3))
        self.tombol(i, "Pindah ke tempat sampah / pulihkan", self.toggle_sampah, "#E8DDD5", INK).pack(fill="x")
        self.tombol(i, "Hapus preview video", self.hapus_preview_permanen, "#F3D8D4", INK).pack(fill="x", pady=(4, 0))
        self.tombol(i, "Hapus rekaman permanen", self.hapus_sesi_permanen, RED).pack(fill="x", pady=(4, 0))
        tk.Checkbutton(i, text="\U0001f5d1 Tampilkan isi tempat sampah", variable=self.tampil_sampah,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       command=self.muat_daftar).pack(anchor="w", pady=(6, 0))
        garis = tk.Frame(i, bg="#EFE6DF", height=1); garis.pack(fill="x", pady=(12, 8))
        tk.Label(i, text="KATEGORI REKAMAN TERPILIH", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(i, textvariable=self.kategori_sesi, bg=PANEL, fg=INK,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(1, 6))
        ttk.Combobox(i, textvariable=self.kategori_baru, values=KATEGORI,
                     state="readonly").pack(fill="x")
        self.tombol(i, "Pindahkan ke kategori ini", self.pindah_kategori, "#E8DDD5", INK).pack(fill="x", pady=(5, 0))
        tk.Label(i, text="Memindahkan hanya mengubah folder dan session.json. Isi raw.db3 tidak pernah disentuh.",
                 bg=PANEL, fg=MUTED, wraplength=250, justify="left").pack(anchor="w", pady=(5, 0))
        b, i = self.card(right, "Tinjau rekaman") ; b.pack(fill="both", expand=True)
        # Kontrol di-pack ke bawah DULU (side="bottom") supaya tidak pernah
        # terpotong saat jendela pendek; kanvas mengambil sisa ruang.
        bawah = tk.Frame(i, bg=PANEL); bawah.pack(side="bottom", fill="x", pady=(10, 0))
        # Kanvas, bukan Label: perlu koordinat klik dan menggambar garis ukur.
        self.kanvas_tinjau = tk.Canvas(i, bg="#27201E", highlightthickness=0, height=420)
        self.kanvas_tinjau.pack(fill="both", expand=True)
        self.kanvas_tinjau.bind("<Button-1>", self._klik_kanvas)
        self._teks_kanvas = self.kanvas_tinjau.create_text(
            12, 12, anchor="nw", fill="white", font=("Segoe UI", 10),
            text="Pilih rekaman lalu tekan Buat preview.")

        row = tk.Frame(bawah, bg=PANEL); row.pack(fill="x")
        self.tombol_ringkas(row, "Buat preview", self.buat_preview, BLUE, width=112).pack(side="left", padx=(0, 4))
        self.btn_putar = self.tombol(row, "\u25b6 Putar", self.putar_preview, "#E8DDD5", INK)
        self.btn_putar.configure(width=78)
        self.btn_putar.pack(side="left", padx=4)
        self.tombol_ringkas(row, "\u23ea", lambda: self._langkah(-1), "#E8DDD5", INK, width=36).pack(side="left", padx=2)
        self.tombol_ringkas(row, "\u23e9", lambda: self._langkah(1), "#E8DDD5", INK, width=36).pack(side="left", padx=2)
        tk.Label(row, text="Kecepatan", bg=PANEL, fg=MUTED).pack(side="left", padx=(10, 4))
        ttk.Combobox(row, textvariable=self.kecepatan, width=5, state="readonly",
                     values=("0.25x","0.5x","1x","2x","4x","8x")).pack(side="left")
        self.tombol_ringkas(row, "Folder", self.buka_sesi, "#E8DDD5", INK, width=68).pack(side="right", padx=(4, 0))

        self.scale_pos = tk.Scale(bawah, from_=0, to=0, orient="horizontal", variable=self.posisi,
                                  label="Posisi frame", bg=PANEL, fg=INK, highlightthickness=0,
                                  command=self._geser_posisi)
        self.scale_pos.pack(fill="x", pady=(6, 0))

        ukur = tk.Frame(bawah, bg=PANEL); ukur.pack(fill="x", pady=(6, 0))
        tk.Checkbutton(ukur, text="Ukur 2 titik (preview lengkap)", variable=self.mode_ukur,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       command=self.ubah_mode_ukur).pack(side="left")
        ttk.Combobox(ukur, textvariable=self.cara_ukur, width=34, state="readonly",
                     values=(CARA_BIDANG, CARA_JARAK)).pack(side="left", padx=8)
        self.tombol(ukur, "Hapus garis", self._hapus_ukur, "#E8DDD5", INK).pack(side="right")
        tk.Label(bawah, textvariable=self.hasil_ukur, bg=PANEL, fg=ACCENT, justify="left",
                 font=("Segoe UI", 10, "bold"), wraplength=700).pack(anchor="w", pady=(4, 0))
        tk.Label(bawah, textvariable=self.durasi, bg=PANEL, fg=MUTED).pack(anchor="w", pady=(6, 4))
        self.scale_awal = tk.Scale(bawah, from_=0, to=0, orient="horizontal", variable=self.awal, label="Awal potongan", bg=PANEL, fg=INK, highlightthickness=0)
        self.scale_akhir = tk.Scale(bawah, from_=0, to=0, orient="horizontal", variable=self.akhir, label="Akhir potongan", bg=PANEL, fg=INK, highlightthickness=0)
        self.scale_awal.pack(fill="x"); self.scale_akhir.pack(fill="x")
        manual = tk.Frame(bawah, bg=PANEL); manual.pack(fill="x", pady=(4, 0))
        tk.Label(manual, text="Atau ketik frame", bg=PANEL, fg=MUTED).pack(side="left")
        ent_awal = tk.Spinbox(manual, from_=0, to=999999, textvariable=self.awal, width=8,
                               command=lambda: self.rentang_manual.set(True))
        ent_awal.pack(side="left", padx=(6, 2)); ent_awal.bind("<FocusOut>", lambda _e: self.rentang_manual.set(True))
        tk.Label(manual, text="s.d.", bg=PANEL, fg=MUTED).pack(side="left")
        ent_akhir = tk.Spinbox(manual, from_=0, to=999999, textvariable=self.akhir, width=8,
                                command=lambda: self.rentang_manual.set(True))
        ent_akhir.pack(side="left", padx=2); ent_akhir.bind("<FocusOut>", lambda _e: self.rentang_manual.set(True))
        self.tombol_ringkas(manual, "Awal = kini", self.tetapkan_awal_kini, "#E8DDD5", INK, width=92).pack(side="left", padx=(8, 2))
        self.tombol_ringkas(manual, "Akhir = kini", self.tetapkan_akhir_kini, "#E8DDD5", INK, width=92).pack(side="left", padx=2)
        self.tombol(bawah, "Simpan rentang potong (non-destruktif)", self.simpan_potong, GREEN).pack(fill="x", pady=(8, 0))
        tk.Label(bawah, text="Putar dan ekspor frame yang sedang dijeda dapat langsung dari RAW. Preview lengkap hanya diperlukan untuk slider/lompat frame dan pengukuran 3-D dari dua klik.", bg=PANEL, fg=MUTED, wraplength=720, justify="left").pack(anchor="w", pady=(6, 0))

    def ui_ekspor(self):
        f = tk.Frame(self.tab_ekspor, bg=BG); f.pack(fill="both", expand=True, padx=30, pady=28)
        b, i = self.card(f, "Ekspor frame RGB-D untuk dataset dan label YOLO") ; b.pack(fill="x")
        tk.Label(i, text="Ambil 1 frame setiap … frame sumber", bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w")
        tk.Spinbox(i, from_=1, to=9999, textvariable=self.langkah_ekspor, width=8).grid(row=0, column=1, sticky="w", padx=10)
        self.tombol(i, "Ekspor frame dari rentang pilihan", self.ekspor, GREEN).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        self.tombol(i, "Buat video MP4 rentang (opsional)", self.ekspor_video_rentang, BLUE).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.tombol(i, "Terapkan interval: bersihkan & ekspor ulang", self.buang_frame_belum_dilabeli, "#F3D8D4", INK).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.tombol(i, "Hapus semua hasil ekspor sesi ini", self.hapus_semua_ekspor, "#F3D8D4", INK).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        tk.Label(i, text="Default 10 berarti mengambil frame 0, 10, 20, dan seterusnya dari rentang pilihan sehingga sampel lebih berbeda. Tombol ‘Terapkan interval’ membuang hanya paket tanpa draft/label, lalu langsung ekspor ulang. Video MP4 hanya untuk ditonton/dibagikan; tidak diperlukan untuk labeling.", bg=PANEL, fg=MUTED, wraplength=750, justify="left").grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.label_ekspor = tk.Label(f, text="Belum ada ekspor dipilih.", bg=BG, fg=ACCENT, justify="left"); self.label_ekspor.pack(anchor="w", pady=(18, 0))

    def ui_label(self):
        f = tk.Frame(self.tab_label, bg=BG); f.pack(fill="both", expand=True, padx=14, pady=14)
        left = tk.Frame(f, bg=BG); left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.kanvas = KanvasLabel(left, self.hitung_ukuran, self._aktif_mask_berubah); self.kanvas.pack(fill="both", expand=True)
        # Panel tetap tanpa scrollbar: kontrol dipadatkan dan disusun vertikal
        # agar alur kerja terlihat sekaligus seperti panel aplikasi desktop.
        right = tk.Frame(f, bg=BG, width=390); right.pack(side="left", fill="y"); right.pack_propagate(False)
        b, i = self.card(right, "Pilih frame ekspor") ; b.pack(fill="x", pady=(0, 7))
        self.list_frame = tk.Listbox(i, height=5, bg="#FFF9F4", fg=INK, relief="flat", selectbackground=ACCENT_SOFT)
        self.list_frame.pack(fill="x"); self.list_frame.bind("<<ListboxSelect>>", lambda e: self.pilih_frame())
        self.tombol_ringkas(i, "Ekspor frame saat ini ke Label", self.ekspor_frame_kini_ke_label, GREEN, width=220).pack(fill="x", pady=(4, 0))
        nav = tk.Frame(i, bg=PANEL); nav.pack(fill="x", pady=(4, 0))
        self.tombol_ringkas(nav, "← Sebelum", lambda: self.pindah_frame_label(-1), "#E8DDD5", INK).pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.tombol_ringkas(nav, "Berikut →", lambda: self.pindah_frame_label(1), "#E8DDD5", INK).pack(side="left", fill="x", expand=True, padx=(2, 0))
        kelola = tk.Frame(i, bg=PANEL); kelola.pack(fill="x", pady=(4, 0))
        self.tombol_ringkas(kelola, "Sampahkan", self.toggle_sampah_frame, "#E8DDD5", INK).pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.tombol_ringkas(kelola, "Hapus permanen", self.hapus_frame_permanen, "#F3D8D4", INK).pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Checkbutton(i, text="\U0001f5d1 Tampilkan sampah frame", variable=self.tampil_sampah_frame,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       command=self.muat_frame).pack(anchor="w", pady=(6, 0))
        b, i = self.card(right, "Label dan depth") ; b.pack(fill="x", pady=(0, 7))
        self.rb_objek = tk.Radiobutton(i, variable=self.mode_label, value="objek", indicatoron=False, command=self.ganti_mode,
                                       bg="#FFF9F4", selectcolor=RED, activebackground=RED, fg=INK, relief="flat", pady=7)
        self.rb_acuan = tk.Radiobutton(i, variable=self.mode_label, value="acuan", indicatoron=False, command=self.ganti_mode,
                                       bg="#FFF9F4", selectcolor=BLUE, activebackground=BLUE, fg=INK, relief="flat", pady=7)
        self.rb_objek.pack(fill="x", pady=2); self.rb_acuan.pack(fill="x", pady=2)
        tk.Scale(i, from_=0, to=0.75, resolution=.05, orient="horizontal", variable=self.depth_alpha,
                 command=lambda _: self.ganti_depth(), label="Overlay depth (samar)", bg=PANEL, fg=INK,
                 highlightthickness=0, length=260).pack(fill="x", pady=(4, 0))
        self.tombol(i, "✨ Rekomendasi tangga: YOLO + SAM 2 + depth", self.usulkan_segmentasi, GREEN).pack(fill="x", pady=(10, 2))
        tk.Label(i, text="Tangga: YOLO memberi kandidat, SAM 2 GPU merapikan batas, depth ternormalisasi memeriksa serpihan/outlier. Batu/ramp memakai depth. Semua perubahan disimpan otomatis.",
                 bg=PANEL, fg=MUTED, wraplength=280, justify="left").pack(anchor="w", pady=(0, 6))
        edit = tk.Frame(i, bg=PANEL); edit.pack(fill="x", pady=(2, 0))
        self.tombol_ringkas(edit, "+ Mask", self.mulai_mask_baru, "#E8DDD5", INK, width=72).pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.tombol_ringkas(edit, "+ Titik", self.mulai_tambah_titik, "#E8DDD5", INK, width=72).pack(side="left", fill="x", expand=True, padx=2)
        self.tombol_ringkas(edit, "− Titik", self.kanvas.undo, "#E8DDD5", INK, width=72).pack(side="left", fill="x", expand=True, padx=(2, 0))
        nomor = tk.Frame(i, bg=PANEL); nomor.pack(fill="x", pady=(4, 0))
        tk.Label(nomor, text="Mask yang diedit", bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w")
        self.spin_nomor_mask = tk.Spinbox(nomor, from_=1, to=999, textvariable=self.nomor_mask, width=5,
                                           command=lambda: self.atur_nomor_mask(senyap=True))
        self.spin_nomor_mask.grid(row=0, column=1, sticky="w", padx=6)
        self.spin_nomor_mask.bind("<Return>", lambda _e: self.atur_nomor_mask(senyap=True))
        self.spin_nomor_mask.bind("<FocusOut>", lambda _e: self.atur_nomor_mask(senyap=True))
        self.tombol_ringkas(nomor, "Pilih mask", self.atur_nomor_mask, "#E8DDD5", INK, width=130).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        nomor.columnconfigure(0, weight=1)
        hapus = tk.Frame(i, bg=PANEL); hapus.pack(fill="x", pady=(5, 0))
        self.btn_hapus_objek = self.tombol_ringkas(hapus, "Hapus mask", self.kanvas.hapus_mask_aktif, "#F3D8D4", INK)
        self.btn_hapus_acuan = self.tombol_ringkas(hapus, "Buang saran", self.buang_rekomendasi, "#F3D8D4", INK)
        self.btn_hapus_objek.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.btn_hapus_acuan.pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Label(i, text="● AUTO-SAVE AKTIF — draft dan label YOLO tersimpan setelah Anda berhenti mengubah mask.",
                 bg=PANEL, fg=GREEN, wraplength=300, justify="left", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(8, 2))
        self.tombol(i, "Bangun folder dataset YOLO", self.bangun_yolo, GREEN).pack(fill="x", pady=(4, 2))
        tk.Label(i, text="Klik kanan titik untuk menghapus titik itu. Tarik titik untuk memindahkan.", bg=PANEL, fg=MUTED, wraplength=300, justify="left").pack(anchor="w", pady=(5, 0))
        tk.Label(i, textvariable=self.ukur_status, bg=PANEL, fg=INK, wraplength=300, justify="left", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(5, 0))
        self.perbarui_konteks_label()

    # ----- rekam -----
    def _mulai_kamera_async(self):
        threading.Thread(target=self._mulai_kamera, daemon=True).start()

    def simpan_preferensi(self):
        tulis_json(self.path_preferensi, {"kategori": self.kategori.get(), "split": self.split.get(),
                                          "kode_adegan": self.kode.get().strip()})

    def ganti_tab(self, _event=None):
        """Matikan stream yang tidak diperlukan agar labeling tetap ringan."""
        pada_rekam = self.tabs.select() == str(self.tab_rekam)
        if not pada_rekam and not self.sedang_rekam and self.cam.hidup:
            self.preview_diminta = False
            self._hentikan_render()
            self.cam.hentikan()
            self.preview_info = None
            self.btn_preview_kamera.configure(text="Aktifkan preview kamera")
            self.status.set("Kamera dihentikan saat labeling/tinjau agar aplikasi tetap ringan.")

    def toggle_preview_kamera(self):
        if self.sedang_rekam:
            return
        if self.cam.hidup:
            self.preview_diminta = False
            self._hentikan_render()
            self.cam.hentikan(); self.preview_info = None
            self.btn_preview_kamera.configure(text="Aktifkan preview kamera")
            self.status.set("Preview kamera dimatikan.")
            return
        self.preview_diminta = True
        self.btn_preview_kamera.configure(text="Menyalakan kamera…", state="disabled")
        self._mulai_kamera_async()

    def _mulai_kamera(self):
        try:
            self.cam.mulai(); self.preview_info = None
            self.cam.mulai_pompa(); self._mulai_render()
            self.q.put(("status", "D435 siap. Preview RGB dan depth aktif.")); self.q.put(("live", None))
            self.q.put(("preset_muat", self._baca_preset_perangkat()))
            self.q.put(("preview_kamera_siap", None))
        except BaseException as e:
            # BaseException, bukan Exception. Pembukaan kamera dulu bisa
            # melempar SystemExit lewat sys.exit() di lapisan bawah; itu lolos
            # dari "except Exception", thread ini mati diam-diam, dan tombol
            # tinggal "Menyalakan kamera..." tanpa satu pun pesan galat.
            # Apa pun yang terjadi di thread ini HARUS sampai ke UI.
            self.q.put(("error", f"D435 tidak dapat dibuka.\n\n{e}"))

    # ---------------- preset depth ----------------
    def _baca_preset_perangkat(self):
        """Query perangkat. WAJIB dari thread pekerja, bukan thread Tk.

        daftar_preset()/preset_sekarang() sendirian cuma ~0,5 ms, tapi keduanya
        butuh mutex perangkat librealsense yang sedang dipegang thread
        streaming. Dipanggil dari thread Tk, keduanya antre di belakang aliran
        dan sempat terukur memblokir jendela ratusan milidetik.
        """
        try:
            return self.cam.daftar_preset(), self.cam.preset_sekarang()
        except Exception:                                       # noqa: BLE001
            return [], None

    def _isi_preset(self, daftar, sekarang):
        """Hanya menyentuh widget Tk - tidak ada panggilan librealsense."""
        if not daftar:
            self.preset_pilih.set("(tidak didukung perangkat)")
            self.cb_preset.configure(values=(), state="disabled")
            return
        self.cb_preset.configure(values=daftar,
                                 state="disabled" if self.sedang_rekam else "readonly")
        self.preset_pilih.set(sekarang if sekarang in daftar else daftar[0])

    def _ganti_preset(self, _event=None):
        """Terapkan preset ke kamera yang sedang hidup, DI THREAD TERPISAH.

        visual_preset boleh disetel saat streaming, jadi aliran tidak perlu
        dinyalakan ulang - preview tidak terputus dan rekaman tidak terpotong.
        Tapi perangkatnya sendiri butuh ~2 detik untuk menulis setelan itu;
        kalau dipanggil dari thread Tk, jendela membeku selama itu.
        """
        if self.sedang_rekam or not self.cam.hidup:
            return
        nama = self.preset_pilih.get()
        self.cb_preset.configure(state="disabled")
        self.status.set(f"Menerapkan preset {nama}…")
        threading.Thread(target=self._preset_worker, args=(nama,), daemon=True).start()

    def _preset_worker(self, nama: str):
        try:
            terpakai = self.cam.ganti_preset(nama)
            self.preview_info = None        # intrinsics/meta dibaca ulang
            self.q.put(("preset_selesai", terpakai))
        except BaseException as e:          # noqa: BLE001
            # Baca ulang keadaan perangkat DI SINI, selagi masih di thread ini.
            self.q.put(("preset_gagal", (str(e), self._baca_preset_perangkat())))

    # ---------------- preview ----------------
    # Dulu SELURUH pekerjaan ini berjalan di thread Tk: wait_for_frames yang
    # memblokir, align depth->RGB, percentile pada 407k piksel, dua colormap,
    # dua resize PIL. Selama itu Tk tidak bisa menggambar ulang atau menjawab
    # klik - itulah jendela yang terasa patah-patah. Sekarang:
    #   pompa KameraRGBD  : menguras pipeline pada 30 fps (thread sendiri)
    #   _render_worker    : align + colormap + resize (thread sendiri, ~10 fps)
    #   live()            : hanya menempel gambar 360x300 yang sudah jadi
    # 20 Hz hasil pengukuran: preview nyata ~18 fps dengan jitter UI p99 21 ms.
    # Di 30 Hz worker tidak sanggup (hanya tercapai 22 fps) dan muncul lonjakan
    # event-loop sampai 2 detik, jadi justru terasa lebih patah.
    HZ_RENDER = 20

    def _mulai_render(self):
        if getattr(self, "_render_thread", None) and self._render_thread.is_alive():
            return
        self._render_henti = threading.Event()
        self._gambar_siap = None
        self._kunci_gambar = threading.Lock()
        self._render_thread = threading.Thread(target=self._render_worker, daemon=True)
        self._render_thread.start()

    def _hentikan_render(self):
        ev = getattr(self, "_render_henti", None)
        if ev is not None:
            ev.set()
        t = getattr(self, "_render_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._render_thread = None
        with getattr(self, "_kunci_gambar", threading.Lock()):
            self._gambar_siap = None

    def _render_worker(self):
        """Ubah frame terbaru jadi dua thumbnail RGB. Di luar thread Tk."""
        periode = 1.0 / self.HZ_RENDER
        while not self._render_henti.is_set():
            t0 = time.perf_counter()
            try:
                if self.cam.hidup and self.preview_diminta:
                    hasil = self._render_sekali()
                    if hasil is not None:
                        with self._kunci_gambar:
                            self._gambar_siap = hasil
            except Exception:                                   # noqa: BLE001
                pass                # preview tidak pernah boleh menjatuhkan app
            sisa = periode - (time.perf_counter() - t0)
            if sisa > 0:
                self._render_henti.wait(sisa)

    @staticmethod
    def _pita(img, teks, di_bawah=False):
        """Pita gelap semi-transparan + teks putih.

        Dulu teksnya digambar putih tebal lalu gelap tipis di atasnya. Di atas
        area terang - dinding putih pada RGB, ujung dekat colormap pada depth -
        garis putihnya melebur dengan latar dan tulisannya jadi tak terbaca.
        Pita gelap membuat kontrasnya tetap sama di atas apa pun.
        """
        h, w = img.shape[:2]
        # Tinggi pita dan huruf mengikuti ukuran gambar: preview kamera kini
        # dirender jauh lebih besar dari 360 px, dan pita 20 px/font 0,42 di
        # sana tidak lagi terbaca.
        tinggi = max(20, round(h * 0.062))
        skala_font = max(0.42, w / 1100)
        y0 = h - tinggi if di_bawah else 0
        petak = img[y0:y0 + tinggi]
        img[y0:y0 + tinggi] = cv2.addWeighted(petak, .30, np.zeros_like(petak), .70, 0)
        cv2.putText(img, teks, (8, y0 + tinggi - max(6, tinggi // 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, skala_font,
                    (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _silang(img):
        """Penanda titik yang jaraknya ditampilkan."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = max(7, min(h, w) // 40)
        for warna, tebal in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.line(img, (cx - r, cy), (cx + r, cy), warna, tebal, cv2.LINE_AA)
            cv2.line(img, (cx, cy - r), (cx, cy + r), warna, tebal, cv2.LINE_AA)

    @staticmethod
    def _jarak_tengah(z_meter):
        """Median depth pada petak kecil di tengah. Median supaya lubang dan
        derau satu piksel tidak mengubah angkanya."""
        h, w = z_meter.shape
        r = max(3, min(h, w) // 24)
        petak = z_meter[h // 2 - r:h // 2 + r + 1, w // 2 - r:w // 2 + r + 1]
        sah = petak[(petak > .15) & (petak < 10)]
        return float(np.median(sah)) if sah.size >= 5 else None

    def _atur_ukuran_preview(self, e):
        """Lebar render preview mengikuti lebar widgetnya (dipicu resize)."""
        w = max(240, min(960, e.width - 8))
        if abs(w - self._preview_w) > 24:
            self._preview_w = w

    def _render_sekali(self):
        _, cn, dn, ca, da = self.cam.ambil()
        rgb = np.asanyarray(ca.get_data())
        dep = np.asanyarray(da.get_data())
        if self.preview_info is None:
            self.preview_info = self.cam.info(cn, dn)
        skala = self.preview_info["depth_scale"]

        # Kecilkan DULU, baru hitung. Statistik pada 1/16 piksel praktis sama
        # dengan pada citra penuh, tapi percentile jadi jauh lebih murah.
        dep_kecil = dep[::2, ::2].astype(np.float32) * skala
        valid = (dep_kecil > .15) & (dep_kecil < 6)
        persen_sah = float(valid.mean() * 100)

        jarak = self._jarak_tengah(dep_kecil)
        teks_jarak = f"{jarak:.2f} m" if jarak is not None else "-- m"

        lebar = max(240, min(960, self._preview_w))
        vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        vis = cv2.resize(vis, (lebar, int(lebar * vis.shape[0] / vis.shape[1])),
                         interpolation=cv2.INTER_AREA)
        self._silang(vis)
        self._pita(vis, f"Titik tengah {teks_jarak}   Depth sah {persen_sah:.0f}%")

        dvis = np.zeros(dep_kecil.shape, dtype=np.uint8)
        lo = hi = None
        if valid.any():
            lo, hi = np.percentile(dep_kecil[valid], (2, 98))
            dvis = np.clip((dep_kecil - lo) * 255 / max(hi - lo, .01), 0, 255).astype(np.uint8)
            dvis[~valid] = 0
        dvis = cv2.cvtColor(cv2.applyColorMap(dvis, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        dvis = cv2.resize(dvis, (lebar, int(lebar * dvis.shape[0] / dvis.shape[1])),
                          interpolation=cv2.INTER_AREA)
        self._silang(dvis)
        self._pita(dvis, f"Titik tengah {teks_jarak}")
        # Skala warna diberi angka; "dekat -> jauh" saja tidak bisa dibaca
        # sebagai ukuran, dan rentangnya berubah tiap frame.
        self._pita(dvis, (f"biru {lo:.2f} m  ->  merah {hi:.2f} m"
                          if lo is not None else "tidak ada depth sah"),
                   di_bawah=True)
        return vis, dvis

    def live(self):
        """Hanya menempel gambar yang sudah jadi. Harus tetap ringan."""
        if not self.winfo_exists(): return
        pada_rekam = self.tabs.select() == str(self.tab_rekam)
        if not pada_rekam or not self.preview_diminta or not self.cam.hidup:
            self.after(250, self.live)
            return
        siap = None
        with getattr(self, "_kunci_gambar", threading.Lock()):
            if getattr(self, "_gambar_siap", None) is not None:
                siap, self._gambar_siap = self._gambar_siap, None
        if siap is not None:
            vis, dvis = siap
            # PhotoImage dibuat SEKALI lalu isinya ditimpa dengan paste().
            # Membuat PhotoImage baru tiap frame berarti mengalokasi dan
            # menyalin dua citra preview lalu menyuruh Tk menukar gambar label -
            # itu yang bikin live() sesekali makan 40-50 ms di thread UI.
            im_rgb, im_dep = Image.fromarray(vis), Image.fromarray(dvis)
            if (getattr(self, "preview_photo", None) is None
                    or self.preview_photo.width() != im_rgb.width
                    or self.preview_photo.height() != im_rgb.height):
                self.preview_photo = ImageTk.PhotoImage(im_rgb)
                self.depth_photo = ImageTk.PhotoImage(im_dep)
                self.label_live.configure(image=self.preview_photo, text="")
                self.label_depth.configure(image=self.depth_photo, text="")
            else:
                self.preview_photo.paste(im_rgb)
                self.depth_photo.paste(im_dep)
        self.after(1000 // self.HZ_RENDER, self.live)

    def toggle_rekam(self):
        if self.sedang_rekam:
            try:
                self._hentikan_render()
                self.cam.hentikan()
                assert self.sesi
                meta = baca_json(self.sesi / "source" / "session.json")
                meta["selesai_iso"] = datetime.now().isoformat(timespec="milliseconds")
                meta["status"] = "selesai"; tulis_json(self.sesi / "source" / "session.json", meta)
                self.sedang_rekam = False; self.btn_rekam.configure(text="● Mulai rekam", bg=RED)
                self.preview_diminta = False
                self.preview_info = None
                self.btn_preview_kamera.configure(text="Aktifkan preview kamera", state="normal")
                self.cb_preset.configure(state="readonly")
                self.status.set("Rekaman asli selesai. Tekan tab Tinjau untuk membuat preview bila diperlukan.")
                self.muat_daftar()
            except Exception as e: messagebox.showerror("Rekaman", str(e), parent=self)
            return
        kategori = self.kategori.get(); kode = nama_aman(self.kode.get()) or kategori.upper()
        self.simpan_preferensi()
        sesi = self.root_data / "rekaman" / kategori / f"{kode}_{stamp()}"
        (sesi / "source").mkdir(parents=True, exist_ok=False)
        try:
            self.cam.hentikan()
            self.cam.mulai(sesi / "source" / nama_rekaman())
            self.cam.mulai_pompa(); self._mulai_render()
            self.preview_diminta = True
            self.preview_info = None
            # info baru ditulis saat stop; bag memuat stream primer lengkap.
            tulis_json(sesi / "source" / "session.json", {"id": sesi.name, "kategori": kategori, "split": self.split.get(), "kode_adegan": kode,
                "mulai_iso": datetime.now().isoformat(timespec="milliseconds"), "status": "merekam",
                # nama sebenarnya, bukan tebakan: cam.mulai() boleh berganti
                # ekstensi kalau librealsense menolak yang pertama.
                "raw_bag": (self.cam.record_path or (sesi / "source" / nama_rekaman())).name,
                "raw_tidak_boleh_diubah": True, "fps_native": self.args.fps})
            self.sesi = sesi; self.sedang_rekam = True
            self.btn_rekam.configure(text="■ Selesai rekaman", bg=ACCENT)
            self.btn_preview_kamera.configure(text="Preview aktif saat rekam", state="disabled")
            # Preset dikunci selama merekam: satu rekaman harus punya satu
            # setelan depth, kalau tidak provenance-nya jadi tidak terbaca.
            self.cb_preset.configure(state="disabled")
            self.after(100, self.live)
            self.status.set(f"Merekam {sesi.name}. Ambil semua sudut yang diperlukan, lalu tekan Selesai rekaman.")
        except BaseException as e:
            # Sama alasannya dengan _mulai_kamera: sys.exit() dari lapisan bawah
            # dulu melewati "except Exception" dan membuat aplikasi tampak
            # membeku tanpa penjelasan.
            self.sedang_rekam = False
            self.preview_diminta = False
            self.btn_rekam.configure(text="\u25cf Mulai rekam", bg=RED)
            self.btn_preview_kamera.configure(text="Aktifkan preview kamera", state="normal")
            self.cb_preset.configure(state="readonly")
            self._buang_sesi_kosong(sesi)
            self.status.set("Rekaman tidak jadi dimulai.")
            messagebox.showerror("Rekaman", f"Tidak dapat memulai rekaman.\n\n{e}", parent=self)

    @staticmethod
    def _buang_sesi_kosong(sesi: Path) -> None:
        """Hapus folder sesi HANYA bila benar-benar masih kosong.

        Rekaman primer (raw.db3/raw.bag) tidak pernah dihapus aplikasi ini,
        jadi penghapusan dibatasi pada folder yang belum berisi satu berkas pun.
        """
        try:
            if any(p.is_file() for p in sesi.rglob("*")):
                return
            shutil.rmtree(sesi)
        except OSError:
            pass

    # ----- daftar / preview / potong -----
    def muat_daftar(self):
        self.list_sesi.delete(0, "end")
        self._map_sesi = []
        for p in self.daftar_sesi():
            if p.parent.name != self.filter_sesi.get():
                continue
            state = self._state(p)
            if state.get("di_sampah") != self.tampil_sampah.get(): continue
            ikon = "\U0001f5d1 " if state.get("di_sampah") else ""
            self._map_sesi.append(p); self.list_sesi.insert("end", f"{ikon}{p.parent.name}  |  {p.name}")
        if self._map_sesi:
            target = self._map_sesi.index(self.sesi) if self.sesi in self._map_sesi else 0
            self.list_sesi.selection_set(target); self.list_sesi.activate(target)
            self.after_idle(self.pilih_sesi)
        self.status.set(f"{len(self._map_sesi)} rekaman {self.filter_sesi.get()} ditampilkan.")

    def pilih_sesi(self):
        sel = self.list_sesi.curselection()
        if not sel: return
        self._tutup_video(); self._hapus_ukur()
        self.sesi = self._map_sesi[sel[0]]
        p = self.sesi / "derived" / "frame_index.csv"
        self.indeks = []
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f: self.indeks = list(csv.DictReader(f))
        n = len(self.indeks)
        self.awal.set(0); self.akhir.set(max(0, n - 1)); self.scale_awal.configure(to=max(0,n-1)); self.scale_akhir.configure(to=max(0,n-1))
        self.durasi.set(f"Sesi: {self.sesi.name}. {'Indeks siap: '+str(n)+' frame.' if n else 'Tekan Buat preview untuk membaca rekaman.'}")
        self.kategori_sesi.set(self.sesi.parent.name)
        self.kategori_baru.set(self.sesi.parent.name)
        self.perbarui_konteks_label(self.sesi.parent.name)
        self.posisi.set(0)
        if self._buka_video():
            self._tampilkan(0)
        else:
            self.kanvas_tinjau.delete("all")
            self.kanvas_tinjau.create_text(12, 12, anchor="nw", fill="white",
                font=("Segoe UI", 10), text="Tekan Putar untuk melihat RAW langsung.")
        self.muat_frame()  # frame ekspor langsung tersedia saat rekaman berganti

    def tetapkan_awal_kini(self):
        if getattr(self, "_bingkai_kini", None) is None:
            messagebox.showinfo("Putar rekaman", "Putar atau jeda rekaman pada frame yang ingin dijadikan awal.", parent=self); return
        self.awal.set(self.posisi.get()); self.rentang_manual.set(True)
        self.status.set(f"Awal ekspor: frame {self.awal.get()}.")

    def tetapkan_akhir_kini(self):
        if getattr(self, "_bingkai_kini", None) is None:
            messagebox.showinfo("Putar rekaman", "Putar atau jeda rekaman pada frame yang ingin dijadikan akhir.", parent=self); return
        self.akhir.set(self.posisi.get()); self.rentang_manual.set(True)
        self.status.set(f"Akhir ekspor: frame {self.akhir.get()}.")

    @staticmethod
    def pindahkan_sesi(sesi: Path, kategori_baru: str, kategori_lama_daftar=KATEGORI) -> Path:
        """Pindahkan satu sesi ke kategori lain. -> path baru.

        Hanya folder dan session.json yang berubah; raw.db3 dipindah apa adanya
        (rename di filesystem yang sama, isinya tidak dibaca maupun ditulis).

        Nama folder berformat <kode>_<stempel>. Kalau <kode> ternyata cuma nama
        kategori lama yang dihuruf-besarkan - yaitu kode otomatis saat kolom
        "kode adegan" dikosongkan - kode itu ikut diganti supaya nama folder
        tidak lagi menyesatkan. Kode yang Anda ketik sendiri dipertahankan.
        """
        if kategori_baru not in kategori_lama_daftar:
            raise ValueError(f"Kategori {kategori_baru!r} tidak dikenal.")
        lama = sesi.parent.name
        if lama == kategori_baru:
            return sesi
        nama = sesi.name
        kode, _, stempel = nama.rpartition("_")
        # stempel berbentuk YYYYmmdd_HHMMSS, jadi rpartition sekali belum cukup
        if "_" in kode and kode.rsplit("_", 1)[-1].isdigit() and len(stempel) == 6:
            kode, tanggal = kode.rsplit("_", 1)
            stempel = f"{tanggal}_{stempel}"
        kode_baru = kategori_baru.upper() if kode == lama.upper() else kode
        tujuan = sesi.parent.parent / kategori_baru / f"{kode_baru}_{stempel}"
        if tujuan.exists():
            raise FileExistsError(f"Sudah ada sesi bernama {tujuan.name} di {kategori_baru}.")
        tujuan.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sesi), str(tujuan))
        meta_path = tujuan / "source" / "session.json"
        meta = baca_json(meta_path, {})
        if meta:
            meta["kategori"] = kategori_baru
            meta["id"] = tujuan.name
            if kode_baru != kode:
                meta["kode_adegan"] = kode_baru
            meta.setdefault("riwayat_kategori", []).append(
                {"dari": lama, "ke": kategori_baru,
                 "nama_lama": nama, "waktu_iso": datetime.now().isoformat(timespec="seconds")})
            tulis_json(meta_path, meta)
        return tujuan

    def pindah_kategori(self):
        if self.sedang_rekam:
            messagebox.showinfo("Sedang merekam", "Selesaikan rekaman dahulu.", parent=self); return
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        baru = self.kategori_baru.get(); lama = self.sesi.parent.name
        if baru == lama:
            messagebox.showinfo("Kategori sama", f"Rekaman ini sudah berada di {lama}.", parent=self); return
        if not messagebox.askyesno("Pindahkan kategori",
                f"Pindahkan\n\n{self.sesi.name}\n\ndari '{lama}' ke '{baru}'?\n\n"
                "Isi rekaman tidak diubah sama sekali.", parent=self):
            return
        try:
            self.sesi = self.pindahkan_sesi(self.sesi, baru)
            self.kategori_sesi.set(baru)
            self.muat_daftar()
            self.status.set(f"Dipindahkan ke {baru}: {self.sesi.name}")
        except Exception as e:                                  # noqa: BLE001
            messagebox.showerror("Pindah kategori", str(e), parent=self)

    def buat_preview(self):
        if not self.sesi: messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        if getattr(self, "_preview_thread", None) and self._preview_thread.is_alive():
            messagebox.showinfo("Sedang berjalan",
                                "Preview untuk rekaman lain masih dibuat. Tunggu sampai selesai.",
                                parent=self); return
        self._hentikan_pemutar()
        self._preview_thread = threading.Thread(target=self._preview_worker, args=(self.sesi,), daemon=True)
        self._preview_thread.start()
        self.status.set("Membuat preview… membaca rekaman dari awal, ini memang lama.")

    @staticmethod
    def _depth_bgr(z_meter, lo: float, hi: float):
        """Depth -> citra warna BGR dengan rentang TETAP.

        Sengaja tidak memakai persentil per frame seperti preview kamera: di
        video tinjau, warna harus berarti jarak yang sama dari awal sampai
        akhir. Kalau rentangnya ikut berubah tiap frame, benda diam terlihat
        berganti warna dan videonya jadi menyesatkan.
        """
        sah = (z_meter > lo) & (z_meter < hi)
        skala = np.clip((z_meter - lo) * 255 / max(hi - lo, .01), 0, 255).astype(np.uint8)
        skala[~sah] = 0
        warna = cv2.applyColorMap(skala, cv2.COLORMAP_TURBO)
        warna[~sah] = (0, 0, 0)                 # lubang depth dibiarkan hitam
        return warna

    def _preview_worker(self, sesi: Path):
        try:
            derived = sesi / "derived"; derived.mkdir(exist_ok=True)
            # Depth ikut disimpan per frame. Tanpa ini, mengukur pada frame yang
            # sedang di-pause mustahil: seek pada .db3 terukur 10 detik sekali
            # panggil DAN mengembalikan frame yang salah, sedangkan mendekode
            # ulang dari awal butuh puluhan detik. Preview toh sudah mendekode
            # seluruh rekaman sekali, jadi menyimpannya di sini nyaris gratis.
            dir_depth = derived / "depth"
            if dir_depth.exists(): shutil.rmtree(dir_depth)
            dir_depth.mkdir()
            out = derived / "preview.mp4"; index = [] ; writer = None; skala_depth = None
            kamera = None
            # Laju video = fps SEBENARNYA dari timestamp rekaman. Dulu preview
            # selalu ditulis pada args.fps (30), padahal rekaman terukur
            # 29,9x fps - videonya jadi sedikit lebih cepat dari aslinya.
            try:
                fps_video = PembacaBag(self.bag(sesi)).fps_terukur() or float(self.args.fps)
            except Exception:                                   # noqa: BLE001
                fps_video = float(self.args.fps)
            for i, (_, color, depth, aligned, profile) in enumerate(PembacaBag(self.bag(sesi)).iter_frame()):
                if i % 25 == 0:
                    self.q.put(("status", f"Membuat preview… {i} frame terbaca "
                                          f"({sesi.name}). Rekaman dibaca dari awal, ini memang lama."))
                bgr = np.asanyarray(color.get_data())
                # Depth ikut masuk video. Sebelumnya preview hanya berisi RGB,
                # jadi tidak ada cara meninjau mutu depth sebelum ekspor -
                # padahal justru itu yang menentukan frame ini berguna atau tidak.
                if skala_depth is None:
                    skala_depth = float(profile.get_device().first_depth_sensor().get_depth_scale())
                z = np.asanyarray(aligned.get_data()).astype(np.float32) * skala_depth
                dep_bgr = self._depth_bgr(z, Z_MIN, Z_MAX)
                persen = float(((z > Z_MIN) & (z < Z_MAX)).mean() * 100)
                self._pita(bgr, f"RGB   frame {i}")
                self._pita(dep_bgr, f"DEPTH {Z_MIN:.2f}-{Z_MAX:.1f} m   sah {persen:.0f}%")
                bgr = np.hstack([bgr, dep_bgr])
                if writer is None:
                    h,w = bgr.shape[:2]; writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w,h))
                    kamera = self._info_profile(profile, color, depth)
                writer.write(bgr)
                # PNG 16-bit = lossless. Depth TIDAK boleh lewat kompresi lossy;
                # satu nilai Z16 yang bergeser berarti jarak yang salah.
                cv2.imwrite(str(dir_depth / f"{i:06d}.png"),
                            np.asanyarray(aligned.get_data()),
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
                index.append({"i": i, "frame": int(depth.get_frame_number()), "timestamp_ms": f"{depth.get_timestamp():.3f}"})
            if writer: writer.release()
            if kamera:
                kamera["fps_video_terukur"] = round(fps_video, 3)
                tulis_json(derived / "kamera.json", kamera)
            with (derived / "frame_index.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["i","frame","timestamp_ms"]); w.writeheader(); w.writerows(index)
            if not index: raise RuntimeError("Tidak menemukan pasangan RGB dan depth dalam rekaman.")
            cap = cv2.VideoCapture(str(out)); ok,img=cap.read(); cap.release()
            if ok: cv2.imwrite(str(derived / "thumbnail.jpg"), img)
            self.q.put(("preview_selesai", (sesi, len(index))))
        except Exception as e: self.q.put(("error", f"Gagal membuat preview: {e}"))

    def simpan_potong(self):
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        a,b = sorted((self.awal.get(), self.akhir.get()))
        if not self.indeks and not self.rentang_manual.get():
            messagebox.showinfo("Tentukan rentang", "Jeda video lalu gunakan tombol Awal/Akhir = frame kini, atau ketik nomor frame yang diinginkan.", parent=self); return
        edit = self.sesi / "edit"; edit.mkdir(exist_ok=True)
        tulis_json(edit / "rentang.json", {"awal_indeks": a, "akhir_indeks": b, "jumlah_total": len(self.indeks) or None,
            "non_destruktif": True, "dibuat_iso": datetime.now().isoformat(timespec="seconds")})
        self.status.set(f"Rentang {a}–{b} disimpan. Rekaman asli tetap utuh.")

    def toggle_sampah(self):
        if not self.sesi: messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        state=self._state(self.sesi); state["di_sampah"] = not state.get("di_sampah", False); state["waktu_ubah_iso"] = datetime.now().isoformat(timespec="seconds")
        self._tulis_state(self.sesi,state)
        # Data TIDAK dipindah maupun dihapus - hanya ditandai. Dengan begitu
        # indeks frame, ekspor, dan label sesi lain tidak terganggu sama
        # sekali, dan pemulihan tinggal membalik tanda.
        self.status.set("Rekaman dipindah ke tempat sampah. Centang 'Tampilkan isi tempat sampah' untuk melihat/memulihkannya."
                        if state["di_sampah"] else
                        "Rekaman dipulihkan dari tempat sampah.")
        self.muat_daftar()

    def hapus_preview_permanen(self):
        """Hapus hanya MP4/thumbnail turunan; RAW dan ekspor frame aman."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman terlebih dahulu.", parent=self); return
        preview = self.sesi / "derived" / "preview.mp4"
        thumbnail = self.sesi / "derived" / "thumbnail.jpg"
        if not preview.exists() and not thumbnail.exists():
            messagebox.showinfo("Tidak ada preview", "Preview video untuk rekaman ini belum ada.", parent=self); return
        if not messagebox.askyesno("Hapus preview video", "Hapus preview.mp4 dan thumbnail?\n\nRekaman RAW, depth, ekspor frame, dan label tidak dihapus.", parent=self):
            return
        self._tutup_video()
        for p in (preview, thumbnail):
            try:
                if p.exists(): p.unlink()
            except OSError as e:
                messagebox.showerror("Hapus preview", str(e), parent=self); return
        self.kanvas_tinjau.delete("all")
        self.kanvas_tinjau.create_text(12, 12, anchor="nw", fill="white", font=("Segoe UI", 10),
                                       text="Preview dihapus. Tekan Putar untuk membaca RAW langsung.")
        self.status.set("Preview video dihapus permanen. RAW dan frame ekspor tetap ada.")

    def hapus_sesi_permanen(self):
        """Hapus satu folder sesi terpilih, termasuk RAW, hanya setelah konfirmasi."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman yang akan dihapus terlebih dahulu.", parent=self); return
        target = self.sesi.resolve()
        rekaman = (self.root_data / "rekaman").resolve()
        if rekaman not in target.parents:
            messagebox.showerror("Target tidak aman", "Folder terpilih bukan sesi rekaman yang valid.", parent=self); return
        if not messagebox.askyesno("Hapus rekaman permanen", f"Hapus PERMANEN seluruh sesi ini?\n\n{target.name}\n\nRAW, preview, ekspor frame, gambar, mask, dan label akan hilang dan tidak dapat dipulihkan.", icon="warning", parent=self):
            return
        self._tutup_video()
        try:
            shutil.rmtree(target)
            self.sesi = None; self.indeks = []; self.frame_paths = []; self.label_path = None; self.label_info = None
            self.list_frame.delete(0, "end"); self.kanvas.delete("all")
            self.kategori_sesi.set("(rekaman dihapus)")
            self.muat_daftar()
            self.status.set("Rekaman dan seluruh turunannya dihapus permanen.")
        except OSError as e:
            messagebox.showerror("Hapus rekaman", str(e), parent=self)

    def buka_sesi(self):
        if self.sesi: self._buka_folder(self.sesi)

    # ----- pemutar preview DI DALAM aplikasi -----
    # Dulu tombol ini hanya memanggil xdg-open, jadi videonya selalu terbuka di
    # pemutar luar pada jendela terpisah. Sekarang diputar di kanvas Tinjau:
    # dekode di thread, thread Tk hanya menempel gambar.
    KECEPATAN = {"0.25x": .25, "0.5x": .5, "1x": 1., "2x": 2., "4x": 4., "8x": 8.}

    def _pemutar_jalan(self) -> bool:
        t = getattr(self, "_pemutar_thread", None)
        return t is not None and t.is_alive()

    def _buka_video(self) -> bool:
        """Siapkan VideoCapture untuk sesi terpilih. -> berhasil?"""
        if self._video_cap is not None:
            return True
        if not self.sesi:
            return False
        preview = self.sesi / "derived" / "preview.mp4"
        if not preview.exists():
            return False
        cap = cv2.VideoCapture(str(preview))
        if not cap.isOpened():
            cap.release()
            return False
        self._video_cap = cap
        self._n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps_video = cap.get(cv2.CAP_PROP_FPS) or float(self.args.fps)
        self.scale_pos.configure(to=max(0, self._n_frame - 1))
        return True

    def _tutup_video(self):
        self._hentikan_pemutar()
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        self._n_frame = 0
        self._depth_frame_cache = (None, None)

    def putar_preview(self):
        if self._pemutar_jalan():
            self._hentikan_pemutar(); return
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        self._pemutar_henti = threading.Event()
        # Jangan paksa pengguna menunggu transkode seluruh rekaman hanya untuk
        # melihat isi RAW. Bila preview.mp4 belum ada, baca .bag/.db3 berurutan
        # langsung; preview lengkap tetap opsional untuk seek dan ekspor rentang.
        langsung = not self._buka_video()
        self._pemutar_thread = threading.Thread(target=self._pemutar_raw_worker if langsung else self._pemutar_worker,
                                                daemon=True)
        self._pemutar_thread.start()
        self.btn_putar.configure(text="\u23f8 Jeda")
        self.after(20, self._gambar_bingkai)
        if langsung:
            self.status.set("Memutar RAW langsung. Siapkan indeks & preview lengkap bila ingin lompat/potong/ukur presisi.")

    def _hentikan_pemutar(self):
        ev = getattr(self, "_pemutar_henti", None)
        if ev is not None: ev.set()
        t = getattr(self, "_pemutar_thread", None)
        if t is not None and t.is_alive(): t.join(timeout=1.5)
        self._pemutar_thread = None
        try: self.btn_putar.configure(text="\u25b6 Putar")
        except tk.TclError: pass

    def _pemutar_worker(self):
        while not self._pemutar_henti.is_set():
            t0 = time.perf_counter()
            laju = self.KECEPATAN.get(self.kecepatan.get(), 1.0)
            with self._kunci_video:
                ok, bgr = self._video_cap.read()
                if not ok:
                    self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                i = int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            with self._kunci_bingkai:
                self._bingkai_siap = (bgr, i)
            sisa = 1.0 / max(1.0, self._fps_video * laju) - (time.perf_counter() - t0)
            if sisa > 0: self._pemutar_henti.wait(sisa)

    def _pemutar_raw_worker(self):
        """Playback pertama yang cepat tanpa membuat MP4/PNG seluruh sesi."""
        try:
            skala_depth = None
            for i, (_, color, depth, aligned, profile) in enumerate(PembacaBag(self.bag(self.sesi)).iter_frame()):
                if self._pemutar_henti.is_set():
                    break
                t0 = time.perf_counter()
                bgr = np.asanyarray(color.get_data()).copy()
                if skala_depth is None:
                    skala_depth = float(profile.get_device().first_depth_sensor().get_depth_scale())
                z = np.asanyarray(aligned.get_data()).astype(np.float32) * skala_depth
                dep = self._depth_bgr(z, Z_MIN, Z_MAX)
                self._pita(bgr, f"RGB  RAW frame {i}")
                self._pita(dep, f"DEPTH {Z_MIN:.2f}-{Z_MAX:.1f} m")
                with self._kunci_bingkai:
                    self._bingkai_siap = (np.hstack([bgr, dep]), i)
                laju = self.KECEPATAN.get(self.kecepatan.get(), 1.0)
                sisa = 1.0 / max(1.0, float(self.args.fps) * laju) - (time.perf_counter() - t0)
                if sisa > 0:
                    self._pemutar_henti.wait(sisa)
        except Exception as e:  # pembacaan RAW gagal harus terlihat di UI
            self.q.put(("status", f"Playback RAW berhenti: {e}"))

    def _ambil_bingkai(self, i: int):
        """Baca satu frame tertentu (dipakai saat jeda/geser). -> BGR atau None."""
        if not self._buka_video(): return None
        with self._kunci_video:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(i, self._n_frame - 1)))
            ok, bgr = self._video_cap.read()
        return bgr if ok else None

    def _langkah(self, arah: int):
        self._hentikan_pemutar()
        self.posisi.set(max(0, min(self.posisi.get() + arah, max(0, self._n_frame - 1))))
        self._tampilkan(self.posisi.get())

    def _geser_posisi(self, _v=None):
        if self._pemutar_jalan(): return          # saat main, skala mengikuti video
        self._tampilkan(self.posisi.get())

    def _tampilkan(self, i: int):
        bgr = self._ambil_bingkai(i)
        if bgr is not None:
            self._bingkai_kini = bgr
            self._gambar_ulang_kanvas()

    def _gambar_bingkai(self):
        if not self.winfo_exists(): return
        if not self._pemutar_jalan():
            try: self.btn_putar.configure(text="\u25b6 Putar")
            except tk.TclError: pass
            return
        paket = None
        with self._kunci_bingkai:
            if self._bingkai_siap is not None:
                paket, self._bingkai_siap = self._bingkai_siap, None
        if paket is not None:
            bgr, i = paket
            self._bingkai_kini = bgr
            self.posisi.set(i)
            self._gambar_ulang_kanvas()
        self.after(15, self._gambar_bingkai)

    # ----- gambar ke kanvas + garis ukur -----
    def _gambar_ulang_kanvas(self):
        bgr = getattr(self, "_bingkai_kini", None)
        if bgr is None: return
        kw = max(1, self.kanvas_tinjau.winfo_width()); kh = max(1, self.kanvas_tinjau.winfo_height())
        h, w = bgr.shape[:2]
        skala = min(kw / w, kh / h)
        self._skala_tampil = skala
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if skala != 1.0:
            rgb = cv2.resize(rgb, (max(1,int(w*skala)), max(1,int(h*skala))),
                             interpolation=cv2.INTER_AREA)
        im = Image.fromarray(rgb)
        if (getattr(self, "video_photo", None) is None
                or self.video_photo.width() != im.width or self.video_photo.height() != im.height):
            self.video_photo = ImageTk.PhotoImage(im)
        else:
            self.video_photo.paste(im)
        self.kanvas_tinjau.delete("all")
        self.kanvas_tinjau.create_image(0, 0, anchor="nw", image=self.video_photo)
        for (p1, p2, teks) in self.garis_ukur:
            self._gambar_garis(p1, p2, teks)
        for t in self.titik_ukur:
            x, y = t[0]*skala, t[1]*skala
            self.kanvas_tinjau.create_oval(x-5, y-5, x+5, y+5, outline="#FFD400", width=2)
        info = f"frame {self.posisi.get()} / {max(0,self._n_frame-1)}"
        if self.mode_ukur.get():
            info += "   MODE UKUR: klik titik 1 lalu titik 2"
        self.kanvas_tinjau.create_text(10, 8, anchor="nw", fill="#FFD400",
                                       font=("Segoe UI", 10, "bold"), text=info)

    def _gambar_garis(self, p1, p2, teks):
        sk = self._skala_tampil
        x1, y1, x2, y2 = p1[0]*sk, p1[1]*sk, p2[0]*sk, p2[1]*sk
        self.kanvas_tinjau.create_line(x1, y1, x2, y2, fill="#000000", width=5)
        self.kanvas_tinjau.create_line(x1, y1, x2, y2, fill="#FFD400", width=2)
        for x, y in ((x1,y1),(x2,y2)):
            self.kanvas_tinjau.create_oval(x-4, y-4, x+4, y+4, fill="#FFD400", outline="black")
        mx, my = (x1+x2)/2, (y1+y2)/2
        self.kanvas_tinjau.create_text(mx+1, my-13, fill="black", font=("Segoe UI", 11, "bold"), text=teks)
        self.kanvas_tinjau.create_text(mx, my-14, fill="#FFD400", font=("Segoe UI", 11, "bold"), text=teks)

    # ----- pengukuran dari dua klik -----
    def _muat_kamera(self):
        """Intrinsics + depth_scale sesi ini, ditulis saat preview dibuat."""
        if not self.sesi: return None
        k = baca_json(self.sesi / "derived" / "kamera.json", {})
        return k or None

    def _muat_depth(self, i: int):
        """Depth selaras-RGB untuk frame ke-i, dari derived/depth/. -> uint16."""
        if self._depth_frame_cache[0] == i:
            return self._depth_frame_cache[1]
        f = self.sesi / "derived" / "depth" / f"{i:06d}.png"
        if not f.exists(): return None
        d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        self._depth_frame_cache = (i, d)
        return d

    @staticmethod
    def _z_di(depth, x: int, y: int, skala: float, r: int = 4):
        """Kedalaman di satu piksel, median petak kecil.

        Median, bukan nilai piksel tunggal: depth D435 berlubang, dan satu
        piksel nol akan membuat jarak yang dihitung ngawur total.
        """
        h, w = depth.shape[:2]
        y0, y1 = max(0, y-r), min(h, y+r+1)
        x0, x1 = max(0, x-r), min(w, x+r+1)
        petak = depth[y0:y1, x0:x1].astype(np.float32) * skala
        sah = petak[(petak > .15) & (petak < 10)]
        return float(np.median(sah)) if sah.size >= 3 else None

    def ubah_mode_ukur(self):
        """Jangan biarkan checkbox ukur tampak aktif tetapi klik tidak bekerja."""
        if self.mode_ukur.get() and (not self.sesi or not (self.sesi / "derived" / "kamera.json").exists()):
            self.mode_ukur.set(False)
            self.hasil_ukur.set("Pengukuran dua titik memerlukan preview lengkap karena depth frame harus dapat dipanggil ulang. Untuk cara cepat, buat mask merah + biru di Label & Ukur; tinggi dihitung otomatis.")
        self._gambar_ulang_kanvas()

    def _klik_kanvas(self, e):
        if not self.mode_ukur.get(): return
        bgr = getattr(self, "_bingkai_kini", None)
        if bgr is None:
            # Dulu klik diam saja tanpa penjelasan - terlihat seperti mode
            # ukur rusak padahal preview-nya yang belum ada.
            self.hasil_ukur.set("Belum ada gambar. Putar RAW dahulu; untuk ukur dua titik gunakan preview lengkap, atau ukur otomatis dari mask di Label & Ukur.")
            return
        sk = getattr(self, "_skala_tampil", 1.0) or 1.0
        x, y = int(e.x / sk), int(e.y / sk)
        h, w = bgr.shape[:2]
        if not (0 <= x < w and 0 <= y < h): return
        # Video berdampingan RGB | DEPTH. Panel kanan koordinatnya sama dengan
        # panel kiri, cuma digeser - jadi klik di panel mana pun boleh.
        lebar_panel = w // 2
        if x >= lebar_panel: x -= lebar_panel
        self.titik_ukur.append((x, y))
        if len(self.titik_ukur) == 2:
            if self.cara_ukur.get() == CARA_BIDANG:
                self._ukur_ke_bidang()
            else:
                self._ukur_dua_titik()
        else:
            self.hasil_ukur.set(
                "Klik 1 = pada BIDANG ACUAN (mis. tapakan bawah). Klik 2 = titik yang diukur tingginya."
                if self.cara_ukur.get() == CARA_BIDANG else "Titik 1 ditandai. Klik titik kedua.")
        self._gambar_ulang_kanvas()

    @staticmethod
    def _masker_petak(bentuk, x: int, y: int, r: int):
        """Masker kotak di sekitar satu klik, untuk diumpankan ke ukur()."""
        m = np.zeros(bentuk[:2], dtype=bool)
        h, w = bentuk[:2]
        m[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)] = True
        return m

    def _intrinsics(self, k) -> dict:
        i = k["intrinsics_rgb_native"]
        return {"depth_scale": float(k["depth_scale"]), "fx": i["fx"], "fy": i["fy"],
                "cx": i["ppx"], "cy": i["ppy"]}

    def _ukur_ke_bidang(self):
        """Metode skripsi: RANSAC bidang acuan lalu jarak tegak lurus.

        Klik pertama menandai BIDANG ACUAN - petak di sekitarnya dipasangi
        bidang dengan pasang_bidang() dari geometri.py. Klik kedua menandai
        objek yang diukur. Perhitungannya diserahkan ke pengukuran_objek.ukur(),
        fungsi yang sama yang dipakai tab Label, supaya angka di sini dan di
        sana tidak mungkin berbeda metode.
        """
        (x1, y1), (x2, y2) = self.titik_ukur
        self.titik_ukur.clear()
        k = self._muat_kamera()
        depth = self._muat_depth(self.posisi.get()) if k else None
        if not k or depth is None:
            self.hasil_ukur.set("Depth/intrinsics frame ini belum tersimpan. "
                                "Tekan Buat preview dengan versi terbaru."); return
        mask_acuan = self._masker_petak(depth.shape, x1, y1, 40)   # bidang: petak 81x81
        mask_objek = self._masker_petak(depth.shape, x2, y2, 15)   # objek : petak 31x31
        # subsample=1: petak objek hanya 31x31, sesudah erosi 7x7 dan saring
        # depth lubang, subsample=2 kerap menyisakan <100 titik sehingga ukur()
        # selalu menolak ("Titik depth objek terlalu sedikit").
        hasil = ukur(mask_objek, mask_acuan, depth, self._intrinsics(k), subsample=1)
        if not hasil.get("ok"):
            self.hasil_ukur.set(f"Tidak dapat diukur: {hasil.get('alasan')}  "
                                "(klik 1 harus pada permukaan datar yang cukup luas)")
            self._gambar_ulang_kanvas(); return
        teks = f"{hasil['tinggi_cm']:.1f} cm"
        self.garis_ukur.append(((x1,y1),(x2,y2), teks))
        self.hasil_ukur.set(
            f"Tinggi ke bidang acuan {hasil['tinggi_cm']:.1f} cm   "
            f"(p05 {hasil['elevasi_p05_cm']:.1f} cm)   |   "
            f"RMS bidang {hasil['rms_bidang_acuan_mm']:.1f} mm   "
            f"titik bidang {hasil['titik_bidang_acuan']}   titik objek {hasil['titik_objek']}   |   "
            f"jarak objek {hasil['jarak_median_objek_m']:.2f} m")

    def _ukur_dua_titik(self):
        (x1, y1), (x2, y2) = self.titik_ukur
        self.titik_ukur.clear()
        k = self._muat_kamera()
        if not k:
            self.hasil_ukur.set("kamera.json tidak ada. Buat ulang preview agar "
                                "intrinsics dan depth ikut tersimpan."); return
        depth = self._muat_depth(self.posisi.get())
        if depth is None:
            self.hasil_ukur.set("Depth frame ini belum tersimpan. Buat ulang preview "
                                "dengan versi terbaru agar folder derived/depth/ terisi."); return
        sk = float(k["depth_scale"]); intr = k["intrinsics_rgb_native"]
        z1 = self._z_di(depth, x1, y1, sk); z2 = self._z_di(depth, x2, y2, sk)
        if z1 is None or z2 is None:
            self.hasil_ukur.set("Tidak ada depth sah di salah satu titik "
                                "(berlubang atau di luar 0,15-10 m). Pilih titik lain."); return
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
        P1 = np.array([(x1-cx)*z1/fx, (y1-cy)*z1/fy, z1])
        P2 = np.array([(x2-cx)*z2/fx, (y2-cy)*z2/fy, z2])
        d = P2 - P1
        jarak = float(np.linalg.norm(d))
        # Galat stereo D435 tumbuh kuadratik; dilaporkan supaya angkanya tidak
        # dibaca lebih presisi daripada yang sebenarnya bisa dijamin sensor.
        galat = sum((z*z*0.1/(0.05*fx)) for z in (z1, z2))
        teks = f"{jarak*100:.1f} cm"
        self.garis_ukur.append(((x1,y1),(x2,y2), teks))
        self.hasil_ukur.set(
            f"Jarak 3-D {jarak*100:.1f} cm  (± {galat*100:.1f} cm)   |   "
            f"tinggi (sumbu Y) {abs(d[1])*100:.1f} cm   lebar (X) {abs(d[0])*100:.1f} cm   "
            f"kedalaman (Z) {abs(d[2])*100:.1f} cm   |   titik pada {z1:.2f} m dan {z2:.2f} m")

    def _hapus_ukur(self):
        self.titik_ukur.clear(); self.garis_ukur.clear()
        self.hasil_ukur.set(""); self._gambar_ulang_kanvas()

    # ----- ekspor -----
    def perbarui_konteks_label(self, kategori=None):
        """Istilah mask mengikuti adegan agar batu/ramp tidak terasa tak didukung."""
        kategori = kategori or (self.sesi.parent.name if self.sesi else self.kategori.get())
        peta = KELAS_MODE.get(kategori, KELAS_MODE["tangga_naik"])
        objek, acuan = peta["objek"], peta["acuan"]
        if hasattr(self, "rb_objek"):
            self.rb_objek.configure(text=f"Mask {objek.replace('_', ' ')}  (MERAH)")
            self.rb_acuan.configure(text=f"Mask {acuan.replace('_', ' ')} / bidang acuan  (BIRU)")

    def rentang(self):
        if not self.sesi: return (0, -1)
        x=baca_json(self.sesi/"edit"/"rentang.json", {})
        if x: return int(x["awal_indeks"]), int(x["akhir_indeks"])
        if self.rentang_manual.get():
            return tuple(sorted((self.awal.get(), self.akhir.get())))
        return (0, len(self.indeks)-1)

    def fps_asli(self, sesi: Path) -> float:
        """FPS SEBENARNYA dari timestamp rekaman, bukan dari --fps.

        --fps hanya yang DIMINTA saat merekam; yang benar-benar tercatat bisa
        berbeda. Terukur pada rekaman Anda: 29,92 dan 29,95 fps, dengan satu
        frame hilang di tiap sesi. Memakai 30 sebagai pembagi membuat langkah
        pengambilan sedikit meleset, dan makin jauh untuk rentang yang panjang.
        """
        try:
            baris = list(csv.DictReader((sesi/"derived"/"frame_index.csv").open(newline="", encoding="utf-8")))
            ts = [float(b["timestamp_ms"]) for b in baris]
            if len(ts) > 1 and ts[-1] > ts[0]:
                return (len(ts) - 1) / ((ts[-1] - ts[0]) / 1000.0)
        except Exception:                                       # noqa: BLE001
            pass
        return float(self.args.fps)

    def ekspor(self):
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        a, b = self.rentang()
        if b < a:
            messagebox.showinfo("Tentukan rentang", "Jeda video lalu tetapkan Awal dan Akhir = frame kini, atau ketik nomor frame.", parent=self); return
        if getattr(self, "_ekspor_thread", None) and self._ekspor_thread.is_alive():
            messagebox.showinfo("Sedang berjalan", "Ekspor lain masih berjalan.", parent=self); return
        sesi = self.sesi
        langkah = max(1, int(self.langkah_ekspor.get()))
        self._ekspor_thread = threading.Thread(target=self._ekspor_worker, args=(sesi, langkah), daemon=True)
        self._ekspor_thread.start()
        self.status.set(f"Mengekspor setiap {langkah} frame sumber… frame yang sudah ada akan dilewati.")

    @staticmethod
    def _punya_label(frame: Path) -> bool:
        """Draft maupun label jadi bukti frame pernah disentuh pengguna."""
        label = frame / "label_yolo_seg.txt"
        if label.exists() and label.read_text(encoding="utf-8").strip():
            return True
        draft = baca_json(frame / "label_draft.json", {})
        return any(poly for daftar in draft.get("poligon", {}).values() for poly in daftar)

    def buang_frame_belum_dilabeli(self):
        """Reset aman untuk mengganti interval tanpa menghapus kerja labeling."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman terlebih dahulu.", parent=self); return
        if getattr(self, "_ekspor_thread", None) and self._ekspor_thread.is_alive():
            messagebox.showinfo("Ekspor berjalan", "Tunggu ekspor yang sedang berjalan selesai.", parent=self); return
        kandidat = [p for p in self.daftar_frame_ekspor() if not self._punya_label(p)]
        if not kandidat:
            messagebox.showinfo("Tidak ada yang dibuang", "Semua frame ekspor sudah memiliki draft/label, atau belum ada frame.", parent=self); return
        if not messagebox.askyesno("Buang frame belum dilabeli",
                                   f"Buang {len(kandidat)} paket frame yang BELUM memiliki draft/label?\n\n"
                                   "Frame yang sudah Anda segmentasi tetap dipertahankan. RAW tidak disentuh.",
                                   icon="warning", parent=self):
            return
        gagal = []
        for p in kandidat:
            try:
                shutil.rmtree(p)
            except OSError:
                gagal.append(p.name)
        self.label_path = None; self.label_info = None
        self.muat_frame()
        if gagal:
            self.status.set(f"{len(kandidat) - len(gagal)} frame dibuang; {len(gagal)} gagal dihapus. Ekspor ulang dibatalkan.")
            return
        self.status.set(f"{len(kandidat)} frame belum dilabeli dibuang. Mengekspor ulang setiap {self.langkah_ekspor.get()} frame…")
        self.ekspor()

    def hapus_semua_ekspor(self):
        """Buang seluruh turunan ekspor sesi aktif, tanpa menyentuh RAW."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman terlebih dahulu.", parent=self); return
        root = self.sesi / "exports"
        if not root.exists():
            messagebox.showinfo("Belum ada ekspor", "Sesi ini belum memiliki hasil ekspor.", parent=self); return
        if getattr(self, "_ekspor_thread", None) and self._ekspor_thread.is_alive():
            messagebox.showinfo("Ekspor berjalan", "Tunggu ekspor yang sedang berjalan selesai.", parent=self); return
        if not messagebox.askyesno("Hapus semua ekspor", "Hapus SEMUA frame ekspor, gambar, depth, IR, mask, label, dan video ekspor sesi ini?\n\nRekaman RAW tidak dihapus. Anda dapat mengekspor ulang dari awal.", icon="warning", parent=self):
            return
        try:
            shutil.rmtree(root)
            self.frame_paths = []; self.label_path = None; self.label_info = None
            self.list_frame.delete(0, "end"); self.kanvas.delete("all")
            self.status.set("Semua hasil ekspor dihapus. RAW tetap aman dan siap diekspor ulang.")
        except OSError as e:
            messagebox.showerror("Hapus ekspor", str(e), parent=self)

    def ekspor_frame_kini_ke_label(self):
        if not self.sesi or getattr(self, "_bingkai_kini", None) is None:
            messagebox.showinfo("Pilih frame video", "Pilih rekaman lalu putar atau tampilkan satu frame terlebih dahulu.", parent=self)
            return
        # PembacaBag memberi indeks deterministik yang sama untuk playback RAW
        # dan ekspor, jadi frame yang sedang dijeda dapat diekspor tanpa harus
        # lebih dulu membangun MP4 dan indeks lengkap.
        i = max(0, self.posisi.get())
        if getattr(self, "_ekspor_thread", None) and self._ekspor_thread.is_alive():
            messagebox.showinfo("Ekspor berjalan", "Tunggu ekspor yang sedang berjalan selesai.", parent=self); return
        self._label_setelah_ekspor = i
        self._ekspor_thread = threading.Thread(target=self._ekspor_worker, args=(self.sesi, 1, (i, i)), daemon=True)
        self._ekspor_thread.start()
        self.status.set(f"Menyalin paket RAW frame {i} ke Label & Ukur…")

    def ekspor_video_rentang(self):
        if not self.sesi or not self.indeks or not (self.sesi / "derived" / "preview.mp4").exists():
            messagebox.showinfo("Butuh preview lengkap", "Siapkan indeks & preview lengkap dahulu agar video bisa diekspor dari frame awal hingga akhir yang dipilih.", parent=self); return
        a, b = self.rentang()
        threading.Thread(target=self._ekspor_video_worker, args=(self.sesi, a, b), daemon=True).start()
        self.status.set(f"Mengekspor video frame {a}–{b}…")

    def _ekspor_video_worker(self, sesi: Path, a: int, b: int):
        try:
            cap = cv2.VideoCapture(str(sesi / "derived" / "preview.mp4"))
            fps = cap.get(cv2.CAP_PROP_FPS) or float(self.args.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            out_dir = sesi / "exports"; out_dir.mkdir(exist_ok=True)
            out = out_dir / f"preview_{a:06d}_{b:06d}.mp4"
            writer = None
            for _i in range(a, b + 1):
                ok, frame = cap.read()
                if not ok: break
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            cap.release()
            if writer is not None: writer.release()
            self.q.put(("status", f"Video rentang selesai: {out}"))
        except Exception as e:
            self.q.put(("error", f"Ekspor video gagal: {e}"))

    @staticmethod
    def _info_profile(profile, color, depth) -> dict:
        dev=profile.get_device(); ds=dev.first_depth_sensor(); cp=color.profile.as_video_stream_profile(); dp=depth.profile.as_video_stream_profile(); ci=cp.intrinsics; di=dp.intrinsics; ex=dp.get_extrinsics_to(cp)
        return {"depth_scale":float(ds.get_depth_scale()), "intrinsics_rgb_native":{"width":ci.width,"height":ci.height,"fx":ci.fx,"fy":ci.fy,"ppx":ci.ppx,"ppy":ci.ppy,"coeffs":list(ci.coeffs)}, "intrinsics_depth_native":{"width":di.width,"height":di.height,"fx":di.fx,"fy":di.fy,"ppx":di.ppx,"ppy":di.ppy,"coeffs":list(di.coeffs)}, "extrinsics_depth_ke_rgb":{"rotation_row_major":list(ex.rotation),"translation_meter":list(ex.translation)}}

    @staticmethod
    def _rencana_picks(sesi: Path, a: int, b: int, langkah: int) -> set[int]:
        """Indeks sumber dengan jarak tetap: 0, 10, 20 … bila langkah=10."""
        del sesi  # Rentang memakai indeks sumber deterministik dari PembacaBag.
        return set(range(a, b + 1, max(1, langkah)))

    def _ekspor_worker(self, sesi: Path, langkah: int, rentang_paksa: tuple[int, int] | None = None):
        """Ekspor yang BISA DILANJUTKAN dan tidak pernah menggandakan frame.

        Tiga hal yang membuatnya begitu:
          * Nama folder ekspor tetap "fps_<N>", tanpa stempel waktu. Dulu tiap
            penekanan tombol membuat folder baru, jadi mengekspor ulang berarti
            satu set frame yang sama tersimpan dua kali.
          * Nama folder frame diambil dari INDEKS SUMBER, bukan nomor urut
            ekspor. Frame yang sama selalu mendarat di nama yang sama, jadi
            menjalankan ulang tinggal melewatinya - frame yang sudah Anda
            labeli tidak tertimpa, dan frame yang Anda buang ke sampah TIDAK
            dibangkitkan lagi (foldernya masih ada, hanya ditandai).
          * Bila seluruh frame rencana sudah lengkap di disk, dekode rekaman
            dilewati sama sekali - ekspor ulang selesai dalam sekejap sebagai
            "0 baru, N dilewati", bukan membaca ulang seluruh bag.
        """
        try:
            a,b = rentang_paksa or self.rentang()
            # Satu folder frame kanonik per sesi, apa pun FPS pilihan. Dengan
            # begitu ekspor 30 FPS lalu 15 FPS tidak menggandakan frame sumber.
            root = sesi / "exports"; frames = root / "frames"
            frames.mkdir(parents=True, exist_ok=True)
            # frame.json ditulis PALING AKHIR, jadi keberadaannya = frame utuh.
            # Folder tanpa frame.json berarti ekspor sebelumnya terputus di
            # tengah frame itu, dan frame itu diulang.
            sudah: dict[int, Path] = {}
            # Folder fps_* adalah format lama. Dibaca juga agar ekspor baru
            # tidak membuat salinan frame yang sudah ada di format tersebut.
            lokasi_lama = [frames, *sorted(root.glob("fps_*/frames"))]
            for lokasi in lokasi_lama:
                for p in lokasi.glob("frame_*"):
                    if not (p/"frame.json").exists(): continue
                    try: sudah.setdefault(int(p.name.split("_", 1)[1]), p)
                    except ValueError: pass
            picks = self._rencana_picks(sesi, a, b, langkah)
            if picks and picks <= sudah.keys() and (root/"export.json").exists():
                self.q.put(("status", "Ekspor sudah lengkap; tidak ada satu frame pun yang diekspor ulang."))
                self.q.put(("ekspor_selesai", (sesi, root, 0, len(picks))))
                return
            lama = baca_json(root / "export.json", {})
            frame_meta = {int(x["index_bag"]): x for x in lama.get("frames", []) if "index_bag" in x}
            meta_export={"sumber_bag":f"../source/{self.bag(sesi).name}","rentang_indeks_terakhir":[a,b],
                         "interval_frame_sumber":langkah,"non_destruktif":True,
                         "penamaan":"exports/frames/frame_<indeks_sumber>; satu frame sumber hanya disimpan sekali",
                         "pemilihan":"setiap N indeks frame sumber; frame RGB/depth tetap satu paket sinkron dari RAW",
                         "frames":[]}
            n=0; dilewati=0
            for i,(native,color,depth,aligned,profile) in enumerate(PembacaBag(self.bag(sesi)).iter_frame()):
                if i<a: continue
                if i>b: break
                if i not in picks: continue
                folder = frames/f"frame_{i:06d}"
                if i in sudah:
                    dilewati += 1
                    lama_folder = sudah[i]
                    frame_meta[i] = {"folder":str(lama_folder.relative_to(root)),"index_bag":i,"timestamp_ms":float(depth.get_timestamp())}
                    continue
                n+=1; folder.mkdir(exist_ok=True)
                if n % 10 == 1:
                    self.q.put(("status", f"Mengekspor… {n} frame baru, {dilewati} dilewati "
                                          f"(indeks {i} dari {b})"))
                rgb=np.asanyarray(color.get_data()); raw=np.asanyarray(depth.get_data()); al=np.asanyarray(aligned.get_data())
                cv2.imwrite(str(folder/"color_raw.png"),rgb); cv2.imwrite(str(folder/"depth_raw.png"),raw); cv2.imwrite(str(folder/"depth_aligned_to_color.png"),al); np.save(folder/"depth_raw.npy",raw); np.save(folder/"depth_aligned_to_color.npy",al)
                for j,nm in ((1,"ir_left_raw.png"),(2,"ir_right_raw.png")):
                    ir=native.get_infrared_frame(j)
                    if ir: cv2.imwrite(str(folder/nm),np.asanyarray(ir.get_data()))
                info=self._info_profile(profile,color,depth); fm={"id":folder.name,"kategori":sesi.parent.name,"index_bag":i,"frame_number":int(depth.get_frame_number()),"timestamp_kamera_ms":float(depth.get_timestamp()),"format_depth":"Z16 native; meter=nilai*depth_scale","raw_bag_sumber":str(self.bag(sesi)),**info}
                tulis_json(folder/"frame.json",fm); frame_meta[i] = {"folder":str(folder.relative_to(root)),"index_bag":i,"timestamp_ms":fm["timestamp_kamera_ms"]}
            meta_export["frames"] = sorted(frame_meta.values(), key=lambda x: x["index_bag"])
            meta_export["jumlah_frame"]=len(meta_export["frames"])
            tulis_json(root/"export.json",meta_export)
            self.q.put(("ekspor_selesai",(sesi,root,n,dilewati)))
        except Exception as e: self.q.put(("error",f"Ekspor gagal: {e}"))

    # ----- label/ukur -----
    def daftar_frame_ekspor(self) -> list[Path]:
        """Frame unik berdasarkan indeks sumber, termasuk format ekspor lama."""
        if not self.sesi:
            return []
        root = self.sesi / "exports"
        lokasi = [root / "frames", *sorted(root.glob("fps_*/frames"))] if root.exists() else []
        unik: dict[int, Path] = {}
        for folder in lokasi:
            for p in folder.glob("frame_*"):
                try:
                    unik.setdefault(int(p.name.split("_", 1)[1]), p)
                except ValueError:
                    continue
        return [unik[i] for i in sorted(unik)]

    def muat_frame(self):
        if not self.sesi: messagebox.showinfo("Pilih sesi", "Pilih rekaman pada tab Tinjau dahulu.", parent=self); return
        semua = self.daftar_frame_ekspor()
        self.frame_paths=[]
        self.list_frame.delete(0,"end")
        for p in semua:
            state = baca_json(p / "frame_state.json", {"di_sampah": False})
            if state.get("di_sampah", False) != self.tampil_sampah_frame.get(): continue
            ikon = "\U0001f5d1 " if state.get("di_sampah") else ""
            self.frame_paths.append(p)
            self.list_frame.insert("end",f"{ikon}{p.parent.parent.name} / {p.name}")
        self.status.set(f"{len(self.frame_paths)} frame ekspor dimuat.")

    def pilih_frame(self):
        s=self.list_frame.curselection()
        if not s:return
        self._buka_frame_ekspor(self.frame_paths[s[0]])

    def _buka_frame_ekspor(self, p: Path):
        bgr=cv2.imread(str(p/"color_raw.png")); dep=np.load(p/"depth_aligned_to_color.npy")
        if bgr is None: return
        self.label_path=p; self.label_info=baca_json(p/"frame.json"); self.kanvas.set_frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),dep)
        draft = baca_json(p / "label_draft.json", {})
        if draft.get("poligon"):
            self.kanvas.poligon = {nama: [list(map(tuple, poly)) for poly in draft["poligon"].get(nama, [])]
                                   for nama in ("objek", "acuan")}
            self.kanvas.aktif_indeks = {nama: (len(daftar) - 1 if daftar else None)
                                        for nama, daftar in self.kanvas.poligon.items()}
            self.kanvas.after(25, self.kanvas.fit)
            self.status.set(f"Draft mask {p.name} dipulihkan otomatis.")
        elif not self._punya_label(p):
            # Frame baru selalu mendapat usulan terlebih dahulu. Pengguna lalu
            # tinggal mengoreksi; draft/label yang sudah ada tidak ditimpa.
            self.after(180, lambda target=p: self._auto_segmentasi(target))
        self.perbarui_konteks_label(self.label_info.get("kategori"))
        self.ukur_status.set("Tandai mask objek dan, untuk tinggi, bidang acuan.")

    def _auto_segmentasi(self, target: Path):
        if target != self.label_path or self._punya_label(target):
            return
        self.status.set(f"Auto-segmentasi YOLO + SAM 2 untuk {target.name}…")
        self.usulkan_segmentasi()

    def pindah_frame_label(self, arah: int):
        if not self.frame_paths:
            self.muat_frame()
        if not self.frame_paths:
            messagebox.showinfo("Belum ada frame", "Ekspor satu frame dari video atau pilih rentang untuk diekspor dahulu.", parent=self); return
        try:
            kini = self.frame_paths.index(self.label_path)
        except ValueError:
            kini = -1 if arah > 0 else 0
        tujuan = max(0, min(kini + arah, len(self.frame_paths) - 1))
        self.list_frame.selection_clear(0, "end"); self.list_frame.selection_set(tujuan); self.list_frame.activate(tujuan)
        self._buka_frame_ekspor(self.frame_paths[tujuan])

    def _aktif_mask_berubah(self, nama: str, indeks: int | None):
        """Sinkronkan nomor di panel ketika titik/mask dipilih langsung."""
        if indeks is None:
            return
        self.mode_label.set(nama)
        self.nomor_mask.set(indeks + 1)

    def atur_nomor_mask(self, senyap=False):
        """Pilih mask bernomor tertentu; urutan/nomor mask tidak diubah."""
        nama, nomor = self.mode_label.get(), self.nomor_mask.get()
        if self.kanvas.pilih_mask(nama, nomor):
            if not senyap:
                self.status.set(f"Mask {nomor} dipilih. + Titik dan klik gambar akan mengubah mask ini.")
            return True
        if not senyap:
            messagebox.showinfo("Nomor mask belum ada",
                                f"Mask {nomor} belum ada. Tekan + Mask untuk membuat mask baru.", parent=self)
        return False

    def mulai_mask_baru(self):
        self.kanvas.poligon_baru(self.mode_label.get())
        self.nomor_mask.set(len(self.kanvas.poligon[self.mode_label.get()]))
        self.kanvas.focus_set()
        self.status.set(f"Mask {self.nomor_mask.get()} baru siap. Klik gambar untuk menambah titik pertama.")

    def mulai_tambah_titik(self):
        if not self.atur_nomor_mask(senyap=True):
            messagebox.showinfo("Pilih mask", "Pilih nomor mask yang ada, atau tekan + Mask untuk membuat mask baru.", parent=self)
            return
        self.kanvas.focus_set()
        self.status.set(f"Klik lokasi pada gambar untuk menambah titik ke mask {self.nomor_mask.get()}.")

    def buang_rekomendasi(self):
        if not any(self.kanvas.poligon.values()):
            return
        if messagebox.askyesno("Buang rekomendasi", "Hapus seluruh mask rekomendasi/yang sedang tampil? Anda dapat membuat mask baru dari nol.", parent=self):
            self.kanvas.poligon = {"objek": [], "acuan": []}
            self.kanvas.render(); self.hitung_ukuran()
            self.status.set("Semua mask rekomendasi dibuang. Mask baru siap dibuat.")

    def jadwalkan_autosave(self):
        if not self.label_path or self.kanvas.rgb is None:
            return
        if self._autosave_setelah is not None:
            self.after_cancel(self._autosave_setelah)
        self._autosave_setelah = self.after(450, self.simpan_draft_label)

    def simpan_draft_label(self):
        self._autosave_setelah = None
        if not self.label_path or self.kanvas.rgb is None:
            return
        tulis_json(self.label_path / "label_draft.json", {
            "versi": 1, "otomatis": True,
            "disimpan_iso": datetime.now().isoformat(timespec="seconds"),
            "poligon": self.kanvas.poligon,
        })
        if any(len(poly) >= 3 for daftar in self.kanvas.poligon.values() for poly in daftar):
            self.simpan_label(senyap=True)
        self.status.set(f"Draft mask otomatis disimpan: {self.label_path.name}")

    def toggle_sampah_frame(self):
        if not self.label_path:
            messagebox.showinfo("Pilih frame", "Pilih frame yang ingin dipindahkan atau dipulihkan.", parent=self)
            return
        state = baca_json(self.label_path / "frame_state.json", {"di_sampah": False})
        state["di_sampah"] = not state.get("di_sampah", False)
        state["waktu_ubah_iso"] = datetime.now().isoformat(timespec="seconds")
        tulis_json(self.label_path / "frame_state.json", state)
        # Hanya frame INI yang ditandai - folder dan isinya tidak dipindah,
        # jadi frame lain, label, dan ekspor yang bisa dilanjutkan tidak
        # terganggu. Pemulihan tinggal membalik tanda lewat tampilan sampah.
        self.status.set("Frame dibuang ke sampah (tidak ikut dataset YOLO). Centang 'Tampilkan sampah frame' untuk memulihkannya."
                        if state["di_sampah"] else
                        "Frame dipulihkan dari sampah.")
        self.label_path = None
        self.muat_frame()

    def hapus_frame_permanen(self):
        """Hapus hanya satu paket turunan ekspor; RAW sesi tetap tidak tersentuh."""
        p = self.label_path
        if not p:
            messagebox.showinfo("Pilih frame", "Pilih frame ekspor yang akan dihapus permanen.", parent=self); return
        exports = (self.sesi / "exports").resolve() if self.sesi else None
        if exports is None or exports not in p.resolve().parents:
            messagebox.showerror("Target tidak aman", "Frame bukan bagian dari exports sesi aktif.", parent=self); return
        if not messagebox.askyesno("Hapus frame permanen", f"Hapus seluruh paket turunan {p.name}?\n\nRGB, depth, IR, mask, dan label pada frame ini akan hilang. RAW tidak dihapus.", parent=self):
            return
        try:
            # Simpan kandidat sesudah frame kini SEBELUM folder dihapus. Dengan
            # begitu alur label maju ke gambar berikutnya, bukan kembali awal.
            try:
                posisi = self.frame_paths.index(p)
            except ValueError:
                posisi = -1
            berikut = self.frame_paths[posisi + 1] if 0 <= posisi < len(self.frame_paths) - 1 else None
            sebelum = self.frame_paths[posisi - 1] if posisi > 0 else None
            shutil.rmtree(p)
            self.label_path = None; self.label_info = None
            self.kanvas.rgb = self.kanvas.depth = None; self.kanvas.delete("all")
            self.muat_frame()
            tujuan = berikut if berikut in self.frame_paths else sebelum if sebelum in self.frame_paths else None
            if tujuan is not None:
                indeks = self.frame_paths.index(tujuan)
                self.list_frame.selection_set(indeks); self.list_frame.activate(indeks)
                self._buka_frame_ekspor(tujuan)
                self.status.set(f"Frame dihapus. Lanjut ke {tujuan.name}.")
            else:
                self.status.set("Frame ekspor dihapus permanen. Rekaman RAW tetap aman.")
        except OSError as e:
            messagebox.showerror("Hapus frame", str(e), parent=self)

    def usulkan_segmentasi(self):
        """Isi kanvas dengan usulan tapakan/riser hasil hitungan depth.

        Ini USULAN, bukan kebenaran: hasilnya langsung bisa diedit seperti
        poligon yang digambar tangan. Tujuannya menghapus pekerjaan menggambar
        sepuluh poligon per frame dari nol, bukan menggantikan penilaian Anda.
        """
        if self.kanvas.rgb is None or self.kanvas.depth is None:
            messagebox.showinfo("Pilih frame", "Pilih satu frame ekspor dahulu.", parent=self); return
        info = self.label_info or {}
        if "intrinsics_rgb_native" not in info:
            messagebox.showinfo("Intrinsics tidak ada",
                                "frame.json frame ini tidak memuat intrinsics.", parent=self); return
        kategori = info.get("kategori", self.sesi.parent.name if self.sesi else "tangga_naik")
        try:
            # Bobot aktif hanya dilatih untuk kelas tangga. Batu/ramp tetap
            # mendapat proposal depth, bukan dipaksa ke model yang salah kelas.
            if kategori == "tangga_naik":
                hasil = usulkan_yolo_depth(self.kanvas.rgb, self.kanvas.depth, self._intrinsics(info))
                # SAM 2 memakai kandidat YOLO sebagai bounding-box prompt.
                # Setiap kandidat tetap dipertahankan bila SAM gagal merapikan.
                rapih = rapikan_sam2(self.kanvas.rgb, {
                    "tapakan": hasil["tapakan"], "bidang_tegak": hasil["bidang_tegak"],
                })
                rapih = verifikasi_depth(rapih, self.kanvas.depth)
                hasil["tapakan"], hasil["bidang_tegak"] = rapih["tapakan"], rapih["bidang_tegak"]
                hasil["sumber"] = "YOLO + SAM 2.1 Tiny + depth (GPU)"
            else:
                hasil = usulkan_segmentasi(self.kanvas.depth, self._intrinsics(info))
                hasil["sumber"] = "depth (bobot YOLO tangga tidak dipakai untuk kategori ini)"
        except Exception as e:                                  # noqa: BLE001
            # Untuk tangga jangan diam-diam mengganti hasil YOLO dengan depth:
            # pengguna harus tahu bila bobot/runtime YOLO bermasalah.
            if kategori == "tangga_naik":
                messagebox.showerror("YOLO tidak tersedia", f"Rekomendasi tangga memakai YOLO.\n\n{e}", parent=self)
                return
            # Batu/ramp belum mempunyai bobot YOLO khusus, sehingga depth tetap
            # menjadi proposal kategori tersebut.
            try:
                hasil = usulkan_segmentasi(self.kanvas.depth, self._intrinsics(info))
                hasil["sumber"] = "depth (belum ada bobot YOLO kategori ini)"
            except Exception as cadangan:
                messagebox.showerror("Rekomendasi mask", str(cadangan), parent=self); return
        # Hasil mentah komponen terhubung tidak berurutan. Untuk tangga,
        # nomor 1 harus berada di tapakan TERBAWAH, bukan di bagian atas layar.
        # Urutan visual bawah->atas juga dipakai untuk riser agar konsisten.
        tapakan = [t["poligon"] for t in hasil.get("tapakan_terurut", [])] or hasil["tapakan"]
        urut_bawah = lambda daftar: sorted(daftar, key=lambda poly: -sum(p[1] for p in poly) / max(1, len(poly)))
        self.kanvas.poligon["acuan"] = [list(map(tuple, p)) for p in urut_bawah(tapakan)]
        self.kanvas.poligon["objek"] = [list(map(tuple, p)) for p in urut_bawah(hasil["bidang_tegak"])]
        self.kanvas.aktif_indeks = {nama: (len(daftar) - 1 if daftar else None)
                                    for nama, daftar in self.kanvas.poligon.items()}
        self._aktif_mask_berubah(self.mode_label.get(), self.kanvas.aktif_indeks[self.mode_label.get()])
        self.kanvas.render(); self.kanvas.on_change()
        urut = hasil.get("tapakan_terurut", [])
        tinggi = [t["tinggi_dari_terbawah_cm"] for t in urut if t["tinggi_dari_terbawah_cm"] is not None]
        beda = [round(b - a, 1) for a, b in zip(tinggi, tinggi[1:])] if len(tinggi) > 1 else []
        self.ukur_status.set(
            f"Rekomendasi {hasil.get('sumber', 'depth')}: {len(hasil['tapakan'])} tapakan (biru), {len(hasil['bidang_tegak'])} riser (merah). "
            + (f"Beda tinggi antar tapakan: {beda} cm." if beda else "")
            + " Periksa lalu koreksi sebelum menyimpan.")

    def ganti_mode(self): self.kanvas.mode=self.mode_label.get(); self.kanvas.render()
    def ganti_depth(self): self.kanvas.depth_alpha=float(self.depth_alpha.get()); self.kanvas.render()

    def _mask(self, nama):
        """Masker gabungan SEMUA instance kelas ini (untuk pengukuran)."""
        if self.kanvas.rgb is None: return None
        m = np.zeros(self.kanvas.rgb.shape[:2], np.uint8)
        for poly in self.kanvas.poligon[nama]:
            if len(poly) >= 3:
                cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        return m

    def simpan_label(self, senyap: bool = False):
        """Tulis label YOLO-seg: SATU BARIS PER INSTANCE, dua kelas semantik.

        Poligon biru (acuan) ikut jadi label - pada dataset lama 'tapakan'
        memang kelas 0 yang dilatih, bukan sekadar alat bantu ukur.
        """
        if not self.label_path or self.kanvas.rgb is None:
            if not senyap:
                messagebox.showinfo("Pilih frame", "Pilih satu frame ekspor dahulu.", parent=self)
            return
        kategori = (self.label_info or {}).get("kategori", "tangga_naik")
        peta = KELAS_MODE.get(kategori, KELAS_MODE["tangga_naik"])
        h, w = self.kanvas.rgb.shape[:2]
        baris, jumlah = [], {}
        for mode in ("acuan", "objek"):
            nama_kelas = peta[mode]; cls = KELAS_YOLO[nama_kelas]
            sah = [poly for poly in self.kanvas.poligon[mode] if len(poly) >= 3]
            jumlah[nama_kelas] = len(sah)
            for poly in sah:
                norm = " ".join(f"{v:.6f}" for pt in poly for v in (pt[0]/w, pt[1]/h))
                baris.append(f"{cls} {norm}")
        if not baris:
            if not senyap:
                messagebox.showwarning("Poligon belum cukup", "Belum ada poligon dengan minimal tiga titik.", parent=self)
            return
        (self.label_path/"label_yolo_seg.txt").write_text("\n".join(baris) + "\n", encoding="utf-8")
        obj = self._mask("objek"); ref = self._mask("acuan")
        if obj is not None and obj.any(): cv2.imwrite(str(self.label_path/"mask_objek.png"), obj*255)
        if ref is not None and ref.any(): cv2.imwrite(str(self.label_path/"mask_acuan.png"), ref*255)
        rinci = ", ".join(f"{k}={v}" for k, v in jumlah.items())
        if not senyap:
            self.status.set(f"Label disimpan untuk {self.label_path.name}: {len(baris)} instance ({rinci}).")

    def bangun_yolo(self):
        """Buat turunan siap Ultralytics tanpa memindahkan/menghapus raw frame."""
        if not self.sesi:
            messagebox.showinfo("Pilih sesi", "Pilih sesi pada tab Tinjau dahulu.", parent=self)
            return
        meta_sesi = baca_json(self.sesi / "source" / "session.json")
        split = meta_sesi.get("split", "train")
        root = self.root_data / "dataset_yolo_seg"
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        count = 0
        sudah_indeks: set[int] = set()
        for frame in self.daftar_frame_ekspor():
            info_frame = baca_json(frame / "frame.json", {})
            indeks = int(info_frame.get("index_bag", -1))
            if indeks in sudah_indeks:
                continue
            sudah_indeks.add(indeks)
            if baca_json(frame / "frame_state.json", {"di_sampah": False}).get("di_sampah", False):
                continue
            label = frame / "label_yolo_seg.txt"
            if not label.exists():
                continue
            stem = f"{self.sesi.name}_{frame.parent.parent.name}_{frame.name}"
            shutil.copy2(frame / "color_raw.png", root / "images" / split / f"{stem}.png")
            shutil.copy2(label, root / "labels" / split / f"{stem}.txt")
            count += 1
        tulis_json(root / "dataset_yolo_seg.yaml", {"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test", "names": {str(v): k for k, v in sorted(KELAS_YOLO.items(), key=lambda x: x[1])}})
        self.status.set(f"Dataset YOLO diperbarui: {count} frame berlabel dari sesi ini masuk split {split}. Raw tetap ada di sesi.")

    def hitung_ukuran(self):
        self.jadwalkan_autosave()
        if not self.label_path or not self.label_info:return
        obj,ref=self._mask("objek"),self._mask("acuan")
        if obj is None or ref is None or obj.sum()<3 or ref.sum()<3:return
        if "intrinsics_rgb_native" not in self.label_info:
            self.ukur_status.set("frame.json frame ini tidak memuat intrinsics kamera. "
                                 "Ekspor ulang frame ini agar metadata kamera ikut tertulis.")
            return
        try:
            # WAJIB lewat _intrinsics(): frame.json menyimpan intrinsics
            # bersarang (intrinsics_rgb_native.ppx dst.), sedangkan ukur()
            # mengharapkan kunci datar cx/cy/fx/fy. Dulu label_info dioper
            # mentah -> KeyError 'cx' tertelan except -> mode ukur di tab ini
            # tidak pernah berhasil.
            hasil=ukur(obj,ref,self.kanvas.depth,self._intrinsics(self.label_info))
            if hasil.get("ok"):
                tulis_json(self.label_path/"hasil_ukur_depth.json",hasil)
                self.ukur_status.set(f"Tinggi: {hasil['tinggi_cm']:.1f} cm\nJarak median objek: {hasil['jarak_median_objek_m']:.2f} m\nRMS bidang: {hasil['rms_bidang_acuan_mm']:.1f} mm")
            else:self.ukur_status.set(hasil.get("alasan","Depth belum cukup."))
        except Exception as e:self.ukur_status.set(f"Belum dapat mengukur: {e}")

    # ----- util / antrian -----
    def _buka_folder(self,p:Path):
        import subprocess
        try: subprocess.Popen(["xdg-open",str(p)])
        except Exception: messagebox.showinfo("Folder",str(p),parent=self)

    def _poll(self):
        try:
            while True:
                k,v=self.q.get_nowait()
                if k=="status":self.status.set(str(v))
                elif k=="error":
                    self.preview_diminta = False
                    self.btn_preview_kamera.configure(text="Aktifkan preview kamera", state="normal")
                    self.status.set(str(v)); messagebox.showerror("ZenExo Studio",str(v),parent=self)
                elif k=="live":self.after(50,self.live)
                elif k=="preset_muat":self._isi_preset(*v)
                elif k=="preset_selesai":
                    self.cb_preset.configure(state="disabled" if self.sedang_rekam else "readonly")
                    self.status.set(f"Preset depth: {v}.")
                elif k=="preset_gagal":
                    pesan, keadaan = v
                    self._isi_preset(*keadaan)
                    messagebox.showerror("Preset", f"Tidak dapat menerapkan preset.\n\n{pesan}", parent=self)
                elif k=="preview_kamera_siap":
                    self.btn_preview_kamera.configure(text="Matikan preview kamera", state="normal")
                elif k=="preview_selesai":
                    sesi,n=v; self.sesi=sesi
                    # Muat ulang indeks lengkap (dengan timestamp) dari CSV yang
                    # baru ditulis - daftar kosong {"i": i} membuat fps_asli()
                    # dan rentang potong bekerja dari data basi.
                    p=sesi/"derived"/"frame_index.csv"
                    if p.exists():
                        with p.open(newline="", encoding="utf-8") as f: self.indeks=list(csv.DictReader(f))
                    else:
                        self.indeks=[{"i":i} for i in range(n)]
                    m=len(self.indeks)
                    self.awal.set(0);self.akhir.set(max(0,m-1));self.scale_awal.configure(to=max(0,m-1));self.scale_akhir.configure(to=max(0,m-1))
                    # Dulu di sini menulis ke self.preview_label - widget itu
                    # tidak pernah ada, AttributeError-nya memutus rantai _poll
                    # sehingga pesan berikutnya (termasuk ekspor_selesai) tidak
                    # pernah lagi diproses. Sekarang tampilkan frame pertama
                    # preview di kanvas tinjau.
                    self._tutup_video()
                    if self._buka_video(): self._tampilkan(0)
                    self.durasi.set(f"Preview siap: {m} frame. Atur awal/akhir, lalu simpan potongan.");self.status.set("Preview dibuat hanya sebagai turunan. raw.bag tetap utuh.")
                elif k=="ekspor_selesai":
                    sesi,root,n,dilewati=v;self.sesi=sesi
                    self.label_ekspor.configure(text=f"Ekspor selesai: {n} frame baru, {dilewati} sudah ada (dilewati)\n{root}")
                    target = getattr(self, "_label_setelah_ekspor", None)
                    if target is not None:
                        self._label_setelah_ekspor = None
                        self.muat_frame()
                        tujuan = root / "frames" / f"frame_{target:06d}"
                        if tujuan.exists():
                            self._buka_frame_ekspor(tujuan)
                            self.tabs.select(self.tab_label)
                            self.status.set(f"Frame {target} siap dilabeli dan diukur.")
                    else:
                        self.status.set(f"Ekspor selesai: {n} baru, {dilewati} dilewati. Buka tab Label lalu tekan Muat frame dari sesi.")
        except queue.Empty:pass
        self.after(60,self._poll)

    def tutup(self):
        if self.sedang_rekam and not messagebox.askyesno("Rekaman masih berlangsung","Selesaikan rekaman dahulu agar raw.bag ditutup dengan benar. Tetap keluar?",parent=self):return
        self.simpan_preferensi()
        self._tutup_video();self._hentikan_render();self.cam.hentikan();self.destroy()


def main():
    ap=argparse.ArgumentParser(description="Studio rekaman RGB-D D435 non-destruktif untuk dataset YOLO segmentation")
    ap.add_argument("--keluar", default=str(Path(__file__).resolve().parents[2] / "dataset" / "studio_rgbd"), help="folder utama dataset studio")
    ap.add_argument("--lebar",type=int,default=848);ap.add_argument("--tinggi",type=int,default=480);ap.add_argument("--fps",type=int,default=30)
    # Daftar pilihan diambil dari lapisan kamera. Dulu daftar di sini punya
    # nama sendiri ("high_accuracy"/"high_density") yang tidak dikenal
    # terapkan_preset(), sehingga keduanya diam-diam memasang preset Default.
    ap.add_argument("--preset",default="jangan",choices=PRESET_PILIHAN);ap.add_argument("--batas-frame",type=int,default=8000)
    # 20 Hz terukur nol lonjakan event-loop >33 ms pada mesin ini. Naikkan bila
    # mesin Anda lebih kuat, turunkan di Jetson kalau preview mulai tersendat.
    ap.add_argument("--fps-preview",type=int,default=20,help="laju render preview (Hz)")
    Studio(ap.parse_args()).mainloop()


if __name__=="__main__":main()
