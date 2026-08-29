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
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 belum tersedia. Jalankan dari .venv proyek.")

from .kamera_rgbd import KameraRGBD, nama_aman, tulis_json
from .pengukuran_objek import ukur


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
KELAS_YOLO = {"batu": 0, "tangga_naik": 1, "ramp_naik": 2}


def baca_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default.copy()
    return json.loads(path.read_text(encoding="utf-8"))


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
                yield native, color, depth, aligned, profile
        finally:
            pipe.stop()

    def indeks(self) -> list[dict]:
        hasil: list[dict] = []
        for i, (_, color, depth, _, _) in enumerate(self.iter_frame()):
            hasil.append({"i": i, "frame": int(depth.get_frame_number()),
                          "timestamp_ms": float(depth.get_timestamp()),
                          "lebar": color.get_width(), "tinggi": color.get_height()})
        return hasil


class KanvasLabel(tk.Canvas):
    """Kanvas label poligon: klik kiri tambah, kanan undo, roda zoom, tengah pan."""

    def __init__(self, master, on_change, **kw):
        super().__init__(master, bg="#241F1D", highlightthickness=0, **kw)
        self.on_change = on_change
        self.rgb: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self.depth_alpha = 0.0
        self.mode = "objek"
        self.poligon = {"objek": [], "acuan": []}
        self._drag = None
        self._photo = None
        self.bind("<Button-1>", self.tambah)
        self.bind("<Button-3>", self.undo)
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<Button-4>", lambda e: self.zoom_langkah(e, 1.15))
        self.bind("<Button-5>", lambda e: self.zoom_langkah(e, 1 / 1.15))
        self.bind("<ButtonPress-2>", self.pan_mulai)
        self.bind("<B2-Motion>", self.pan)
        self.bind("<Configure>", lambda e: self.fit_bila_baru())

    def set_frame(self, rgb: np.ndarray, depth: np.ndarray):
        self.rgb, self.depth = rgb, depth
        self.poligon = {"objek": [], "acuan": []}
        self.after(20, self.fit)

    def fit_bila_baru(self):
        if self.rgb is not None and not self.poligon["objek"] and not self.poligon["acuan"]:
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
        return out

    def render(self):
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        img = self.gambar_tampil()
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        self._photo = ImageTk.PhotoImage(Image.fromarray(img).resize(size, Image.LANCZOS))
        self.delete("all")
        self.create_image(self.ox, self.oy, anchor="nw", image=self._photo)
        specs = (("objek", "#6DC1FF", "#D7F0FF"), ("acuan", "#90D179", "#E1F7D9"))
        for nama, garis, titik in specs:
            pts = [(x * self.scale + self.ox, y * self.scale + self.oy) for x, y in self.poligon[nama]]
            if len(pts) >= 3:
                self.create_polygon(pts, fill=garis, outline=garis, stipple="gray50", width=2)
            elif len(pts) >= 2:
                self.create_line(pts, fill=garis, width=2)
            for x, y in pts:
                self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=titik, outline=garis)

    def canvas_ke_gambar(self, x, y):
        if self.rgb is None:
            return None
        gx, gy = (x - self.ox) / self.scale, (y - self.oy) / self.scale
        h, w = self.rgb.shape[:2]
        return (gx, gy) if 0 <= gx < w and 0 <= gy < h else None

    def tambah(self, e):
        p = self.canvas_ke_gambar(e.x, e.y)
        if p:
            self.poligon[self.mode].append(p)
            self.render(); self.on_change()

    def undo(self, _e=None):
        if self.poligon[self.mode]:
            self.poligon[self.mode].pop()
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
        self.render()

    def pan_mulai(self, e): self._drag = (e.x, e.y, self.ox, self.oy)
    def pan(self, e):
        if self._drag:
            x, y, ox, oy = self._drag
            self.ox, self.oy = ox + e.x - x, oy + e.y - y
            self.render()


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
        self.sesi: Path | None = None
        self.indeks: list[dict] = []
        self.frame_paths: list[Path] = []
        self.label_path: Path | None = None
        self.label_info: dict | None = None
        self.preview_photo = None
        self.title("ZenExo Studio — Rekam, Tinjau, Ekspor, Label")
        self.geometry("1380x860"); self.minsize(1100, 720); self.configure(bg=BG)
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
        self.fps_ekspor = IntVar(value=args.fps)
        self.awal = IntVar(value=0); self.akhir = IntVar(value=0)
        self.depth_alpha = DoubleVar(value=0.28)
        self.mode_label = StringVar(value="objek")
        self.ukur_status = StringVar(value="Tandai objek (biru) dan bidang acuan/lantai (hijau).")
        self.tampil_sampah = BooleanVar(value=False)
        self.tampil_sampah_frame = BooleanVar(value=False)
        self._siapkan_root(); self._buat_ui()
        self.tabs.bind("<<NotebookTabChanged>>", self.ganti_tab)
        self.after(30, self._poll)
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
    def bag(self, sesi: Path) -> Path: return sesi / "source" / "raw.bag"

    def daftar_sesi(self) -> list[Path]:
        hasil = []
        for k in KATEGORI:
            hasil.extend(sorted((self.root_data / "rekaman" / k).glob("*"), reverse=True))
        return [p for p in hasil if p.is_dir() and self.bag(p).exists()]

    # ----- UI -----
    def _gaya(self):
        """Gaya tenang dan konsisten; tetap 100% widget Tkinter native."""
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        s.configure("TNotebook.Tab", background="#E9DED5", foreground=MUTED,
                    padding=(18, 10), font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        s.configure("TCombobox", fieldbackground="#FFF9F4", background="#FFF9F4",
                    foreground=INK, padding=5, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE)

    def card(self, parent, judul):
        b = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1, bd=0)
        tk.Label(b, text=judul.upper(), bg=PANEL, fg=ACCENT, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        garis = tk.Frame(b, bg="#EFE6DF", height=1); garis.pack(fill="x", padx=16, pady=(0, 11))
        isi = tk.Frame(b, bg=PANEL); isi.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        return b, isi

    def tombol(self, parent, text, cmd, color=ACCENT, fg="white"):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg=fg, activebackground=color,
                         activeforeground=fg, relief="flat", bd=0, padx=14, pady=10,
                         font=("Segoe UI", 10, "bold"), cursor="hand2", highlightthickness=0)

    def _buat_ui(self):
        top = tk.Frame(self, bg=BG); top.pack(fill="x", padx=22, pady=(18, 10))
        brand = tk.Frame(top, bg=BG); brand.pack(side="left")
        tk.Label(brand, text="ZenExo Studio", bg=BG, fg=INK, font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(brand, text="Rekam RGB-D mentah • sortir • ekspor • label • ukur", bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(1, 0))
        badge = tk.Label(top, text="D435  •  RGB + DEPTH", bg=ACCENT_SOFT, fg=ACCENT,
                         font=("Segoe UI", 9, "bold"), padx=12, pady=7)
        badge.pack(side="right", padx=(12, 0))
        tk.Label(top, textvariable=self.status, bg=BG, fg=ACCENT, font=("Segoe UI", 10), wraplength=500, justify="right").pack(side="right")
        self.tabs = ttk.Notebook(self); self.tabs.pack(fill="both", expand=True, padx=16, pady=(0, 16))
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
        i.columnconfigure(1, weight=1)
        self.btn_rekam = self.tombol(i, "● Mulai rekam", self.toggle_rekam, RED)
        self.btn_rekam.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 4))
        self.btn_preview_kamera = self.tombol(i, "Aktifkan preview kamera", self.toggle_preview_kamera, "#E8DDD5", INK)
        self.btn_preview_kamera.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        tk.Label(i, text="Kode adegan kosong = nama otomatis. Jika Anda mengisi misalnya Taman, nilai terakhir akan diingat sampai diganti.", bg=PANEL, fg=MUTED, justify="left", wraplength=490).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        tk.Label(i, text="Cukup tekan Mulai rekam, ambil berbagai sudut, lalu tekan Selesai rekaman. Semua stream mentah RGB, Z16 depth, IR, timestamp, intrinsics, dan extrinsics tersimpan dalam raw.bag.", bg=PANEL, fg=MUTED, justify="left", wraplength=490).grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))
        b, i = self.card(kanan, "Preview kamera dan data") ; b.pack(fill="both", expand=True)
        tk.Label(i, text="• Rekaman asli tidak pernah dipotong atau dihapus.\n• Potong hanya membuat rentang virtual.\n• Frame hasil ekspor membawa RGB, depth native Z16, depth selaras RGB, IR, timestamp, dan metadata kamera.\n• Jika kalibrasi berubah, ekspor dapat dibuat ulang dari raw.bag yang sama.", bg=PANEL, fg=INK, justify="left", wraplength=430, font=("Segoe UI", 11)).pack(anchor="nw")
        previews = tk.Frame(i, bg=PANEL); previews.pack(fill="both", expand=True, pady=(18, 0))
        previews.columnconfigure((0, 1), weight=1)
        tk.Label(previews, text="RGB", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(previews, text="DEPTH (warna = jarak)", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.label_live = tk.Label(previews, text="Menunggu RGB…", bg="#27201E", fg="white")
        self.label_depth = tk.Label(previews, text="Menunggu depth…", bg="#27201E", fg="white")
        self.label_live.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.label_depth.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

    def ui_tinjau(self):
        f = tk.Frame(self.tab_tinjau, bg=BG); f.pack(fill="both", expand=True, padx=18, pady=18)
        left = tk.Frame(f, bg=BG, width=300); left.pack(side="left", fill="y", padx=(0, 12)); right = tk.Frame(f, bg=BG); right.pack(side="left", fill="both", expand=True)
        b, i = self.card(left, "Rekaman") ; b.pack(fill="both", expand=True)
        self.list_sesi = tk.Listbox(i, bg="#FFF9F4", fg=INK, relief="flat", selectbackground=ACCENT_SOFT, activestyle="none", height=22)
        self.list_sesi.pack(fill="both", expand=True); self.list_sesi.bind("<<ListboxSelect>>", lambda e: self.pilih_sesi())
        self.tombol(i, "Muat daftar", self.muat_daftar, "#E8DDD5", INK).pack(fill="x", pady=(10, 3))
        self.tombol(i, "Pindah ke tempat sampah / pulihkan", self.toggle_sampah, "#E8DDD5", INK).pack(fill="x")
        b, i = self.card(right, "Tinjau hanya jika Anda meminta") ; b.pack(fill="both", expand=True)
        self.preview_label = tk.Label(i, text="Pilih rekaman lalu tekan Buat preview.", bg="#27201E", fg="white")
        self.preview_label.pack(fill="both", expand=True)
        row = tk.Frame(i, bg=PANEL); row.pack(fill="x", pady=(10, 0))
        self.tombol(row, "Buat / buka preview", self.buat_preview, BLUE).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.tombol(row, "Putar preview", self.putar_preview, "#E8DDD5", INK).pack(side="left", fill="x", expand=True, padx=4)
        self.tombol(row, "Buka folder", self.buka_sesi, "#E8DDD5", INK).pack(side="left", fill="x", expand=True, padx=(4, 0))
        tk.Label(i, textvariable=self.durasi, bg=PANEL, fg=MUTED).pack(anchor="w", pady=(12, 4))
        self.scale_awal = tk.Scale(i, from_=0, to=0, orient="horizontal", variable=self.awal, label="Awal potongan", bg=PANEL, fg=INK, highlightthickness=0)
        self.scale_akhir = tk.Scale(i, from_=0, to=0, orient="horizontal", variable=self.akhir, label="Akhir potongan", bg=PANEL, fg=INK, highlightthickness=0)
        self.scale_awal.pack(fill="x"); self.scale_akhir.pack(fill="x")
        self.tombol(i, "Simpan rentang potong (non-destruktif)", self.simpan_potong, GREEN).pack(fill="x", pady=(8, 0))

    def ui_ekspor(self):
        f = tk.Frame(self.tab_ekspor, bg=BG); f.pack(fill="both", expand=True, padx=30, pady=28)
        b, i = self.card(f, "Ekspor frame RGB-D untuk dataset dan label YOLO") ; b.pack(fill="x")
        tk.Label(i, text="FPS ekspor (maksimum mengikuti FPS kamera)", bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w")
        tk.Spinbox(i, from_=1, to=self.args.fps, textvariable=self.fps_ekspor, width=8).grid(row=0, column=1, sticky="w", padx=10)
        self.tombol(i, "Ekspor frame dari rentang potong", self.ekspor, GREEN).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        tk.Label(i, text="Setiap frame disimpan pada sesi yang sama: exports/<nama>/frames/frame_xxxxxx/. Tidak ada frame dari luar rentang yang diekspor. Gunakan tab Label untuk membuka frame tersebut, memberi poligon YOLO, dan mencocokkannya dengan depth.", bg=PANEL, fg=MUTED, wraplength=750, justify="left").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.label_ekspor = tk.Label(f, text="Belum ada ekspor dipilih.", bg=BG, fg=ACCENT, justify="left"); self.label_ekspor.pack(anchor="w", pady=(18, 0))

    def ui_label(self):
        f = tk.Frame(self.tab_label, bg=BG); f.pack(fill="both", expand=True, padx=14, pady=14)
        left = tk.Frame(f, bg=BG); left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.kanvas = KanvasLabel(left, self.hitung_ukuran); self.kanvas.pack(fill="both", expand=True)
        right = tk.Frame(f, bg=BG, width=340); right.pack(side="left", fill="y")
        b, i = self.card(right, "Pilih frame ekspor") ; b.pack(fill="x", pady=(0, 10))
        self.list_frame = tk.Listbox(i, height=11, bg="#FFF9F4", fg=INK, relief="flat", selectbackground=ACCENT_SOFT)
        self.list_frame.pack(fill="x"); self.list_frame.bind("<<ListboxSelect>>", lambda e: self.pilih_frame())
        self.tombol(i, "Muat frame dari sesi", self.muat_frame, "#E8DDD5", INK).pack(fill="x", pady=(8, 0))
        self.tombol(i, "Pindah/pulihkan frame dari sampah", self.toggle_sampah_frame, "#E8DDD5", INK).pack(fill="x", pady=(4, 0))
        b, i = self.card(right, "Label dan depth") ; b.pack(fill="x", pady=(0, 10))
        for mode, text, color in (("objek", "Objek / label YOLO (biru)", BLUE), ("acuan", "Bidang acuan / lantai (hijau)", GREEN)):
            tk.Radiobutton(i, text=text, variable=self.mode_label, value=mode, indicatoron=False, command=self.ganti_mode,
                           bg="#FFF9F4", selectcolor=color, activebackground=color, fg=INK, relief="flat", pady=7).pack(fill="x", pady=2)
        tk.Scale(i, from_=0, to=0.75, resolution=.05, variable=self.depth_alpha, command=lambda _: self.ganti_depth(), label="Overlay depth (samar)", bg=PANEL, fg=INK, highlightthickness=0).pack(fill="x", pady=(8, 0))
        row = tk.Frame(i, bg=PANEL); row.pack(fill="x", pady=(8, 0))
        self.tombol(row, "Undo", self.kanvas.undo, "#E8DDD5", INK).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.tombol(row, "Hapus objek", lambda: self.kanvas.bersihkan("objek"), "#F3D8D4", INK).pack(side="left", fill="x", expand=True, padx=3)
        self.tombol(row, "Hapus acuan", lambda: self.kanvas.bersihkan("acuan"), "#E8DDD5", INK).pack(side="left", fill="x", expand=True, padx=(3, 0))
        self.tombol(i, "Simpan label YOLO segmentation", self.simpan_label, ACCENT).pack(fill="x", pady=(10, 2))
        self.tombol(i, "Bangun folder dataset YOLO", self.bangun_yolo, GREEN).pack(fill="x", pady=(4, 2))
        tk.Label(i, text="Klik kiri: tambah titik • Klik kanan: hapus titik terakhir • Roda: zoom • tahan roda tengah: geser", bg=PANEL, fg=MUTED, wraplength=300, justify="left").pack(anchor="w", pady=(7, 0))
        b, i = self.card(right, "Cocokkan dengan depth") ; b.pack(fill="x")
        tk.Label(i, textvariable=self.ukur_status, bg=PANEL, fg=INK, wraplength=300, justify="left", font=("Segoe UI", 10, "bold")).pack(anchor="w")

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
            self.cam.hentikan()
            self.preview_info = None
            self.btn_preview_kamera.configure(text="Aktifkan preview kamera")
            self.status.set("Kamera dihentikan saat labeling/tinjau agar aplikasi tetap ringan.")

    def toggle_preview_kamera(self):
        if self.sedang_rekam:
            return
        if self.cam.hidup:
            self.preview_diminta = False
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
            self.q.put(("status", "D435 siap. Preview RGB dan depth aktif.")); self.q.put(("live", None))
            self.q.put(("preview_kamera_siap", None))
        except Exception as e: self.q.put(("error", f"D435 tidak dapat dibuka: {e}"))

    def live(self):
        if not self.winfo_exists(): return
        # Preview hanya pada tab Rekam dan sekitar 5 FPS. Rendering dua gambar
        # setiap 45 ms membuat Tkinter memakai CPU berlebihan saat tidak perlu.
        if self.tabs.select() != str(self.tab_rekam) or not self.preview_diminta or not self.cam.hidup:
            self.after(250, self.live)
            return
        if self.cam.hidup:
            try:
                _, cn, dn, ca, da = self.cam.ambil()
                rgb = np.asanyarray(ca.get_data())
                dep = np.asanyarray(da.get_data())
                if self.preview_info is None:
                    self.preview_info = self.cam.info(cn, dn)
                info = self.preview_info
                z = dep.astype(np.float32) * info["depth_scale"]
                valid = (z > .15) & (z < 6)
                vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cv2.putText(vis, f"Depth sah {valid.mean()*100:.0f}%", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 3, cv2.LINE_AA)
                cv2.putText(vis, f"Depth sah {valid.mean()*100:.0f}%", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (40,80,40), 1, cv2.LINE_AA)
                dvis = np.zeros_like(dep, dtype=np.uint8)
                if valid.any():
                    lo, hi = np.percentile(z[valid], (2, 98))
                    dvis = np.clip((z - lo) * 255 / max(hi - lo, .01), 0, 255).astype(np.uint8)
                    dvis[~valid] = 0
                dvis = cv2.cvtColor(cv2.applyColorMap(dvis, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                cv2.putText(dvis, "Depth: dekat -> jauh", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 3, cv2.LINE_AA)
                cv2.putText(dvis, "Depth: dekat -> jauh", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (35,35,35), 1, cv2.LINE_AA)
                small_rgb = Image.fromarray(vis); small_rgb.thumbnail((360, 300))
                small_depth = Image.fromarray(dvis); small_depth.thumbnail((360, 300))
                self.preview_photo = ImageTk.PhotoImage(small_rgb)
                self.depth_photo = ImageTk.PhotoImage(small_depth)
                self.label_live.configure(image=self.preview_photo, text="")
                self.label_depth.configure(image=self.depth_photo, text="")
            except Exception: pass
        self.after(200, self.live)

    def toggle_rekam(self):
        if self.sedang_rekam:
            try:
                self.cam.hentikan()
                assert self.sesi
                meta = baca_json(self.sesi / "source" / "session.json")
                meta["selesai_iso"] = datetime.now().isoformat(timespec="milliseconds")
                meta["status"] = "selesai"; tulis_json(self.sesi / "source" / "session.json", meta)
                self.sedang_rekam = False; self.btn_rekam.configure(text="● Mulai rekam", bg=RED)
                self.preview_diminta = False
                self.preview_info = None
                self.btn_preview_kamera.configure(text="Aktifkan preview kamera", state="normal")
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
            self.cam.mulai(sesi / "source" / "raw.bag")
            self.preview_diminta = True
            self.preview_info = None
            # info baru ditulis saat stop; bag memuat stream primer lengkap.
            tulis_json(sesi / "source" / "session.json", {"id": sesi.name, "kategori": kategori, "split": self.split.get(), "kode_adegan": kode,
                "mulai_iso": datetime.now().isoformat(timespec="milliseconds"), "status": "merekam",
                "raw_bag": "raw.bag", "raw_tidak_boleh_diubah": True, "fps_native": self.args.fps})
            self.sesi = sesi; self.sedang_rekam = True
            self.btn_rekam.configure(text="■ Selesai rekaman", bg=ACCENT)
            self.btn_preview_kamera.configure(text="Preview aktif saat rekam", state="disabled")
            self.after(100, self.live)
            self.status.set(f"Merekam {sesi.name}. Ambil semua sudut yang diperlukan, lalu tekan Selesai rekaman.")
        except Exception as e:
            messagebox.showerror("Rekaman", f"Tidak dapat memulai rekaman: {e}", parent=self)

    # ----- daftar / preview / potong -----
    def muat_daftar(self):
        self.list_sesi.delete(0, "end")
        self._map_sesi = []
        for p in self.daftar_sesi():
            state = self._state(p)
            if state.get("di_sampah") != self.tampil_sampah.get(): continue
            self._map_sesi.append(p); self.list_sesi.insert("end", f"{p.parent.name}  |  {p.name}")
        self.status.set(f"{len(self._map_sesi)} rekaman ditampilkan.")

    def pilih_sesi(self):
        sel = self.list_sesi.curselection()
        if not sel: return
        self.sesi = self._map_sesi[sel[0]]
        p = self.sesi / "derived" / "frame_index.csv"
        self.indeks = []
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f: self.indeks = list(csv.DictReader(f))
        n = len(self.indeks)
        self.awal.set(0); self.akhir.set(max(0, n - 1)); self.scale_awal.configure(to=max(0,n-1)); self.scale_akhir.configure(to=max(0,n-1))
        self.durasi.set(f"Sesi: {self.sesi.name}. {'Indeks siap: '+str(n)+' frame.' if n else 'Tekan Buat preview untuk membaca rekaman.'}")

    def buat_preview(self):
        if not self.sesi: messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        threading.Thread(target=self._preview_worker, args=(self.sesi,), daemon=True).start()
        self.status.set("Membuat preview dan indeks dari raw.bag…")

    def _preview_worker(self, sesi: Path):
        try:
            derived = sesi / "derived"; derived.mkdir(exist_ok=True)
            out = derived / "preview.mp4"; index = [] ; writer = None
            for i, (_, color, depth, _aligned, _profile) in enumerate(PembacaBag(self.bag(sesi)).iter_frame()):
                bgr = np.asanyarray(color.get_data())
                if writer is None:
                    h,w = bgr.shape[:2]; writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), self.args.fps, (w,h))
                writer.write(bgr)
                index.append({"i": i, "frame": int(depth.get_frame_number()), "timestamp_ms": f"{depth.get_timestamp():.3f}"})
            if writer: writer.release()
            with (derived / "frame_index.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["i","frame","timestamp_ms"]); w.writeheader(); w.writerows(index)
            if not index: raise RuntimeError("Tidak menemukan pasangan RGB dan depth dalam rekaman.")
            cap = cv2.VideoCapture(str(out)); ok,img=cap.read(); cap.release()
            if ok: cv2.imwrite(str(derived / "thumbnail.jpg"), img)
            self.q.put(("preview_selesai", (sesi, len(index))))
        except Exception as e: self.q.put(("error", f"Gagal membuat preview: {e}"))

    def simpan_potong(self):
        if not self.sesi or not self.indeks: messagebox.showinfo("Preview belum siap", "Buat preview dahulu agar rentang frame diketahui.", parent=self); return
        a,b = sorted((self.awal.get(), self.akhir.get()))
        edit = self.sesi / "edit"; edit.mkdir(exist_ok=True)
        tulis_json(edit / "rentang.json", {"awal_indeks": a, "akhir_indeks": b, "jumlah_total": len(self.indeks),
            "non_destruktif": True, "dibuat_iso": datetime.now().isoformat(timespec="seconds")})
        self.status.set(f"Rentang {a}–{b} disimpan. Rekaman asli tetap utuh.")

    def toggle_sampah(self):
        if not self.sesi: messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        state=self._state(self.sesi); state["di_sampah"] = not state.get("di_sampah", False); state["waktu_ubah_iso"] = datetime.now().isoformat(timespec="seconds")
        self._tulis_state(self.sesi,state); self.status.set("Status tempat sampah diubah; data tidak dihapus."); self.muat_daftar()

    def buka_sesi(self):
        if self.sesi: self._buka_folder(self.sesi)

    def putar_preview(self):
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self)
            return
        preview = self.sesi / "derived" / "preview.mp4"
        if not preview.exists():
            messagebox.showinfo("Preview belum ada", "Tekan Buat preview dahulu. Preview tidak dibuat otomatis agar proses cepat.", parent=self)
            return
        import subprocess
        try:
            subprocess.Popen(["xdg-open", str(preview)])
        except Exception as e:
            messagebox.showerror("Preview", f"Tidak dapat membuka preview: {e}", parent=self)

    # ----- ekspor -----
    def rentang(self):
        if not self.sesi: return (0, -1)
        x=baca_json(self.sesi/"edit"/"rentang.json", {})
        if x: return int(x["awal_indeks"]), int(x["akhir_indeks"])
        return (0, len(self.indeks)-1)

    def ekspor(self):
        if not self.sesi or not self.indeks: messagebox.showinfo("Preview belum siap", "Pilih rekaman dan buat preview dahulu.", parent=self); return
        fps=max(1,min(self.args.fps,int(self.fps_ekspor.get()))); sesi=self.sesi
        threading.Thread(target=self._ekspor_worker,args=(sesi,fps),daemon=True).start(); self.status.set("Mengekspor raw frame RGB-D dari rentang potong…")

    @staticmethod
    def _info_profile(profile, color, depth) -> dict:
        dev=profile.get_device(); ds=dev.first_depth_sensor(); cp=color.profile.as_video_stream_profile(); dp=depth.profile.as_video_stream_profile(); ci=cp.intrinsics; di=dp.intrinsics; ex=dp.get_extrinsics_to(cp)
        return {"depth_scale":float(ds.get_depth_scale()), "intrinsics_rgb_native":{"width":ci.width,"height":ci.height,"fx":ci.fx,"fy":ci.fy,"ppx":ci.ppx,"ppy":ci.ppy,"coeffs":list(ci.coeffs)}, "intrinsics_depth_native":{"width":di.width,"height":di.height,"fx":di.fx,"fy":di.fy,"ppx":di.ppx,"ppy":di.ppy,"coeffs":list(di.coeffs)}, "extrinsics_depth_ke_rgb":{"rotation_row_major":list(ex.rotation),"translation_meter":list(ex.translation)}}

    def _ekspor_worker(self, sesi: Path, fps_out: int):
        try:
            a,b=self.rentang(); langkah=self.args.fps/fps_out; nama=f"fps_{fps_out}_{stamp()}"; root=sesi/"exports"/nama; frames=root/"frames"; frames.mkdir(parents=True)
            meta_export={"sumber_bag":"../../source/raw.bag","rentang_indeks":[a,b],"fps_asli":self.args.fps,"fps_ekspor":fps_out,"non_destruktif":True,"frames":[]}; next_pick=float(a); n=0
            for i,(native,color,depth,aligned,profile) in enumerate(PembacaBag(self.bag(sesi)).iter_frame()):
                if i<a: continue
                if i>b: break
                if i+1e-6 < next_pick: continue
                next_pick += langkah; n+=1; folder=frames/f"frame_{n:06d}"; folder.mkdir()
                rgb=np.asanyarray(color.get_data()); raw=np.asanyarray(depth.get_data()); al=np.asanyarray(aligned.get_data())
                cv2.imwrite(str(folder/"color_raw.png"),rgb); cv2.imwrite(str(folder/"depth_raw.png"),raw); cv2.imwrite(str(folder/"depth_aligned_to_color.png"),al); np.save(folder/"depth_raw.npy",raw); np.save(folder/"depth_aligned_to_color.npy",al)
                for j,nm in ((1,"ir_left_raw.png"),(2,"ir_right_raw.png")):
                    ir=native.get_infrared_frame(j)
                    if ir: cv2.imwrite(str(folder/nm),np.asanyarray(ir.get_data()))
                info=self._info_profile(profile,color,depth); fm={"id":folder.name,"kategori":sesi.parent.name,"index_bag":i,"frame_number":int(depth.get_frame_number()),"timestamp_kamera_ms":float(depth.get_timestamp()),"format_depth":"Z16 native; meter=nilai*depth_scale","raw_bag_sumber":str(self.bag(sesi)),**info}
                tulis_json(folder/"frame.json",fm); meta_export["frames"].append({"folder":folder.name,"index_bag":i,"timestamp_ms":fm["timestamp_kamera_ms"]})
            tulis_json(root/"export.json",meta_export)
            self.q.put(("ekspor_selesai",(sesi,root,n)))
        except Exception as e: self.q.put(("error",f"Ekspor gagal: {e}"))

    # ----- label/ukur -----
    def muat_frame(self):
        if not self.sesi: messagebox.showinfo("Pilih sesi", "Pilih rekaman pada tab Tinjau dahulu.", parent=self); return
        semua=[]
        for root in sorted((self.sesi/"exports").glob("*/frames"),reverse=True) if (self.sesi/"exports").exists() else []: semua.extend(sorted(root.glob("frame_*")))
        self.frame_paths=[p for p in semua if baca_json(p / "frame_state.json", {"di_sampah": False}).get("di_sampah", False) == self.tampil_sampah_frame.get()]
        self.list_frame.delete(0,"end")
        for p in self.frame_paths: self.list_frame.insert("end",f"{p.parent.parent.name} / {p.name}")
        self.status.set(f"{len(self.frame_paths)} frame ekspor dimuat.")

    def pilih_frame(self):
        s=self.list_frame.curselection()
        if not s:return
        p=self.frame_paths[s[0]]; bgr=cv2.imread(str(p/"color_raw.png")); dep=np.load(p/"depth_aligned_to_color.npy")
        if bgr is None: return
        self.label_path=p; self.label_info=baca_json(p/"frame.json"); self.kanvas.set_frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),dep); self.ukur_status.set("Tandai poligon objek dan, untuk tinggi, bidang acuan.")

    def toggle_sampah_frame(self):
        if not self.label_path:
            messagebox.showinfo("Pilih frame", "Pilih frame yang ingin dipindahkan atau dipulihkan.", parent=self)
            return
        state = baca_json(self.label_path / "frame_state.json", {"di_sampah": False})
        state["di_sampah"] = not state.get("di_sampah", False)
        state["waktu_ubah_iso"] = datetime.now().isoformat(timespec="seconds")
        tulis_json(self.label_path / "frame_state.json", state)
        self.status.set("Status frame diubah; data raw dan label tidak dihapus.")
        self.label_path = None
        self.muat_frame()

    def ganti_mode(self): self.kanvas.mode=self.mode_label.get(); self.kanvas.render()
    def ganti_depth(self): self.kanvas.depth_alpha=float(self.depth_alpha.get()); self.kanvas.render()

    def _mask(self,nama):
        if self.kanvas.rgb is None:return None
        m=np.zeros(self.kanvas.rgb.shape[:2],np.uint8); pts=self.kanvas.poligon[nama]
        if len(pts)>=3:cv2.fillPoly(m,[np.round(pts).astype(np.int32)],1)
        return m

    def simpan_label(self):
        if not self.label_path or self.kanvas.rgb is None: messagebox.showinfo("Pilih frame", "Pilih satu frame ekspor dahulu.", parent=self);return
        pts=self.kanvas.poligon["objek"]
        if len(pts)<3: messagebox.showwarning("Poligon belum cukup", "Label YOLO membutuhkan minimal tiga titik objek.", parent=self);return
        h,w=self.kanvas.rgb.shape[:2]; kategori=self.label_info.get("kategori", "batu") if self.label_info else "batu"; cls=KELAS_YOLO[kategori]
        norm=" ".join(f"{v:.6f}" for p in pts for v in (p[0]/w,p[1]/h)); (self.label_path/"label_yolo_seg.txt").write_text(f"{cls} {norm}\n",encoding="utf-8")
        obj=self._mask("objek"); ref=self._mask("acuan"); cv2.imwrite(str(self.label_path/"mask_objek.png"),obj*255)
        if ref is not None and ref.any():cv2.imwrite(str(self.label_path/"mask_acuan.png"),ref*255)
        self.status.set(f"Label YOLO disimpan untuk {self.label_path.name}. Kelas: {kategori} ({cls}).")

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
        for frame in (self.sesi / "exports").glob("*/frames/frame_*"):
            if baca_json(frame / "frame_state.json", {"di_sampah": False}).get("di_sampah", False):
                continue
            label = frame / "label_yolo_seg.txt"
            if not label.exists():
                continue
            stem = f"{self.sesi.name}_{frame.parent.parent.name}_{frame.name}"
            shutil.copy2(frame / "color_raw.png", root / "images" / split / f"{stem}.png")
            shutil.copy2(label, root / "labels" / split / f"{stem}.txt")
            count += 1
        tulis_json(root / "dataset_yolo_seg.yaml", {"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test", "names": {str(v): k for k, v in KELAS_YOLO.items()}})
        self.status.set(f"Dataset YOLO diperbarui: {count} frame berlabel dari sesi ini masuk split {split}. Raw tetap ada di sesi.")

    def hitung_ukuran(self):
        if not self.label_path or not self.label_info:return
        obj,ref=self._mask("objek"),self._mask("acuan")
        if obj is None or ref is None or obj.sum()<3 or ref.sum()<3:return
        try:
            hasil=ukur(obj,ref,self.kanvas.depth,self.label_info)
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
                elif k=="preview_kamera_siap":
                    self.btn_preview_kamera.configure(text="Matikan preview kamera", state="normal")
                elif k=="preview_selesai":
                    sesi,n=v; self.sesi=sesi; self.indeks=[{"i":i} for i in range(n)]; self.awal.set(0);self.akhir.set(n-1);self.scale_awal.configure(to=n-1);self.scale_akhir.configure(to=n-1)
                    thumb=cv2.imread(str(sesi/"derived"/"thumbnail.jpg"));
                    if thumb is not None:
                        img=Image.fromarray(cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB));img.thumbnail((780,480));self.preview_photo=ImageTk.PhotoImage(img);self.preview_label.configure(image=self.preview_photo,text="")
                    self.durasi.set(f"Preview siap: {n} frame. Atur awal/akhir, lalu simpan potongan.");self.status.set("Preview dibuat hanya sebagai turunan. raw.bag tetap utuh.")
                elif k=="ekspor_selesai":
                    sesi,root,n=v;self.sesi=sesi;self.label_ekspor.configure(text=f"Ekspor selesai: {n} frame\n{root}");self.status.set("Ekspor selesai. Buka tab Label lalu tekan Muat frame dari sesi.")
        except queue.Empty:pass
        self.after(60,self._poll)

    def tutup(self):
        if self.sedang_rekam and not messagebox.askyesno("Rekaman masih berlangsung","Selesaikan rekaman dahulu agar raw.bag ditutup dengan benar. Tetap keluar?",parent=self):return
        self.simpan_preferensi()
        self.cam.hentikan();self.destroy()


def main():
    ap=argparse.ArgumentParser(description="Studio rekaman RGB-D D435 non-destruktif untuk dataset YOLO segmentation")
    ap.add_argument("--keluar", default=str(Path(__file__).resolve().parents[2] / "dataset" / "studio_rgbd"), help="folder utama dataset studio")
    ap.add_argument("--lebar",type=int,default=848);ap.add_argument("--tinggi",type=int,default=480);ap.add_argument("--fps",type=int,default=30)
    ap.add_argument("--preset",default="jangan",choices=("jangan","bawaan","high_accuracy","high_density"));ap.add_argument("--batas-frame",type=int,default=8000)
    Studio(ap.parse_args()).mainloop()


if __name__=="__main__":main()
