"""RGB-D Labelling Studio: rekam RGB-D D435 -> tinjau -> potong virtual -> ekspor -> label.

Prinsip keselamatan data
------------------------
`source/raw.*` (umumnya ``raw.db3`` pada SDK modern) adalah rekaman primer dan
TIDAK pernah diubah/dihapus oleh
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
import collections
import csv
import math
import json
import os
import queue
import re
import shutil
import sys
import threading
from collections import OrderedDict
import time
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont

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
    from . import catatan_rekaman as CR
    from . import salin_label, skala_ui
    from .atribut_tangga import AtributTangga
    from . import tempat_sampah as TS
    from .hitung_data import JendelaJumlahData, status_label
    from .ui_bantu import jadikan_modal, kolom_gulir
    from . import visual_depth, ir_selaras, sinkron, tema
    from . import sudut_kamera
    from .segmentasi_rfdetr_depth import usulkan as usulkan_rfdetr_depth
    from .segmentasi_convnext_depth import (hangatkan as hangatkan_convnext, panaskan_utas_ini,
                                         model_siap as model_convnext_siap, peta_kelas as peta_kelas_convnext,
                                         usulkan as usulkan_convnext_depth, lepas as lepas_convnext)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from studio_rgbd.kamera_rgbd import (KameraRGBD, cari_rekaman, nama_aman, nama_rekaman,
                                         tulis_json)
    from studio_rgbd.resolusi_kamera import PRESET_PILIHAN
    from studio_rgbd.geometri import Z_MAX, Z_MIN
    from studio_rgbd.pengukuran_objek import ukur
    from studio_rgbd.segmentasi_otomatis import usulkan as usulkan_segmentasi
    from studio_rgbd import catatan_rekaman as CR
    from studio_rgbd import sudut_kamera
    from studio_rgbd import salin_label, skala_ui
    from studio_rgbd.atribut_tangga import AtributTangga
    from studio_rgbd import tempat_sampah as TS
    from studio_rgbd.hitung_data import JendelaJumlahData, status_label
    from studio_rgbd.ui_bantu import jadikan_modal, kolom_gulir
    from studio_rgbd import visual_depth, ir_selaras, sinkron, tema
    from studio_rgbd.segmentasi_rfdetr_depth import usulkan as usulkan_rfdetr_depth
    from studio_rgbd.segmentasi_convnext_depth import (hangatkan as hangatkan_convnext, panaskan_utas_ini,
                                         model_siap as model_convnext_siap, peta_kelas as peta_kelas_convnext,
                                         usulkan as usulkan_convnext_depth, lepas as lepas_convnext)


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
WARNA_DATA_TEST = "#407EA3"          # warna TEST tab Split; tema gelap memetakannya ke biru muda
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
# Nomor 4 dahulu dipakai kelas lantai. Kelas itu ditarik: letak lantai kini
# diturunkan secara geometris dari riser paling ujung oleh pelacak, sehingga
# tidak perlu dilabeli tangan. Nomornya tetap dicadangkan, tidak dipakai ulang,
# supaya label lama yang sudah memuat angka 4 tidak berubah arti.
KELAS_YOLO = {"tapakan": 0, "bidang_tegak": 1, "batu": 2, "ramp": 3}
# Poligon biru selalu permukaan datar (tapakan). Poligon merah artinya
# bergantung kategori adegan yang sedang direkam.
KELAS_MODE = {
    "tangga_naik": {"acuan": "tapakan", "objek": "bidang_tegak"},
    "batu":        {"acuan": "tapakan", "objek": "batu"},
    "ramp_naik":   {"acuan": "tapakan", "objek": "ramp"},
}
# Semua jenis poligon yang dapat disunting di kanvas, dalam urutan tampilan.
MODE_MASK = ("objek", "acuan")


def baris_label_yolo(poligon: dict, kategori: str, w: int, h: int) -> tuple[list[str], dict]:
    """Baris label YOLO-seg (satu per poligon >= 3 titik) dan jumlah per kelas, dari poligon kanvas.

    Satu-satunya tempat label YOLO disusun, dipakai saat menyimpan label dan saat
    membangun folder dataset YOLO, sehingga keduanya selalu sama dengan poligon
    yang terlihat. Kunci poligon di luar MODE_MASK (mis. "lantai" peninggalan
    kelas yang sudah ditarik) diabaikan.
    """
    peta = KELAS_MODE.get(kategori, KELAS_MODE["tangga_naik"])
    baris, jumlah = [], {}
    for mode in ("acuan", "objek"):
        nama_kelas = peta[mode]
        cls = KELAS_YOLO[nama_kelas]
        sah = [poly for poly in (poligon or {}).get(mode, []) if len(poly) >= 3]
        jumlah[nama_kelas] = len(sah)
        for poly in sah:
            baris.append(f"{cls} " + " ".join(f"{v:.6f}" for pt in poly for v in (pt[0] / w, pt[1] / h)))
    return baris, jumlah


# ------------------------------------------------------------ alat potong
# Penghalang (tiang, pegangan, kaki orang) yang menutupi tangga dibuang dari
# mask dengan menarik satu pita di atasnya. Pita dikurangkan dari setiap
# poligon secara raster (resolusi SKALA_POTONG kali) lalu dikembalikan ke
# poligon; shapely tidak tersedia di .venv, OpenCV sudah.
SKALA_POTONG = 6
LUAS_MIN_POTONGAN = 12.0      # px^2 citra; serpihan lebih kecil dari ini dibuang


def pita_potong(p0, p1, lebar: float) -> np.ndarray | None:
    """Empat sudut pita selebar `lebar` piksel citra dengan sumbu p0->p1."""
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    arah = b - a
    panjang = float(np.hypot(*arah))
    if panjang < 2 or lebar <= 0:
        return None
    n = np.array([-arah[1], arah[0]]) / panjang * (lebar / 2)
    return np.array([a + n, b + n, b - n, a - n])


def _sambung_cincin(cincin: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Gabung beberapa cincin jadi SATU poligon lewat jembatan selebar nol.

    Cara yang sama dipakai Ultralytics (merge_multi_segment) untuk mask COCO
    yang terdiri dari beberapa bagian: format YOLO-seg hanya mengenal satu
    poligon per instance. Jembatan ditarik antara pasangan titik terdekat.
    """
    hasil = list(cincin[0])
    sisa = [list(c) for c in cincin[1:]]
    while sisa:
        terbaik = None
        A = np.asarray(hasil)
        for k, c in enumerate(sisa):
            C = np.asarray(c)
            d = ((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(int(d.argmin()), d.shape)
            if terbaik is None or d[i, j] < terbaik[0]:
                terbaik = (d[i, j], k, int(i), int(j))
        _, k, i, j = terbaik
        c = sisa.pop(k)
        hasil = hasil[:i + 1] + c[j:] + c[:j + 1] + hasil[i:]
    return hasil


def pecah_jembatan(poly) -> list[list[tuple[float, float]]]:
    """Pecah poligon bersambung jembatan (lihat _sambung_cincin) kembali ke cincin aslinya.

    Jembatan = sisi a->b yang juga ditempuh balik b->a. Dipakai untuk MENGGAMBAR
    dan merasterisasi mask, supaya garis sambungan tidak tampak sebagai garis
    melintang di atas penghalang dan tidak ikut terisi 1 px di mask PNG. Label
    YOLO tetap memakai poligon bersambung (satu baris per instance).
    """
    tumpuk = [[tuple(map(float, q)) for q in poly]]
    hasil = []
    while tumpuk:
        r = tumpuk.pop()
        n = len(r)
        sisi = {}
        for k in range(n):
            sisi.setdefault((r[k], r[(k + 1) % n]), []).append(k)
        pecah = None
        for k in range(n):
            a, b = r[k], r[(k + 1) % n]
            if a == b:
                continue
            balik = [m for m in sisi.get((b, a), []) if m != k]
            if balik:
                pecah = (k, balik[0])
                break
        if pecah is None:
            if n >= 3:
                hasil.append(r)
            continue
        k, _ = pecah
        r = r[k:] + r[:k]                                   # sisi jembatan kini r[0] -> r[1]
        m = next(i for i in range(1, n) if r[i] == r[1] and r[(i + 1) % n] == r[0])
        dalam, luar = r[1:m], [r[0]] + r[m + 2:]
        for bagian in (dalam, luar):
            bagian = [q for i, q in enumerate(bagian) if i == 0 or q != bagian[i - 1]]
            if len(bagian) >= 3:
                tumpuk.append(bagian)
    return hasil


def isi_poligon(m: np.ndarray, poly, nilai=1) -> None:
    """fillPoly yang memahami jembatan: setiap cincin diisi, lubang tetap lubang (aturan genap-ganjil)."""
    cincin = pecah_jembatan(poly)
    if cincin:
        cv2.fillPoly(m, [np.round(np.asarray(c)).astype(np.int32) for c in cincin], nilai)


def _rapikan_hasil(cincin, sumber_sisi, sumber_titik, tol=0.8):
    """Tempelkan titik hasil raster ke geometri asli supaya tepinya presisi, tidak bergerigi.

    Titik ditempel ke (1) vertex asli mask/penghalang atau perpotongan sisinya,
    lalu (2) ke sisi asli terdekat; titik segaris kemudian dibuang.
    """
    def proyeksi(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return a
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
        return (a[0] + t * dx, a[1] + t * dy)
    keluar = []
    for p in cincin:
        terbaik = None
        for q in sumber_titik:
            d = math.hypot(q[0] - p[0], q[1] - p[1])
            if d <= tol and (terbaik is None or d < terbaik[0]):
                terbaik = (d, q)
        if terbaik is None:
            for a, b in sumber_sisi:
                q = proyeksi(p, a, b)
                d = math.hypot(q[0] - p[0], q[1] - p[1])
                if d <= tol and (terbaik is None or d < terbaik[0]):
                    terbaik = (d, q)
        q = terbaik[1] if terbaik else p
        q = (round(float(q[0]), 2), round(float(q[1]), 2))
        if not keluar or keluar[-1] != q:
            keluar.append(q)
    while len(keluar) > 1 and keluar[0] == keluar[-1]:
        keluar.pop()
    # Buang titik yang segaris dengan kedua tetangganya (sisa tangga piksel).
    berubah = True
    while berubah and len(keluar) > 3:
        berubah = False
        for i in range(len(keluar)):
            a, p, b = keluar[i - 1], keluar[i], keluar[(i + 1) % len(keluar)]
            q = proyeksi(p, a, b)
            if math.hypot(q[0] - p[0], q[1] - p[1]) < 0.25:
                keluar.pop(i); berubah = True
                break
    return keluar


def _perpotongan(a, b, c, d):
    r = (b[0] - a[0], b[1] - a[1]); s_ = (d[0] - c[0], d[1] - c[1])
    den = r[0] * s_[1] - r[1] * s_[0]
    if abs(den) < 1e-12:
        return None
    t = ((c[0] - a[0]) * s_[1] - (c[1] - a[1]) * s_[0]) / den
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (a[0] + t * r[0], a[1] + t * r[1])
    return None


def kurangi_pita(poly, pita: np.ndarray, satukan: bool = True):
    """Kurangkan pita dari satu poligon.

    Mengembalikan None bila poligon tidak tersentuh (titiknya dibiarkan persis),
    atau daftar poligon hasil: kosong bila habis terpotong, satu poligon, atau
    beberapa bila terbelah dan `satukan` False. Lubang (pita yang tidak sampai
    menembus tepi) selalu disambung ke cincin luarnya agar potongan tetap ada.
    """
    if len(poly) < 3:
        return None
    cincin_asal = pecah_jembatan(poly)
    if not cincin_asal:
        return None
    P = np.asarray(poly, float)
    S = SKALA_POTONG
    x0, y0 = np.floor(P.min(0)) - 2
    x1, y1 = np.ceil(P.max(0)) + 2
    w, h = int((x1 - x0) * S) + 1, int((y1 - y0) * S) + 1
    asal = np.array([x0, y0])
    ke_raster = lambda Q: np.round((np.asarray(Q, float) - asal) * S).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [ke_raster(c) for c in cincin_asal], 255)   # jembatan tidak ikut terisi
    b = np.zeros_like(m)
    cv2.fillPoly(b, [ke_raster(pita)], 255)
    if not cv2.countNonZero(cv2.bitwise_and(m, b)):
        return None
    m[b > 0] = 0
    kontur, hirarki = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not kontur:
        return []
    hirarki = hirarki[0]
    Q = [tuple(map(float, q)) for q in np.asarray(pita, float)]
    sisi_P = [(c[i], c[(i + 1) % len(c)]) for c in cincin_asal for i in range(len(c))]
    sisi_Q = [(Q[i], Q[(i + 1) % len(Q)]) for i in range(len(Q))]
    titik_asli = [q for c in cincin_asal for q in c] + Q
    for a, b2 in sisi_P:
        for c, d in sisi_Q:
            x = _perpotongan(a, b2, c, d)
            if x is not None:
                titik_asli.append(x)
    ke_citra = lambda c: _rapikan_hasil([tuple(q) for q in (c.reshape(-1, 2) / S + asal)],
                                        sisi_P + sisi_Q, titik_asli)
    sederhana = lambda c: cv2.approxPolyDP(c, 0.5 * S, True)
    bagian = []
    for i, c in enumerate(kontur):
        if hirarki[i][3] != -1 or cv2.contourArea(c) < LUAS_MIN_POTONGAN * S * S:
            continue
        cincin = [ke_citra(sederhana(c))]
        anak = hirarki[i][2]
        while anak != -1:                                   # lubang di dalam cincin ini
            if cv2.contourArea(kontur[anak]) >= LUAS_MIN_POTONGAN * S * S:
                cincin.append(ke_citra(sederhana(kontur[anak])))
            anak = hirarki[anak][0]
        cincin = [r for r in cincin if len(r) >= 3]
        if cincin and cincin[0] is not None and len(cincin[0]) >= 3:
            bagian.append((cv2.contourArea(c), cincin))
    if not bagian:
        return []
    bagian.sort(key=lambda t: -t[0])                       # bagian terbesar jadi pangkal jembatan
    if satukan:
        return [_sambung_cincin([r for _, cin in bagian for r in cin])]
    return [_sambung_cincin(cin) for _, cin in bagian]


def baca_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default.copy()
    return json.loads(path.read_text(encoding="utf-8"))


# Penanda status label di daftar frame dan pita di bawah klip video. Label buatan
# Claude (auto-label massal, lihat runs/label_claude/tulis_final.py di repo
# Train) bertanda "oleh_claude"; "perlu_dicek" = keyakinan model rendah atau
# usulan lama yang tidak dikenali pengusul baru. Begitu disunting atau ditandai
# diperiksa, frame berstatus "diperiksa" seperti biasa.
IKON_LABEL = {"diperiksa": "\u2714 ", "claude_cek": "\U0001f916\u26a0 ", "claude": "\U0001f916 ",
              "claude_kosong": "\U0001f916\u2205 ", "usulan": "\u25cc ", "": "\u25cb "}
WARNA_LABEL = {"diperiksa": "#1E8E3E", "claude_cek": "#D93025", "claude": "#7C3AED", "claude_kosong": "#64748B",
               "usulan": "#B0A8A0"}
KETERANGAN_LABEL = ("\u2714 diperiksa   \U0001f916 Claude   \U0001f916\u26a0 perlu dicek\n"
                    "\U0001f916\u2205 Claude: tanpa tangga   \u25cc usulan lama   \u25cb belum ada label   \u2205 LATAR = mask kosong")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _shade(hexwarna: str, faktor: float) -> str:
    """Gelapkan/terangkan warna #RRGGBB (dipakai untuk keadaan hover tombol)."""
    h = hexwarna.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"#{min(255, int(r * faktor)):02x}{min(255, int(g * faktor)):02x}{min(255, int(b * faktor)):02x}"


def periksa_rekaman(bag: Path) -> str | None:
    """Alasan rekaman tidak bisa dibuka, atau None bila tampak utuh.

    librealsense MENGGANTUNG SELAMANYA di pipeline.start() bila .db3 tidak berisi
    frame (perekaman terputus; mis. 20260926_234424: hanya 16 KB, 1 pesan,
    status 'merekam'), dan itu terjadi di kode C sehingga Python tidak dapat
    menghentikannya. Karena itu isi .db3 diperiksa lebih dulu lewat SQLite
    (baca-saja, berhenti pada pesan pertama yang cocok, jadi cepat juga untuk
    rekaman 9 GB).
    """
    bag = Path(bag)
    if not bag.exists():
        return "berkas rekaman mentah tidak ada"
    if bag.stat().st_size < 1_000_000:
        return f"berkas rekaman hanya {bag.stat().st_size // 1024} KB (perekaman tidak selesai)"
    if bag.suffix == ".db3":
        import sqlite3
        try:
            c = sqlite3.connect(f"file:{bag}?mode=ro", uri=True, timeout=2)
            topik = {nama: tid for tid, nama in c.execute("select id, name from topics")}
            for jenis in ("Color", "Depth"):
                tid = next((t for n, t in topik.items() if f"/{jenis}_" in n and n.endswith("/image/data")), None)
                if tid is None or c.execute("select 1 from messages where topic_id=? limit 1", (tid,)).fetchone() is None:
                    return f"rekaman tidak berisi frame {jenis} (perekaman tidak selesai)"
        except sqlite3.Error as e:
            return f"berkas rekaman rusak ({e})"
    return None


class PembacaBag:
    """Membaca .bag secara offline; tidak menyentuh rekaman asli."""

    def __init__(self, bag: Path):
        self.bag = bag
        alasan = periksa_rekaman(bag)
        if alasan:
            # Jangan pernah serahkan rekaman rusak ke librealsense: start() menggantung.
            raise RuntimeError(f"Rekaman tidak dapat dibuka: {alasan}")

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


NAMA_ARAH = {-1: "sebelumnya", 1: "berikutnya"}


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
        self._ukuran = (skala_ui.px(104), skala_ui.px(32))
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
        w, h, r = max(1, w), max(1, h), skala_ui.px(7)
        warna = tema.latar(self.color if self.enabled else "#D8D0CB")
        if self._hover and self.enabled:
            warna = _shade(warna, 1.18 if tema.GELAP else 0.88)
        garis = tema.latar("#D8CAC1") if self.enabled and self.color == "#E8DDD5" else warna
        self.create_rectangle(r, 0, w-r, h, fill=warna, outline="")
        self.create_rectangle(0, r, w, h-r, fill=warna, outline="")
        for x, y, mulai in ((0, 0, 90), (w-2*r, 0, 0), (w-2*r, h-2*r, 270), (0, h-2*r, 180)):
            self.create_arc(x, y, x+2*r, y+2*r, start=mulai, extent=90,
                            style="pieslice", fill=warna, outline=warna)
        # Garis sangat tipis hanya pada tombol sekunder agar tetap terpisah
        # dari panel tanpa memberi kesan kotak berat.
        self.create_rectangle(r, 0, w-r, h-1, outline=garis)
        self.create_text(w / 2, h / 2, text=self.text, fill=tema.tinta(self.fg if self.enabled else MUTED),
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
        for k in ("width", "height"):
            if k in kw:
                kw[k] = skala_ui.px(kw[k])
        result = super().configure(cnf, **kw)
        self._gambar()
        return result


def _cerahkan(rgb: np.ndarray, kekuatan: float) -> np.ndarray:
    """Terangkan frame gelap: auto-eksposur, gamma, lalu CLAHE pada kanal L (LAB).

    CLAHE saja hanya menaikkan rerata kecerahan frame malam 211803 dari ~6 ke
    ~21 (dari 255), masih terlalu gelap untuk melabeli. Karena itu kanal L
    lebih dulu direntangkan dari persentil 1-99,5% ke rentang penuh
    (auto-eksposur per frame), bayangan diangkat dengan gamma, baru CLAHE
    menonjolkan tepi lokal. Pada frame tergelap (rerata 4) hasilnya ~92.
    Derau sensor ikut terlihat: itu memang yang terekam, bukan dibuat-buat.
    ``kekuatan`` 0 = tidak berubah, 1 = penuh.
    """
    if kekuatan <= 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lo, hi = np.percentile(l, (1, 99.5))
    hi = max(float(hi), float(lo) + 8.0)
    lf = np.clip((l.astype(np.float32) - lo) / (hi - lo), 0, 1) ** (1.0 / (1.0 + 1.5 * kekuatan))
    l2 = cv2.createCLAHE(clipLimit=2.0 + 2.0 * kekuatan, tileGridSize=(8, 8)).apply((lf * 255).astype(np.uint8))
    l2 = cv2.addWeighted(l, 1 - kekuatan, l2, kekuatan, 0) if kekuatan < 1 else l2
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)


def golong_arah_permukaan(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Golongkan piksel menurut arah permukaan dari depth: 1 menghadap atas, 2 menghadap kamera, 3 lain, 0 tanpa depth.

    Pada frame gelap riser dan tread sama-sama hitam di RGB, tetapi arah
    permukaannya berbeda 90 derajat dan depth tidak terpengaruh cahaya.
    Normal dihitung dari titik 3-D (depth dihaluskan), lalu dikelompokkan
    k-means menjadi tiga arah dominan: yang normalnya paling ke atas (sumbu
    Y kamera ke bawah) adalah permukaan datar, yang paling menghadap kamera
    adalah permukaan tegak di depan kamera. Ini ARAH permukaan, bukan label:
    dinding yang menghadap kamera ikut tergolong "menghadap kamera".
    """
    sah = depth_m > 0.15
    if int(sah.sum()) < 500:
        return np.zeros(depth_m.shape, np.uint8)
    z = depth_m.astype(np.float32).copy()
    z[~sah] = float(np.median(z[sah]))
    z = cv2.GaussianBlur(z, (0, 0), 3.0)
    h, w = z.shape
    L = 5
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    P = np.dstack([(u - cx) * z / fx, (v - cy) * z / fy, z])
    n = np.cross(np.roll(P, -L, 1) - np.roll(P, L, 1), np.roll(P, -L, 0) - np.roll(P, L, 0))
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    n[(n * P).sum(2) > 0] *= -1
    contoh = n[sah][np.random.default_rng(0).integers(0, int(sah.sum()), 6000)].astype(np.float32)
    _, _, pusat = cv2.kmeans(contoh, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
                             3, cv2.KMEANS_PP_CENTERS)
    pusat /= np.linalg.norm(pusat, axis=1, keepdims=True)
    atas = int(np.argmin(pusat[:, 1]))
    kamera = min((i for i in range(3) if i != atas), key=lambda i: pusat[i, 2])
    kelas = np.argmax(n.reshape(-1, 3) @ pusat.T, axis=1).reshape(h, w)
    peta = np.full((h, w), 3, np.uint8)
    peta[kelas == atas] = 1
    peta[kelas == kamera] = 2
    peta[~sah] = 0
    peta = cv2.medianBlur(peta, 7)
    for c in (1, 2):                                   # bercak derau kecil dibuang
        nk, lab, stat, _ = cv2.connectedComponentsWithStats((peta == c).astype(np.uint8), 8)
        kecil = np.flatnonzero(stat[:, cv2.CC_STAT_AREA] < 800)
        peta[np.isin(lab, kecil[kecil > 0])] = 0
    return peta


def arah_dari_kelas(peta_kelas: np.ndarray) -> np.ndarray:
    """Peta kelas model (1 riser, 2 tread) ke kode lapisan arah (1 tread/atas, 2 riser/hadap kamera)."""
    out = np.zeros(peta_kelas.shape, np.uint8)
    out[peta_kelas == 2] = 1
    out[peta_kelas == 1] = 2
    return out


def garis_pemandu(peta: np.ndarray) -> list[np.ndarray]:
    """Garis batas pemandu, dalam koordinat citra: tepi luar tangga dan tepi tiap tread.

    Kontur gabungan (riser+tread) memberi batas tangga terhadap dinding/lantai;
    kontur tread memberi batas tread-riser. Tepi tread yang bersebelahan dengan
    latar jatuh tepat di kontur gabungan, sehingga tidak tampil ganda.
    """
    ker = np.ones((5, 5), np.uint8)
    garis = []
    for biner in ((peta == 1) | (peta == 2), peta == 1):
        b = cv2.morphologyEx(biner.astype(np.uint8), cv2.MORPH_OPEN, ker)
        kontur, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        garis += [cv2.approxPolyDP(k, 1.0, True).reshape(-1, 2).astype(np.float32)
                  for k in kontur if cv2.contourArea(k) >= 400]
    return garis


def _lapisan_arah(rgb: np.ndarray, peta: np.ndarray, kekuatan: float) -> np.ndarray:
    """Warnai tread biru dan riser merah; garis batasnya digambar Canvas (garis_pemandu)."""
    if kekuatan <= 0 or peta is None:
        return rgb
    warna = np.zeros_like(rgb)
    warna[peta == 1] = (90, 169, 255)
    warna[peta == 2] = (255, 90, 90)
    warna[peta == 3] = (140, 140, 140)
    out = rgb.copy()
    ada = peta > 0
    out[ada] = cv2.addWeighted(rgb, 1 - 0.55 * kekuatan, warna, 0.55 * kekuatan, 0)[ada]
    return out


def _pertajam(rgb: np.ndarray, kekuatan: float) -> np.ndarray:
    """Unsharp mask: satu keluarga dengan penajaman Laplace.

    Kernel Laplace langsung (mis. [[0,-1,0],[-1,5,-1],[0,-1,0]]) menajamkan
    SETIAP piksel bertetangga, termasuk derau sensor satu-dua piksel pada
    citra gelap -- persis yang paling banyak dikuatkan justru bukan tepi
    anak tangga. Unsharp mask memakai selisih citra terhadap versi halusnya
    (Gaussian blur) sebagai "detail", sehingga radius blur menentukan skala
    detail yang ditajamkan; radius 1,5 px di sini menekankan tepi tanpa
    terlalu menajamkan derau satu piksel. ``kekuatan`` adalah seberapa besar
    detail itu ditambahkan kembali.
    """
    if kekuatan <= 0:
        return rgb
    halus = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(rgb, 1 + kekuatan, halus, -kekuatan, 0)


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
        self.depth_mode = "bidang"   # lihat visual_depth.MODE
        self.pakai_ir = False        # latar inframerah selaras, bukan RGB
        self.warna_bidang = 0.0      # 0..1 kekuatan warna ketinggian di atas latar
        self.garis_3d = False        # garis lipatan ujung/pangkal dari depth (visual_depth.garis_tepi)
        self.nomor_anak_tangga = []  # (x, y, teks, warna) koordinat citra; diisi panel_ukuran dari pelacakan video
        # Bayangan frame acuan (sebelum/sesudah): hanya tampilan, tidak ikut klik/simpan. Diisi Studio tiap frame dibuka.
        self.bayangan_rgb: np.ndarray | None = None
        self.bayangan_poligon: dict | None = None
        self.bayangan_alpha = 0.35
        self.penyedia_depth_vis = None   # Studio: fungsi(mode) -> citra hasil prefetch atau None
        self._depth_vis = self._depth_vis_key = self._depth_vis_obj = None
        self.kecerahan = 0.0     # 0 = asli, 1 = CLAHE penuh (lihat _cerahkan)
        self.ketajaman = 0.0     # 0 = asli, unsharp mask (lihat _pertajam)
        self.bantu_arah = 0.0    # 0 = mati; pemandu riser/tread (lihat _lapisan_arah, garis_pemandu)
        self.intrinsik = None    # dict fx, fy, cx, cy, depth_scale frame aktif; diisi Studio
        # Studio mengisi penyedia_peta: fungsi tanpa argumen yang mengembalikan
        # peta kelas model (0/1 riser/2 tread) frame aktif, atau None.
        self.penyedia_peta = None
        self._peta_arah = self._peta_arah_key = None
        self._garis_pemandu: list[np.ndarray] = []
        self.mask_alpha = 0.42
        self.mode = "objek"
        # Tiap kelas menampung BANYAK poligon: satu anak tangga = satu instance.
        # Dulu hanya satu poligon per kelas, jadi tangga dengan lima anak tangga
        # mustahil dilabeli sesuai dataset lama yang rata-rata 4,75 instance.
        self.poligon = {nama: [] for nama in MODE_MASK}
        # Mask aktif tidak selalu yang terakhir: pengguna dapat memilih mask
        # nomor 2 lalu menambah/menghapus titik hanya pada mask tersebut.
        self.aktif_indeks = {nama: None for nama in MODE_MASK}
        # Klik area kosong setelah mask selesai membuat instance baru. Flag ini
        # berubah hanya jika operator sengaja memilih/edit mask yang sudah ada.
        self._mask_dipilih_eksplisit = False
        self._drag = None
        self._drag_titik = None
        self._seleksi_awal = None
        self._blok_awal = None
        self._blok_kotak = None
        self.lapor_blok = None
        # Alat potong: pita di atas penghalang dikurangkan dari semua mask.
        self.lebar_potong = 14.0          # piksel citra
        self.satukan_potongan = True      # anak tangga yang terbelah tetap 1 mask
        self._potong_awal = None          # titik awal pita, koordinat citra
        self._potong_item = None
        self.lapor_potong = None
        # Alat masker penghalang: garis bentuk penghalang sembarang (klik titik
        # atau tarik bebas), lalu bentuk itu dikurangkan dari semua mask.
        self._penghalang: list[tuple[float, float]] = []
        self._penghalang_tarik = None     # None | {"mode": "titik", "i"} | {"mode": "laso", "layar"}
        self.saat_ganti_alat = None       # Studio menyelaraskan tombol alat bila kanvas berpindah alat sendiri
        self.saat_tukar = None            # Studio menukar warna mask yang diklik dengan alat Tukar
        self.bind("<Double-Button-1>", self._klik_ganda)
        # Dipanggil saat Ctrl+Z tidak punya langkah lagi di frame ini; Studio
        # memakainya untuk kembali ke frame yang baru disunting (riwayat lintas frame).
        self.saat_undo_habis = None
        # Alat tarik tanpa modifier, untuk papan sentuh yang sulit menahan
        # Ctrl atau Shift sambil menarik satu jari.
        self.alat = "normal"      # "normal" | "geser" | "blok" | "potong" | "penghalang" | "tukar"
        self.touchpad = True      # gulir dua jari menggeser; Ctrl+gulir memperbesar
        self._seleksi_kotak = None
        self.titik_dipilih: set[tuple[str, int, int]] = set()
        self.magnet_aktif = True
        # Cukup dekat untuk merapatkan batas, tetapi tidak sampai menarik
        # vertex ke garis/titik lain saat operator sedang membetulkan detail.
        self.jarak_magnet = 7  # piksel layar, tetap nyaman pada setiap zoom
        self._riwayat: list[dict[str, list[list[tuple[float, float]]]]] = []
        self._riwayat_pos = -1
        # True bila keadaan poligon saat ini hasil Ctrl+Z/redo, bukan suntingan
        # baru. Studio memakainya untuk membedakan "membatalkan usulan" dari
        # "sengaja menghapus semua mask" -- keduanya bisa berakhir di poligon
        # kosong yang sama persis.
        self.dari_undo = False
        self._render_terjadwal = False
        self._render_setelah = None
        self._zoom_cepat = False
        self._zoom_tajam_setelah = None
        self._photo = None
        self._photo_key = None
        self._canvas_background_key = None
        self._tampil_cache = None
        self._tampil_key = None
        self.bind("<Button-1>", self.tambah)
        self.bind("<B1-Motion>", self.geser_titik)
        self.bind("<ButtonRelease-1>", self.selesai_geser_titik)
        self.bind("<Shift-Button-1>", self.mulai_seleksi_titik)
        self.bind("<Shift-B1-Motion>", self.ubah_seleksi_titik)
        self.bind("<Shift-ButtonRelease-1>", self.selesai_seleksi_titik)
        self.bind("<Button-3>", self.hapus_titik_dipilih)
        # Ctrl+tarik = blok hapus. Dibedakan dari Shift+tarik yang hanya
        # MEMILIH titik: blok langsung menghapus begitu tombol dilepas,
        # sehingga membersihkan bercak tidak perlu dua langkah.
        self.bind("<Control-Button-1>", self.mulai_blok_hapus)
        self.bind("<Control-B1-Motion>", self.ubah_blok_hapus)
        self.bind("<Control-ButtonRelease-1>", self.selesai_blok_hapus)
        # Gulir dua jari menggeser pada mode touchpad dan memperbesar pada mode
        # tetikus. Ctrl+gulir selalu memperbesar, karena gerakan cubit papan
        # sentuh dikirim sebagai Ctrl+gulir. Gulir mendatar memakai Button-6/7.
        self.bind("<MouseWheel>", lambda e: self.roda(e, 1 if e.delta > 0 else -1))
        self.bind("<Button-4>", lambda e: self.roda(e, 1))
        self.bind("<Button-5>", lambda e: self.roda(e, -1))
        self.bind("<Control-MouseWheel>", lambda e: self.zoom_langkah(e, 1.15 if e.delta > 0 else 1 / 1.15))
        self.bind("<Control-Button-4>", lambda e: self.zoom_langkah(e, 1.15))
        self.bind("<Control-Button-5>", lambda e: self.zoom_langkah(e, 1 / 1.15))
        self.bind("<Shift-MouseWheel>", lambda e: self.geser_layar(110 if e.delta > 0 else -110, 0))
        self.bind("<Shift-Button-4>", lambda e: self.geser_layar(110, 0))
        self.bind("<Shift-Button-5>", lambda e: self.geser_layar(-110, 0))
        try:    # Tk 8.6 belum mengenal tombol gulir mendatar; Shift+gulir tetap ada
            self.bind("<Button-6>", lambda e: self.geser_layar(110, 0))
            self.bind("<Button-7>", lambda e: self.geser_layar(-110, 0))
        except tk.TclError:
            pass
        # Papan sentuh umumnya tidak punya tombol tengah, jadi Alt+tarik juga
        # menggeser gambar.
        self.bind("<Alt-Button-1>", self.pan_mulai)
        self.bind("<Alt-B1-Motion>", self.pan)
        # Satu handler untuk kedua pola keysym ("z" dan "Z"), memutuskan
        # undo/redo dari BIT Shift sungguhan (event.state), bukan dari huruf
        # keysym. Caps Lock mengubah huruf keysym tanpa menekan Shift --
        # membedakan lewat huruf membuat Ctrl+Z dengan Caps Lock menyala
        # salah terbaca sebagai Ctrl+Shift+Z (redo), yang biasanya tidak ada
        # yang bisa diulang dan terasa seperti "Ctrl+Z tidak bekerja".
        self.bind("<Control-z>", self._ctrl_z)
        self.bind("<Control-Z>", self._ctrl_z)
        self.bind("<Control-Shift-Z>", lambda _e: self.redo_riwayat())
        self.bind("<ButtonPress-2>", self.pan_mulai)
        self.bind("<B2-Motion>", self.pan)
        self.bind("<Configure>", lambda e: self.fit_bila_baru())

    def set_frame(self, rgb: np.ndarray, depth: np.ndarray):
        self.rgb, self.depth = rgb, depth
        self.poligon = {nama: [] for nama in MODE_MASK}
        self.aktif_indeks = {nama: None for nama in MODE_MASK}
        self._mask_dipilih_eksplisit = False
        self.titik_dipilih.clear()
        self._penghalang = []                     # garis penghalang frame lama tidak berlaku
        self._penghalang_tarik = None
        self.bayangan_rgb = self.bayangan_poligon = None
        self._photo = self._photo_key = self._tampil_cache = self._tampil_key = None
        self._canvas_background_key = None
        self.reset_riwayat()
        # Digambar sekali, sesudah pemanggil selesai memulihkan poligon draft.
        # Dulu fit dijadwalkan 20 ms lalu dan pemulihan draft menjadwalkan fit
        # lagi 25 ms lalu, sehingga di antara keduanya layar sempat menampilkan
        # frame baru tanpa mask lalu dengan mask: berkedip saat next/prev.
        self._fit_tertunda = True
        self.after_idle(self._fit_bila_tertunda)

    def _fit_bila_tertunda(self):
        if getattr(self, "_fit_tertunda", False):
            self.fit()

    def _salin_poligon(self):
        return {nama: [[tuple(titik) for titik in poly] for poly in daftar]
                for nama, daftar in self.poligon.items()}

    def reset_riwayat(self):
        self._riwayat = [self._salin_poligon()]
        self._riwayat_pos = 0
        self.dari_undo = False

    def catat_riwayat(self):
        self.dari_undo = False
        keadaan = self._salin_poligon()
        if self._riwayat_pos >= 0 and keadaan == self._riwayat[self._riwayat_pos]:
            return
        del self._riwayat[self._riwayat_pos + 1:]
        self._riwayat.append(keadaan)
        # Batasi memori bila satu frame memiliki banyak vertex.
        if len(self._riwayat) > 80:
            self._riwayat.pop(0)
        self._riwayat_pos = len(self._riwayat) - 1

    def _pulihkan_riwayat(self, posisi):
        self.poligon = {nama: [[tuple(titik) for titik in poly] for poly in daftar]
                        for nama, daftar in self._riwayat[posisi].items()}
        for nama, daftar in self.poligon.items():
            self.aktif_indeks[nama] = min(self.aktif_indeks.get(nama) or 0, len(daftar) - 1) if daftar else None
        self.titik_dipilih.clear()
        self.dari_undo = True
        self.render(); self.on_change()

    def _ctrl_z(self, e):
        return self.redo_riwayat() if (e.state & 0x1) else self.undo_riwayat()

    def undo_riwayat(self, _e=None):
        # Selama garis penghalang sedang dibuat, Ctrl+Z mundur satu titik garis
        # itu, bukan membatalkan suntingan mask di belakangnya.
        if self.alat == "penghalang" and self._penghalang:
            self.mundur_penghalang()
            return "break"
        if self._riwayat_pos <= 0:
            if self.saat_undo_habis is not None:
                self.saat_undo_habis()
            return "break"
        self._riwayat_pos -= 1
        self._pulihkan_riwayat(self._riwayat_pos)
        return "break"

    def redo_riwayat(self, _e=None):
        if self._riwayat_pos >= len(self._riwayat) - 1:
            return "break"
        self._riwayat_pos += 1
        self._pulihkan_riwayat(self._riwayat_pos)
        return "break"

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
        self._mask_dipilih_eksplisit = True
        self._beritahu_aktif(nama, indeks)
        self.render()
        return True

    def poligon_baru(self, nama: str | None = None):
        """Mulai instance baru tanpa menghapus mask yang sudah ada."""
        nama = nama or self.mode
        if not self.poligon[nama] or self.poligon[nama][-1]:
            self.poligon[nama].append([])
        self.aktif_indeks[nama] = len(self.poligon[nama]) - 1
        self._mask_dipilih_eksplisit = False
        self._beritahu_aktif(nama, self.aktif_indeks[nama])
        self.catat_riwayat(); self.render(); self.on_change()

    def hapus_mask_aktif(self):
        """Hapus instance aktif mode aktif, bukan seluruh kelas mask."""
        daftar = self.poligon[self.mode]
        if daftar:
            indeks = self.aktif_indeks.get(self.mode)
            indeks = len(daftar) - 1 if indeks is None else min(indeks, len(daftar) - 1)
            daftar.pop(indeks)
            self.aktif_indeks[self.mode] = min(indeks, len(daftar) - 1) if daftar else None
            self._beritahu_aktif(self.mode, self.aktif_indeks[self.mode])
            self.catat_riwayat(); self.render(); self.on_change()

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
        self._fit_tertunda = False
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        self.scale = max(0.05, min((self.winfo_width() - 28) / w, (self.winfo_height() - 28) / h))
        self.ox = (self.winfo_width() - w * self.scale) / 2
        self.oy = (self.winfo_height() - h * self.scale) / 2
        self.render()

    def gambar_tampil(self) -> np.ndarray:
        assert self.rgb is not None
        key = (id(self.rgb), id(self.depth), round(self.depth_alpha, 3), self.depth_mode, self.pakai_ir,
               round(self.warna_bidang, 3), self.garis_3d,
              round(self.kecerahan, 3), round(self.ketajaman, 3), round(self.bantu_arah, 3),
              id(self.bayangan_rgb), round(self.bayangan_alpha, 3))
        if self._tampil_key == key and self._tampil_cache is not None:
            return self._tampil_cache
        # Tajam setelah cerah: menajamkan derau mentah pada frame gelap lebih
        # kuat daripada menajamkan tepi yang CLAHE sudah bantu tonjolkan.
        latar = self.rgb
        if self.pakai_ir and self.penyedia_depth_vis is not None:
            # Inframerah selaras sebagai gambar dasar (frame gelap); RGB bila IR tidak ada.
            ir = self.penyedia_depth_vis("ir")
            if ir is not None:
                ir_rgb = cv2.cvtColor(ir, cv2.COLOR_GRAY2RGB) if ir.ndim == 2 else ir.copy()
                # Piksel tanpa data IR (tepi hasil geser ke RGB, lubang depth) diisi
                # RGB asli agar tidak muncul bar/lubang hitam; gambar tetap penuh.
                kosong = ir_rgb.max(2) == 0
                if kosong.any():
                    ir_rgb[kosong] = self.rgb[kosong]
                latar = ir_rgb
        out = _pertajam(_cerahkan(latar, self.kecerahan), self.ketajaman)
        if self.bayangan_rgb is not None and self.bayangan_rgb.shape == out.shape:
            lama = _pertajam(_cerahkan(self.bayangan_rgb, self.kecerahan), self.ketajaman)
            out = cv2.addWeighted(out, 1 - self.bayangan_alpha, lama, self.bayangan_alpha, 0)
        if self.warna_bidang > 0 and self.depth is not None and self.intrinsik is not None:
            bid = self.penyedia_depth_vis("bidang") if self.penyedia_depth_vis else None
            if bid is None:
                bid = visual_depth.gambar(self.depth, "bidang", self.intrinsik)
            out = visual_depth.warnai(out, bid, self.warna_bidang)
        if self.bantu_arah > 0 and self.depth is not None and self.intrinsik is not None:
            # Dihitung sekali per frame (~0,1 s). Kuncinya objek depth itu sendiri,
            # bukan id(): id bisa dipakai ulang oleh array frame berikutnya.
            if self._peta_arah_key is not self.depth:
                # Utamakan peta model RGB-D: pada 211803 garis batasnya median
                # 1,4 px (terang) / 2,9 px (gelap) dari label manual, sedangkan
                # arah normal depth 5,7 / 6,7 px dan recall riser 0,27 / 0,54.
                # Arah normal hanya cadangan bila model tidak tersedia.
                peta = None
                if self.penyedia_peta is not None:
                    try:
                        peta = self.penyedia_peta()
                    except Exception:                       # noqa: BLE001
                        peta = None
                if peta is not None:
                    self._peta_arah = arah_dari_kelas(peta)
                else:
                    i = self.intrinsik
                    self._peta_arah = golong_arah_permukaan(self.depth.astype(np.float32) * float(i["depth_scale"]),
                                                            i["fx"], i["fy"], i["cx"], i["cy"])
                self._garis_pemandu = garis_pemandu(self._peta_arah)
                self._peta_arah_key = self.depth
            out = _lapisan_arah(out, self._peta_arah, self.bantu_arah)
        if self.depth is not None and self.depth_alpha > 0:
            # Relief 3-D menampakkan bentuk anak tangga dari depth, juga pada frame
            # yang gelap total di kamera RGB (lihat visual_depth). Dihitung sekali
            # per frame dan per mode.
            kunci = (self.depth_mode, id(self.intrinsik))
            if self._depth_vis_key != kunci or self._depth_vis_obj is not self.depth:
                vis = self.penyedia_depth_vis(self.depth_mode) if self.penyedia_depth_vis else None
                if vis is None and self.depth_mode not in visual_depth.BUTUH_BERKAS:
                    vis = visual_depth.gambar(self.depth, self.depth_mode, self.intrinsik)
                if vis is not None and vis.ndim == 2:
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
                self._depth_vis = vis
                self._depth_vis_key, self._depth_vis_obj = kunci, self.depth
            if self._depth_vis is not None:                   # IR tidak ada: tampilkan RGB saja
                # Overlay hanya di piksel yang punya data; tepi hasil geser dan lubang
                # depth menampilkan gambar dasar, bukan bar hitam/transparan.
                campur = cv2.addWeighted(out, 1 - self.depth_alpha, self._depth_vis, self.depth_alpha, 0)
                ada = self._depth_vis.max(2) > 0
                out = np.where(ada[..., None], campur, out)
        if self.garis_3d and self.penyedia_depth_vis is not None:
            # Lapisan teratas: lipatan bentuk 3-D tidak boleh tertutup overlay lain.
            out = visual_depth.gambar_garis(out, self.penyedia_depth_vis("garis_3d"))
        self._tampil_key, self._tampil_cache = key, out
        return out

    def render(self):
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        # Pada zoom besar yang dibuat hanya bagian citra yang terlihat, ditambah
        # cadangan 35% layar di tiap sisi. Dulu seluruh citra diperbesar: pada
        # zoom 4x itu 3392x1920 piksel (~0,1-0,15 s per langkah zoom), padahal
        # yang tampak hanya selebar kanvas. Selama geser tetap di dalam
        # cadangan, foto lama dipakai ulang dan hanya dipindah (Canvas.move),
        # sehingga tidak ada area kosong yang sempat terlihat.
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        margin = 12
        s = self.scale
        dasar = (id(self.rgb), id(self.depth), round(self.depth_alpha, 3), self.depth_mode, self.pakai_ir,
                 round(self.warna_bidang, 3), self.garis_3d, round(self.kecerahan, 3),
                 round(self.ketajaman, 3), round(self.bantu_arah, 3), id(self.bayangan_rgb),
                 round(self.bayangan_alpha, 3), round(s, 5))
        lama = self._photo_key if self._photo is not None else None
        if w * h * s * s <= 2_500_000:
            wilayah = (0, 0, w, h)
        else:
            lihat = (max(0, int((-self.ox) / s)), max(0, int((-self.oy) / s)),
                     min(w, int(math.ceil((cw - self.ox) / s)) + 1), min(h, int(math.ceil((ch - self.oy) / s)) + 1))
            if lihat[2] <= lihat[0] or lihat[3] <= lihat[1]:
                wilayah = lama[-1] if lama is not None and lama[:-1] == dasar else (0, 0, w, h)
            elif (lama is not None and lama[:-1] == dasar and lama[-1][0] <= lihat[0] and lama[-1][1] <= lihat[1]
                  and lama[-1][2] >= lihat[2] and lama[-1][3] >= lihat[3]):
                wilayah = lama[-1]
            else:
                px, py = cw * .35 / s, ch * .35 / s
                wilayah = (max(0, int(lihat[0] - px)), max(0, int(lihat[1] - py)),
                           min(w, int(lihat[2] + px) + 1), min(h, int(lihat[3] + py) + 1))
        key = dasar + (wilayah,)
        # Drag titik bisa memanggil render puluhan kali/detik. Gambar dasar
        # cukup dibuat sekali; yang berubah hanya garis poligon di atasnya.
        if self._photo_key != key or self._photo is None:
            x0, y0, x1, y1 = wilayah
            size = (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))
            # LANCZOS di setiap tick roda mouse mahal. Ketika zoom masih
            # berlangsung gunakan BILINEAR, lalu render LANCZOS sekali setelah
            # pengguna berhenti.
            resample = Image.BILINEAR if self._zoom_cepat or size[0] * size[1] > 3_000_000 else Image.LANCZOS
            potong = np.ascontiguousarray(self.gambar_tampil()[y0:y1, x0:x1])
            self._photo = ImageTk.PhotoImage(Image.fromarray(potong).resize(size, resample))
            self._photo_key = key
        # Foto dasar mahal untuk digambar ulang. Saat titik digeser, yang
        # berubah hanyalah overlay sehingga gambar RGB tetap dipertahankan.
        gambar_x, gambar_y = self.ox + self._photo_key[-1][0] * s, self.oy + self._photo_key[-1][1] * s
        background_key = (self._photo_key, round(gambar_x, 2), round(gambar_y, 2))
        if self._canvas_background_key != background_key:
            self.delete("gambar")
            self.create_image(gambar_x, gambar_y, anchor="nw", image=self._photo, tags=("gambar",))
            self._canvas_background_key = background_key
        self.delete("overlay")
        g1, g2, g3, g4, g5 = map(skala_ui.px, (1, 2, 3, 4, 5))
        if self.bayangan_poligon:
            # Garis putus-putus di bawah mask aktif: hanya acuan, tidak bisa diklik.
            for nama, warna in (("objek", "#FFB3B3"), ("acuan", "#B3D6FF")):
                for poly in self.bayangan_poligon.get(nama, []):
                    for c in pecah_jembatan(poly) or [poly]:
                        pts = [v for x, y in list(c) + [c[0]] for v in (x * s + self.ox, y * s + self.oy)]
                        if len(pts) >= 6:
                            self.create_line(*pts, fill="#000000", width=g3, tags=("overlay",))
                            self.create_line(*pts, fill=warna, width=g1, dash=(g5 * 2, g4), tags=("overlay",))
        # merah = sisi tinggi (riser), biru = permukaan datar (tapakan). Warna
        # ini bukan hiasan: yang biru dipakai RANSAC sebagai BIDANG ACUAN, dan
        # yang merah sebagai objek yang diukur tingginya terhadap bidang itu.
        if self.bantu_arah > 0 and self._garis_pemandu and self._peta_arah_key is self.depth:
            # Garis pemandu vektor: tetap setipis 1 px pada zoom berapa pun, dan
            # putus-putus kuning agar tidak tertukar dengan mask yang disunting.
            for g in self._garis_pemandu:
                pts = (g * self.scale + (self.ox, self.oy)).ravel().tolist()
                pts += pts[:2]
                self.create_line(*pts, fill="#000000", width=3, tags=("overlay",))
                self.create_line(*pts, fill="#FFE34D", width=1, dash=(5, 3), tags=("overlay",))
        specs = (("objek", "#FF5A5A", "#FFD9D9"), ("acuan", "#5AA9FF", "#D7E9FF"))
        cepat = self._drag_titik is not None or self._zoom_cepat
        for nama, garis, titik in specs:
            for idx, poly in enumerate(self.poligon[nama], start=1):
                pts = [(x * self.scale + self.ox, y * self.scale + self.oy) for x, y in poly]
                if pts and (max(x for x, _ in pts) < -margin or min(x for x, _ in pts) > cw + margin
                            or max(y for _, y in pts) < -margin or min(y for _, y in pts) > ch + margin):
                    continue
                aktif = idx - 1 == self.aktif_indeks.get(nama)
                if len(pts) >= 3:
                    # Mask yang terbelah penghalang disimpan sebagai satu poligon
                    # bersambung jembatan; digambar per cincin supaya jembatan
                    # tidak tampak sebagai garis melintang di atas penghalang.
                    cincin = [[(x * self.scale + self.ox, y * self.scale + self.oy) for x, y in c]
                              for c in pecah_jembatan(poly)] or [pts]
                    lubang = [any(i != j and len(c2) >= 3 and cv2.pointPolygonTest(
                                  np.asarray(c2, np.float32), c[0], False) > 0
                                  for j, c2 in enumerate(cincin)) for i, c in enumerate(cincin)]
                    pola = "gray12" if self.mask_alpha <= .20 else "gray25" if self.mask_alpha <= .38 else "gray50" if self.mask_alpha <= .62 else "gray75"
                    for c, adalah_lubang in zip(cincin, lubang):
                        # Opacity 0 berarti hanya garis: titik tetap mudah dipilih
                        # tanpa menutupi warna RGB/depth di belakangnya.
                        if cepat:
                            # Halo gelap di bawah garis: pada citra gelap/berderau
                            # atau saat banyak mask bertumpuk, garis warna pastel
                            # (merah/biru muda) tipis 1-2 px mudah lebur ke latar.
                            # Halo memberi kontras tetap terlepas dari isi citra.
                            self.create_line(c + [c[0]], fill="#000000",
                                             width=g4 if aktif else g3, tags=("overlay",))
                            self.create_line(c + [c[0]], fill="#FFFFFF" if aktif else garis,
                                             width=g2 if aktif else g1, tags=("overlay",))
                        elif self.mask_alpha <= .01 or adalah_lubang:
                            self.create_polygon(c, fill="", outline="#000000",
                                                width=g5 if aktif else g4, tags=("overlay",))
                            self.create_polygon(c, fill="", outline="#FFFFFF" if aktif else garis,
                                                width=g3 if aktif else g2, tags=("overlay",))
                        else:
                            self.create_polygon(c, fill="", outline="#000000",
                                                width=g5 if aktif else g4, tags=("overlay",))
                            self.create_polygon(c, fill=garis, outline="#FFFFFF" if aktif else garis,
                                                stipple=pola, width=g3 if aktif else g2, tags=("overlay",))
                    utama = max(cincin, key=len)
                    cx = sum(p[0] for p in utama) / len(utama); cy = sum(p[1] for p in utama) / len(utama)
                    if not cepat:
                        self.create_text(cx, cy, text=str(idx), fill=garis if self.mask_alpha <= .01 else "white",
                                         font=("Segoe UI", 11, "bold"), tags=("overlay",))
                        self.create_text(cx, cy, text=str(idx), fill="",
                                         font=("Segoe UI", 11, "bold"), tags=("overlay",))
                elif len(pts) >= 2:
                    self.create_line(pts, fill="#000000", width=g4, tags=("overlay",))
                    self.create_line(pts, fill=garis, width=g2, tags=("overlay",))
                for it, (x, y) in enumerate(pts):
                    dipilih = (nama, idx - 1, it) in self.titik_dipilih
                    if cepat and not aktif and not dipilih:
                        continue
                    # Cincin hitam tipis di luar outline warna: titik tetap
                    # jelas terpisah dari garis mask dan dari latar gelap.
                    luar, dalam = skala_ui.px(7 if dipilih else 5), skala_ui.px(6 if dipilih else 4)
                    self.create_oval(x - luar, y - luar, x + luar, y + luar,
                                     fill="", outline="#000000", width=g1, tags=("overlay",))
                    self.create_oval(x - dalam, y - dalam, x + dalam, y + dalam,
                                     fill=titik, outline="#FFD400" if dipilih else garis,
                                     width=g2 if dipilih else g1, tags=("overlay",))
        for x, y, teks, warna in self.nomor_anak_tangga:
            # Nomor anak tangga dari pelacakan video utuh (panel_ukuran): teks
            # berhalo gelap agar terbaca di atas mask dan citra terang/gelap.
            cx, cy = x * self.scale + self.ox, y * self.scale + self.oy
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2)):
                self.create_text(cx + dx, cy + dy, text=teks, fill="#000000",
                                 font=("Segoe UI", 12, "bold"), tags=("overlay",))
            self.create_text(cx, cy, text=teks, fill=warna, font=("Segoe UI", 12, "bold"), tags=("overlay",))
        if self._penghalang:
            self._gambar_penghalang()
        if self._drag_titik is not None:
            self._gambar_lup(*self._drag_titik["layar"])

    def render_nanti(self, jeda=33):
        """Koaleskan redraw saat roda/drag; UI tidak mengantre ratusan resize."""
        if not self._render_terjadwal:
            self._render_terjadwal = True
            # Maksimum ~30 redraw/detik ketika drag/zoom, sehingga event
            # mouse tidak menumpuk ketika satu frame berisi banyak poligon.
            self._render_setelah = self.after(jeda, self._render_tertunda)

    def _render_tertunda(self):
        self._render_terjadwal = False
        self._render_setelah = None
        self.render()

    def _titik_dekat(self, x, y, batas=8, nama=None):
        batas = skala_ui.px(batas)
        """Cari titik hanya di kelas aktif bila ``nama`` diberikan.

        Ini mencegah vertex merah dan biru yang berdekatan saling mengambil
        klik/drag. Pengguna memilih kelasnya dulu lewat A atau S.
        """
        terbaik = None
        sumber = ((nama, self.poligon[nama]),) if nama is not None else self.poligon.items()
        for nama, daftar in sumber:
            for ip, poly in enumerate(daftar):
                for it, (px, py) in enumerate(poly):
                    d = (px * self.scale + self.ox - x) ** 2 + (py * self.scale + self.oy - y) ** 2
                    if d <= batas ** 2 and (terbaik is None or d < terbaik[0]):
                        terbaik = (d, nama, ip, it)
        return None if terbaik is None else terbaik[1:]

    def _titik_magnet(self, titik, kecualikan=()):
        """Kembalikan vertex terdekat untuk menyambungkan dua batas mask."""
        if not self.magnet_aktif:
            return titik
        kecualikan = set(kecualikan)
        batas = skala_ui.px(self.jarak_magnet) / max(self.scale, .01)
        terbaik = None
        x, y = titik
        for nama, daftar in self.poligon.items():
            for ip, poly in enumerate(daftar):
                for it, kandidat in enumerate(poly):
                    if (nama, ip, it) in kecualikan:
                        continue
                    jarak = (kandidat[0] - x) ** 2 + (kandidat[1] - y) ** 2
                    if jarak <= batas ** 2 and (terbaik is None or jarak < terbaik[0]):
                        terbaik = (jarak, kandidat)
        return terbaik[1] if terbaik is not None else titik

    def _sisi_terdekat(self, x, y, nama=None, batas=8, aktif_saja=False, kecualikan=()):
        batas = skala_ui.px(batas)
        """Cari sisi terdekat dan titik proyeksinya dalam koordinat gambar."""
        terbaik = None
        sumber = self.poligon.items() if nama is None else ((nama, self.poligon[nama]),)
        kecualikan = set(kecualikan)
        for nama_kini, daftar in sumber:
            aktif = self.aktif_indeks.get(nama_kini)
            indeks = [aktif] if aktif_saja and aktif is not None and 0 <= aktif < len(daftar) else range(len(daftar))
            for ip in indeks:
                poly = daftar[ip]
                if len(poly) < 2:
                    continue
                # Poligon lengkap memiliki sisi penutup terakhir -> pertama.
                jumlah_sisi = len(poly) if len(poly) >= 3 else len(poly) - 1
                for i in range(jumlah_sisi):
                    j = (i + 1) % len(poly)
                    if (nama_kini, ip, i) in kecualikan or (nama_kini, ip, j) in kecualikan:
                        continue
                    a, b = poly[i], poly[j]
                    ax, ay = a[0] * self.scale + self.ox, a[1] * self.scale + self.oy
                    bx, by = b[0] * self.scale + self.ox, b[1] * self.scale + self.oy
                    dx, dy = bx - ax, by - ay
                    panjang2 = dx * dx + dy * dy
                    if panjang2 <= 1e-6:
                        continue
                    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / panjang2))
                    sx, sy = ax + t * dx, ay + t * dy
                    jarak2 = (x - sx) ** 2 + (y - sy) ** 2
                    if jarak2 <= batas ** 2 and (terbaik is None or jarak2 < terbaik[0]):
                        # Proyeksi dalam koordinat citra menjaga titik tepat
                        # berada di atas garis, apa pun tingkat zoom-nya.
                        gx, gy = (sx - self.ox) / self.scale, (sy - self.oy) / self.scale
                        terbaik = (jarak2, nama_kini, ip, i, (gx, gy))
        return None if terbaik is None else terbaik[1:]

    def _magnet_garis(self, titik, kecualikan=()):
        if not self.magnet_aktif:
            return titik
        x = titik[0] * self.scale + self.ox
        y = titik[1] * self.scale + self.oy
        sisi = self._sisi_terdekat(x, y, batas=self.jarak_magnet, kecualikan=kecualikan)
        return sisi[3] if sisi is not None else titik

    def rapikan_semua_magnet(self):
        """Satukan kelompok vertex yang sudah berdekatan pada label lama."""
        titik = [(nama, ip, it, xy)
                 for nama, daftar in self.poligon.items()
                 for ip, poly in enumerate(daftar)
                 for it, xy in enumerate(poly)]
        if len(titik) < 2:
            return 0, 0
        batas = skala_ui.px(self.jarak_magnet) / max(self.scale, .01)
        induk = list(range(len(titik)))

        def akar(i):
            while induk[i] != i:
                induk[i] = induk[induk[i]]
                i = induk[i]
            return i

        def gabung(a, b):
            a, b = akar(a), akar(b)
            if a != b:
                induk[b] = a

        # Grid kecil menghindari pemeriksaan semua pasangan pada frame dengan
        # ratusan titik.
        sel: dict[tuple[int, int], list[int]] = {}
        for i, (_, _, _, (x, y)) in enumerate(titik):
            gx, gy = int(np.floor(x / batas)), int(np.floor(y / batas))
            for yy in range(gy - 1, gy + 2):
                for xx in range(gx - 1, gx + 2):
                    for j in sel.get((xx, yy), []):
                        u, v = titik[j][3]
                        if (u - x) ** 2 + (v - y) ** 2 <= batas ** 2:
                            gabung(i, j)
            sel.setdefault((gx, gy), []).append(i)

        kelompok: dict[int, list[int]] = {}
        for i in range(len(titik)):
            kelompok.setdefault(akar(i), []).append(i)
        grup = [anggota for anggota in kelompok.values() if len(anggota) > 1]
        for anggota in grup:
            # Median tahan terhadap satu vertex salah posisi di dekat batas.
            xs = [titik[i][3][0] for i in anggota]; ys = [titik[i][3][1] for i in anggota]
            target = (float(np.median(xs)), float(np.median(ys)))
            for i in anggota:
                nama, ip, it, _ = titik[i]
                self.poligon[nama][ip][it] = target
        # Setelah vertex yang hampir sama disatukan, rapikan juga vertex yang
        # berada sangat dekat dengan sisi mask lain agar sambungannya rapat.
        ke_garis = 0
        for nama, ip, it, _ in titik:
            if ip >= len(self.poligon[nama]) or it >= len(self.poligon[nama][ip]):
                continue
            asal = self.poligon[nama][ip][it]
            akhir = self._magnet_garis(asal, {(nama, ip, it)})
            if akhir != asal:
                self.poligon[nama][ip][it] = akhir
                ke_garis += 1
        if grup or ke_garis:
            self.titik_dipilih.clear()
            self.catat_riwayat(); self.render(); self.on_change()
        return len(grup), sum(len(x) for x in grup) + ke_garis

    def _gambar_lup(self, x, y):
        """Lup kecil pada titik aktif agar batas masking presisi saat zoom."""
        if self.rgb is None:
            return
        # Bebas dari batas citra: lup tetap mengikuti kursor persis saat
        # menggeser vertex di tepi, alih-alih melompat ke pojok (0, 0).
        gx, gy = self.canvas_ke_gambar_bebas(x, y)
        # Bidang yang lebih kecil dengan pembesaran lebih tinggi membuat
        # posisi vertex jauh lebih jelas, terutama saat dua batas berdekatan.
        r, faktor = 18, skala_ui.px(6)
        h, w = self.rgb.shape[:2]
        x0, y0, x1, y1 = max(0, int(gx-r)), max(0, int(gy-r)), min(w, int(gx+r)), min(h, int(gy+r))
        potong = self.gambar_tampil()[y0:y1, x0:x1]
        if potong.size == 0:
            return
        potong = cv2.resize(potong, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_NEAREST)
        foto = ImageTk.PhotoImage(Image.fromarray(potong))
        self._lup_photo = foto
        lx, ly = min(self.winfo_width()-potong.shape[1]-8, x+18), max(8, y-potong.shape[0]-18)
        self.create_rectangle(lx-3, ly-3, lx+potong.shape[1]+3, ly+potong.shape[0]+3, fill="#FFFDFC", outline="#FFD400", width=2, tags=("overlay",))
        self.create_image(lx, ly, anchor="nw", image=foto, tags=("overlay",))
        self.create_line(lx+potong.shape[1]/2-8, ly+potong.shape[0]/2, lx+potong.shape[1]/2+8, ly+potong.shape[0]/2, fill="#FFD400", width=2, tags=("overlay",))
        self.create_line(lx+potong.shape[1]/2, ly+potong.shape[0]/2-8, lx+potong.shape[1]/2, ly+potong.shape[0]/2+8, fill="#FFD400", width=2, tags=("overlay",))
        self.create_text(lx + 5, ly + 5, anchor="nw", text=f"{gx:.1f}, {gy:.1f}",
                         fill="#201A18", font=("Segoe UI", 8, "bold"), tags=("overlay",))

    def canvas_ke_gambar(self, x, y):
        if self.rgb is None:
            return None
        gx, gy = (x - self.ox) / self.scale, (y - self.oy) / self.scale
        h, w = self.rgb.shape[:2]
        return (gx, gy) if 0 <= gx < w and 0 <= gy < h else None

    def canvas_ke_gambar_bebas(self, x, y) -> tuple[float, float]:
        """Sama seperti :meth:`canvas_ke_gambar`, tanpa batas dan tanpa ``None``.

        Titik yang sedang digeser sering melewati tepi citra sesaat --
        gerakan tetikus jarang berhenti persis di piksel terakhir yang valid,
        apalagi untuk vertex yang memang berada di tepi. ``canvas_ke_gambar``
        mengembalikan ``None`` di situ, dan pemanggil yang menjadikannya acuan
        arah gerak (bukan posisi akhir) salah bila jatuh ke (0, 0): (0, 0)
        adalah SUDUT KIRI-ATAS citra, bukan "tidak ada gerakan". Posisi akhir
        titik tetap dijepit ke batas citra di tempat lain (:meth:`geser_titik`),
        jadi nilai bebas ini aman dipakai untuk menghitung SELISIH gerak.
        """
        return ((x - self.ox) / self.scale, (y - self.oy) / self.scale)

    def tambah(self, e):
        # Alat tarik yang sedang aktif menggantikan tugas modifier.
        if self.alat == "geser":
            return self.pan_mulai(e)
        if self.alat == "blok":
            return self.mulai_blok_hapus(e)
        if self.alat == "potong":
            return self.mulai_potong(e)
        if self.alat == "penghalang":
            return self.titik_penghalang(e)
        if self.alat == "tukar":
            kena = self.mask_di(e.x, e.y)
            if kena is not None and self.saat_tukar is not None:
                self.saat_tukar(*kena)
            return
        dekat = self._titik_dekat(e.x, e.y, nama=self.mode)
        if dekat is not None:
            nama, ip, it = dekat
            self.aktif_indeks[nama] = ip
            self._mask_dipilih_eksplisit = True
            self._beritahu_aktif(nama, ip)
            kunci = (nama, ip, it)
            if kunci not in self.titik_dipilih:
                self.titik_dipilih = {kunci}
            # Titik kembar di ujung jembatan (mask yang terbelah penghalang)
            # ikut bergerak, supaya sambungan tidak robek menjadi celah tipis.
            for n_, p_, t_ in list(self.titik_dipilih):
                if p_ < len(self.poligon[n_]) and t_ < len(self.poligon[n_][p_]):
                    xy = tuple(self.poligon[n_][p_][t_])
                    self.titik_dipilih |= {(n_, p_, j) for j, q in enumerate(self.poligon[n_][p_])
                                           if tuple(q) == xy}
            kelompok = [(n, p, t) for n, p, t in self.titik_dipilih
                         if p < len(self.poligon[n]) and t < len(self.poligon[n][p])]
            asal = {k: tuple(self.poligon[k[0]][k[1]][k[2]]) for k in kelompok}
            # Acuan arah gerak, BUKAN posisi akhir -- lihat canvas_ke_gambar_bebas.
            # Titik yang dipegang seringkali persis di tepi citra; awal (0, 0)
            # dari fallback lama membuat delta gerak salah total sejak tarikan
            # pertama, dan vertex melompat ke pojok kiri-atas citra.
            self._drag_titik = {"utama": kunci, "kelompok": kelompok, "asal": asal,
                                "awal": self.canvas_ke_gambar_bebas(e.x, e.y),
                                "layar": (e.x, e.y)}
            self.render()
            return
        p = self.canvas_ke_gambar(e.x, e.y)
        if p:
            sebelumnya_tidak_dipilih = self.aktif_indeks.get(self.mode) is None
            aktif = self._aktif(self.mode)
            # Prioritasnya sisi mask yang sedang dipilih, bukan garis lain
            # yang kebetulan lebih dekat pada gambar.
            sisi = self._sisi_terdekat(e.x, e.y, self.mode, aktif_saja=True)
            titik_baru = self._titik_magnet(p)
            if sisi is not None:
                _nama, ip, setelah, titik_baru = sisi
                self.aktif_indeks[self.mode] = ip
                self._mask_dipilih_eksplisit = True
                aktif = self.poligon[self.mode][ip]
                # Klik di atas garis selalu menyisipkan titik di sisi itu;
                # urutan polygon terjaga sehingga bagian atas tidak bolong.
                aktif.insert(setelah + 1, titik_baru)
                self._beritahu_aktif(self.mode, ip)
            else:
                # Mask baru hanya dimulai setelah operator dengan sengaja
                # unselect memakai klik kanan di area kosong. Klik kiri biasa
                # tetap meneruskan mask yang sedang dipilih.
                if len(aktif) >= 3 and not self._mask_dipilih_eksplisit and sebelumnya_tidak_dipilih:
                    self.poligon[self.mode].append([])
                    self.aktif_indeks[self.mode] = len(self.poligon[self.mode]) - 1
                    aktif = self.poligon[self.mode][-1]
                    self._beritahu_aktif(self.mode, self.aktif_indeks[self.mode])
                if titik_baru == p:
                    titik_baru = self._magnet_garis(p)
                aktif.append(titik_baru)
            self._urutkan_dari_bawah(self.mode, aktif)
            self.titik_dipilih.clear()
            self.catat_riwayat(); self.render(); self.on_change()

    def geser_titik(self, e):
        if self.alat == "geser":
            return self.pan(e)
        # Shift atau Ctrl dapat dilepas sebelum tombol tetikus, sehingga sisa
        # tarikan jatuh ke binding polos ini. Tarikan yang sedang berjalan
        # diteruskan ke pemiliknya, bukan diperlakukan sebagai geser titik.
        if self._blok_awal is not None:
            return self.ubah_blok_hapus(e)
        if self._potong_awal is not None:
            return self.ubah_potong(e)
        if self._penghalang_tarik is not None:
            return self.tarik_penghalang(e)
        if self._seleksi_awal is not None:
            return self.ubah_seleksi_titik(e)
        if self._drag_titik is None:
            return
        # Bebas dari batas citra: dulu geser berhenti (titik terasa "macet")
        # begitu kursor melewati tepi citra, karena canvas_ke_gambar mengem-
        # balikan None di situ dan pembaruan posisi dilewati begitu saja.
        p = self.canvas_ke_gambar_bebas(e.x, e.y)
        drag = self._drag_titik
        dx, dy = p[0] - drag["awal"][0], p[1] - drag["awal"][1]
        utama_asal = drag["asal"][drag["utama"]]
        calon_utama = (utama_asal[0] + dx, utama_asal[1] + dy)
        tempel = self._titik_magnet(calon_utama, drag["kelompok"])
        if tempel == calon_utama:
            tempel = self._magnet_garis(calon_utama, drag["kelompok"])
        dx, dy = tempel[0] - utama_asal[0], tempel[1] - utama_asal[1]
        # Bila beberapa titik dipilih, batasi translasi berdasarkan
        # SELURUH grup. Dulu hanya titik utama yang aman, sehingga vertex
        # lain dapat terdorong keluar dari citra seperti pada screenshot.
        h, w = self.rgb.shape[:2]
        xs = [drag["asal"][k][0] for k in drag["kelompok"]]
        ys = [drag["asal"][k][1] for k in drag["kelompok"]]
        dx = min(max(dx, -min(xs)), (w - 1) - max(xs))
        dy = min(max(dy, -min(ys)), (h - 1) - max(ys))
        for nama, ip, it in drag["kelompok"]:
            x0, y0 = drag["asal"][(nama, ip, it)]
            self.poligon[nama][ip][it] = (x0 + dx, y0 + dy)
        drag["layar"] = (e.x, e.y)
        self.render_nanti()

    def selesai_geser_titik(self, _e=None):
        # Pelepasan tombol tetikus sesudah Shift atau Ctrl dilepas tidak pernah
        # sampai ke handler bermodifier. Tanpa penerusan ini kotak tertinggal di
        # kanvas, seleksi tidak jadi, dan blok hapus tampak tidak bekerja.
        if _e is not None and self._blok_awal is not None:
            return self.selesai_blok_hapus(_e)
        if _e is not None and self._potong_awal is not None:
            return self.selesai_potong(_e)
        if self._penghalang_tarik is not None:
            return self.lepas_penghalang(_e)
        if _e is not None and self._seleksi_awal is not None:
            return self.selesai_seleksi_titik(_e)
        if self.alat == "geser":
            self._drag = None
            return "break"
        if self._drag_titik is not None:
            drag = self._drag_titik
            # Saat titik digeser melewati mask lain, nomor tetap mengikuti
            # urutan vertikal (bawah = 1) setelah mouse dilepas.
            for nama in {k[0] for k in drag["kelompok"]}:
                kunci = next(k for k in drag["kelompok"] if k[0] == nama)
                if 0 <= kunci[1] < len(self.poligon[nama]):
                    poly = self.poligon[nama][kunci[1]]
                    self._urutkan_dari_bawah(nama, poly)
            self._drag_titik = None
            self.titik_dipilih.clear()
            self.catat_riwayat(); self.render(); self.on_change()

    def mulai_seleksi_titik(self, e):
        """Shift+tarik membuat kotak seleksi tanpa menambah vertex baru."""
        self.focus_set()
        self._seleksi_awal = (e.x, e.y)
        self._seleksi_kotak = self.create_rectangle(e.x, e.y, e.x, e.y,
                                                    outline="#FFD400", dash=(4, 2), width=2)
        return "break"

    def ubah_seleksi_titik(self, e):
        if self._seleksi_awal is not None and self._seleksi_kotak is not None:
            self.coords(self._seleksi_kotak, *self._seleksi_awal, e.x, e.y)
        return "break"

    def selesai_seleksi_titik(self, e):
        if self._seleksi_awal is None:
            return "break"
        x0, y0 = self._seleksi_awal
        kiri, kanan = sorted((x0, e.x)); atas, bawah = sorted((y0, e.y))
        self.titik_dipilih = {
            (nama, ip, it)
            for nama, daftar in self.poligon.items()
            for ip, poly in enumerate(daftar)
            for it, (px, py) in enumerate(poly)
            if kiri <= px * self.scale + self.ox <= kanan and atas <= py * self.scale + self.oy <= bawah
        }
        if self._seleksi_kotak is not None:
            self.delete(self._seleksi_kotak)   # tanpa ini kotak tertinggal menumpuk di kanvas
        self._seleksi_awal = self._seleksi_kotak = None
        self.render()
        return "break"

    # ------------------------------------------------------------ blok hapus
    def mulai_blok_hapus(self, e):
        """Ctrl+tarik membuat kotak yang isinya langsung dihapus saat dilepas."""
        self.focus_set()
        self._blok_awal = (e.x, e.y)
        self._blok_kotak = self.create_rectangle(e.x, e.y, e.x, e.y,
                                                 outline="#E4573C", dash=(3, 2), width=2)
        return "break"

    def ubah_blok_hapus(self, e):
        if getattr(self, "_blok_awal", None) is not None and self._blok_kotak is not None:
            self.coords(self._blok_kotak, *self._blok_awal, e.x, e.y)
        return "break"

    def selesai_blok_hapus(self, e):
        """Hapus poligon yang berada di dalam kotak, lalu titik yang tersisa.

        Poligon yang MAYORITAS titiknya di dalam kotak dibuang seluruhnya,
        bukan disisakan sebagai pecahan. Menyisakan pecahan membuat pemakainya
        harus menghapus dua kali untuk satu bercak, dan itu kebalikan dari
        tujuan fitur ini. Poligon yang hanya tersenggol sebagian kehilangan
        titik yang tersentuh saja, sehingga koreksi halus tetap mungkin.
        """
        if getattr(self, "_blok_awal", None) is None:
            return "break"
        x0, y0 = self._blok_awal
        if self._blok_kotak is not None:
            self.delete(self._blok_kotak)
        self._blok_awal = self._blok_kotak = None
        kiri, kanan = sorted((x0, e.x)); atas, bawah = sorted((y0, e.y))
        if kanan - kiri < 4 or bawah - atas < 4:      # klik tidak sengaja
            return "break"
        di_dalam = lambda px, py: (kiri <= px * self.scale + self.ox <= kanan
                                   and atas <= py * self.scale + self.oy <= bawah)
        n_poly = n_titik = 0
        for nama, daftar in self.poligon.items():
            baru = []
            for poly in daftar:
                kena = [i for i, (px, py) in enumerate(poly) if di_dalam(px, py)]
                if not kena:
                    baru.append(poly); continue
                if len(kena) > len(poly) / 2:
                    n_poly += 1                        # buang seluruh poligon
                    continue
                sisa = [t for i, t in enumerate(poly) if i not in set(kena)]
                n_titik += len(kena)
                if len(sisa) >= 3:
                    baru.append(sisa)
                else:
                    n_poly += 1                        # sisa tidak lagi poligon
            self.poligon[nama] = baru
            self.aktif_indeks[nama] = (len(baru) - 1) if baru else None
        self.titik_dipilih = set()
        self.catat_riwayat(); self.render(); self.on_change()
        if getattr(self, "lapor_blok", None):
            self.lapor_blok(n_poly, n_titik)
        return "break"

    # ------------------------------------------------------------ potong
    def mulai_potong(self, e):
        """Alat potong: tarik garis di sepanjang penghalang (mis. tiang)."""
        self.focus_set()
        self._potong_awal = self.canvas_ke_gambar_bebas(e.x, e.y)
        self._gambar_pita(self._potong_awal)
        return "break"

    def ubah_potong(self, e):
        if self._potong_awal is not None:
            self._gambar_pita(self.canvas_ke_gambar_bebas(e.x, e.y))
        return "break"

    def _gambar_pita(self, ujung):
        if self._potong_item is not None:
            self.delete(self._potong_item)
            self._potong_item = None
        pita = pita_potong(self._potong_awal, ujung, self.lebar_potong)
        if pita is None:
            # Belum ada arah: tampilkan lingkaran selebar pita sebagai pratinjau ukuran.
            x, y = self._potong_awal[0] * self.scale + self.ox, self._potong_awal[1] * self.scale + self.oy
            r = max(2.0, self.lebar_potong * self.scale / 2)
            self._potong_item = self.create_oval(x - r, y - r, x + r, y + r, outline="#FFD400", width=2)
            return
        titik = [v for x, y in pita for v in (x * self.scale + self.ox, y * self.scale + self.oy)]
        self._potong_item = self.create_polygon(*titik, fill="#FFD400", stipple="gray50",
                                                outline="#FFD400", width=2)

    def atur_lebar_potong(self, faktor: float):
        self.lebar_potong = float(min(200.0, max(2.0, self.lebar_potong * faktor)))
        if self._potong_awal is not None:
            self._gambar_pita(self._potong_awal)
        return self.lebar_potong

    def selesai_potong(self, e):
        """Kurangkan pita dari SEMUA mask (merah dan biru) yang dilewatinya.

        Satu tarikan menggantikan menghapus titik satu per satu di sekitar
        penghalang. Anak tangga yang terbelah dua oleh tiang tetap satu mask
        (disambung jembatan selebar nol) kecuali `satukan_potongan` dimatikan.
        """
        if self._potong_awal is None:
            return "break"
        awal, self._potong_awal = self._potong_awal, None
        if self._potong_item is not None:
            self.delete(self._potong_item)
            self._potong_item = None
        pita = pita_potong(awal, self.canvas_ke_gambar_bebas(e.x, e.y), self.lebar_potong)
        if pita is None:                                   # klik tanpa tarikan
            return "break"
        # Pita tidak langsung memotong: ia menjadi garis penghalang 4 titik
        # yang masih bisa disunting (geser titik, sisip titik di sisi, klik
        # kanan hapus) lalu diterapkan dengan Enter, sama seperti alat Halang.
        self._penghalang = [tuple(map(float, p)) for p in pita]
        self.alat = "penghalang"
        self.config(cursor="pencil")
        if self.saat_ganti_alat:
            self.saat_ganti_alat("penghalang", "pita")
        self.render()
        return "break"

    # ------------------------------------------------- masker penghalang
    def _klik_ganda(self, e):
        # Klik ganda menutup garis penghalang. Pada alat lain klik kedua tetap
        # diperlakukan sebagai klik biasa, karena Tk menggantikan Button-1
        # kedua dengan Double-Button-1.
        if self.alat == "penghalang":
            return self.selesai_penghalang()
        if self.alat == "tukar":
            return None                # klik kedua jangan membalikkan lagi mask yang baru ditukar
        return self.tambah(e)

    def _layar(self, p):
        return p[0] * self.scale + self.ox, p[1] * self.scale + self.oy

    def _titik_penghalang_dekat(self, x, y, batas=9) -> int | None:
        batas = skala_ui.px(batas)
        terbaik = None
        for i, p in enumerate(self._penghalang):
            px, py = self._layar(p)
            d = math.hypot(px - x, py - y)
            if d <= batas and (terbaik is None or d < terbaik[0]):
                terbaik = (d, i)
        return None if terbaik is None else terbaik[1]

    def _sisi_penghalang_dekat(self, x, y, batas=6) -> tuple[int, tuple[float, float]] | None:
        batas = skala_ui.px(batas)
        """Sisi garis penghalang terdekat (termasuk sisi penutup) -> (indeks awal sisi, titik proyeksi)."""
        n = len(self._penghalang)
        if n < 2:
            return None
        terbaik = None
        for i in range(n if n >= 3 else 1):
            a, b = self._layar(self._penghalang[i]), self._layar(self._penghalang[(i + 1) % n])
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy
            if L2 < 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / L2))
            qx, qy = a[0] + t * dx, a[1] + t * dy
            d = math.hypot(qx - x, qy - y)
            if d <= batas and (terbaik is None or d < terbaik[0]):
                terbaik = (d, i, ((qx - self.ox) / self.scale, (qy - self.oy) / self.scale))
        return None if terbaik is None else (terbaik[1], terbaik[2])

    def titik_penghalang(self, e):
        """Penyuntingan garis penghalang, sama seperti mask biasa.

        Klik dekat titik = pegang dan geser titik itu; klik di sisi = sisipkan
        titik di sisi itu (lalu bisa langsung digeser); klik di tempat kosong =
        tambah titik di akhir, dan bila ditarik menjadi laso bebas. Potongan
        baru diterapkan dengan Enter / klik ganda / tombol Terapkan.
        """
        self.focus_set()
        i = self._titik_penghalang_dekat(e.x, e.y)
        if i is None:
            sisi = self._sisi_penghalang_dekat(e.x, e.y)
            if sisi is not None:
                i = sisi[0] + 1
                self._penghalang.insert(i, sisi[1])
        if i is not None:
            self._penghalang_tarik = {"mode": "titik", "i": i, "layar": (e.x, e.y),
                                      "awal": self.canvas_ke_gambar_bebas(e.x, e.y),
                                      "asal": tuple(self._penghalang[i])}
        else:
            self._penghalang.append(self._tempel_magnet(self.canvas_ke_gambar_bebas(e.x, e.y)))
            self._penghalang_tarik = {"mode": "laso", "layar": (e.x, e.y)}
        self.render()
        return "break"

    def tarik_penghalang(self, e):
        """Geser titik yang dipegang, atau gambar laso bebas (titik tiap ~5 px layar)."""
        t = self._penghalang_tarik
        if t["mode"] == "titik":
            if 0 <= t["i"] < len(self._penghalang):
                # Seperti menggeser titik mask: ikut arah gerak (bukan melompat ke
                # kursor), menempel magnet ke titik/garis mask, dan memakai lup.
                p = self.canvas_ke_gambar_bebas(e.x, e.y)
                calon = (t["asal"][0] + p[0] - t["awal"][0], t["asal"][1] + p[1] - t["awal"][1])
                self._penghalang[t["i"]] = self._tempel_magnet(calon)
                t["layar"] = (e.x, e.y)
                self.render_nanti()
        elif math.hypot(e.x - t["layar"][0], e.y - t["layar"][1]) >= 5:
            self._penghalang.append(self.canvas_ke_gambar_bebas(e.x, e.y))
            t["layar"] = (e.x, e.y)
            self.render_nanti()
        return "break"

    def _tempel_magnet(self, p):
        q = self._titik_magnet(p)
        return self._magnet_garis(p) if q == p else q

    def lepas_penghalang(self, _e=None):
        # Tidak ada yang diterapkan saat tombol dilepas: garis tetap bisa disunting.
        self._penghalang_tarik = None
        self.render()
        return "break"

    def hapus_titik_penghalang(self, e) -> bool:
        """Klik kanan pada titik garis penghalang menghapus titik itu."""
        i = self._titik_penghalang_dekat(e.x, e.y, batas=12)
        if i is None:
            return False
        self._penghalang.pop(i)
        self.render()
        return True

    def mundur_penghalang(self) -> bool:
        """Backspace: buang titik terakhir garis penghalang."""
        if not self._penghalang:
            return False
        self._penghalang.pop()
        self.render()
        return True

    def batal_penghalang(self) -> bool:
        if not self._penghalang:
            return False
        self._penghalang = []
        self._penghalang_tarik = None
        self.render()
        return True

    def selesai_penghalang(self):
        titik = [p for i, p in enumerate(self._penghalang) if i == 0 or p != self._penghalang[i - 1]]
        self._penghalang, self._penghalang_tarik = [], None
        if len(titik) < 3:
            self.render()
            if self.lapor_potong:
                self.lapor_potong(-1, 0, 0)
            return "break"
        self._kurangi_semua(np.asarray(titik, float))
        return "break"

    def _gambar_penghalang(self):
        # Digambar dengan gaya yang sama dengan mask merah/biru (halo hitam,
        # isi berpola, titik bercincin), hanya warnanya kuning.
        pts = [self._layar(p) for p in self._penghalang]
        garis, isi = "#FFD400", "#FFF3B0"
        if len(pts) >= 3:
            self.create_polygon(pts, fill="", outline="#000000", width=5, tags=("overlay",))
            self.create_polygon(pts, fill=garis, outline=garis, stipple="gray25", width=3, tags=("overlay",))
        elif len(pts) == 2:
            self.create_line(pts, fill="#000000", width=5, tags=("overlay",))
            self.create_line(pts, fill=garis, width=3, tags=("overlay",))
        t = self._penghalang_tarik or {}
        dipegang = t.get("i") if t.get("mode") == "titik" else None
        for i, (x, y) in enumerate(pts):
            r = 6 if i == dipegang else 4
            self.create_oval(x - r - 1, y - r - 1, x + r + 1, y + r + 1, fill="", outline="#000000", width=1, tags=("overlay",))
            self.create_oval(x - r, y - r, x + r, y + r, fill=isi, outline=garis, width=2 if i == dipegang else 1, tags=("overlay",))
        if dipegang is not None and "layar" in t:
            self._gambar_lup(*t["layar"])

    def _kurangi_semua(self, bentuk: np.ndarray) -> None:
        """Kurangkan satu bentuk (pita atau garis penghalang) dari SEMUA mask."""
        n_kena = n_belah = n_habis = 0
        for nama, daftar in self.poligon.items():
            baru = []
            for poly in daftar:
                hasil = kurangi_pita(poly, bentuk, satukan=False)
                if hasil is None:
                    baru.append(poly); continue
                n_kena += 1
                if not hasil:
                    n_habis += 1
                elif len(hasil) > 1:
                    n_belah += 1
                    if self.satukan_potongan:
                        hasil = [_sambung_cincin(hasil)]
                baru.extend(hasil)
            if baru != daftar:
                # Nomor tetap dari bawah ke atas seperti alat lain.
                baru.sort(key=lambda q: -sum(y for _, y in q) / len(q) if q else float("inf"))
                self.poligon[nama] = baru
                self.aktif_indeks[nama] = (len(baru) - 1) if baru else None
        self.titik_dipilih = set()
        self.catat_riwayat(); self.render(); self.on_change()
        if self.lapor_potong:
            self.lapor_potong(n_kena, n_belah, n_habis)

    def gabung_mask_dipilih(self) -> list[tuple[str, int]]:
        """Satukan mask yang titiknya terpilih (Shift+tarik) jadi SATU mask per kelas.

        Untuk memperbaiki anak tangga yang terlanjur dilabeli sebagai dua mask
        terpisah di kiri-kanan penghalang. Hasilnya sama dengan alat potong
        bermode "tetap 1 mask": cincin-cincin disambung jembatan selebar nol.
        Mengembalikan [(kelas, jumlah mask yang digabung)].
        """
        per_kelas: dict[str, set[int]] = {}
        for nama, ip, _it in self.titik_dipilih:
            if 0 <= ip < len(self.poligon[nama]) and len(self.poligon[nama][ip]) >= 3:
                per_kelas.setdefault(nama, set()).add(ip)
        laporan = []
        for nama, indeks in per_kelas.items():
            if len(indeks) < 2:
                continue
            daftar = self.poligon[nama]
            cincin = sorted((daftar[i] for i in indeks),
                            key=lambda q: -abs(cv2.contourArea(np.asarray(q, np.float32))))
            gabungan = _sambung_cincin(cincin)
            self.poligon[nama] = [p for i, p in enumerate(daftar) if i not in indeks] + [gabungan]
            self._urutkan_dari_bawah(nama, gabungan)
            self.mode = nama
            self._mask_dipilih_eksplisit = True
            self._beritahu_aktif(nama, self.aktif_indeks[nama])
            laporan.append((nama, len(indeks)))
        if laporan:
            self.titik_dipilih = set()
            self.catat_riwayat(); self.render(); self.on_change()
        return laporan

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
        self.catat_riwayat(); self.render(); self.on_change()

    def hapus_titik_dipilih(self, e):
        """Klik kanan hanya menghapus vertex yang dipilih, tidak yang terakhir."""
        if self.alat == "penghalang":
            # Di alat Halang klik kanan hanya menyentuh garis penghalang, tidak mask.
            self.hapus_titik_penghalang(e)
            return "break"
        dekat = self._titik_dekat(e.x, e.y, batas=14, nama=self.mode)
        if dekat is None:
            # Klik kanan kosong adalah aksi unselect yang eksplisit. Dengan
            # demikian klik kiri berikutnya aman untuk memulai mask baru.
            self.aktif_indeks[self.mode] = None
            self._mask_dipilih_eksplisit = False
            self.titik_dipilih.clear()
            self._beritahu_aktif(self.mode, None)
            self.render()
            return
        nama, ip, it = dekat
        poly = self.poligon[nama][ip]
        poly.pop(it)
        # Instance kosong tidak perlu dibiarkan sebagai mask semu.
        if not poly:
            self.poligon[nama].pop(ip)
        daftar = self.poligon[nama]
        self.aktif_indeks[nama] = min(ip, len(daftar) - 1) if daftar else None
        self.titik_dipilih.clear()
        self._beritahu_aktif(nama, self.aktif_indeks[nama])
        self.catat_riwayat(); self.render(); self.on_change()

    def _urutkan_dari_bawah(self, nama: str, aktif: list):
        """Nomor mask selalu dimulai dari bidang yang paling bawah di gambar."""
        daftar = self.poligon[nama]
        if not daftar:
            return

        def urutan(poly):
            # Koordinat Y besar berarti letaknya lebih bawah. Mask kosong
            # sementara ditempatkan terakhir hingga menerima titik pertama.
            return -sum(y for _, y in poly) / len(poly) if poly else float("inf")

        daftar.sort(key=urutan)
        self.aktif_indeks[nama] = daftar.index(aktif)

    def mask_di(self, x, y, batas=6):
        """(kelas, indeks) mask di titik kanvas (x, y), merah maupun biru.

        Tidak bergantung pada kelas yang sedang dipilih: mask yang salah warna
        justru sering berada di kelas lain. Bila beberapa mask menumpuk, yang
        terkecil menang karena yang besar masih bisa dikenai di tempat lain.
        """
        if self.rgb is None:
            return None
        titik = ((x - self.ox) / self.scale, (y - self.oy) / self.scale)
        toleransi = skala_ui.px(batas) / self.scale
        terbaik = None
        for nama, daftar in self.poligon.items():
            for ip, poly in enumerate(daftar):
                if len(poly) < 3:
                    continue
                kontur = np.array(poly, np.float32)
                jarak = cv2.pointPolygonTest(kontur, titik, True)
                if jarak >= -toleransi:
                    kunci = (jarak < 0, cv2.contourArea(kontur))
                    if terbaik is None or kunci < terbaik[0]:
                        terbaik = (kunci, nama, ip)
        return None if terbaik is None else terbaik[1:]

    def pindah_kelas(self, dari: str, indeks: int, ke: str) -> None:
        """Pindahkan satu mask utuh beserta titiknya ke kelas lain.

        Kesalahan yang sering terjadi adalah kelasnya, bukan bentuknya, jadi
        mask tidak perlu digambar ulang.
        """
        daftar = self.poligon[dari]
        poly = daftar.pop(indeks)
        aktif = self.aktif_indeks.get(dari)
        if aktif == indeks:
            aktif = (len(daftar) - 1) if daftar else None
        elif aktif is not None and aktif > indeks:
            aktif -= 1                                     # nomor di belakangnya maju satu
        self.aktif_indeks[dari] = aktif
        self.poligon[ke].append(poly)
        self.mode = ke
        self._mask_dipilih_eksplisit = True
        self._urutkan_dari_bawah(ke, poly)
        self.titik_dipilih.clear()
        self.catat_riwayat(); self.render(); self.on_change()
        self._beritahu_aktif(ke, self.aktif_indeks.get(ke))

    def ubah_kelas_aktif(self, ke: str) -> str | None:
        """Pindahkan mask aktif ke kelas ``ke``; kembalikan kelas asalnya, atau None bila tidak ada yang sah."""
        dari = self.mode
        if ke == dari or ke not in self.poligon:
            return None
        daftar = self.poligon[dari]
        indeks = self.aktif_indeks.get(dari)
        if indeks is None or not (0 <= indeks < len(daftar)) or len(daftar[indeks]) < 3:
            return None
        self.pindah_kelas(dari, indeks, ke)
        return dari

    def bersihkan(self, nama):
        self.poligon[nama] = []
        self.catat_riwayat(); self.render(); self.on_change()

    def zoom(self, e):
        self.zoom_langkah(e, 1.15 if e.delta > 0 else 1 / 1.15)

    def zoom_langkah(self, e, faktor):
        return self.zoom_di(e.x, e.y, faktor)

    def zoom_tengah(self, faktor: float):
        """Perbesar dari tengah kanvas, untuk tombol dan papan ketik.

        Papan sentuh tidak mengirim gerakan cubit ke Tk, dan menahan Ctrl sambil
        menggulir dua jari tidak selalu nyaman, jadi zoom harus tetap ada tanpa
        tetikus maupun modifier.
        """
        self.zoom_di(self.winfo_width() // 2, self.winfo_height() // 2, faktor)
        return "break"

    def zoom_di(self, ex: int, ey: int, faktor: float):
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        sebelumnya = self.canvas_ke_gambar(ex, ey)
        if sebelumnya is None:
            # Jika panning sebelumnya membawa gambar keluar layar, wheel
            # tidak boleh memperbesar area kosong berwarna coklat.
            self.fit()
            return
        # Lup 6× memberi presisi lebih baik daripada membuat gambar dasar
        # sampai 8× (yang terlalu berat untuk Tk pada resolusi D435).
        self.scale = max(0.08, min(4.0, self.scale * faktor))
        if sebelumnya:
            self.ox, self.oy = ex - sebelumnya[0] * self.scale, ey - sebelumnya[1] * self.scale
        if self.ox + w * self.scale < 0 or self.oy + h * self.scale < 0 or self.ox > self.winfo_width() or self.oy > self.winfo_height():
            self.fit()
            return
        self._zoom_cepat = True
        if self._zoom_tajam_setelah is not None:
            self.after_cancel(self._zoom_tajam_setelah)
        self._zoom_tajam_setelah = self.after(160, self._zoom_selesai)
        self.render_nanti(80)

    def _zoom_selesai(self):
        """Ganti preview cepat menjadi gambar tajam hanya sekali per zoom."""
        self._zoom_tajam_setelah = None
        self._zoom_cepat = False
        self._photo_key = None
        self._canvas_background_key = None
        # Jangan menunggu event mouse berikutnya: beberapa driver Tk dapat
        # kehilangan layer PhotoImage setelah resize cepat.
        self.render()

    def muat_ulang_gambar(self):
        """Pulihkan layer RGB bila compositor Tk sempat hanya menampilkan mask."""
        if self.rgb is None:
            return
        self._photo = self._photo_key = None
        self._canvas_background_key = None
        self.render()

    def pan_mulai(self, e): self._drag = (e.x, e.y, self.ox, self.oy)
    def pan(self, e):
        if self._drag:
            x, y, ox, oy = self._drag
            baru_x, baru_y = ox + e.x - x, oy + e.y - y
            dx, dy = baru_x - self.ox, baru_y - self.oy
            self.ox, self.oy = baru_x, baru_y
            # Panning tidak membutuhkan resize atau membangun ulang ribuan
            # titik; Canvas dapat menggeser dua layer secara native.
            self.move("gambar", dx, dy); self.move("overlay", dx, dy)
            self._canvas_background_key = None
            # Setelah gerakan berhenti sejenak, muat tile baru yang menutup
            # viewport. Selama drag, Canvas tetap memakai geser native cepat.
            self.render_nanti(80)

    def roda(self, e, arah: int):
        """Gulir: menggeser pada mode touchpad, memperbesar pada mode tetikus."""
        if self.touchpad:
            return self.geser_layar(0, arah * 110)
        return self.zoom_langkah(e, 1.15 if arah > 0 else 1 / 1.15)

    def geser_layar(self, dx: int, dy: int):
        """Geser gambar sejauh dx, dy piksel layar lewat jalur cepat Canvas."""
        if self.rgb is None:
            return "break"
        self.ox, self.oy = self.ox + dx, self.oy + dy
        self.move("gambar", dx, dy); self.move("overlay", dx, dy)
        self._canvas_background_key = None
        self.render_nanti(80)
        return "break"

    def set_alat(self, nama: str) -> None:
        """Pilih alat tarik; kursor ikut berubah supaya keadaannya terlihat."""
        self.alat = nama if nama in ("normal", "geser", "blok", "potong", "penghalang", "tukar") else "normal"
        if self.alat in ("normal", "blok") and self._penghalang:
            # Garis setengah jadi dibuang saat kembali menyunting mask; saat
            # menggeser gambar atau membuat pita, garis tetap disimpan.
            self.batal_penghalang()
        self.config(cursor={"geser": "fleur", "blok": "X_cursor", "potong": "tcross",
                            "penghalang": "pencil", "tukar": "exchange"}.get(self.alat, ""))


class Studio(tk.Tk):
    def __init__(self, args):
        # className = StartupWMClass di desktop/studio-rgbd.desktop: dock mengelompokkan
        # jendela ini dengan ikon peluncurnya, bukan sebagai jendela "Tk" tanpa ikon.
        # Tk 8.6.17 (Ubuntu 26.04) menulis kelas WM_CLASS hanya huruf pertama kapital
        # ("StudioRGBD" -> "Studiorgbd"), jadi nama kelas ditulis dalam bentuk itu.
        super().__init__(className="Studiorgbd")
        try:
            # Ikon orisinal (desktop/buat_ikon.py), bukan tangkapan layar aplikasi.
            self._ikon = tk.PhotoImage(file=str(Path(__file__).with_name("ikon_studio.png")))
            self.iconphoto(True, self._ikon)
        except tk.TclError:
            pass
        self.args = args
        self.root_data = Path(args.keluar).expanduser().resolve()
        self.cam = KameraRGBD(args.lebar, args.tinggi, args.fps, args.preset, args.batas_frame,
                               exposure_us=getattr(args, 'exposure', 0))
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
        self._riwayat_frame: "OrderedDict[str, dict]" = OrderedDict()   # Ctrl+Z lintas frame
        self.label_path: Path | None = None
        self.label_info: dict | None = None
        self.preview_photo = None
        self._preview_w = 480               # lebar render preview; mengikuti widget
        tema.pasang(self)                       # tema gelap: terjemahkan warna sebelum widget dibuat
        skala_ui.pasang()                       # angka piksel kode ikut skala layar
        self._skala_diberitahu = skala_ui.FAKTOR
        self.after(15000, self._pantau_skala)
        self.title("RGB-D Labelling Studio — Rekam, Tinjau, Ekspor, Label")
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
        # Frame tetangga dibaca lebih dulu di thread latar. Disk dataset ini
        # terukur lambat (rata-rata ~56 ms per permintaan baca, sesekali
        # 0,8-2 s untuk satu PNG saat partisi penuh), sehingga membaca frame
        # baru baru saat Space ditekan membuat navigasi tersendat.
        self._cache_frame: "OrderedDict[Path, dict]" = OrderedDict()
        self._kunci_cache = threading.Lock()
        self._syarat_prefetch = threading.Condition()
        self._antrean_prefetch: list[Path] = []
        threading.Thread(target=self._pekerja_prefetch, daemon=True).start()
        self._jam_x: tuple[int, float] | None = None       # (waktu event X, jam lokal) untuk membuang antrean auto-repeat
        self._navigasi_selesai = 0.0
        self.depth_alpha = DoubleVar(value=0.28)
        self.mask_alpha = DoubleVar(value=0.42)
        self.kecerahan = DoubleVar(value=float(preferensi.get("kecerahan", 0.0)))
        self.ketajaman = DoubleVar(value=float(preferensi.get("ketajaman", 0.0)))
        self.bantu_arah = DoubleVar(value=float(preferensi.get("bantu_arah", 0.0)))
        self.depth_mode = StringVar(value=preferensi.get("depth_mode", "bidang"))
        self.pakai_ir = BooleanVar(value=bool(preferensi.get("pakai_ir", False)))
        self.warna_bidang = DoubleVar(value=float(preferensi.get("warna_bidang", 0.0)))
        self.garis_3d = BooleanVar(value=bool(preferensi.get("garis_3d", False)))
        self.bayangan_aktif = BooleanVar(value=bool(preferensi.get("bayangan_aktif", False)))
        self.bayangan_alpha = DoubleVar(value=float(preferensi.get("bayangan_alpha", 0.35)))
        self.bayangan_label = BooleanVar(value=bool(preferensi.get("bayangan_label", True)))
        self.frame_acuan = IntVar(value=int(preferensi.get("frame_acuan", -1)))     # -1 sebelumnya, +1 berikutnya
        self.salin_selaras = BooleanVar(value=bool(preferensi.get("salin_selaras", True)))
        self.salin_otomatis = BooleanVar(value=bool(preferensi.get("salin_otomatis", False)))
        self.magnet_titik = BooleanVar(value=True)
        # Model pengusul di GPU: auto-usulan frame tanpa label, peta model Bantu arah,
        # tombol Rekomendasi/Batch. Mati = model tidak dimuat (dan dilepas bila sudah).
        self.model_gpu = BooleanVar(value=bool(preferensi.get("model_gpu", True)))
        self._model_gpu_aktif = bool(self.model_gpu.get())   # salinan untuk thread prefetch
        self.mode_label = StringVar(value="objek")
        self.kontrol_label = StringVar(value=preferensi.get("kontrol_label", "mudah"))
        # Navigasi frame dikunci sampai operator sengaja memilih warna mask.
        # Ini menghindari frame berganti saat sedang menyunting titik.
        self._mode_label_dipilih = False
        self.ukur_status = StringVar(value="Tandai sisi tinggi/riser (merah) dan permukaan datar/tapakan (biru).")
        self.alat_label = StringVar(value="normal")
        self._tahan_status = False        # autosave berikutnya tidak menimpa hasil salin/tukar di bilah status
        self.mode_touchpad = BooleanVar(value=bool(preferensi.get("mode_touchpad", True)))
        self.lebar_potong = DoubleVar(value=float(preferensi.get("lebar_potong", 14.0)))
        self.satukan_potongan = BooleanVar(value=bool(preferensi.get("satukan_potongan", True)))
        self.tampil_sampah = BooleanVar(value=False)
        self.tampil_sampah_frame = BooleanVar(value=False)
        self._siapkan_root(); self._buat_ui()
        self.tabs.bind("<<NotebookTabChanged>>", self.ganti_tab)
        self.after(30, self._poll)
        self.after(120, self.muat_daftar)   # daftar tangga langsung terlihat saat aplikasi dibuka
        if self._model_gpu_aktif:
            self.after(500, self._hangatkan_model_async)
        self.status.set("Siap untuk labeling. Kamera dinyalakan hanya saat Preview Kamera atau Mulai rekam ditekan."
                        + ("" if self._model_gpu_aktif else " Auto-label/model GPU mati (tab Otomatis)."))

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
        # Rekaman tanpa video mentah tetap ditampilkan bila punya frame ekspor
        # (mis. hasil Tarik dataset dari Hugging Face): masih bisa dilabel ulang.
        return [p for p in hasil if p.is_dir() and (self.bag(p).exists() or self._ada_ekspor(p))]

    @staticmethod
    def _ada_ekspor(sesi: Path) -> bool:
        fr = sesi / "exports" / "frames"
        return fr.exists() and any(d.is_dir() for d in fr.iterdir())

    # ----- UI -----
    def _gaya(self):
        """Gaya desktop yang ringan: warna lembut, garis tipis, radius kecil."""
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        L, T = tema.latar, tema.tinta
        s.configure("TNotebook", background=L(BG), borderwidth=0, tabmargins=(2, 0, 2, 0))
        s.configure("Treeview", rowheight=tkfont.Font(font="TkDefaultFont").metrics("linespace") + skala_ui.px(5))
        s.configure("TNotebook.Tab", background=L("#EEE7E2"), foreground=T(MUTED),
                    padding=(skala_ui.px(22), skala_ui.px(11)), font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", L(PANEL))], foreground=[("selected", T(ACCENT))])
        s.configure("TCombobox", fieldbackground=L("#FFFDFC"), background=L("#FFFDFC"),
                    foreground=T(INK), padding=skala_ui.px(5), bordercolor=L("#DED7D2"), lightcolor=L("#DED7D2"), darkcolor=L("#DED7D2"),
                    arrowcolor=T(INK))
        s.map("TCombobox", fieldbackground=[("readonly", L("#FFFDFC"))], foreground=[("readonly", T(INK))],
              selectbackground=[("readonly", L("#FFFDFC"))], selectforeground=[("readonly", T(INK))])
        if tema.GELAP:
            garis = "#3A332F"
            s.configure("TNotebook", bordercolor=garis, lightcolor=garis, darkcolor=garis)
            s.configure("TNotebook.Tab", bordercolor=garis, lightcolor=garis, darkcolor=garis)
            s.map("TNotebook.Tab", lightcolor=[("selected", L(PANEL))], bordercolor=[("selected", garis)])
            for gaya in ("TScrollbar", "Vertical.TScrollbar", "Horizontal.TScrollbar"):
                s.configure(gaya, background="#37302C", troughcolor="#1D1A18", bordercolor="#1D1A18",
                            arrowcolor=T(INK), lightcolor="#37302C", darkcolor="#37302C")
            s.configure("Treeview", background="#1D1A18", fieldbackground="#1D1A18", foreground=T(INK))
            s.configure("Treeview.Heading", background="#2C2724", foreground=T(INK))
            s.map("Treeview", background=[("selected", "#4A3A31")])
            for gaya in ("TCheckbutton", "TRadiobutton", "TLabel", "TFrame"):
                s.configure(gaya, background=L(PANEL), foreground=T(INK))

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
        tk.Label(brand, text="RGB-D Labelling Studio", bg=BG, fg=INK, font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(brand, text="Rekam RGB-D mentah • sortir • ekspor • label • ukur", bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(1, 0))
        tk.Label(brand, text="by Khairuramdhani", bg=BG, fg=MUTED, font=("Segoe UI", 10, "italic")).pack(anchor="w", pady=(1, 0))
        badge = tk.Label(top, text="D435  •  RGB + DEPTH", bg=ACCENT_SOFT, fg=ACCENT,
                         font=("Segoe UI", 9, "bold"), padx=12, pady=7)
        badge.pack(side="right", padx=(12, 0))
        self.tombol_ringkas(top, "\u2600  Tema terang" if tema.GELAP else "\U0001f319  Tema gelap", self.ganti_tema,
                             "#E8DDD5", INK, width=118).pack(side="right")
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
        self.tab_uji = tk.Frame(self.tabs, bg=BG); self.tab_uji_berkas = tk.Frame(self.tabs, bg=BG)
        self.tab_split = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(self.tab_rekam, text="  1. Rekam  "); self.tabs.add(self.tab_tinjau, text="  2. Tinjau & Potong  ")
        self.tabs.add(self.tab_ekspor, text="  3. Ekspor Frame  "); self.tabs.add(self.tab_label, text="  4. Label & Ukur  ")
        # Tab 5 berbeda tujuan dari tab 1-4: bukan membuat dataset, melainkan
        # MENGUJI model terlatih pada aliran kamera langsung. Dipisah supaya
        # alur dataset tidak tercampur dengan alur pengujian.
        self.tabs.add(self.tab_uji, text="  5. Uji Sistem Tangga  ")
        # Tab 6 menguji model pada BERKAS, bukan aliran kamera. Dipisah dari
        # tab 5 karena hasilnya dapat diulang persis dan karena itu dipakai
        # membandingkan antar-model, sedangkan aliran kamera tidak pernah sama
        # dua kali.
        self.tabs.add(self.tab_uji_berkas, text="  6. Uji Model pada Berkas  ")
        # Tab 7 membagi frame berlabel ke train/val/test; skrip latih membaca
        # berkas yang sama lewat --split, jadi yang tampil di sini = yang dilatih.
        self.tabs.add(self.tab_split, text="  7. Split Dataset  ")
        self.ui_rekam(); self.ui_tinjau(); self.ui_ekspor(); self.ui_label(); self.ui_uji()
        self.ui_uji_berkas(); self.ui_split()

    def ui_split(self):
        """Panel pembagian train/val/test dan push dataset ke Hugging Face."""
        try:
            try:
                from .tab_split import TabSplit
            except ImportError:
                from tab_split import TabSplit
            self.panel_split = TabSplit(self.tab_split, self)
        except Exception as e:                                    # noqa: BLE001
            self.panel_split = None
            tk.Label(self.tab_split, text=f"Panel split dataset gagal dimuat:\n{e}",
                     bg=BG, fg=MUTED, justify="left").pack(padx=24, pady=24)

    def ui_uji_berkas(self):
        """Panel pengujian model pada gambar, video, atau folder frame RGB-D."""
        try:
            from .uji_berkas import UjiBerkas
        except ImportError:
            from uji_berkas import UjiBerkas
        try:
            self.panel_uji_berkas = UjiBerkas(self.tab_uji_berkas, self)
        except Exception as e:                                    # noqa: BLE001
            self.panel_uji_berkas = None
            tk.Label(self.tab_uji_berkas, text=f"Panel uji berkas gagal dimuat:\n{e}",
                     bg=BG, fg=MUTED, justify="left").pack(padx=24, pady=24)

    def ui_uji(self):
        """Pasang panel uji realtime. Impor ditunda supaya Studio tetap bisa
        dibuka di mesin tanpa torch -- pengguna yang hanya melabeli tidak perlu
        memasang seluruh tumpukan deep learning."""
        try:
            from .uji_realtime import UjiRealtime
        except ImportError:
            from uji_realtime import UjiRealtime
        try:
            self.panel_uji = UjiRealtime(self.tab_uji, self)
        except Exception as e:
            tk.Label(self.tab_uji, text=f"Panel uji realtime tidak dapat dimuat:\n{e}",
                     bg=BG, fg=MUTED, justify="left").pack(padx=30, pady=30)

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
        # Uji galat depth D435 terhadap meteran (dinding datar); memakai kamera preview ini.
        self.tombol(i, "Uji depth dinding (meteran)…", self.buka_uji_depth, "#E8DDD5", INK).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
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

    def _kolom_gulir(self, induk, lebar: int) -> tk.Frame:
        return kolom_gulir(induk, lebar, BG, padx=(0, 12))

    def ui_tinjau(self):
        f = tk.Frame(self.tab_tinjau, bg=BG); f.pack(fill="both", expand=True, padx=18, pady=18)
        # Kolom kiri bisa di-scroll: daftar rekaman, tombolnya, dan kartu catatan
        # tidak muat pada layar pendek, dan tanpa ini tombol bawah hilang.
        left = self._kolom_gulir(f, 330)
        right = tk.Frame(f, bg=BG); right.pack(side="left", fill="both", expand=True)
        self._ui_catatan_rekaman(left)
        b, i = self.card(left, "Rekaman") ; b.pack(fill="both", expand=True)
        tk.Label(i, text="Kategori", bg=PANEL, fg=MUTED).pack(anchor="w")
        pilih_kategori = ttk.Combobox(i, textvariable=self.filter_sesi, values=KATEGORI,
                                      state="readonly")
        pilih_kategori.pack(fill="x", pady=(2, 8))
        pilih_kategori.bind("<<ComboboxSelected>>", lambda _e: self.muat_daftar())
        self.list_sesi = tk.Listbox(i, bg="#FFF9F4", fg=INK, relief="flat", selectbackground=ACCENT_SOFT,
                                    selectforeground=INK, activestyle="none", height=8, exportselection=False)
        self.list_sesi.pack(fill="both", expand=True); self.list_sesi.bind("<<ListboxSelect>>", lambda e: self.pilih_sesi())
        # Nama rekaman bisa disalin: Ctrl+C / klik kanan = nama folder, Ctrl+Shift+C = jalur lengkap.
        self.list_sesi.bind("<Control-c>", lambda e: self._salin_rekaman())
        self.list_sesi.bind("<Control-C>", lambda e: self._salin_rekaman(penuh=bool(e.state & 0x1)))
        self.list_sesi.bind("<Button-3>", self._salin_rekaman_kursor)
        # Tombol dipadatkan menjadi dua baris agar kartu catatan di bawahnya
        # tidak mendorong tombol keluar layar pada jendela pendek.
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(6, 0))
        self.tombol_ringkas(baris, "↻ Muat ulang", self.muat_daftar, "#E8DDD5", INK, width=96).pack(side="left", expand=True, fill="x")
        self.tombol_ringkas(baris, "🗑 Sampah/pulihkan", self.toggle_sampah, "#E8DDD5", INK, width=130).pack(side="left", expand=True, fill="x", padx=(3, 0))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(3, 0))
        self.tombol_ringkas(baris, "Hapus preview", self.hapus_preview_permanen, "#F3D8D4", INK, width=110).pack(side="left", expand=True, fill="x")
        self.tombol_ringkas(baris, "Hapus rekaman", self.hapus_sesi_permanen, RED, width=110).pack(side="left", expand=True, fill="x", padx=(3, 0))
        # Tandai rekaman sebagai data uji (split_dataset.json, sama dengan set TEST di tab 7. Split Dataset).
        self.tombol_ringkas(i, "🧪 Data test: tandai / lepas", self.toggle_data_test, WARNA_DATA_TEST, "white",
                            width=220).pack(fill="x", pady=(3, 0))
        self.tombol_ringkas(i, "♻ Tempat sampah Studio…", self.buka_tempat_sampah, "#E8DDD5", INK,
                            width=220).pack(fill="x", pady=(3, 0))
        self.tombol_ringkas(i, "📊 Jumlah data berlabel", self.buka_jumlah_data, "#E8DDD5", INK,
                            width=220).pack(fill="x", pady=(3, 0))
        tk.Checkbutton(i, text="\U0001f5d1 Tampilkan isi tempat sampah", variable=self.tampil_sampah,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       command=self.muat_daftar).pack(anchor="w", pady=(4, 0))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(4, 0))
        tk.Label(baris, text="Kategori:", bg=PANEL, fg=MUTED).pack(side="left")
        tk.Label(baris, textvariable=self.kategori_sesi, bg=PANEL, fg=INK,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(3, 0))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(2, 0))
        ttk.Combobox(baris, textvariable=self.kategori_baru, values=KATEGORI, state="readonly",
                     width=12).pack(side="left", fill="x", expand=True)
        self.tombol_ringkas(baris, "Pindahkan", self.pindah_kategori, "#E8DDD5", INK, width=80).pack(side="left", padx=(3, 0))
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
        self.tombol_ringkas(row, "Isi meteran", lambda: self.buka_pemutar_ukur(mulai=self.posisi.get()),
                            ACCENT, "white", width=96).pack(side="right", padx=4)

        self.scale_pos = tk.Scale(bawah, from_=0, to=0, orient="horizontal", variable=self.posisi,
                                  label="Posisi frame", bg=PANEL, fg=INK, highlightthickness=0,
                                  command=self._geser_posisi)
        self.scale_pos.pack(fill="x", pady=(6, 0))
        # Pita penanda status label frame ekspor di sepanjang klip: hijau = diperiksa,
        # ungu = dilabel Claude, merah = dilabel Claude dan perlu dicek. Klik = lompat
        # ke frame bertanda terdekat.
        self.pita_label = tk.Canvas(bawah, height=12, bg=PANEL, highlightthickness=0, cursor="hand2")
        self.pita_label.pack(fill="x")
        self.pita_label.bind("<Configure>", lambda e: self._gambar_pita_label())
        self.pita_label.bind("<Button-1>", self._klik_pita_label)
        self._pita_frame: list[tuple[int, str]] = []

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
        # Awal dan akhir berdampingan: bertumpuk, keduanya ikut mendorong tombol
        # simpan keluar layar pada laptop pendek.
        potong = tk.Frame(bawah, bg=PANEL); potong.pack(fill="x")
        self.scale_awal = tk.Scale(potong, from_=0, to=0, orient="horizontal", variable=self.awal, label="Awal potongan", bg=PANEL, fg=INK, highlightthickness=0)
        self.scale_akhir = tk.Scale(potong, from_=0, to=0, orient="horizontal", variable=self.akhir, label="Akhir potongan", bg=PANEL, fg=INK, highlightthickness=0)
        self.scale_awal.pack(side="left", fill="x", expand=True, padx=(0, 6)); self.scale_akhir.pack(side="left", fill="x", expand=True)
        manual = tk.Frame(bawah, bg=PANEL); manual.pack(fill="x", pady=(4, 0))
        tk.Label(manual, text="Ketik frame", bg=PANEL, fg=MUTED).pack(side="left")
        ent_awal = tk.Spinbox(manual, from_=0, to=999999, textvariable=self.awal, width=7,
                               command=lambda: self.rentang_manual.set(True))
        ent_awal.pack(side="left", padx=(6, 2)); ent_awal.bind("<FocusOut>", lambda _e: self.rentang_manual.set(True))
        tk.Label(manual, text="s.d.", bg=PANEL, fg=MUTED).pack(side="left")
        ent_akhir = tk.Spinbox(manual, from_=0, to=999999, textvariable=self.akhir, width=7,
                                command=lambda: self.rentang_manual.set(True))
        ent_akhir.pack(side="left", padx=2); ent_akhir.bind("<FocusOut>", lambda _e: self.rentang_manual.set(True))
        self.tombol_ringkas(manual, "Awal = kini", self.tetapkan_awal_kini, "#E8DDD5", INK, width=84).pack(side="left", padx=(8, 2))
        self.tombol_ringkas(manual, "Akhir = kini", self.tetapkan_akhir_kini, "#E8DDD5", INK, width=84).pack(side="left", padx=2)
        self.tombol(bawah, "Simpan rentang potong (non-destruktif)", self.simpan_potong, GREEN).pack(fill="x", pady=(8, 0))

    # ------------------------------------------------ catatan rekaman & scene
    def _ui_catatan_rekaman(self, induk) -> None:
        """Kartu warna stabilo, catatan/lokasi, dan scene rekaman terpilih."""
        self._catatan_setelah = None
        self._sesi_catatan = None
        self._cache_catatan: dict = {}
        b, i = self.card(induk, "Catatan rekaman & scene"); b.pack(side="bottom", fill="x", pady=(10, 0))
        # Kotak warna kecil (Frame, bukan Button yang lebar minimalnya besar) agar
        # tombol hapus warna tidak terdorong keluar kartu seperti sebelumnya.
        self.info_waktu_rekaman = tk.Label(i, text="", bg=PANEL, fg=ACCENT, justify="left", anchor="w",
                                           wraplength=270, font=("Segoe UI", 9))
        self.info_waktu_rekaman.pack(fill="x", pady=(0, 4))
        # Ceklis rekaman selesai: semua frame sudah dilabel dan dicek.
        self.selesai_var = BooleanVar(value=False)
        tk.Checkbutton(i, text="✔ Rekaman ini sudah benar semua", variable=self.selesai_var, bg=PANEL, fg="#1E5B2A",
                       selectcolor=PANEL, activebackground=PANEL, font=("Segoe UI", 9, "bold"),
                       command=self.atur_selesai).pack(anchor="w")
        self.info_progres = tk.Label(i, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 8), anchor="w", justify="left",
                                     wraplength=270)
        self.info_progres.pack(fill="x")
        self.tombol_ringkas(i, "⏱ Cek sinkron RGB–depth–IR", self.cek_sinkron, "#E8DDD5", INK,
                            width=200).pack(anchor="w", pady=(2, 0))
        # Stabilo = warna TANGGA fisik bila rekaman punya ID tangga (semua rekaman
        # dan scene tangga itu ikut berwarna sama); tanpa tangga, warna rekaman ini.
        self.label_stabilo = tk.Label(i, text="Stabilo", bg=PANEL, fg=MUTED, wraplength=270, justify="left", anchor="w")
        self.label_stabilo.pack(fill="x", pady=(4, 0))
        nama_warna = list(CR.WARNA.items())
        for awal in (0, 5):
            baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(2, 0))
            for nama, w in nama_warna[awal:awal + 5]:
                kotak = tk.Frame(baris, bg=w, width=26, height=20, cursor="hand2",
                                 highlightbackground="#CFC4BC", highlightthickness=1)
                kotak.pack(side="left", padx=2)
                kotak.bind("<Button-1>", lambda _e, n=nama: self.atur_warna_sesi(n))
            if awal == 5:
                self.tombol_ringkas(baris, "✕ Hapus warna", lambda: self.atur_warna_sesi(None),
                                    "#E8DDD5", INK, width=100).pack(side="left", padx=(6, 0))
        # Tangga fisik: rekaman yang merekam tangga yang sama diberi ID sama, agar
        # pembagian train/val/test bisa per tangga (tab Split).
        tk.Label(i, text="Tangga fisik: ketik nama lalu Enter (nama sama = otomatis terhubung)", bg=PANEL,
                 fg=MUTED).pack(anchor="w", pady=(6, 0))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x")
        # Bisa diketik: nama yang sudah ada dipakai ulang (tanpa beda huruf besar/spasi),
        # nama baru otomatis menjadi tangga baru. Dulu hanya bisa memilih dari daftar
        # atau "+ Baru", yang membuat ID ganda bila namanya diketik ulang.
        self.pilih_tangga = ttk.Combobox(baris, width=24)
        self.pilih_tangga.pack(side="left", fill="x", expand=True)
        self.pilih_tangga.bind("<<ComboboxSelected>>", lambda _e: self.atur_tangga_sesi(self.pilih_tangga.get()))
        self.pilih_tangga.bind("<Return>", lambda _e: self.tangga_dari_ketikan())
        self.pilih_tangga.bind("<KP_Enter>", lambda _e: self.tangga_dari_ketikan())
        self.pilih_tangga.bind("<FocusOut>", lambda _e: self.tangga_dari_ketikan())
        self.tombol_ringkas(baris, "+ Baru", self.tangga_baru, GREEN, width=60).pack(side="left", padx=(3, 0))
        # Atribut tangga fisik: lokasi (pembagian latih/uji yang adil), luar/dalam ruang, bahan.
        self.atribut_tangga = AtributTangga(i, self.root_data, sesudah=lambda: self._segarkan_item_sesi(self.sesi)
                                            if self.sesi else None, bg=PANEL, fg_muted=MUTED)
        self.atribut_tangga.pack(fill="x", pady=(3, 0))
        self.tombol_ringkas(i, "🪜 Kelompokkan rekaman per tangga…", self.buka_kelompok_tangga, "#E8DDD5", INK,
                            width=240).pack(fill="x", pady=(3, 0))
        tk.Label(i, text="Catatan / lokasi (tersimpan otomatis)", bg=PANEL, fg=MUTED).pack(anchor="w", pady=(6, 0))
        self.teks_catatan = tk.Text(i, height=3, wrap="word", bg="#FFF9F4", fg=INK, relief="flat",
                                    font=("Segoe UI", 9), undo=True)
        self.teks_catatan.pack(fill="x")
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(4, 0))
        tk.Label(baris, text="Cahaya terukur (lux)", bg=PANEL, fg=MUTED).pack(side="left")
        self.lux_var = StringVar()
        isian = tk.Entry(baris, textvariable=self.lux_var, width=8, relief="flat", bg="#FFF9F4")
        isian.pack(side="left", padx=4)
        isian.bind("<KeyRelease>", lambda _e: self._jadwal_simpan_catatan())
        isian.bind("<FocusOut>", lambda _e: self._simpan_catatan_sekarang())
        self.tombol_ringkas(baris, "📷 Dari foto HP", self.lux_dari_foto, "#E8DDD5", INK, width=110).pack(side="left", padx=(4, 0))
        self.lux_sumber_lbl = tk.Label(i, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 8), wraplength=270,
                                       justify="left", anchor="w")
        self.lux_sumber_lbl.pack(fill="x")
        self.teks_catatan.bind("<KeyRelease>", lambda _e: self._jadwal_simpan_catatan())
        self.teks_catatan.bind("<FocusOut>", lambda _e: self._simpan_catatan_sekarang())
        tk.Label(i, text="Scene (klik ganda = lompat ke sana)", bg=PANEL, fg=MUTED).pack(anchor="w", pady=(6, 0))
        self.list_scene = tk.Listbox(i, height=3, bg="#FFF9F4", fg=INK, relief="flat", activestyle="none",
                                     exportselection=False, font=("Segoe UI", 9))
        self.list_scene.pack(fill="x")
        self.list_scene.bind("<Double-Button-1>", lambda _e: self.lompat_scene())
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(4, 0))
        self.tombol_ringkas(baris, "+ Dari awal–akhir potongan", self.tambah_scene, GREEN, width=170).pack(side="left")
        self.tombol_ringkas(baris, "Nama", self.ganti_nama_scene, "#E8DDD5", INK, width=50).pack(side="left", padx=2)
        self.tombol_ringkas(baris, "Hapus", self.hapus_scene, "#F3D8D4", INK, width=50).pack(side="left")
        self.tombol_ringkas(baris, "Tangga", self.tangga_scene, "#E8DDD5", INK, width=60).pack(side="left", padx=2)

    def _catatan_cache(self, sesi: Path) -> dict:
        """catatan.json per rekaman, dibaca ulang hanya bila berkasnya berubah (disk dataset lambat)."""
        f = Path(sesi) / CR.NAMA_BERKAS
        try:
            kunci = f.stat().st_mtime_ns
        except OSError:
            kunci = None
        lama = self._cache_catatan.get(sesi)
        if lama is None or lama[0] != kunci:
            lama = (kunci, CR.baca(sesi))
            self._cache_catatan[sesi] = lama
        return lama[1]

    def _teks_item_sesi(self, p: Path, ikon: str = "") -> str:
        d = self._catatan_cache(p)
        tanda = ("  📝" if d["catatan"] else "") + (f"  ◆{len(d['scene'])}" if d["scene"] else "")
        if d.get("tangga"):
            tanda += f"  🪜{d['tangga']}"
        if not self.bag(p).exists():
            tanda += "  (tanpa video)"
        elif self.bag(p).stat().st_size < 1_000_000:
            tanda += "  ⚠(rusak/kosong)"
        cek = "✔ " if d.get("selesai") else ""
        uji = ""
        if self._set_split(p) == "test":
            # Lokasi + nomor tangga seperti di sheet Ukur Tangga, untuk mencocokkan meteran.
            tid = d.get("tangga")
            nama = (self._daftar_tangga().get(tid) or {}).get("nama", "") if tid else ""
            uji = f"   🧪 DATA TEST · {nama}" if nama else "   🧪 DATA TEST · (lokasi belum diisi)"
            tanda = tanda.replace(f"  🪜{tid}", "") if tid else tanda
        return f"{ikon}{cek}{p.name.replace('TANGGA_NAIK_', '')}{uji}{tanda}"

    def _warnai_item_sesi(self, i: int, p: Path) -> None:
        w = CR.WARNA.get(CR.warna_rekaman(self._catatan_cache(p), self._daftar_tangga()))
        uji = self._set_split(p) == "test"
        # Latar pastel tangga tidak diubah tema gelap, jadi teksnya harus tetap gelap (warna di luar peta tema).
        fg = ("#1F4E79" if uji else "#1A1A1A") if w else (WARNA_DATA_TEST if uji else INK)
        self.list_sesi.itemconfig(i, bg=w or "#FFF9F4", selectbackground=w or ACCENT_SOFT, fg=fg, selectforeground=fg)

    # ---- data test: set rekaman pada split_dataset.json (satu sumber dengan tab 7. Split Dataset)
    def _pustaka_split(self):
        panel = getattr(self, "panel_split", None)
        if panel is not None:
            return panel.sd
        try:
            from .tab_split import _pustaka
        except ImportError:
            from tab_split import _pustaka
        return _pustaka()[0]

    def _data_split(self) -> dict:
        """split_dataset.json, dibaca ulang hanya bila berkasnya berubah."""
        f = self.root_data / "split_dataset.json"
        try:
            kunci = f.stat().st_mtime_ns
        except OSError:
            kunci = None
        lama = getattr(self, "_cache_split", None)
        if lama is None or lama[0] != kunci:
            try:
                data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            except (OSError, ValueError):
                data = {}
            lama = (kunci, dict(data.get("rekaman") or {}))
            self._cache_split = lama
        return lama[1]

    def _set_split(self, p: Path) -> str | None:
        return self._data_split().get(Path(p).name)

    def _dialog_lokasi_uji(self):
        """Lokasi dan nomor tangga persis seperti di sheet Ukur Tangga. -> (lokasi, nomor), () = lewati, None = batal."""
        daftar = self._daftar_tangga()
        tid = CR.baca(self.sesi).get("tangga") if self.sesi else None
        lokasi_awal, nomor_awal = "", 1
        if tid and tid in daftar:
            lokasi_awal = daftar[tid].get("lokasi", "")
            m = re.search(r"Tangga\s+(\d+)\s*$", daftar[tid].get("nama", ""))
            nomor_awal = int(m.group(1)) if m else 1
        semua_lokasi = sorted({v.get("lokasi", "") for v in daftar.values() if v.get("lokasi")}, key=str.casefold)
        hasil = {"nilai": None}
        dlg = tk.Toplevel(self); dlg.title("Data test: lokasi"); dlg.configure(bg=PANEL)
        dlg.transient(self); dlg.resizable(False, False)
        tk.Label(dlg, text="Tulis persis seperti nama lokasi dan nomor tangga di sheet Ukur Tangga,\n"
                           "supaya ukuran meterannya bisa dicocokkan.", bg=PANEL, fg=MUTED, justify="left",
                 font=("Segoe UI", 10)).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 8))
        tk.Label(dlg, text="Lokasi", bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=14, pady=4)
        pilih = ttk.Combobox(dlg, values=semua_lokasi, width=34, font=("Segoe UI", 10))
        pilih.set(lokasi_awal); pilih.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=4)
        tk.Label(dlg, text="Tangga ke-", bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w", padx=14, pady=4)
        nomor = tk.Spinbox(dlg, from_=1, to=99, width=6, bg="#FFF9F4", fg=INK, relief="flat", font=("Segoe UI", 10),
                           buttonbackground="#E8DDD5", insertbackground=INK)
        nomor.delete(0, "end"); nomor.insert(0, str(nomor_awal)); nomor.grid(row=2, column=1, sticky="w", pady=4)
        tombol = tk.Frame(dlg, bg=PANEL); tombol.grid(row=3, column=0, columnspan=2, sticky="e", padx=14, pady=(10, 12))

        def simpan(_e=None):
            lok = " ".join(pilih.get().split())
            if not lok:
                messagebox.showinfo("Lokasi kosong", "Isi lokasi, atau tekan Lewati.", parent=dlg); return
            try:
                n = int(nomor.get())
            except ValueError:
                messagebox.showinfo("Nomor tangga", "Nomor tangga harus angka.", parent=dlg); return
            hasil["nilai"] = (lok, n); dlg.destroy()

        def lewati():
            hasil["nilai"] = (); dlg.destroy()
        self.tombol_ringkas(tombol, "Simpan", simpan, WARNA_DATA_TEST, "white", width=90).pack(side="right")
        self.tombol_ringkas(tombol, "Lewati", lewati, "#E8DDD5", INK, width=80).pack(side="right", padx=4)
        self.tombol_ringkas(tombol, "Batal", dlg.destroy, "#E8DDD5", INK, width=70).pack(side="right")
        dlg.bind("<Return>", simpan); dlg.bind("<Escape>", lambda _e: dlg.destroy())
        jadikan_modal(dlg, self)
        pilih.focus_set()
        self.wait_window(dlg)
        return hasil["nilai"]

    def toggle_data_test(self):
        """Tandai/lepas rekaman terpilih sebagai DATA TEST (tidak boleh dipakai melatih atau menyetel)."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        nama = self.sesi.name
        panel = getattr(self, "panel_split", None)
        if panel is not None:                       # tab Split memegang salinan di memori: ubah lewat dia
            data = panel.data = panel.sd.baca(panel.berkas)
        else:
            sd = self._pustaka_split()
            berkas = self.root_data / "split_dataset.json"
            data = sd.baca(berkas)
        lama = data["rekaman"].get(nama)
        if lama == "test":
            data["rekaman"].pop(nama, None)
            pesan = f"{nama}: tanda DATA TEST dilepas (set rekaman kini kosong)."
        else:
            pilihan = self._dialog_lokasi_uji()
            if pilihan is None:
                return                              # batal
            data["rekaman"][nama] = "test"
            pesan = (f"{nama}: ditandai DATA TEST" + (f" (sebelumnya {lama.upper()})" if lama else "")
                     + ". Rekaman ini tidak dipakai melatih model.")
            if pilihan:
                lokasi, nomor = pilihan
                tid, _baru = CR.tangga_dari_nama(self.root_data, f"{lokasi} · Tangga {nomor}", lokasi)
                self._simpan_catatan_sekarang()
                d = CR.baca(self.sesi); d["tangga"] = tid; CR.tulis(self.sesi, d)
                self._muat_catatan_ui()
                pesan += f" Lokasi: {lokasi} · Tangga {nomor} ({tid})."
        try:
            if panel is not None:
                panel._simpan()                     # tab 7 menghitung ulang saat dibuka (ganti_tab)
            else:
                sd.tulis(berkas, data)
        except Exception as e:                      # noqa: BLE001
            messagebox.showerror("Data test", f"Gagal menyimpan split_dataset.json:\n{e}", parent=self); return
        self._cache_split = None
        self._segarkan_item_sesi(self.sesi)
        self.status.set(pesan)

    def _daftar_tangga(self) -> dict:
        """tangga.json, dibaca ulang hanya bila berubah."""
        f = self.root_data / CR.BERKAS_TANGGA
        try:
            kunci = f.stat().st_mtime_ns
        except OSError:
            kunci = None
        if getattr(self, "_cache_tangga", (None, None))[0] != kunci or not hasattr(self, "_cache_tangga"):
            self._cache_tangga = (kunci, CR.baca_tangga(self.root_data))
        return self._cache_tangga[1]

    def _segarkan_item_sesi(self, sesi: Path) -> None:
        if sesi in getattr(self, "_map_sesi", []):
            i = self._map_sesi.index(sesi)
            ikon = "\U0001f5d1 " if self._state(sesi).get("di_sampah") else ""
            terpilih = i in self.list_sesi.curselection()
            self.list_sesi.delete(i); self.list_sesi.insert(i, self._teks_item_sesi(sesi, ikon))
            self._warnai_item_sesi(i, sesi)
            if terpilih:
                self.list_sesi.selection_set(i)

    def _muat_catatan_ui(self) -> None:
        self._sesi_catatan = self.sesi
        d = self._catatan_cache(self.sesi) if self.sesi else {"catatan": "", "scene": []}
        self.teks_catatan.delete("1.0", "end"); self.teks_catatan.insert("1.0", d["catatan"])
        w = CR.waktu_rekaman(self.sesi) if self.sesi else None
        if w:
            mulai, selesai = w
            teks = f"🕒 Direkam {CR.format_waktu(mulai)}"
            if selesai:
                dur = int((selesai - mulai).total_seconds())
                teks += f"\n     sampai {selesai:%H.%M.%S} ({dur // 60} m {dur % 60} d)"
            self.info_waktu_rekaman.config(text=teks)
        else:
            self.info_waktu_rekaman.config(text="🕒 Waktu rekaman tidak tercatat")
        self.lux_var.set(d.get("lux", ""))
        self.selesai_var.set(bool(d.get("selesai")))
        tid = d.get("tangga")
        self.label_stabilo.config(text=(f"Stabilo tangga {tid} (semua rekaman/scene tangga ini ikut berwarna sama)"
                                        if tid else "Stabilo rekaman ini (beri ID tangga agar warnanya per tangga)"))
        self._segarkan_info_tinjau()
        self._isi_pilihan_tangga(d.get("tangga") or None)
        self.lux_sumber_lbl.config(text=("Sumber: " + d["lux_sumber"]) if d.get("lux_sumber")
                                   else "Isi dari lux meter di lokasi, atau hitung dari foto HP.")
        self.teks_catatan.edit_reset()
        self.list_scene.delete(0, "end")
        for sc in d["scene"]:
            self.list_scene.insert("end", f"{sc['nama']}   ({sc['awal']}–{sc['akhir']})"
                                          + (f"  · {sc['tangga']}" if sc.get("tangga") else ""))
            w_ = CR.WARNA.get(CR.warna_untuk_tangga(sc.get("tangga"), self._daftar_tangga()))
            if w_:
                self.list_scene.itemconfig("end", bg=w_)

    def _jadwal_simpan_catatan(self) -> None:
        if self._catatan_setelah is not None:
            self.after_cancel(self._catatan_setelah)
        self._catatan_setelah = self.after(600, self._simpan_catatan_sekarang)

    def _simpan_catatan_sekarang(self) -> None:
        """Tulis catatan ke rekaman yang SEDANG dimuat di kartu, bukan yang baru dipilih."""
        if getattr(self, "_catatan_setelah", None) is not None:
            self.after_cancel(self._catatan_setelah)
            self._catatan_setelah = None
        sesi = getattr(self, "_sesi_catatan", None)
        if sesi is None or not sesi.exists():
            return
        teks = self.teks_catatan.get("1.0", "end").strip()
        lux = self.lux_var.get().strip().replace(",", ".")
        d = CR.baca(sesi)
        if d["catatan"] == teks and d.get("lux", "") == lux:
            return
        if d.get("lux", "") != lux and not getattr(self, "_lux_dari_foto", False):
            d["lux_sumber"] = "lux meter / diisi manual" if lux else ""
        d["catatan"], d["lux"] = teks, lux
        CR.tulis(sesi, d)
        self._segarkan_item_sesi(sesi)

    def _ubah_catatan(self, ubah) -> None:
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        self._simpan_catatan_sekarang()
        d = CR.baca(self.sesi)
        ubah(d)
        CR.tulis(self.sesi, d)
        self._segarkan_item_sesi(self.sesi)
        self._muat_catatan_ui()

    def lux_dari_foto(self) -> None:
        """Perkiraan lux rekaman dari EXIF foto HP yang diambil di lokasi tangga."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        f = filedialog.askopenfilename(parent=self, title="Foto HP di lokasi tangga (JPG asli)",
                                       filetypes=[("Foto", "*.jpg *.jpeg *.JPG *.JPEG *.tif *.tiff"), ("Semua", "*")])
        if not f:
            return
        try:
            lux, ket = CR.lux_dari_foto(f)
        except Exception as e:                                 # noqa: BLE001
            messagebox.showerror("Lux dari foto", str(e), parent=self); return
        self._simpan_catatan_sekarang()
        self._ubah_catatan(lambda d: d.update(lux=f"{lux:.0f}", lux_sumber=f"{ket}; {Path(f).name}"))
        self.status.set(f"Perkiraan cahaya lokasi ≈ {lux:.0f} lux dari {Path(f).name} ({ket}).")

    # ------------------------------------------------ tangga fisik
    TANPA_TANGGA = "— belum ada tangga —"

    def _isi_pilihan_tangga(self, tid: str | None) -> None:
        daftar = CR.baca_tangga(self.root_data)
        self.pilih_tangga.configure(values=[self.TANPA_TANGGA] + [CR.label_tangga(k, daftar) for k in daftar])
        self.pilih_tangga.set(CR.label_tangga(tid, daftar) if tid else self.TANPA_TANGGA)
        if getattr(self, "atribut_tangga", None) is not None:
            self.atribut_tangga.tampilkan(tid)

    @staticmethod
    def _id_dari_label(label: str) -> str:
        return "" if not label or label.startswith("—") else label.split(" · ")[0].strip()

    def atur_tangga_sesi(self, label: str) -> None:
        self._ubah_catatan(lambda d: d.update(tangga=self._id_dari_label(label)))

    def tangga_dari_ketikan(self) -> None:
        """Kaitkan rekaman ke tangga bernama sesuai ketikan; nama baru menjadi tangga baru."""
        if not self.sesi:
            return
        teks = " ".join(self.pilih_tangga.get().split())
        daftar = CR.baca_tangga(self.root_data)
        lama = CR.baca(self.sesi).get("tangga") or ""
        if teks == (CR.label_tangga(lama, daftar) if lama else self.TANPA_TANGGA):
            return                                       # tidak berubah (mis. fokus pindah saja)
        if not teks or teks.startswith("—"):
            self._ubah_catatan(lambda d: d.update(tangga=""))
            self.status.set("Tangga fisik dilepas dari rekaman ini.")
            return
        tid, baru = CR.tangga_dari_nama(self.root_data, teks)
        self._ubah_catatan(lambda d: d.update(tangga=tid))
        label = CR.label_tangga(tid, CR.baca_tangga(self.root_data))
        if baru:
            self.status.set(f"Tangga baru {label} dibuat dan dipasang. Ketik nama yang sama di rekaman lain "
                            "untuk menghubungkannya.")
        else:
            n = sum(1 for p in getattr(self, "_map_sesi", []) if CR.baca(p).get("tangga") == tid)
            self.status.set(f"Terhubung ke {label} ({n} rekaman).")

    def tangga_baru(self) -> None:
        """Daftarkan tangga fisik baru lalu pasang ke rekaman terpilih."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        daftar = CR.baca_tangga(self.root_data)
        nama = simpledialog.askstring("Tangga baru", f"Nama tangga {CR.id_tangga_baru(daftar)}\n"
                                      "(mis. Rusunawa tangga 1, Gedung F tangga darurat):", parent=self)
        if not nama or not nama.strip():
            return
        if CR.cari_tangga(daftar, nama) is None:
            lokasi = simpledialog.askstring("Tangga baru", "Lokasi/keterangan (boleh kosong):", parent=self) or ""
        else:
            lokasi = ""
        tid, baru = CR.tangga_dari_nama(self.root_data, nama, lokasi)
        self._ubah_catatan(lambda d: d.update(tangga=tid))
        label = CR.label_tangga(tid, CR.baca_tangga(self.root_data))
        self.status.set(f"Tangga {label} dibuat dan dipasang ke rekaman ini." if baru
                        else f"Nama itu sudah dipakai {label}; rekaman ini dihubungkan ke sana, bukan tangga baru.")

    def buka_kelompok_tangga(self) -> None:
        """Jendela kartu bergambar untuk menandai banyak rekaman sekaligus sebagai tangga fisik yang sama."""
        lama = getattr(self, "_jendela_kelompok", None)
        if lama is not None and lama.winfo_exists():
            lama.lift(); lama.muat(); return
        try:
            from .kelompok_tangga import JendelaKelompokTangga
        except ImportError:
            from kelompok_tangga import JendelaKelompokTangga
        kat = self.filter_sesi.get() if self.filter_sesi.get() in KATEGORI else "tangga_naik"
        self._jendela_kelompok = JendelaKelompokTangga(self, kat)

    def tangga_scene(self) -> None:
        """Tangga untuk scene terpilih (satu rekaman bisa melewati beberapa tangga)."""
        i = self._scene_terpilih()
        if i is None or not self.sesi:
            messagebox.showinfo("Pilih scene", "Pilih scene di daftar dahulu.", parent=self); return
        daftar = CR.baca_tangga(self.root_data)
        if not daftar:
            messagebox.showinfo("Belum ada tangga", "Buat tangga dahulu dengan tombol + Baru.", parent=self); return
        dlg = tk.Toplevel(self); dlg.title("Tangga scene"); dlg.configure(bg=PANEL); dlg.transient(self)
        tk.Label(dlg, text="Tangga untuk scene ini:", bg=PANEL, fg=INK).pack(padx=14, pady=(12, 4), anchor="w")
        pilih = ttk.Combobox(dlg, state="readonly", width=34,
                             values=["— ikut tangga rekaman —"] + [CR.label_tangga(k, daftar) for k in daftar])
        sc = CR.baca(self.sesi)["scene"][i]
        pilih.set(CR.label_tangga(sc.get("tangga"), daftar) if sc.get("tangga") else "— ikut tangga rekaman —")
        pilih.pack(padx=14, fill="x")

        def simpan():
            tid = self._id_dari_label(pilih.get())
            dlg.destroy()
            self._ubah_catatan(lambda d: d["scene"][i].update(tangga=tid))
        tk.Button(dlg, text="Simpan", command=simpan).pack(pady=10)
        jadikan_modal(dlg, self)
        dlg.wait_window()

    def atur_warna_sesi(self, warna: str | None) -> None:
        """Stabilo: warna tangga fisik bila rekaman punya tangga, selain itu warna rekaman ini."""
        if not self.sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        tid = CR.baca(self.sesi).get("tangga")
        if tid:
            daftar = CR.baca_tangga(self.root_data)
            if tid in daftar:
                daftar[tid]["warna"] = warna
                CR.tulis_tangga(self.root_data, daftar)
                for p_ in list(getattr(self, "_map_sesi", [])):
                    self._segarkan_item_sesi(p_)
                self._muat_catatan_ui()
                self.status.set(f"Warna tangga {CR.label_tangga(tid, daftar)} diganti; semua rekaman/scene tangga ini ikut.")
                return
        self._ubah_catatan(lambda d: d.update(warna=warna))

    def atur_selesai(self) -> None:
        """Ceklis rekaman selesai; diperingatkan bila masih ada frame yang belum diperiksa."""
        if not self.sesi:
            self.selesai_var.set(False); return
        nilai = bool(self.selesai_var.get())
        if nilai:
            total, diperiksa = self._hitung_progres(self.sesi)
            if total and diperiksa < total and not messagebox.askyesno(
                    "Tandai selesai", f"Baru {diperiksa} dari {total} frame ekspor yang ditandai diperiksa.\n"
                    "Tetap tandai rekaman ini sudah benar semua?", parent=self):
                self.selesai_var.set(False); return
        self._ubah_catatan(lambda d: d.update(selesai=nilai, selesai_iso=datetime.now().isoformat(timespec="seconds")
                                              if nilai else ""))

    def _segarkan_info_tinjau(self) -> None:
        """Frame diperiksa, ukuran berkas, dan status sinkron rekaman terpilih (dihitung di thread latar)."""
        sesi_kini = self.sesi
        if not sesi_kini:
            self.info_progres.config(text="")
            return
        d = self._catatan_cache(sesi_kini)
        self.info_progres.config(text="Menghitung frame diperiksa dan ukuran berkas…")

        def hitung():
            # Thread latar TIDAK boleh menyentuh Tk; hasil dikirim lewat antrean self.q.
            t, dp = self._hitung_progres(sesi_kini)
            baris = [f"Frame diperiksa: {dp}/{t}" + (f"  ({d['selesai_iso'][:16].replace('T', ' ')} ditandai selesai)"
                                                    if d.get("selesai") and d.get("selesai_iso") else "")]
            baris.append(CR.teks_ukuran(CR.ukuran_rekaman(sesi_kini)))
            baris.append(sinkron.teks_ringkas(sinkron.baca(sesi_kini).get("ringkasan")))
            self.q.put(("progres_rekaman", (sesi_kini, "\n".join(baris))))
        threading.Thread(target=hitung, daemon=True).start()

    def cek_sinkron(self) -> None:
        """Baca ulang rekaman mentah: apakah RGB, depth, dan IR tiap frame ekspor satu jepretan (sinkron.py)."""
        sesi = self.sesi
        if not sesi:
            return
        if getattr(self, "_sinkron_jalan", None):
            self.status.set(f"Pemeriksaan sinkron {self._sinkron_jalan} masih berjalan…")
            return
        if not (sesi / "exports" / "frames").is_dir():
            messagebox.showinfo("Cek sinkron", "Rekaman ini belum punya frame ekspor.", parent=self)
            return
        bag = self.bag(sesi)
        alasan = periksa_rekaman(bag)
        if alasan:
            messagebox.showinfo("Cek sinkron", f"Rekaman mentah tidak dapat dibaca: {alasan}.\n\n"
                                "Kesesuaian waktu hanya bisa diperiksa dari rekaman mentah.", parent=self)
            return
        self._sinkron_jalan = sesi.name
        self.status.set(f"Memeriksa sinkron {sesi.name}: membaca ulang rekaman mentah…")

        def kerja():
            try:
                r = sinkron.periksa_sesi(sesi, bag, lambda n: self.q.put(
                    ("status", f"Memeriksa sinkron {sesi.name}: {n} frame rekaman dibaca…")))
            except Exception as e:                              # noqa: BLE001
                r = {"galat": str(e)}
            self.q.put(("sinkron_selesai", (sesi, r)))
        threading.Thread(target=kerja, daemon=True).start()

    def _sinkron_frame(self, sesi: Path) -> dict:
        """Isi exports/sinkron_waktu.json, di-cache per rekaman menurut waktu ubah berkas."""
        f = sesi / "exports" / sinkron.BERKAS
        try:
            mt = f.stat().st_mtime
        except OSError:
            return {}
        cache = getattr(self, "_cache_sinkron", None)
        if cache is None:
            cache = self._cache_sinkron = {}
        if cache.get(sesi, (None,))[0] != mt:
            cache[sesi] = (mt, sinkron.baca(sesi))
        return cache[sesi][1]

    def _hitung_progres(self, sesi: Path) -> tuple[int, int]:
        """(frame ekspor bukan sampah, frame yang ditandai diperiksa manual)."""
        fr = sesi / "exports" / "frames"
        total = diperiksa = 0
        if fr.exists():
            for d in fr.iterdir():
                if not d.is_dir() or baca_json(d / "frame_state.json", {}).get("di_sampah"):
                    continue
                total += 1
                j = baca_json(d / "label_draft.json", {})
                diperiksa += bool(j.get("diperiksa_manual", False))
        return total, diperiksa

    def tambah_scene(self) -> None:
        awal, akhir = int(self.awal.get()), int(self.akhir.get())
        if akhir <= awal:
            messagebox.showinfo("Rentang scene", "Atur Awal dan Akhir potongan dahulu (Awal = kini, Akhir = kini).",
                                parent=self); return
        n = len(self._catatan_cache(self.sesi)["scene"]) + 1 if self.sesi else 1
        nama = simpledialog.askstring("Scene baru", f"Nama scene untuk frame {awal}–{akhir}\n"
                                      "(mis. Tangga A, Bordes, Tangga gedung F):",
                                      initialvalue=f"Scene {n}", parent=self)
        if nama and nama.strip():
            self._ubah_catatan(lambda d: d["scene"].append({"nama": nama.strip(), "awal": awal, "akhir": akhir}))

    def _scene_terpilih(self) -> int | None:
        sel = self.list_scene.curselection()
        return sel[0] if sel else None

    def ganti_nama_scene(self) -> None:
        i = self._scene_terpilih()
        if i is None or not self.sesi:
            return
        lama = CR.baca(self.sesi)["scene"][i]["nama"]
        nama = simpledialog.askstring("Ganti nama scene", "Nama baru:", initialvalue=lama, parent=self)
        if nama and nama.strip():
            self._ubah_catatan(lambda d: d["scene"][i].update(nama=nama.strip()))

    def hapus_scene(self) -> None:
        i = self._scene_terpilih()
        if i is None or not self.sesi:
            return
        if messagebox.askyesno("Hapus scene", "Hapus scene ini? Frame dan label tidak tersentuh.", parent=self):
            self._ubah_catatan(lambda d: d["scene"].pop(i))

    def lompat_scene(self) -> None:
        i = self._scene_terpilih()
        if i is None or not self.sesi:
            return
        sc = CR.baca(self.sesi)["scene"][i]
        self.awal.set(int(sc["awal"])); self.akhir.set(int(sc["akhir"])); self.rentang_manual.set(True)
        self.posisi.set(int(sc["awal"]))
        try:
            self._tampilkan(int(sc["awal"]))
        except Exception:                                      # noqa: BLE001
            pass
        self.status.set(f"Scene {sc['nama']}: rentang potong {sc['awal']}–{sc['akhir']} siap diekspor.")

    def _perbarui_info_rekaman(self, p: Path) -> None:
        sesi = p.parent.parent.parent
        d = self._catatan_cache(sesi)
        sc = CR.scene_untuk(d, CR.indeks_frame(p.name))
        baris = []
        if d["catatan"]:
            baris.append("📍 " + CR.ringkas(d, 140))
        if sc:
            baris.append(f"Scene: {sc['nama']}")
        tid = CR.tangga_untuk(d, CR.indeks_frame(p.name))
        if tid:
            baris.append("🪜 Tangga " + CR.label_tangga(tid, CR.baca_tangga(self.root_data)))
        with self._kunci_cache:
            g = (self._cache_frame.get(p) or {}).get("geser")
        teks_g = ir_selaras.teks_geser(g)
        if teks_g:
            baris.append(f"↕ Rolling shutter: {teks_g} (sudah dikoreksi)")
        sk = sinkron.info_frame(self._sinkron_frame(sesi), p.name) or sinkron.info_dari_frame_json(self.label_info or {})
        if sk:
            baris.append(sk)
        wf = CR.waktu_frame(self.label_info or {})
        if wf:
            baris.append(f"🕒 Diambil {CR.format_waktu(wf)}")
        cahaya = (f"Cahaya lokasi: {'≈' if 'foto HP' in d.get('lux_sumber', '') else ''}{d['lux']} lux"
                  if d.get("lux") else "Cahaya lokasi: belum diukur")
        if self.kanvas.rgb is not None:
            l, kat = CR.kecerahan(self.kanvas.rgb)
            cahaya += f"  •  kecerahan gambar {l:.0f}/255 ({kat})"
        baris.append(cahaya)
        w_ = CR.warna_untuk_tangga(tid, self._daftar_tangga()) if tid else CR.warna_rekaman(d, self._daftar_tangga())
        teks, latar = "\n".join(baris), CR.WARNA.get(w_) or PANEL
        if self.info_rekaman.cget("text") == teks and self.info_rekaman.cget("bg") == latar:
            return                                      # tidak berubah: jangan gambar ulang
        self.info_rekaman.config(text=teks, bg=latar)
        self.kotak_info_rekaman.config(bg=latar)

    def ui_ekspor(self):
        f = tk.Frame(self.tab_ekspor, bg=BG); f.pack(fill="both", expand=True, padx=30, pady=28)
        b, i = self.card(f, "Ekspor frame RGB-D untuk dataset dan label YOLO") ; b.pack(fill="x")
        tk.Label(i, text="Ambil 1 frame setiap … frame sumber", bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w")
        tk.Spinbox(i, from_=1, to=9999, textvariable=self.langkah_ekspor, width=8).grid(row=0, column=1, sticky="w", padx=10)
        self.tombol(i, "Ekspor frame dari rentang pilihan", self.ekspor, GREEN).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        self.tombol(i, "Buat video MP4 rentang (opsional)", self.ekspor_video_rentang, BLUE).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.tombol(i, "Terapkan interval: bersihkan & ekspor ulang", self.buang_frame_belum_dilabeli, "#F3D8D4", INK).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.tombol(i, "Hapus semua hasil ekspor sesi ini", self.hapus_semua_ekspor, "#F3D8D4", INK).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        self.label_ekspor = tk.Label(f, text="Belum ada ekspor dipilih.", bg=BG, fg=ACCENT, justify="left"); self.label_ekspor.pack(anchor="w", pady=(18, 0))

    def ui_label(self):
        f = tk.Frame(self.tab_label, bg=BG); f.pack(fill="both", expand=True, padx=14, pady=14)
        left = tk.Frame(f, bg=BG); left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        # Yang dipakai tiap frame ada di atas gambar, tidak perlu dicari dengan menggulir panel.
        bilah = tk.Frame(left, bg=PANEL, highlightthickness=1, highlightbackground=LINE, padx=6, pady=4)
        bilah.pack(fill="x", pady=(0, 6))
        bilah_kelas = tk.Frame(bilah, bg=PANEL); bilah_kelas.pack(fill="x")
        bilah_salin = tk.Frame(bilah, bg=PANEL); bilah_salin.pack(fill="x", pady=(4, 0))
        self.kanvas = KanvasLabel(left, self.hitung_ukuran, self._aktif_mask_berubah); self.kanvas.pack(fill="both", expand=True)
        self.kanvas.saat_tukar = self.tukar_warna_mask
        self.kanvas.saat_undo_habis = self._undo_frame_lain
        self.kanvas.penyedia_peta = self._peta_model_kini
        self.kanvas.lapor_blok = self._lapor_blok
        self.kanvas.lapor_potong = self._lapor_potong
        self.kanvas.saat_ganti_alat = self._kanvas_ganti_alat
        self.kanvas.lebar_potong = float(self.lebar_potong.get())
        self.kanvas.satukan_potongan = bool(self.satukan_potongan.get())
        self.kanvas.touchpad = bool(self.mode_touchpad.get())
        self.kanvas.depth_mode = self.depth_mode.get()
        self.kanvas.penyedia_depth_vis = self._depth_vis_kini
        self.kanvas.pakai_ir = bool(self.pakai_ir.get())
        self.kanvas.warna_bidang = float(self.warna_bidang.get())
        self.kanvas.garis_3d = bool(self.garis_3d.get())
        # Ctrl+C menyalin nama frame yang sedang dibuka. Berguna saat melaporkan
        # frame bermasalah: namanya cukup panjang untuk salah ketik.
        self.kanvas.bind("<Control-c>", lambda _e: self.salin_nama_frame())
        self.kanvas.bind("<Control-C>", lambda _e: self.salin_nama_frame(penuh=True))
        self.kanvas.bind("<KeyPress>", self._shortcut_label, add="+")
        # Fokus dapat berpindah ke spinbox/panel; shortcut tetap harus hidup
        # selama pengguna berada di tab Label.
        self.bind_all("<KeyPress>", self._shortcut_label, add="+")
        # Salin/tempel di kotak isian tetap jalan saat Caps Lock menyala (Tk hanya
        # memetakan Control-c/v/x huruf kecil ke <<Copy>>/<<Paste>>/<<Cut>>).
        for kelas in ("Entry", "Text", "Spinbox", "TEntry", "TCombobox", "TSpinbox"):
            for huruf, peristiwa in (("C", "<<Copy>>"), ("V", "<<Paste>>"), ("X", "<<Cut>>")):
                self.bind_class(kelas, f"<Control-{huruf}>", lambda e, p=peristiwa: e.widget.event_generate(p) or "break")
        for kelas in ("Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox"):
            for huruf in ("a", "A"):
                self.bind_class(kelas, f"<Control-{huruf}>", lambda e: (e.widget.select_range(0, "end"), e.widget.icursor("end"), "break")[-1])
        # Kotak pilihan dan slider menahan fokus keyboard setelah dipakai, sehingga
        # Space/panah tidak lagi berpindah frame (panah malah mengganti pilihan
        # atau menggeser slider). Di tab Label fokus dikembalikan ke kanvas.
        self.bind_class("TCombobox", "<<ComboboxSelected>>", self._fokus_kanvas_label, add="+")
        self.bind_class("Scale", "<ButtonRelease-1>", self._fokus_kanvas_label, add="+")
        self.bind_class("Checkbutton", "<ButtonRelease-1>", self._fokus_kanvas_label, add="+")
        # Panel kanan bisa di-scroll. Dulu tanpa scrollbar karena semua kontrol
        # muat; setelah slider Cerahkan/Pertajam/Bantu riser-tread dan tombol
        # alat ditambahkan, bagian bawahnya keluar layar pada monitor yang
        # lebih pendek dan tidak dapat dijangkau sama sekali.
        luar = tk.Frame(f, bg=BG, width=390); luar.pack(side="left", fill="y"); luar.pack_propagate(False)
        gulir = tk.Canvas(luar, bg=BG, highlightthickness=0)
        bilah = ttk.Scrollbar(luar, orient="vertical", command=gulir.yview)
        gulir.configure(yscrollcommand=bilah.set)
        bilah.pack(side="right", fill="y"); gulir.pack(side="left", fill="both", expand=True)
        right = tk.Frame(gulir, bg=BG)
        jendela = gulir.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda _e: gulir.configure(scrollregion=gulir.bbox("all")))
        gulir.bind("<Configure>", lambda e: gulir.itemconfigure(jendela, width=e.width))
        self._panel_gulir = gulir

        def gulir_panel(e, langkah):
            # Hanya bila kursor di atas panel ini; kanvas gambar punya zoom/geser sendiri.
            w = self.winfo_containing(e.x_root, e.y_root)
            # Widget yang bisa digulir sendiri (daftar frame, teks) tetap menggulir
            # dirinya saja; tanpa ini daftar frame dan panel bergerak bersamaan.
            if isinstance(w, (tk.Listbox, tk.Text)):
                return None
            while w is not None and w is not gulir:
                w = getattr(w, "master", None)
            if w is None or gulir.yview() == (0.0, 1.0):
                return None
            gulir.yview_scroll(langkah, "units")
            return "break"
        self.bind_all("<Button-4>", lambda e: gulir_panel(e, -3), add="+")
        self.bind_all("<Button-5>", lambda e: gulir_panel(e, 3), add="+")
        self.bind_all("<MouseWheel>", lambda e: gulir_panel(e, -3 if e.delta > 0 else 3), add="+")
        b, i = self.card(right, "Pilih frame ekspor") ; b.pack(fill="x", pady=(0, 7))
        # Catatan rekaman dan scene frame aktif, supaya tahu "rekaman ini di
        # mana" sebelum mulai melabel. Disunting di tab Tinjau.
        # Tinggi TETAP: isi info berganti per frame (4-12 baris) dan diisi dua kali
        # (sesudah geser rolling shutter selesai diukur di thread latar). Dulu
        # tingginya ikut berubah sehingga daftar frame dan tombol di bawahnya
        # melompat naik-turun setiap next/prev.
        baris_px = tkfont.Font(font=("Segoe UI", 9)).metrics("linespace")
        self.kotak_info_rekaman = tk.Frame(i, bg=PANEL, height=int(baris_px * 10 / skala_ui.FAKTOR) + 6)
        self.kotak_info_rekaman.pack(fill="x", pady=(0, 4))
        self.kotak_info_rekaman.pack_propagate(False)
        self.info_rekaman = tk.Label(self.kotak_info_rekaman, text="", bg=PANEL, fg=ACCENT, justify="left", anchor="nw",
                                     wraplength=250, font=("Segoe UI", 9))
        self.info_rekaman.pack(fill="both", expand=True)
        # extended: Shift+klik / seret memblok rentang, Ctrl+klik menambah satu,
        # supaya banyak frame bisa dibuang ke sampah sekaligus. exportselection
        # dimatikan agar pilihan tidak lenyap saat teks lain dipilih.
        self.list_frame = tk.Listbox(i, height=8, bg="#FFF9F4", fg=INK, relief="flat", selectbackground=ACCENT_SOFT,
                                     selectforeground=INK,
                                     selectmode="extended", exportselection=False)
        self.list_frame.pack(fill="x"); self.list_frame.bind("<<ListboxSelect>>", lambda e: self.pilih_frame())
        tk.Label(i, text=KETERANGAN_LABEL, bg=PANEL, fg=MUTED, justify="left", anchor="w",
                 font=("Segoe UI", 8)).pack(fill="x", pady=(2, 0))
        # Status dan tombol periksa tepat di bawah daftar frame, selalu terlihat:
        # dulu berada di kartu bawah sehingga harus menggulir panel setiap frame.
        self.lencana_periksa = tk.Label(i, text="—  belum ada frame dipilih", bg=PANEL, fg=MUTED,
                                        font=("Segoe UI", 9, "bold"), pady=4, anchor="w", justify="left",
                                        wraplength=250, height=3)
        self.lencana_periksa.pack(fill="x", pady=(4, 2))
        periksa = tk.Frame(i, bg=PANEL); periksa.pack(fill="x")
        self.tombol_ringkas(periksa, "\u25c0", lambda: self.pindah_frame_label(-1), "#E8DDD5", INK, width=34).pack(side="left")
        self.tombol_ringkas(periksa, "\u2714 Diperiksa & lanjut \u25b6 (Enter)", self._periksa_lalu_lanjut, "#7FA96B", "#FFFFFF",
                             width=150).pack(side="left", fill="x", expand=True, padx=3)
        self.tombol_ringkas(periksa, "\u25b6", lambda: self.pindah_frame_label(1), "#E8DDD5", INK, width=34).pack(side="left")
        self.tombol_ringkas(i, "\u2714 / \u2716 Tandai atau cabut diperiksa (tanpa pindah)", self.toggle_periksa,
                             "#E8DDD5", INK, width=250).pack(fill="x", pady=(3, 0))
        # Klik kanan pada daftar menyalin nama frame yang ditunjuk KURSOR,
        # bukan yang sedang terpilih; keduanya sering berbeda saat menelusuri.
        self.list_frame.bind("<Button-3>", self._salin_dari_kursor)
        self.list_frame.bind("<Control-c>", lambda e: self._salin_frame_terpilih())
        self.list_frame.bind("<Control-C>", lambda e: self._salin_frame_terpilih(penuh=bool(e.state & 0x1)))
        self.tombol_ringkas(i, "Ekspor frame saat ini ke Label", self.ekspor_frame_kini_ke_label, GREEN, width=220).pack(fill="x", pady=(4, 0))
        nav = tk.Frame(i, bg=PANEL); nav.pack(fill="x", pady=(4, 0))
        self.tombol_ringkas(nav, "↻ Muat ulang gambar", self.kanvas.muat_ulang_gambar, "#E8DDD5", INK,
                             width=180).pack(fill="x")
        kelola = tk.Frame(i, bg=PANEL); kelola.pack(fill="x", pady=(4, 0))
        self.btn_sampah_frame = self.tombol_ringkas(kelola, "🗑 Sampahkan", self.toggle_sampah_frame, "#E8DDD5", INK)
        self.btn_sampah_frame.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.tombol_ringkas(kelola, "Hapus", self.hapus_frame_permanen, "#F3D8D4", INK).pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Checkbutton(i, text="\U0001f5d1 Tampilkan sampah frame (lalu ♻ Pulihkan)", variable=self.tampil_sampah_frame,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       command=self._ganti_tampilan_sampah_frame).pack(anchor="w", pady=(6, 0))
        b, i = self.card(right, "Label dan depth") ; b.pack(fill="x", pady=(0, 7))
        panel_label = ttk.Notebook(i); panel_label.pack(fill="both", expand=True)
        edit_i = tk.Frame(panel_label, bg=PANEL); otomatis_i = tk.Frame(panel_label, bg=PANEL)
        panel_label.add(edit_i, text="Edit mask")
        panel_label.add(otomatis_i, text="Otomatis")
        gpu = tk.Frame(otomatis_i, bg=PANEL, highlightthickness=1, highlightbackground=LINE, padx=6, pady=4)
        gpu.pack(fill="x", pady=(6, 2))
        tk.Checkbutton(gpu, text="Auto-label & model GPU", variable=self.model_gpu, command=self.ganti_model_gpu,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.info_model_gpu = tk.Label(gpu, text="", bg=PANEL, fg=MUTED, wraplength=290, justify="left",
                                       font=("Segoe UI", 8))
        self.info_model_gpu.pack(anchor="w")
        self._perbarui_info_model_gpu()
        kontrol = tk.Frame(otomatis_i, bg=PANEL); kontrol.pack(fill="x", pady=(6, 2))
        tk.Label(kontrol, text="PILIH KONTROL", bg=PANEL, fg=ACCENT, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Radiobutton(kontrol, text="Mudah: A/S/Space", variable=self.kontrol_label, value="mudah",
                       command=self.ganti_kontrol_label, bg=PANEL, fg=INK, selectcolor=PANEL,
                       activebackground=PANEL, font=("Segoe UI", 8)).pack(side="left")
        tk.Radiobutton(kontrol, text="Klasik: PgUp/PgDn/←→", variable=self.kontrol_label, value="klasik",
                       command=self.ganti_kontrol_label, bg=PANEL, fg=INK, selectcolor=PANEL,
                       activebackground=PANEL, font=("Segoe UI", 8)).pack(side="left", padx=(4, 0))
        self.label_bantuan_kontrol = tk.Label(otomatis_i, text=self.teks_kontrol_label(), bg=PANEL, fg=MUTED,
                                              wraplength=300, justify="left", font=("Segoe UI", 8))
        self.label_bantuan_kontrol.pack(anchor="w", pady=(1, 5))
        self.rb_objek = tk.Radiobutton(bilah_kelas, variable=self.mode_label, value="objek", indicatoron=False, command=self.ganti_mode,
                                       bg="#FFF9F4", selectcolor=RED, activebackground=RED, fg=INK, relief="flat", pady=5)
        self.rb_acuan = tk.Radiobutton(bilah_kelas, variable=self.mode_label, value="acuan", indicatoron=False, command=self.ganti_mode,
                                       bg="#FFF9F4", selectcolor=BLUE, activebackground=BLUE, fg=INK, relief="flat", pady=5)
        self.rb_objek.pack(side="left", fill="x", expand=True, padx=1)
        self.rb_acuan.pack(side="left", fill="x", expand=True, padx=1)
        tk.Frame(bilah_kelas, bg=LINE, width=1).pack(side="left", fill="y", padx=5)
        for nilai, teks in (("tukar", "⇄ Tukar warna (T)"), ("normal", "✏ Titik"), ("geser", "✋ Geser (G)"), ("blok", "▭ Blok (X)"),
                            ("potong", "✂ Pita (C)"), ("penghalang", "⬠ Halang (V)")):
            tk.Radiobutton(bilah_kelas, text=teks, variable=self.alat_label, value=nilai, indicatoron=False,
                           command=lambda: self.pilih_alat(self.alat_label.get(), dari_tombol=True),
                           bg="#FFF9F4", selectcolor="#E8DDD5", activebackground="#E8DDD5", fg=INK,
                           relief="flat", pady=5, font=("Segoe UI", 8)).pack(side="left", expand=True, fill="x", padx=1)
        cek = dict(bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL, wraplength=300, justify="left")
        tk.Label(bilah_salin, text="Salin label dari:", bg=PANEL, fg=INK).pack(side="left")
        for arah, teks in ((-1, "◀ Sebelumnya (P)"), (1, "Berikutnya (N) ▶")):
            self.tombol_ringkas(bilah_salin, teks, lambda a=arah: self.salin_label_tetangga(a), "#E8DDD5", INK,
                                width=118).pack(side="left", padx=(4, 0))
        tk.Frame(bilah_salin, bg=LINE, width=1).pack(side="left", fill="y", padx=6)
        sakelar = dict(indicatoron=False, bg="#FFF9F4", selectcolor="#E8DDD5", activebackground="#E8DDD5", fg=INK,
                       relief="flat", padx=8, pady=5, font=("Segoe UI", 8))
        tk.Checkbutton(bilah_salin, text="Bayangan (B)", variable=self.bayangan_aktif, command=self.ganti_bayangan,
                       **sakelar).pack(side="left", padx=1)
        tk.Label(bilah_salin, text="dari", bg=PANEL, fg=MUTED).pack(side="left", padx=3)
        for nilai, teks in ((-1, "sebelumnya"), (1, "berikutnya")):
            tk.Radiobutton(bilah_salin, text=teks, variable=self.frame_acuan, value=nilai, command=self.ganti_bayangan,
                           **sakelar).pack(side="left", padx=1)
        tk.Scale(edit_i, from_=.1, to=.85, resolution=.05, orient="horizontal", variable=self.bayangan_alpha,
                 command=lambda _: self.ganti_kekuatan_bayangan(), label="Kekuatan bayangan", bg=PANEL, fg=INK,
                 highlightthickness=0, length=220).pack(fill="x")
        tk.Checkbutton(edit_i, text="Garis label frame acuan (putus-putus)", variable=self.bayangan_label,
                       command=self.ganti_bayangan, **cek).pack(anchor="w")
        tk.Checkbutton(edit_i, text="Salinan mengikuti gerak kamera", variable=self.salin_selaras,
                       command=self.simpan_preferensi, **cek).pack(anchor="w")
        tk.Checkbutton(edit_i, text="Frame tanpa label langsung mulai dari label frame acuan",
                       variable=self.salin_otomatis, command=self.simpan_preferensi, **cek).pack(anchor="w", pady=(0, 4))
        tk.Scale(edit_i, from_=0, to=1.0, resolution=.05, orient="horizontal", variable=self.depth_alpha,
                 command=lambda _: self.ganti_depth(), label="Overlay depth (1,0 = depth murni)", bg=PANEL, fg=INK,
                 highlightthickness=0, length=220).pack(fill="x", pady=(2, 0))
        tk.Checkbutton(edit_i, text="🌙 Inframerah sebagai latar (frame gelap)", variable=self.pakai_ir,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       command=self.ganti_latar_ir).pack(anchor="w", pady=(4, 0))
        tk.Checkbutton(edit_i, text="📐 Garis lipatan 3-D: kuning = ujung tread, biru = pangkal riser",
                       variable=self.garis_3d, bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       wraplength=250, justify="left", command=self.ganti_garis_3d).pack(anchor="w")
        tk.Scale(edit_i, from_=0, to=1.0, resolution=.05, orient="horizontal", variable=self.warna_bidang,
                 command=lambda _: self.ganti_warna_bidang(), label="Warna ketinggian (tiap anak tangga satu warna)",
                 bg=PANEL, fg=INK, highlightthickness=0, length=220).pack(fill="x", pady=(1, 0))
        baris_depth = tk.Frame(edit_i, bg=PANEL); baris_depth.pack(fill="x", pady=(0, 2))
        tk.Label(baris_depth, text="Tampilan bantu", bg=PANEL, fg=MUTED).pack(side="left")
        pilih_depth = ttk.Combobox(baris_depth, state="readonly", width=30,
                                   values=list(visual_depth.MODE.values()))
        pilih_depth.set(visual_depth.MODE.get(self.depth_mode.get(), visual_depth.MODE["bidang"]))
        pilih_depth.pack(side="left", padx=4, fill="x", expand=True)
        pilih_depth.bind("<<ComboboxSelected>>", lambda _e: self.ganti_mode_depth(pilih_depth.get()))
        tk.Label(edit_i, text="Frame gelap: Inframerah selaras atau Bidang + overlay 1,0 menampilkan anak tangga dari sensor IR/depth.",
                 bg=PANEL, fg=MUTED, wraplength=300, justify="left", font=("Segoe UI", 8)).pack(anchor="w")
        tk.Scale(edit_i, from_=0, to=0.85, resolution=.05, orient="horizontal", variable=self.mask_alpha,
                 command=lambda _: self.ganti_opasitas_mask(), label="Opacity mask", bg=PANEL, fg=INK,
                 highlightthickness=0, length=220).pack(fill="x", pady=(1, 0))
        tk.Scale(edit_i, from_=0, to=1.0, resolution=.1, orient="horizontal", variable=self.kecerahan,
                 command=lambda _: self.ganti_kecerahan(), label="Cerahkan (frame gelap/malam)", bg=PANEL, fg=INK,
                 highlightthickness=0, length=220).pack(fill="x", pady=(1, 0))
        tk.Scale(edit_i, from_=0, to=2.0, resolution=.1, orient="horizontal", variable=self.ketajaman,
                 command=lambda _: self.ganti_ketajaman(), label="Pertajam (unsharp mask)", bg=PANEL, fg=INK,
                 highlightthickness=0, length=220).pack(fill="x", pady=(1, 0))
        tk.Scale(edit_i, from_=0, to=1.0, resolution=.1, orient="horizontal", variable=self.bantu_arah,
                 command=lambda _: self.ganti_bantu_arah(),
                 label="Pemandu riser/tread model RGB-D (biru tread, merah riser, garis kuning batas)", bg=PANEL, fg=INK,
                 highlightthickness=0, length=220).pack(fill="x", pady=(1, 0))
        alat_f = tk.Frame(edit_i, bg=PANEL); alat_f.pack(fill="x", pady=(4, 0))
        zoom_f = tk.Frame(alat_f, bg=PANEL); zoom_f.pack(fill="x", pady=(2, 0))
        for teks, perintah in (("🔍−  ( - )", lambda: self.zoom_tombol(1 / 1.25)),
                               ("🔍+  ( + )", lambda: self.zoom_tombol(1.25)),
                               ("⤢ Muat ( 0 )", self.muat_pas)):
            tk.Button(zoom_f, text=teks, command=perintah, bg="#FFF9F4", fg=INK, relief="flat",
                      font=("Segoe UI", 8), pady=4).pack(side="left", expand=True, fill="x", padx=1)
        potong_f = tk.Frame(alat_f, bg=PANEL); potong_f.pack(fill="x", pady=(2, 0))
        tk.Scale(potong_f, from_=2, to=120, resolution=1, orient="horizontal", variable=self.lebar_potong,
                 command=lambda _: self.ganti_lebar_potong(), label="Lebar pita potong (px citra)  [ / ]",
                 bg=PANEL, fg=INK, highlightthickness=0, length=220).pack(fill="x")
        tk.Checkbutton(potong_f, text="Anak tangga yang terbelah tetap 1 mask",
                       variable=self.satukan_potongan, command=self.ganti_satukan_potongan, bg=PANEL, fg=INK,
                       selectcolor=PANEL, activebackground=PANEL, font=("Segoe UI", 8)).pack(anchor="w")
        terap_f = tk.Frame(potong_f, bg=PANEL); terap_f.pack(fill="x", pady=(2, 0))
        self.tombol_ringkas(terap_f, "✂ Terapkan potongan (Enter)", self.terapkan_penghalang, "#F3D8D4", INK,
                            width=150).pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.tombol_ringkas(terap_f, "Batal (Esc)", self.batal_penghalang, "#E8DDD5", INK,
                            width=70).pack(side="left", fill="x", expand=True, padx=(2, 0))
        self.tombol_ringkas(potong_f, "⧉ Gabung mask terpilih (M)", self.gabung_mask, "#E8DDD5", INK,
                            width=220).pack(fill="x", pady=(2, 0))
        tk.Checkbutton(alat_f, text="Mode touchpad: gulir = geser, Ctrl+gulir = zoom",
                       variable=self.mode_touchpad, command=self.ganti_mode_touchpad, bg=PANEL, fg=INK,
                       selectcolor=PANEL, activebackground=PANEL, font=("Segoe UI", 8)).pack(anchor="w")
        magnet = tk.Frame(edit_i, bg=PANEL); magnet.pack(fill="x", pady=(1, 0))
        tk.Checkbutton(magnet, text="Magnet titik & garis", variable=self.magnet_titik,
                       command=self.ganti_magnet_titik, bg=PANEL, fg=INK, selectcolor=PANEL,
                       activebackground=PANEL).pack(side="left")
        self.tombol_ringkas(magnet, "🧲 Rapikan semua", self.rapikan_magnet_semua,
                             "#E8DDD5", INK, width=138).pack(side="right")
        self.tombol(otomatis_i, "✨ Rekomendasi tangga: ConvNeXt RGB-D", self.usulkan_segmentasi, GREEN).pack(fill="x", pady=(8, 2))
        self.tombol(otomatis_i, "⚡ Batch auto-label frame baru", self.batch_auto_label, BLUE).pack(fill="x", pady=(4, 2))
        tk.Label(otomatis_i, text="X lalu tarik = blok hapus (atau Ctrl+tarik) • C lalu tarik sepanjang "
                                  "tiang = pita, [ ] ubah lebar • V lalu klik titik/tarik laso mengelilingi "
                                  "penghalang • garis kuning bisa disunting: tarik titik, klik sisi = sisip, klik kanan "
                                  "= hapus titik • Enter/klik ganda = potong, Esc batal • "
                                  "Shift+tarik mengenai beberapa mask lalu M = gabung jadi 1 • G lalu tarik = geser "
                                  "(atau Alt+tarik / tombol tengah) • Esc kembali ke titik • "
                                  "Shift+tarik pilih titik • Ctrl+C nama frame • Ctrl+Shift+C jalur lengkap",
                 bg=PANEL, fg=MUTED, wraplength=300, justify="left").pack(anchor="w", pady=(4, 2))
        self.tombol(otomatis_i, "Bangun folder dataset YOLO", self.bangun_yolo, GREEN).pack(fill="x", pady=(4, 2))
        for teks, perintah, lebar in (("Redo", self.kanvas.redo_riwayat, 50), ("Undo", self.kanvas.undo_riwayat, 50),
                                      ("+ Titik", self.mulai_tambah_titik, 54), ("+ Mask", self.mulai_mask_baru, 54)):
            self.tombol_ringkas(bilah_salin, teks, perintah, "#E8DDD5", INK, width=lebar).pack(side="right", padx=(3, 0))
        nomor = tk.Frame(edit_i, bg=PANEL); nomor.pack(fill="x", pady=(4, 0))
        tk.Label(nomor, text="Mask yang diedit", bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w")
        self.spin_nomor_mask = tk.Spinbox(nomor, from_=1, to=999, textvariable=self.nomor_mask, width=5,
                                           command=lambda: self.atur_nomor_mask(senyap=True))
        self.spin_nomor_mask.grid(row=0, column=1, sticky="w", padx=6)
        self.spin_nomor_mask.bind("<Return>", lambda _e: self.atur_nomor_mask(senyap=True))
        self.spin_nomor_mask.bind("<FocusOut>", lambda _e: self.atur_nomor_mask(senyap=True))
        self.tombol_ringkas(nomor, "Pilih mask", self.atur_nomor_mask, "#E8DDD5", INK, width=130).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        nomor.columnconfigure(0, weight=1)
        hapus = tk.Frame(edit_i, bg=PANEL); hapus.pack(fill="x", pady=(5, 0))
        self.btn_hapus_objek = self.tombol_ringkas(hapus, "Hapus mask", self.kanvas.hapus_mask_aktif, "#F3D8D4", INK)
        self.btn_hapus_acuan = self.tombol_ringkas(hapus, "Buang saran", self.buang_rekomendasi, "#F3D8D4", INK)
        self.btn_hapus_objek.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.btn_hapus_acuan.pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Label(edit_i, text="● AUTO-SAVE AKTIF", bg=PANEL, fg=GREEN,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(6, 1))
        tk.Label(edit_i, text="Klik kanan lepas/hapus titik • Ctrl+Z undo • Ctrl+Shift+Z redo", bg=PANEL, fg=MUTED, wraplength=300, justify="left").pack(anchor="w", pady=(2, 0))
        tk.Label(edit_i, textvariable=self.ukur_status, bg=PANEL, fg=INK, wraplength=300, justify="left", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(3, 0))
        # Ukuran meteran per anak tangga, dengan nomor dari pelacakan video utuh:
        # pada frame ekspor saja anak tangga ke-1, 2, 3 sulit dibedakan.
        b, i = self.card(right, "Ukuran tangga (meteran)") ; b.pack(fill="x", pady=(0, 7))
        try:
            try:
                from .panel_ukuran import PanelUkuran
            except ImportError:
                from panel_ukuran import PanelUkuran
            self.panel_ukuran = PanelUkuran(i, self)
        except Exception as e:                                   # noqa: BLE001 -- tab Label tetap dapat dipakai
            self.panel_ukuran = None
            tk.Label(i, text=f"Panel ukuran tidak dapat dimuat: {e}", bg=PANEL, fg=MUTED, wraplength=300,
                     justify="left").pack(anchor="w")
        self.perbarui_konteks_label()

    # ----- rekam -----
    def _mulai_kamera_async(self):
        threading.Thread(target=self._mulai_kamera, daemon=True).start()

    def tukar_warna_mask(self, nama: str, indeks: int):
        """Merah jadi biru atau sebaliknya untuk satu mask tertentu."""
        ke = "acuan" if nama == "objek" else "objek"
        self.kanvas.pindah_kelas(nama, indeks, ke)
        self._selesai_pindah_kelas(nama, ke)

    def tukar_di_kursor(self):
        """T: tukar mask di bawah kursor; tanpa mask di sana, alat Tukar menunggu klik."""
        k = self.kanvas
        x, y = k.winfo_pointerx() - k.winfo_rootx(), k.winfo_pointery() - k.winfo_rooty()
        kena = k.mask_di(x, y) if 0 <= x < k.winfo_width() and 0 <= y < k.winfo_height() else None
        if kena is None:
            return self.pilih_alat("tukar")
        self.tukar_warna_mask(*kena)
        return "break"

    def ubah_kelas_mask(self, ke: str):
        """Shift+A/S: pindahkan mask aktif ke kelas yang disebut."""
        dari = self.kanvas.ubah_kelas_aktif(ke)
        if dari is None:
            self.status.set("Pilih dulu mask yang mau diubah kelasnya, dengan mengklik salah satu titiknya.")
            return "break"
        self._selesai_pindah_kelas(dari, ke)
        return "break"

    def _selesai_pindah_kelas(self, dari: str, ke: str) -> None:
        nama = {"objek": "merah", "acuan": "biru"}
        self.mode_label.set(ke)
        self._mode_label_dipilih = True
        self.kanvas.mode = ke
        indeks = self.kanvas.aktif_indeks.get(ke)
        if indeks is not None:
            self.nomor_mask.set(indeks + 1)
        self.simpan_draft_label()
        self.kanvas.focus_set()
        self.status.set(f"Mask dipindah dari {nama[dari]} ke {nama[ke]}. Ctrl+Z membatalkan.")
        self._tahan_status = True

    def zoom_tombol(self, faktor: float):
        self.kanvas.zoom_tengah(faktor)
        self.kanvas.focus_set()
        return "break"

    def muat_pas(self):
        """Kembalikan gambar agar seluruhnya terlihat."""
        self.kanvas.fit()
        self.kanvas.focus_set()
        return "break"

    def pilih_alat(self, nama: str, dari_tombol: bool = False):
        """Alat tarik tanpa modifier, supaya papan sentuh tidak perlu menahan Ctrl.

        Menekan tombol yang sama dua kali kembali ke alat titik, sehingga satu
        tombol cukup untuk masuk dan keluar.
        """
        if not dari_tombol and nama != "normal" and self.kanvas.alat == nama:
            nama = "normal"
        self.kanvas.set_alat(nama)
        self.alat_label.set(self.kanvas.alat)
        self.kanvas.focus_set()
        self.status.set({
            "geser": "Alat GESER aktif: tarik satu jari untuk menggeser gambar. G atau Esc kembali ke titik.",
            "blok": "Alat BLOK HAPUS aktif: tarik kotak untuk membuang mask di dalamnya. X atau Esc kembali ke titik.",
            "penghalang": ("Alat HALANG aktif: klik titik mengelilingi penghalang (atau tarik laso). Tarik titik "
                           "untuk menggeser, klik sisi untuk sisip titik, klik kanan hapus titik. "
                           "Enter / klik ganda = potong semua mask di dalamnya, Esc batal."),
            "potong": (f"Alat PITA aktif: tarik garis di sepanjang tiang (lebar {self.kanvas.lebar_potong:.0f} px, "
                       f"[ ] mengubah). Pita jadi garis kuning yang bisa dirapikan, lalu Enter untuk memotong."),
            "tukar": "Alat TUKAR aktif: klik mask yang salah warna, merah jadi biru atau sebaliknya. T atau Esc kembali ke titik.",
            "normal": "Alat titik aktif: klik menambah titik, tarik memindahkannya.",
        }[self.kanvas.alat])
        return "break"

    def ganti_lebar_potong(self):
        self.kanvas.lebar_potong = float(self.lebar_potong.get())
        self.simpan_preferensi()

    def ganti_satukan_potongan(self):
        self.kanvas.satukan_potongan = bool(self.satukan_potongan.get())
        self.simpan_preferensi()
        self.status.set("Mask yang terbelah penghalang tetap 1 instance (disambung garis tipis)."
                        if self.kanvas.satukan_potongan else
                        "Mask yang terbelah penghalang dipecah jadi beberapa mask terpisah.")

    def _lebar_potong_tombol(self, faktor: float):
        lebar = self.kanvas.atur_lebar_potong(faktor)
        self.lebar_potong.set(round(lebar))
        self.simpan_preferensi()
        self.status.set(f"Lebar pita potong {lebar:.0f} px.")
        return "break"

    def _kanvas_ganti_alat(self, nama: str, sebab: str = "") -> None:
        self.alat_label.set(nama)
        if sebab == "pita":
            self.status.set("Pita jadi garis penghalang kuning: tarik titik/sisi untuk merapikan, klik kanan hapus "
                            "titik, lalu Enter (atau klik ganda) untuk memotong. Esc batal.")

    def terapkan_penghalang(self):
        if not self.kanvas._penghalang:
            self.status.set("Belum ada garis penghalang. Tekan V lalu klik titik mengelilingi penghalang, "
                            "atau C lalu tarik pita.")
        else:
            self.kanvas.selesai_penghalang()
        self.kanvas.focus_set()
        return "break"

    def batal_penghalang(self):
        if self.kanvas.batal_penghalang():
            self.status.set("Garis penghalang dibatalkan.")
        self.kanvas.focus_set()
        return "break"

    def gabung_mask(self):
        laporan = self.kanvas.gabung_mask_dipilih()
        if not laporan:
            self.status.set("Gabung mask: Shift+tarik kotak yang mengenai titik dari 2 mask atau lebih "
                            "(kelas sama) dulu, lalu tekan M.")
        else:
            nama = {"objek": "merah", "acuan": "biru"}
            self.status.set("Gabung mask: " + ", ".join(f"{n} mask {nama.get(k, k)} jadi 1" for k, n in laporan)
                            + ". Ctrl+Z membatalkan.")
        self.kanvas.focus_set()
        return "break"

    def _lapor_potong(self, n_kena: int, n_belah: int, n_habis: int) -> None:
        if n_kena < 0:
            self.status.set("Halang: butuh minimal 3 titik untuk membentuk penghalang."); return
        if not n_kena:
            self.status.set("Potong: pita tidak mengenai mask mana pun."); return
        bagian = [f"{n_kena} mask dipotong"]
        if n_belah:
            bagian.append(f"{n_belah} terbelah" + (" (tetap 1 mask)" if self.kanvas.satukan_potongan else " jadi beberapa mask"))
        if n_habis:
            bagian.append(f"{n_habis} habis terbuang")
        self.status.set("Potong: " + ", ".join(bagian) + ". Ctrl+Z membatalkan.")

    def ganti_mode_touchpad(self):
        self.kanvas.touchpad = bool(self.mode_touchpad.get())
        self.simpan_preferensi()
        self.status.set("Mode touchpad: gulir dua jari menggeser, Ctrl+gulir memperbesar."
                        if self.kanvas.touchpad else
                        "Mode tetikus: gulir memperbesar, Shift+gulir menggeser mendatar.")

    def ganti_tema(self):
        """Simpan tema lawan lalu tawarkan membuka ulang Studio (warna dipasang saat mulai)."""
        baru = "terang" if tema.GELAP else "gelap"
        tema.simpan_tema(baru)
        if messagebox.askyesno("Tema tampilan", f"Tema {baru} disimpan.\n\nBuka ulang Studio sekarang untuk menerapkannya?\n"
                               "(Draft label dan catatan disimpan dulu seperti saat menutup biasa.)", parent=self):
            self._buka_ulang = True
            self.tutup()

    def simpan_preferensi(self):
        tulis_json(self.path_preferensi, {"kategori": self.kategori.get(), "split": self.split.get(),
                                          "kode_adegan": self.kode.get().strip(),
                                          "kontrol_label": self.kontrol_label.get(),
                                          "mode_touchpad": bool(self.mode_touchpad.get()),
                                          "lebar_potong": float(self.lebar_potong.get()),
                                          "satukan_potongan": bool(self.satukan_potongan.get()),
                                          "kecerahan": float(self.kecerahan.get()),
                                          "ketajaman": float(self.ketajaman.get()),
                                          "bantu_arah": float(self.bantu_arah.get()),
                                          "depth_mode": self.depth_mode.get(),
                                          "pakai_ir": bool(self.pakai_ir.get()),
                                          "warna_bidang": float(self.warna_bidang.get()),
                                          "garis_3d": bool(self.garis_3d.get()),
                                          "bayangan_aktif": bool(self.bayangan_aktif.get()),
                                          "bayangan_alpha": float(self.bayangan_alpha.get()),
                                          "bayangan_label": bool(self.bayangan_label.get()),
                                          "frame_acuan": int(self.frame_acuan.get()),
                                          "salin_selaras": bool(self.salin_selaras.get()),
                                          "salin_otomatis": bool(self.salin_otomatis.get()),
                                          "model_gpu": bool(self.model_gpu.get())})

    def ganti_tab(self, _event=None):
        """Matikan stream yang tidak diperlukan agar labeling tetap ringan."""
        # Catatan/scene dan label bisa berubah di tab lain; tab Split memuat ulang sendiri.
        if self.tabs.select() == str(getattr(self, "tab_split", "")) and getattr(self, "panel_split", None):
            self.panel_split.muat_ulang()
        # Rekaman yang terakhir dipilih di tab Split ikut terpilih di Tinjau/Label/Ekspor,
        # agar ✔-nya bisa langsung diubah tanpa mencari ulang di daftar.
        dari_split = getattr(self, "_sesi_dari_split", None)
        if (dari_split is not None and self.tabs.select() != str(getattr(self, "tab_split", ""))
                and dari_split != self.sesi and dari_split in getattr(self, "_map_sesi", [])):
            i = self._map_sesi.index(dari_split)
            self.list_sesi.selection_clear(0, "end"); self.list_sesi.selection_set(i)
            self.list_sesi.activate(i); self.list_sesi.see(i)
            self.pilih_sesi()
            if self.tabs.select() == str(self.tab_label):
                self.muat_frame()
        elif (self.tabs.select() == str(self.tab_label) and self.sesi is not None
              and self.label_path not in getattr(self, "frame_paths", [])):
            self.muat_frame()                           # tab Label tidak boleh kosong saat dibuka
        # Frame yang terakhir dilihat di tab Split dibuka di tab Label (rekaman yang sama).
        frame_split = getattr(self, "_frame_dari_split", None)
        if (frame_split is not None and self.tabs.select() == str(self.tab_label)
                and frame_split != self.label_path and frame_split in getattr(self, "frame_paths", [])):
            self._pilih_indeks_frame(self.frame_paths.index(frame_split))
        if self.tabs.select() != str(getattr(self, "tab_split", "")):
            self._frame_dari_split = None
        self._sesi_dari_split = None
        # Uji realtime memakai kamera secara eksklusif; hentikan begitu pengguna
        # pindah ke tab lain agar device tidak diperebutkan dengan preview/rekam.
        if self.tabs.select() != str(self.tab_uji) and getattr(self, 'panel_uji', None) is not None:
            if getattr(self.panel_uji, 'jalan', False):
                self.panel_uji.berhenti()
        pada_rekam = self.tabs.select() == str(self.tab_rekam)
        if not pada_rekam and not self.sedang_rekam and self.cam.hidup:
            self.preview_diminta = False
            self._hentikan_render()
            self.cam.hentikan()
            self.preview_info = None
            self.btn_preview_kamera.configure(text="Aktifkan preview kamera")
            self.status.set("Kamera dihentikan saat labeling/tinjau agar aplikasi tetap ringan.")

    def buka_uji_depth(self):
        if self.sedang_rekam:
            self.status.set("Selesaikan rekaman dulu sebelum uji depth dinding.")
            return
        jendela = getattr(self, "_jendela_uji_depth", None)
        if jendela is not None and jendela.winfo_exists():
            jendela.lift()
            return
        from .uji_depth_dinding import JendelaUjiDepth
        self._jendela_uji_depth = JendelaUjiDepth(self)

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

    def _perbarui_sudut(self, dep_kecil):
        """Sudut tunduk kamera dari depth (sudut_kamera.py) tiap 0,2 s, dihaluskan median 5 bacaan.

        Selama merekam setiap bacaan disimpan; ringkasannya ditulis ke session.json saat selesai."""
        sekarang = time.monotonic()
        if sekarang - getattr(self, "_sudut_t", 0.0) >= 0.2:
            self._sudut_t = sekarang
            i = self.preview_info
            h = sudut_kamera.sudut_tunduk(dep_kecil, i["fx"] / 2, i["fy"] / 2, i["cx"] / 2, i["cy"] / 2)
            riwayat = self.__dict__.setdefault("_sudut_riwayat", collections.deque(maxlen=5))
            if h is None:
                riwayat.clear()
            else:
                riwayat.append(h)
                if self.sedang_rekam:
                    self.__dict__.setdefault("_sudut_rekam", []).append((time.time(), h["tunduk"], h["samping"]))
            if riwayat:
                self._sudut_teks = sudut_kamera.teks({k: float(np.median([r[k] for r in riwayat]))
                                                      for k in ("tunduk", "samping", "porsi")})
            else:
                self._sudut_teks = sudut_kamera.teks(None)
        return getattr(self, "_sudut_teks", sudut_kamera.teks(None))

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
        self._pita(vis, self._perbarui_sudut(dep_kecil), di_bawah=True)

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
        # Uji realtime memakai kamera secara eksklusif; hentikan begitu pengguna
        # pindah ke tab lain agar device tidak diperebutkan dengan preview/rekam.
        if self.tabs.select() != str(self.tab_uji) and getattr(self, 'panel_uji', None) is not None:
            if getattr(self.panel_uji, 'jalan', False):
                self.panel_uji.berhenti()
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
                sudut = sudut_kamera.ringkas(getattr(self, "_sudut_rekam", []))
                if sudut is not None:
                    meta["sudut_kamera"] = sudut
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
            self._sudut_rekam = []
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
            self._map_sesi.append(p); self.list_sesi.insert("end", self._teks_item_sesi(p, ikon))
            self._warnai_item_sesi(len(self._map_sesi) - 1, p)
        if self._map_sesi:
            target = self._map_sesi.index(self.sesi) if self.sesi in self._map_sesi else 0
            self.list_sesi.selection_set(target); self.list_sesi.activate(target)
            self.after_idle(self.pilih_sesi)
        self.status.set(f"{len(self._map_sesi)} rekaman {self.filter_sesi.get()} ditampilkan.")

    def pilih_sesi(self):
        sel = self.list_sesi.curselection()
        if not sel: return
        self._simpan_catatan_sekarang()
        self._tutup_video(); self._hapus_ukur()
        self.sesi = self._map_sesi[sel[0]]
        self._muat_catatan_ui()
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
        alasan = periksa_rekaman(self.bag(self.sesi))
        if alasan:
            messagebox.showerror("Rekaman rusak", f"Preview tidak dapat dibuat: {alasan}.\n\n"
                                 "Rekaman ini sebaiknya dipindah ke tempat sampah.", parent=self); return
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
        """Bangun turunan preview secara atomik.

        MP4 baru tidak pernah menggantikan preview lama sebelum VideoWriter
        ditutup dan file tersebut terbukti dapat dibaca ulang. Ini mencegah
        `moov atom not found` bila worker gagal/Studio ditutup di tengah proses.
        """
        writer = None
        out_tmp = None
        depth_tmp = None
        try:
            derived = sesi / "derived"; derived.mkdir(exist_ok=True)
            # Depth ikut disimpan per frame. Tanpa ini, mengukur pada frame yang
            # sedang di-pause mustahil: seek pada .db3 terukur 10 detik sekali
            # panggil DAN mengembalikan frame yang salah, sedangkan mendekode
            # ulang dari awal butuh puluhan detik. Preview toh sudah mendekode
            # seluruh rekaman sekali, jadi menyimpannya di sini nyaris gratis.
            dir_depth = derived / "depth"
            depth_tmp = derived / "depth.sedang"
            if depth_tmp.exists(): shutil.rmtree(depth_tmp)
            depth_tmp.mkdir()
            out = derived / "preview.mp4"
            out_tmp = derived / "preview.sedang.mp4"
            if out_tmp.exists(): out_tmp.unlink()
            index = []; skala_depth = None
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
                    h,w = bgr.shape[:2]; writer = cv2.VideoWriter(str(out_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w,h))
                    if not writer.isOpened():
                        raise RuntimeError("VideoWriter preview tidak dapat dibuka.")
                    kamera = self._info_profile(profile, color, depth)
                writer.write(bgr)
                # PNG 16-bit = lossless. Depth TIDAK boleh lewat kompresi lossy;
                # satu nilai Z16 yang bergeser berarti jarak yang salah.
                cv2.imwrite(str(depth_tmp / f"{i:06d}.png"),
                            np.asanyarray(aligned.get_data()),
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
                index.append({"i": i, "frame": int(depth.get_frame_number()), "timestamp_ms": f"{depth.get_timestamp():.3f}"})
            if writer is None:
                raise RuntimeError("Tidak menemukan pasangan RGB dan depth dalam rekaman.")
            writer.release(); writer = None
            # Jangan menerbitkan MP4 yang belum lengkap: buka ulang terlebih dahulu.
            cap = cv2.VideoCapture(str(out_tmp)); ok,img = cap.read(); cap.release()
            if not ok:
                raise RuntimeError("Preview sementara tidak dapat dibaca ulang; preview lama dipertahankan.")
            # Hanya setelah valid, ganti turunan lama. RAW tidak tersentuh.
            os.replace(out_tmp, out)
            if dir_depth.exists(): shutil.rmtree(dir_depth)
            os.replace(depth_tmp, dir_depth); depth_tmp = None
            if kamera:
                kamera["fps_video_terukur"] = round(fps_video, 3)
                tulis_json(derived / "kamera.json", kamera)
            with (derived / "frame_index.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["i","frame","timestamp_ms"]); w.writeheader(); w.writerows(index)
            if ok: cv2.imwrite(str(derived / "thumbnail.jpg"), img)
            self.q.put(("preview_selesai", (sesi, len(index))))
        except Exception as e:
            # release di sini menulis indeks MP4 sementara bila error terjadi
            # setelah beberapa frame, tetapi file itu tidak pernah dipublikasikan.
            if writer is not None:
                writer.release()
            if out_tmp is not None and out_tmp.exists():
                out_tmp.unlink()
            if depth_tmp is not None and depth_tmp.exists():
                shutil.rmtree(depth_tmp)
            self.q.put(("error", f"Gagal membuat preview: {e}"))

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
        if not messagebox.askyesno("Hapus rekaman", f"Hapus seluruh sesi ini?\n\n{target.name}\n\nFoldernya (RAW, preview, ekspor, label) "
                                   "dipindah ke Tempat sampah Studio dan masih bisa dipulihkan. Ruang disk baru kembali "
                                   "setelah tempat sampah dikosongkan.", icon="warning", parent=self):
            return
        self._tutup_video()
        try:
            TS.buang(self.root_data, target, "rekaman")
            self.sesi = None; self.indeks = []; self.frame_paths = []; self.label_path = None; self.label_info = None
            self.list_frame.delete(0, "end"); self.kanvas.delete("all")
            self.kategori_sesi.set("(rekaman dihapus)")
            self.muat_daftar()
            self.status.set(f"{target.name} dipindah ke Tempat sampah Studio (bisa dipulihkan).")
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
        self._gambar_pita_label()
        return True

    def _x_pita(self, i: int) -> float:
        """Posisi x indeks frame i pada pita, selaras dengan jalur slider scale_pos."""
        lebar = max(self.pita_label.winfo_width(), 1)
        n = max(int(float(self.scale_pos.cget("to"))), 1)
        tepi = int(self.scale_pos.cget("sliderlength")) / 2 + 2
        return tepi + (lebar - 2 * tepi) * min(max(i / n, 0.0), 1.0)

    def _gambar_pita_label(self) -> None:
        if not hasattr(self, "pita_label"):
            return
        self.pita_label.delete("all")
        self._pita_frame = []
        if not self.sesi or float(self.scale_pos.cget("to")) <= 0:
            return
        for p in sorted((self.sesi / "exports" / "frames").glob("frame_*")):
            i = CR.indeks_frame(p.name)
            st = status_label(p)
            if i is None or st not in WARNA_LABEL or baca_json(p / "frame_state.json", {}).get("di_sampah"):
                continue
            self._pita_frame.append((i, st))
            x = self._x_pita(i)
            tinggi = 12 if st in ("claude_cek", "diperiksa") else 9
            self.pita_label.create_rectangle(x - 1, 12 - tinggi, x + 1, 12, fill=WARNA_LABEL[st], outline="")

    def _klik_pita_label(self, e) -> None:
        if not self._pita_frame:
            return
        i, _ = min(self._pita_frame, key=lambda f: abs(self._x_pita(f[0]) - e.x))
        self.posisi.set(i)
        self._tampilkan(i)

    def _tutup_video(self):
        self._hentikan_pemutar()
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        self._n_frame = 0
        self._depth_frame_cache = (None, None)

    def buka_tempat_sampah(self):
        """Frame/rekaman yang dihapus: pulihkan, hapus selamanya, atau kosongkan."""
        lama = getattr(self, "_jendela_sampah", None)
        if lama is not None and lama.winfo_exists():
            lama.lift(); lama.muat(); return
        self._jendela_sampah = TS.JendelaTempatSampah(self)

    def buka_jumlah_data(self):
        """Jendela jumlah frame berlabel per rekaman dan perkiraan data train/val/test."""
        lama = getattr(self, "_jendela_jumlah", None)
        if lama is not None and lama.winfo_exists():
            lama.lift(); lama.hitung(); return
        kat = self.filter_sesi.get() if self.filter_sesi.get() in KATEGORI else "tangga_naik"
        self._jendela_jumlah = JendelaJumlahData(self, kat, KATEGORI)

    def buka_pemutar_ukur(self, sesi: Path | None = None, mulai: int | None = None):
        """Jendela "Putar & isi meteran": rekaman diputar dengan nomor anak tangga dari pelacak."""
        sesi = sesi or self.sesi
        if not sesi:
            messagebox.showinfo("Pilih rekaman", "Pilih rekaman dahulu.", parent=self); return
        self._hentikan_pemutar()
        try:
            from .pemutar_ukur import buka
        except ImportError:
            from pemutar_ukur import buka
        buka(self, Path(sesi), mulai)

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
                "Klik 1 pada bidang acuan, klik 2 pada titik yang diukur."
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
            self.rb_objek.configure(text=f"Merah: {objek.replace('_', ' ')}")
            self.rb_acuan.configure(text=f"Biru: {acuan.replace('_', ' ')}")

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
        """Draft maupun label jadi bukti frame pernah disentuh pengguna.

        Frame yang ditandai diperiksa tanpa poligon juga dihitung. Itu contoh
        latar yang disengaja (lantai atau bordes tanpa tangga), bukan frame
        yang belum dilabeli, sehingga tidak boleh ikut dibuang.
        """
        label = frame / "label_yolo_seg.txt"
        if label.exists() and label.read_text(encoding="utf-8").strip():
            return True
        draft = baca_json(frame / "label_draft.json", {})
        if draft.get("diperiksa_manual"):
            return True
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
                                   "Frame yang sudah dilabeli tetap dipertahankan.",
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
        if not messagebox.askyesno("Hapus semua ekspor", "Hapus SEMUA frame ekspor, gambar, depth, IR, mask, label, dan video ekspor sesi ini?\n\n"
                                   "Foldernya dipindah ke Tempat sampah Studio dan masih bisa dipulihkan. Rekaman RAW tidak "
                                   "dihapus.", icon="warning", parent=self):
            return
        try:
            TS.buang(self.root_data, root, "ekspor")
            self.frame_paths = []; self.label_path = None; self.label_info = None
            self.list_frame.delete(0, "end"); self.kanvas.delete("all")
            self.status.set("Semua hasil ekspor dipindah ke Tempat sampah Studio (bisa dipulihkan). RAW tetap aman.")
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
            # Paket yang pernah dibuang/dipindahkan tidak boleh tersisa hanya
            # sebagai entri metadata; daftar ekspor harus merefleksikan disk.
            frame_meta = {
                int(x["index_bag"]): x for x in lama.get("frames", [])
                if "index_bag" in x and "folder" in x and (root / x["folder"]).is_dir()
            }
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
                ts_ir={}; ir_basi={}
                for j,nm,sisi in ((1,"ir_left_raw.png","kiri"),(2,"ir_right_raw.png","kanan")):
                    ir=native.get_infrared_frame(j)
                    if not ir: continue
                    # Frameset pertama kadang membawa IR basi (105348: 201 ms lebih awal dari
                    # depth). IR seperti itu tidak ditulis: menyesatkan saat melabel.
                    if abs(float(ir.get_timestamp())-float(depth.get_timestamp()))>1.0:
                        ir_basi[sisi]=round(float(ir.get_timestamp())-float(depth.get_timestamp()),1); continue
                    cv2.imwrite(str(folder/nm),np.asanyarray(ir.get_data())); ts_ir[sisi]=float(ir.get_timestamp())
                # Cap waktu tiap sensor ikut dicatat agar kesesuaian RGB-depth-IR
                # per frame bisa diperiksa tanpa rekaman mentah (lihat sinkron.py).
                info=self._info_profile(profile,color,depth); fm={"id":folder.name,"kategori":sesi.parent.name,"index_bag":i,"frame_number":int(depth.get_frame_number()),"timestamp_kamera_ms":float(depth.get_timestamp()),"timestamp_rgb_ms":float(color.get_timestamp()),"frame_number_rgb":int(color.get_frame_number()),"timestamp_ir_ms":ts_ir,**({"ir_basi_dilewati_ms":ir_basi} if ir_basi else {}),"format_depth":"Z16 native; meter=nilai*depth_scale","raw_bag_sumber":str(self.bag(sesi)),**info}
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

    def _pilih_indeks_frame(self, i: int) -> None:
        """Sorot dan buka frame ke-i pada daftar, dijepit ke rentang yang sah."""
        if not self.frame_paths:
            # Suntingan yang masih menunggu autosave disimpan ke frame lamanya
            # sebelum label_path dikosongkan; tanpa ini suntingan itu hilang.
            if self._autosave_setelah is not None:
                self.after_cancel(self._autosave_setelah)
                self._autosave_setelah = None
                self.simpan_draft_label()
            self.label_path = None
            self.kanvas.delete("all")
            return
        i = max(0, min(i, len(self.frame_paths) - 1))
        self.list_frame.selection_clear(0, "end")
        self.list_frame.selection_set(i)
        self.list_frame.activate(i)
        self.list_frame.see(i)
        self._buka_frame_ekspor(self.frame_paths[i])

    def muat_frame(self, lanjut_ke: int | None = None):
        """Muat ulang daftar frame.

        `lanjut_ke` menahan posisi setelah sebuah frame hilang dari daftar,
        misalnya karena dibuang ke sampah. Tanpa itu daftar dimuat ulang dan
        pilihan jatuh kembali ke awal, sehingga membuang beberapa frame
        berturut-turut memaksa menggulir dari atas setiap kali. Karena frame
        yang dibuang lenyap dari daftar, indeks yang sama kini menunjuk frame
        BERIKUTNYA -- persis perilaku yang diharapkan.
        """
        if not self.sesi: messagebox.showinfo("Pilih sesi", "Pilih rekaman pada tab Tinjau dahulu.", parent=self); return
        semua = self.daftar_frame_ekspor()
        self.frame_paths=[]
        catatan_sesi = self._catatan_cache(self.sesi)
        self.list_frame.delete(0,"end")
        for p in semua:
            state = baca_json(p / "frame_state.json", {"di_sampah": False})
            if state.get("di_sampah", False) != self.tampil_sampah_frame.get(): continue
            ikon = "\U0001f5d1 " if state.get("di_sampah") else ""
            self.frame_paths.append(p)
            self.list_frame.insert("end", self._teks_item_frame(p, ikon, catatan_sesi))
            self._warnai_item_frame(len(self.frame_paths) - 1, p)
        self.status.set(f"{len(self.frame_paths)} frame ekspor dimuat.")
        if lanjut_ke is not None:
            self._pilih_indeks_frame(lanjut_ke)
        elif self.label_path in self.frame_paths:
            i = self.frame_paths.index(self.label_path)          # frame yang sedang dibuka tetap tersorot
            self.list_frame.selection_clear(0, "end"); self.list_frame.selection_set(i)
            self.list_frame.activate(i); self.list_frame.see(i)
        elif self.frame_paths and self.tabs.select() == str(self.tab_label):
            # Dulu daftar dimuat tanpa membuka frame apa pun, sehingga tab Label kosong
            # sampai frame diklik atau Enter ditekan. Kini frame terakhir yang dibuka di
            # rekaman ini (atau frame pertama) langsung dibuka. Hanya saat tab Label
            # terlihat: membuka frame bisa memicu usulan otomatis di GPU.
            self._buka_frame_ingatan()

    def _buka_frame_ingatan(self) -> None:
        ingat = getattr(self, "_frame_terakhir", {}).get(str(self.sesi))
        self._pilih_indeks_frame(self.frame_paths.index(ingat) if ingat in self.frame_paths else 0)

    def _teks_item_frame(self, p: Path, ikon: str = "", catatan_sesi: dict | None = None) -> str:
        if catatan_sesi is None:
            catatan_sesi = self._catatan_cache(self.sesi)
        sc = CR.scene_untuk(catatan_sesi, CR.indeks_frame(p.name))
        st_ = status_label(p)
        teks = f"{ikon}{IKON_LABEL.get(st_, '')}{p.name}"
        if st_ == "diperiksa" and not any(len(q) >= 3 for v in baca_json(p / "label_draft.json", {}).get("poligon", {}).values()
                                          for q in v):
            teks += "  · ∅ LATAR"                       # diperiksa tanpa tangga: contoh latar, mask kosong
        return teks + (f"  · {sc['nama']}" if sc else "")

    def _warnai_item_frame(self, i: int, p: Path) -> None:
        warna = WARNA_LABEL.get(status_label(p))
        if warna:
            self.list_frame.itemconfig(i, fg=warna)

    def _segarkan_item_frame(self) -> None:
        """Perbarui ikon/warna frame aktif di daftar setelah statusnya berubah (simpan, tandai diperiksa)."""
        if not self.label_path or self.label_path not in getattr(self, "frame_paths", []):
            return
        i = self.frame_paths.index(self.label_path)
        state = baca_json(self.label_path / "frame_state.json", {"di_sampah": False})
        ikon = "\U0001f5d1 " if state.get("di_sampah") else ""
        teks = self._teks_item_frame(self.label_path, ikon)
        warna = WARNA_LABEL.get(status_label(self.label_path), "")
        # Hanya bila berubah: menghapus-menyisipkan baris terpilih setiap kali
        # frame dibuka membuat sorotan daftar berkedip saat next/prev.
        if self.list_frame.get(i) == teks and (self.list_frame.itemcget(i, "fg") or "") == warna:
            return
        terpilih = i in self.list_frame.curselection()
        self.list_frame.delete(i)
        self.list_frame.insert(i, teks)
        self._warnai_item_frame(i, self.label_path)
        if terpilih:
            self.list_frame.selection_set(i)

    def pilih_frame(self):
        s=self.list_frame.curselection()
        if not s:return
        if len(s) > 1:
            # Saat memblok, frame tidak dibuka satu per satu: menyeret di atas
            # 50 baris akan memuat 50 frame dan membuat aplikasi tersendat.
            self.status.set(f"{len(s)} frame dipilih. Tekan Sampahkan untuk membuang/memulihkan semuanya.")
            return
        self._buka_frame_ekspor(self.frame_paths[s[0]])

    def _buka_frame_ekspor(self, p: Path):
        # Selesaikan dulu autosave yang masih tertunda milik frame yang
        # DITINGGALKAN, sebelum label_path/poligon diganti ke frame baru.
        # Autosave ditunda 450 ms (jadwalkan_autosave); tanpa flush ini,
        # berpindah frame lebih cepat dari itu membuang edit yang belum
        # sempat tersimpan -- termasuk hapus blok -- sehingga frame lama
        # kembali ke versi sebelum diedit saat dibuka lagi, riwayat Undo-nya
        # ikut hilang (direset ke versi lama), dan bila draft itu belum
        # pernah tersimpan sama sekali, auto-segmentasi berjalan ulang
        # karena frame dianggap belum berlabel.
        if self._autosave_setelah is not None:
            self.after_cancel(self._autosave_setelah)
            self._autosave_setelah = None
            self.simpan_draft_label()
        data = self._baca_frame(p)
        if data is None: return
        self._simpan_riwayat_frame()
        bgr, dep = data["bgr"], data["dep"]
        # Pilihan A/S dipertahankan antar-frame. Penguncian hanya
        # diperlukan sebelum navigasi pertama, bukan setiap kali frame baru
        # dibuka; jika tidak tombol panah berhenti setelah satu perpindahan.
        self.label_path=p; self.label_info=baca_json(p/"frame.json")
        if not hasattr(self, "_frame_terakhir"):
            self._frame_terakhir = {}
        self._frame_terakhir[str(self.sesi)] = p        # dibuka lagi saat kembali ke rekaman ini
        self._usulan_aktif = None                     # riwayat undo frame lama tidak berlaku lagi
        try:
            self.kanvas.intrinsik = self._intrinsics(self.label_info)
        except (KeyError, TypeError, ValueError):
            self.kanvas.intrinsik = None               # frame tanpa intrinsik: lapisan arah tidak tersedia
        self.kanvas.set_frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),dep)
        self._perbarui_bayangan()
        draft = baca_json(p / "label_draft.json", {})
        if draft.get("poligon"):
            self.kanvas.poligon = {nama: [list(map(tuple, poly)) for poly in draft["poligon"].get(nama, [])]
                                   for nama in MODE_MASK}
            self.kanvas.aktif_indeks = {nama: (len(daftar) - 1 if daftar else None)
                                        for nama, daftar in self.kanvas.poligon.items()}
            self.kanvas.reset_riwayat()
            self.kanvas.fit()                           # sekali, sudah dengan mask (lihat set_frame)
            self.status.set(f"Draft mask {p.name} dipulihkan otomatis.")
        self._pulihkan_riwayat_frame(p)
        if not draft.get("poligon") and not self._punya_label(p):
            # Frame baru selalu mendapat usulan terlebih dahulu. Pengguna lalu
            # tinggal mengoreksi; draft/label yang sudah ada tidak ditimpa.
            self.after(180, lambda target=p: self._mulai_frame_baru(target))
        self.perbarui_lencana_periksa()
        self._perbarui_info_rekaman(p)
        if getattr(self, "panel_ukuran", None) is not None:
            try:
                self.panel_ukuran.frame_dibuka(p)
            except Exception as e:                               # noqa: BLE001 -- jangan ganggu pelabelan
                self.status.set(f"Nomor anak tangga tidak dapat ditampilkan: {e}")
        self.perbarui_konteks_label(self.label_info.get("kategori"))
        self.ukur_status.set(("Tekan A (merah) atau S (biru), lalu klik/tarik titik. Space berpindah frame."
                              if self.kontrol_label.get() == "mudah" else
                              "PgUp merah, PgDn biru, panah pindah frame."))
        self.after(40, self.kanvas.focus_set)

    # ------------------------------------------------ baca frame + prefetch
    _UKURAN_CACHE = 10

    def _baca_berkas_frame(self, p: Path) -> dict | None:
        bgr = cv2.imread(str(p / "color_raw.png"))
        if bgr is None:
            return None
        return {"bgr": bgr, "dep": np.load(p / "depth_aligned_to_color.npy"), "peta": None}

    def _simpan_cache(self, p: Path, data: dict) -> None:
        with self._kunci_cache:
            self._cache_frame[p] = data
            self._cache_frame.move_to_end(p)
            while len(self._cache_frame) > self._UKURAN_CACHE:
                self._cache_frame.popitem(last=False)

    def _baca_frame(self, p: Path) -> dict | None:
        """Frame dari cache prefetch bila ada, dari disk bila belum; lalu pesan tetangganya."""
        with self._kunci_cache:
            data = self._cache_frame.get(p)
        if data is None:
            data = self._baca_berkas_frame(p)
            if data is not None:
                self._simpan_cache(p, data)
        self._pesan_prefetch(p)
        return data

    def _pesan_prefetch(self, p: Path) -> None:
        try:
            i = self.frame_paths.index(p)
        except ValueError:
            return
        # Frame aktif dulu (hanya model geser rolling shutter untuk baris info),
        # lalu tetangga; maju lebih diutamakan: arah label yang biasa adalah Space/next.
        urut = [p] + [self.frame_paths[j] for j in (i + 1, i - 1, i + 2, i + 3)
                      if 0 <= j < len(self.frame_paths)]
        with self._syarat_prefetch:
            self._antrean_prefetch = urut
            self._syarat_prefetch.notify()

    def _ir_mentah(self, p: Path, data: dict | None) -> np.ndarray | None:
        """IR kiri selaras-RGB lewat depth, belum dikoreksi rolling shutter (disimpan di cache frame)."""
        if data is not None and "ir0" in data:
            return data["ir0"]
        try:
            ir = ir_selaras.selaraskan(p, data["dep"] if data else self.kanvas.depth, koreksi=False)
        except (OSError, KeyError, ValueError, cv2.error):
            ir = None
        if data is not None:
            data["ir0"] = ir
        return ir

    def _geser_rgb(self, p: Path, data: dict | None) -> tuple:
        """Model geser rolling shutter depth/IR -> RGB untuk frame p (ir_selaras.ukur_geser), di cache frame."""
        if data is not None and "geser" in data:
            return data["geser"]
        ir = self._ir_mentah(p, data)
        try:
            g = ir_selaras.geser_frame(p, ir=ir) if ir is not None else ir_selaras.NOL
        except (OSError, KeyError, ValueError, cv2.error):
            g = ir_selaras.NOL
        if data is not None:
            data["geser"] = g
        return g

    def _depth_rgb(self, p: Path, data: dict | None) -> np.ndarray:
        """Depth selaras yang sudah dikoreksi ke posisi RGB, untuk PENGUKURAN (tepi tanpa sumber = 0)."""
        if data is not None and "dep_rgb" in data:
            return data["dep_rgb"]
        dep = data["dep"] if data is not None else self.kanvas.depth
        hasil = ir_selaras.terapkan(dep, self._geser_rgb(p, data), terdekat=True)
        if data is not None:
            data["dep_rgb"] = hasil
        return hasil

    def _vis_frame(self, p: Path, data: dict | None, intrinsik: dict, mode: str) -> np.ndarray | None:
        """Tampilan bantu satu frame di posisi RGB: dihitung dari depth apa adanya lalu dikoreksi
        rolling shutter dengan tepi terisi (lihat visual_depth.hitung_dari_folder)."""
        dep = data["dep"] if data is not None else self.kanvas.depth
        return visual_depth.hitung_dari_folder(p, dep, intrinsik, mode, geser=self._geser_rgb(p, data),
                                               ir=self._ir_mentah(p, data))

    def _depth_vis_kini(self, mode: str) -> np.ndarray | None:
        """Tampilan bantu frame aktif dari cache prefetch; IR dihitung di sini bila belum ada."""
        with self._kunci_cache:
            data = self._cache_frame.get(self.label_path) if self.label_path else None
        vis = (data or {}).get("vis", {}).get(mode)
        # Dihitung sekali per frame lalu disimpan: menggeser slider warna tidak
        # boleh menghitung ulang bidang (~90 ms) di setiap geseran.
        if vis is None and self.label_path is not None and self.kanvas.intrinsik is not None:
            try:
                vis = self._vis_frame(self.label_path, data, self.kanvas.intrinsik, mode)
            except (OSError, KeyError, ValueError, cv2.error):
                vis = None
            if data is not None and vis is not None:
                data.setdefault("vis", {})[mode] = vis
        return vis

    def _peta_model_kini(self) -> np.ndarray | None:
        """Peta kelas model untuk frame aktif (penyedia pemandu kanvas)."""
        p = self.label_path
        if p is None or self.kanvas.rgb is None or not self._model_gpu_aktif:
            return None                              # kanvas memakai arah normal depth (CPU)
        if (self.label_info or {}).get("kategori", "tangga_naik") != "tangga_naik":
            return None
        with self._kunci_cache:
            data = self._cache_frame.get(p)
        if data is not None and data.get("peta") is not None:
            return data["peta"]
        peta = peta_kelas_convnext(cv2.cvtColor(self.kanvas.rgb, cv2.COLOR_RGB2BGR), self.kanvas.depth,
                                   self._intrinsics(self.label_info or {}))
        if data is not None:
            data["peta"] = peta
        return peta

    def _pekerja_prefetch(self) -> None:
        """Thread latar: baca frame tetangga, dan hitung peta model bila model siap."""
        while True:
            with self._syarat_prefetch:
                while not self._antrean_prefetch:
                    self._syarat_prefetch.wait()
                p = self._antrean_prefetch.pop(0)
            try:
                with self._kunci_cache:
                    data = self._cache_frame.get(p)
                if data is None:
                    data = self._baca_berkas_frame(p)
                    if data is None:
                        continue
                    self._simpan_cache(p, data)
                if p == self.label_path:
                    # Frame aktif: tampilan bantunya dihitung thread UI saat digambar;
                    # di sini cukup model geser, lalu baris info diperbarui lewat antrean.
                    self._geser_rgb(p, data)
                    self.q.put(("info_frame", p))
                    continue
                perlu = set()
                if self.kanvas.depth_alpha > 0:
                    perlu.add(self.kanvas.depth_mode)
                if self.kanvas.pakai_ir:
                    perlu.add("ir")
                if self.kanvas.warna_bidang > 0:
                    perlu.add("bidang")
                if self.kanvas.garis_3d:
                    perlu.add("garis_3d")
                vis = data.setdefault("vis", {})
                info = baca_json(p / "frame.json", {}) if perlu - set(vis) else {}
                for mode in perlu - set(vis):
                    if "intrinsics_rgb_native" in info:
                        v = self._vis_frame(p, data, self._intrinsics(info), mode)
                        if v is not None:
                            vis[mode] = v
                # Peta model hanya untuk lapisan Bantu arah; saat lapisan itu mati
                # menghitungnya membuang ~30 ms per tetangga dan membuat frame
                # berikutnya belum siap ketika next ditekan (thread UI lalu menghitung sendiri).
                if (data.get("peta") is None and self.kanvas.bantu_arah > 0 and self._model_gpu_aktif
                        and model_convnext_siap()):
                    info = baca_json(p / "frame.json", {})
                    if info.get("kategori", "tangga_naik") == "tangga_naik" and "intrinsics_rgb_native" in info:
                        data["peta"] = peta_kelas_convnext(data["bgr"], data["dep"], self._intrinsics(info))
            except Exception:                               # noqa: BLE001
                continue                                    # prefetch hanya percepatan; jalur utama tetap membaca sendiri

    # -------------------------------------------- bayangan & salin frame tetangga
    def _frame_tetangga(self, arah: int) -> Path | None:
        """Frame di sebelah frame aktif pada daftar (urutan waktu rekaman): -1 sebelumnya, +1 berikutnya."""
        if self.label_path is None or self.label_path not in self.frame_paths:
            return None
        j = self.frame_paths.index(self.label_path) + arah
        return self.frame_paths[j] if 0 <= j < len(self.frame_paths) else None

    def _data_frame(self, p: Path) -> dict | None:
        """Data frame lain dari cache/disk, tanpa memindahkan prefetch ke sekitarnya."""
        with self._kunci_cache:
            data = self._cache_frame.get(p)
        if data is None:
            data = self._baca_berkas_frame(p)
            if data is not None:
                self._simpan_cache(p, data)
        return data

    def _perbarui_bayangan(self) -> None:
        k = self.kanvas
        k.bayangan_rgb = k.bayangan_poligon = None
        k.bayangan_alpha = float(self.bayangan_alpha.get())
        p = self._frame_tetangga(self.frame_acuan.get()) if self.bayangan_aktif.get() else None
        data = self._data_frame(p) if p is not None else None
        if data is None:
            return
        rgb = cv2.cvtColor(data["bgr"], cv2.COLOR_BGR2RGB)
        if self.pakai_ir.get() and k.intrinsik is not None:
            try:
                ir = self._vis_frame(p, data, k.intrinsik, "ir")     # samakan dengan latar frame aktif
            except (OSError, KeyError, ValueError, cv2.error):
                ir = None
            if ir is not None:
                ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2RGB) if ir.ndim == 2 else ir.copy()
                kosong = ir.max(2) == 0
                ir[kosong] = rgb[kosong]
                rgb = ir
        k.bayangan_rgb = rgb
        if self.bayangan_label.get():
            k.bayangan_poligon = baca_json(p / "label_draft.json", {}).get("poligon") or None

    def ganti_bayangan(self) -> None:
        self._perbarui_bayangan()
        self.kanvas.render(); self.simpan_preferensi(); self._fokus_kanvas_label()
        if self.bayangan_aktif.get() and self.kanvas.bayangan_rgb is None and self.label_path is not None:
            self.status.set(f"Tidak ada frame {NAMA_ARAH[self.frame_acuan.get()]} di rekaman ini, jadi belum ada bayangan.")

    def ganti_kekuatan_bayangan(self) -> None:
        self.kanvas.bayangan_alpha = float(self.bayangan_alpha.get())
        self.kanvas.render(); self.simpan_preferensi()

    def _salin_dari_tetangga(self, arah: int) -> bool:
        """Mulai frame aktif dari label frame tetangga. Hasilnya berstatus usulan sampai disunting/diperiksa."""
        k, p = self.kanvas, self.label_path
        asal = self._frame_tetangga(arah)
        poligon = baca_json(asal / "label_draft.json", {}).get("poligon") if asal is not None else None
        if not poligon or not any(poligon.get(nama) for nama in MODE_MASK):
            self.status.set(f"Frame {NAMA_ARAH[arah]} belum punya label untuk disalin.")
            return False
        H = None
        if self.salin_selaras.get():
            data = self._data_frame(asal)
            try:
                H = salin_label.homografi(cv2.cvtColor(data["bgr"], cv2.COLOR_BGR2RGB), k.rgb) if data is not None else None
            except cv2.error:
                H = None
        h, w = k.rgb.shape[:2]
        hasil = salin_label.pindahkan({nama: poligon.get(nama, []) for nama in MODE_MASK}, H, w, h)
        # Jalur yang sama dengan usulan model (usulkan_segmentasi): salinan belum disunting
        # bukan label diperiksa, dan Ctrl+Z memulihkan berkas label sebelumnya persis.
        if self._autosave_setelah is not None:
            self.after_cancel(self._autosave_setelah)
            self.simpan_draft_label()
        self._usulan_aktif = {"frame": p, "sebelum": k._salin_poligon(), "berkas": self._potret_berkas_label(p)}
        k.poligon = hasil
        self._usulan_aktif["usulan"] = k._salin_poligon()
        k.aktif_indeks = {nama: (len(daftar) - 1 if daftar else None) for nama, daftar in k.poligon.items()}
        k.catat_riwayat()
        self._aktif_mask_berubah(self.mode_label.get(), k.aktif_indeks[self.mode_label.get()])
        k.render(); k.on_change()
        selaras = ("mengikuti gerak kamera" if H is not None else
                   "di posisi yang sama (gerak kamera tidak terbaca)" if self.salin_selaras.get() else "di posisi yang sama")
        self.status.set(f"Label {asal.name} disalin {selaras}: {len(hasil['objek'])} merah, {len(hasil['acuan'])} biru. "
                        "Silakan ubah; Ctrl+Z membatalkan.")
        self._tahan_status = True
        return True

    def salin_label_tetangga(self, arah: int):
        if self.label_path is None or self.kanvas.rgb is None:
            return "break"
        if self.status_periksa():
            self.status.set("Frame ini sudah ✔ diperiksa, jadi tidak ditimpa. Cabut dulu dengan Shift+Enter bila ingin menggantinya.")
            return "break"
        self._salin_dari_tetangga(arah)
        self.kanvas.focus_set()
        return "break"

    def _mulai_frame_baru(self, target: Path) -> None:
        """Isi awal frame tanpa label: dari frame acuan bila diminta, kalau tidak dari model."""
        if target != self.label_path or self._punya_label(target):
            return
        if not (self.salin_otomatis.get() and self._salin_dari_tetangga(self.frame_acuan.get())):
            self._auto_segmentasi(target)

    def _pantau_skala(self) -> None:
        """Skala layar bisa berubah saat Studio terbuka (monitor/skala tampilan); Tk tidak menata ulang sendiri."""
        baru = skala_ui.baca_faktor()
        if baru != self._skala_diberitahu:
            self._skala_diberitahu = baru
            if baru != skala_ui.FAKTOR:
                messagebox.showinfo("Skala layar berubah", "Skala layar berubah sejak Studio dibuka, jadi tulisan bisa tampak "
                                    "terlalu kecil atau besar.\n\nTutup lalu buka lagi Studio agar ukurannya kembali pas. "
                                    "Label sudah tersimpan otomatis.", parent=self)
        self.after(15000, self._pantau_skala)

    def _auto_segmentasi(self, target: Path):
        if target != self.label_path or self._punya_label(target) or not self._model_gpu_aktif:
            return
        self.status.set(f"Auto-segmentasi ConvNeXt RGB-D untuk {target.name}…")
        self.usulkan_segmentasi()

    def _perbarui_info_model_gpu(self) -> None:
        if not hasattr(self, "info_model_gpu"):
            return
        if self._model_gpu_aktif:
            teks = ("Nyala: frame tanpa label langsung diberi usulan, Bantu arah memakai peta model. "
                    + ("Model sudah di GPU." if model_convnext_siap() else "Model dimuat saat diperlukan."))
        else:
            teks = ("Mati: tidak ada auto-label saat membuka frame, Rekomendasi/Batch menunggu dinyalakan, "
                    "Bantu arah memakai depth (CPU). Model tidak di GPU.")
        self.info_model_gpu.configure(text=teks)

    def ganti_model_gpu(self) -> None:
        """Sakelar Auto-label & model GPU: mati melepas model dari GPU, nyala memuatnya lagi."""
        if not self.model_gpu.get() and getattr(self, "_batch_thread", None) and self._batch_thread.is_alive():
            self.model_gpu.set(True)
            messagebox.showinfo("Batch berjalan", "Tunggu auto-label batch selesai sebelum mematikan model GPU.",
                                parent=self)
            return
        self._model_gpu_aktif = bool(self.model_gpu.get())
        self.simpan_preferensi()
        self.kanvas._peta_arah_key = None            # pemandu dihitung ulang dari sumber yang berlaku
        self.kanvas.render()
        if self._model_gpu_aktif:
            self.status.set("Auto-label & model GPU dinyalakan; memuat model…")
            self._hangatkan_model_async()
            if self.label_path is not None and not self._punya_label(self.label_path):
                self.after(180, lambda target=self.label_path: self._auto_segmentasi(target))
        else:
            # Thread terpisah: lepas() menunggu inferensi prefetch yang sedang berjalan.
            threading.Thread(target=lambda: (lepas_convnext(), self.q.put(("model_dilepas", None))),
                             daemon=True).start()
        self._perbarui_info_model_gpu()

    def _pastikan_model_gpu(self) -> bool:
        """Untuk tombol yang memakai model: tawarkan menyalakan bila sakelar mati."""
        if self._model_gpu_aktif:
            return True
        if not messagebox.askyesno("Model GPU mati",
                                   "Auto-label & model GPU sedang dimatikan (tab Otomatis).\n\n"
                                   "Nyalakan sekarang untuk menjalankan model?", parent=self):
            return False
        self.model_gpu.set(True)
        self.ganti_model_gpu()
        return True

    def _hangatkan_model_async(self):
        def kerja():
            try:
                self.q.put(("model_siap", hangatkan_convnext()))
            except Exception as e:  # noqa: BLE001
                self.q.put(("model_gagal", str(e)))
        self._model_warm_thread = threading.Thread(target=kerja, daemon=True)
        self._model_warm_thread.start()

    def _rekomendasi_tangga_data(self, rgb, depth, info):
        """Versi tanpa Tk untuk worker batch; sama dengan saran tangga UI."""
        k = self._intrinsics(info)
        # Kanvas/SAM memakai RGB, sedangkan Ultralytics menerima ndarray
        # OpenCV BGR. Training juga membaca color_raw.png lewat OpenCV (BGR).
        # Jalur batch memakai pengusul yang sama dengan jalur UI; kalau
        # berbeda, label hasil batch dan label hasil klik tidak sebanding.
        # depth Z16 mentah; k membawa depth_scale agar model menerima meter.
        # Poligon langsung dari peta kelas model, tanpa SAM 2: pada 52 frame
        # terperiksa 211803, SAM 2 + verifikasi depth menurunkan Dice usulan
        # dari 0,887 ke 0,841 (bukti/final_d435/eksperimen_semua_data_20260925/auto_label).
        hasil = usulkan_convnext_depth(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth, k)
        self._sumber_batch = hasil.get("sumber", "ConvNeXt RGB-D")
        return {"acuan": hasil["tapakan"], "objek": hasil["bidang_tegak"]}

    def batch_auto_label(self):
        if not self.sesi:
            messagebox.showinfo("Pilih sesi", "Pilih sesi pada tab Tinjau dahulu.", parent=self); return
        if getattr(self, "_batch_thread", None) and self._batch_thread.is_alive():
            messagebox.showinfo("Batch berjalan", "Tunggu auto-label batch selesai.", parent=self); return
        if not self._pastikan_model_gpu():
            return
        if getattr(self, "_model_warm_thread", None) and self._model_warm_thread.is_alive():
            self.status.set("Model ConvNeXt sedang dimuat ke GPU; batch akan tersedia sesaat lagi."); return
        target = [p for p in self.daftar_frame_ekspor() if not self._punya_label(p)]
        if not target:
            messagebox.showinfo("Tidak ada frame baru", "Semua frame sudah memiliki draft atau label.", parent=self); return
        if not messagebox.askyesno("Batch auto-label", f"Buat draft + label otomatis untuk {len(target)} frame baru?\n\nLabel/draft manual tidak akan ditimpa.", parent=self): return
        sesi = self.sesi
        self._batch_thread = threading.Thread(target=self._batch_auto_worker, args=(sesi, target), daemon=True)
        self._batch_thread.start()
        self.status.set(f"Batch auto-label dimulai untuk {len(target)} frame…")

    def _batch_auto_worker(self, sesi: Path, target: list[Path]):
        jadi = gagal = 0
        for no, p in enumerate(target, 1):
            try:
                if self._punya_label(p):
                    continue
                info = baca_json(p / "frame.json", {})
                if info.get("kategori", sesi.parent.name) != "tangga_naik":
                    continue
                bgr = cv2.imread(str(p / "color_raw.png")); depth = np.load(p / "depth_aligned_to_color.npy")
                if bgr is None or "intrinsics_rgb_native" not in info:
                    raise ValueError("RGB/depth/intrinsics tidak lengkap")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                poligon = self._rekomendasi_tangga_data(rgb, depth, info)
                h, w = rgb.shape[:2]
                baris = []
                for mode, cls in (("acuan", 0), ("objek", 1)):
                    for poly in poligon[mode]:
                        if len(poly) >= 3:
                            baris.append(str(cls) + " " + " ".join(f"{v:.6f}" for x, y in poly for v in (x / w, y / h)))
                tulis_json(p / "label_draft.json", {"versi": 2, "otomatis": True,
                                                    "diperiksa_manual": False,
                                                    # Dulu tertulis "+ SAM2 + depth" walau SAM 2 sudah tidak dipakai.
                                                    "sumber": "batch " + getattr(self, "_sumber_batch", "ConvNeXt RGB-D"),
                                                       "disimpan_iso": datetime.now().isoformat(timespec="seconds"), "poligon": poligon})
                if baris:
                    (p / "label_yolo_seg.txt").write_text("\n".join(baris) + "\n", encoding="utf-8")
                jadi += 1
            except Exception as e:  # noqa: BLE001
                gagal += 1; self.q.put(("status", f"Batch melewati {p.name}: {e}"))
            if no == 1 or no % 3 == 0:
                self.q.put(("status", f"Batch auto-label {no}/{len(target)} — {jadi} selesai, {gagal} gagal"))
        self.q.put(("batch_auto_selesai", (jadi, gagal)))

    def pindah_frame_label(self, arah: int):
        # Menahan Space/D mengirim tekan-ulang lebih cepat dari frame dimuat; antrean itu dulu
        # terus menggeser frame setelah tombol dilepas. Tekan-ulang < 60 ms sesudah frame
        # terakhir selesai dibuka dibuang.
        if time.monotonic() - getattr(self, "_pindah_terakhir", 0.0) < 0.06:
            return
        try:
            self._pindah_frame_label(arah)
        finally:
            self._pindah_terakhir = time.monotonic()

    def _pindah_frame_label(self, arah: int):
        if not self.frame_paths:
            self.muat_frame()
        if not self.frame_paths:
            messagebox.showinfo("Belum ada frame", "Ekspor satu frame dari video atau pilih rentang untuk diekspor dahulu.", parent=self); return
        try:
            kini = self.frame_paths.index(self.label_path)
        except ValueError:
            kini = -1 if arah > 0 else 0
        tujuan = max(0, min(kini + arah, len(self.frame_paths) - 1))
        # Di ujung daftar tujuan sama dengan frame sekarang. Dulu frame itu dibuka
        # ulang dari awal (gambar dimuat lagi, kanvas dikosongkan lalu digambar
        # ulang), sehingga menahan Space di frame terakhir membuat layar berkedip.
        if tujuan == kini:
            self.status.set("Sudah frame terakhir." if arah > 0 else "Sudah frame pertama.")
            return
        self.list_frame.selection_clear(0, "end"); self.list_frame.selection_set(tujuan); self.list_frame.activate(tujuan)
        self._buka_frame_ekspor(self.frame_paths[tujuan])

    def _di_jendela_utama(self, event) -> bool:
        """Peristiwa berasal dari jendela utama, bukan dari jendela lain (pemutar ukur, dialog)."""
        try:
            return event.widget.winfo_toplevel() is self
        except (AttributeError, KeyError, tk.TclError):
            return False

    def _shortcut_label(self, event):
        """Shortcut label tetap berfungsi saat fokus berada di panel mana pun."""
        if self.tabs.select() != str(self.tab_label) or not self._di_jendela_utama(event):
            return None
        # Saat mengetik di kotak isian, huruf dan angka adalah isi, bukan pintasan.
        fokus = self.focus_get()
        if isinstance(fokus, (tk.Entry, tk.Spinbox, tk.Text, ttk.Entry, ttk.Combobox, ttk.Spinbox)):
            return None
        ctrl = bool(event.state & 0x4)
        shift = bool(event.state & 0x1)
        if ctrl and event.keysym.lower() == "z":
            # `shift` sudah dari event.state (bit Shift sungguhan), bukan dari
            # huruf keysym -- lihat KanvasLabel._ctrl_z untuk alasannya.
            return self.kanvas.redo_riwayat() if shift else self.kanvas.undo_riwayat()
        if shift and event.keysym in {"A", "S"}:
            return self.ubah_kelas_mask({"A": "objek", "S": "acuan"}[event.keysym])
        if shift and event.keysym in {"Prior", "Page_Up", "KP_Prior"}:
            return self.ubah_kelas_mask("objek")
        if shift and event.keysym in {"Next", "Page_Down", "KP_Next"}:
            return self.ubah_kelas_mask("acuan")
        if event.keysym in {"plus", "equal", "KP_Add"}:
            return self.zoom_tombol(1.25)
        if event.keysym in {"minus", "KP_Subtract"}:
            return self.zoom_tombol(1 / 1.25)
        if event.keysym in {"0", "KP_0"}:
            return self.muat_pas()
        if event.keysym == "Escape":
            # Esc pertama membatalkan garis penghalang yang belum jadi; Esc
            # berikutnya baru kembali ke alat titik.
            if self.kanvas.alat == "penghalang" and self.kanvas.batal_penghalang():
                self.status.set("Garis penghalang dibatalkan. Esc lagi kembali ke alat titik.")
                return "break"
            return self.pilih_alat("normal")
        if (not ctrl and event.keysym in ("Return", "KP_Enter")
                and self.kanvas.alat == "penghalang" and self.kanvas._penghalang):
            # Enter menutup garis penghalang, BUKAN menandai frame diperiksa.
            return self.kanvas.selesai_penghalang()
        if not ctrl and event.keysym in ("Return", "KP_Enter"):
            # Enter: label ini sudah dicek manusia -> tandai lalu lanjut frame
            # berikutnya. Shift+Enter mencabut tanda (kembali usulan otomatis).
            if shift:
                self.tandai_diperiksa(False)
                return "break"
            nama = self.label_path.name if self.label_path else None
            self.tandai_diperiksa(True, senyap=True)
            self.pindah_frame_keyboard(1, event)
            if nama:
                self.status.set(f"✔ {nama} ditandai diperiksa manual, lanjut ke frame berikutnya (Shift+Enter mencabut).")
            return "break"
        if not ctrl and event.keysym.lower() == "g":
            return self.pilih_alat("geser")
        if not ctrl and event.keysym.lower() == "x":
            return self.pilih_alat("blok")
        if not ctrl and event.keysym.lower() == "c":
            return self.pilih_alat("potong")
        if not ctrl and event.keysym.lower() == "v":
            return self.pilih_alat("penghalang")
        if not ctrl and event.keysym.lower() == "m":
            return self.gabung_mask()
        if not ctrl and event.keysym.lower() == "t":
            return self.tukar_di_kursor()
        if not ctrl and event.keysym.lower() == "b":
            self.bayangan_aktif.set(not self.bayangan_aktif.get())
            self.ganti_bayangan()
            return "break"
        if not ctrl and event.keysym.lower() in ("p", "n"):
            return self.salin_label_tetangga(-1 if event.keysym.lower() == "p" else 1)
        if not ctrl and event.keysym == "BackSpace" and self.kanvas.alat == "penghalang":
            self.kanvas.mundur_penghalang()
            return "break"
        if not ctrl and event.keysym in ("bracketleft", "bracketright"):
            return self._lebar_potong_tombol(1 / 1.2 if event.keysym == "bracketleft" else 1.2)
        if self.kontrol_label.get() == "mudah":
            if event.keysym.lower() == "a":
                return self.pilih_mode_keyboard("objek")
            if event.keysym.lower() == "s":
                return self.pilih_mode_keyboard("acuan")
            if event.keysym == "space":
                return self.pindah_frame_keyboard(1, event)
            if event.keysym.lower() == "d":
                return self.pindah_frame_keyboard(-1, event)
        else:
            if event.keysym in {"Prior", "Page_Up", "KP_Prior"}:
                return self.pilih_mode_keyboard("objek")
            if event.keysym in {"Next", "Page_Down", "KP_Next"}:
                return self.pilih_mode_keyboard("acuan")
            if event.keysym == "Left":
                return self.pindah_frame_keyboard(-1, event)
            if event.keysym == "Right":
                return self.pindah_frame_keyboard(1, event)
        return None

    def teks_kontrol_label(self):
        alat = (" • G = geser • X = blok hapus • C = potong penghalang • Esc = titik • +/− = zoom • 0 = muat"
                " • T = tukar merah/biru mask di bawah kursor • B = bayangan frame acuan"
                " • P/N = salin label dari frame sebelumnya/berikutnya")
        return (("A = merah • S = biru • Shift+A/S ubah kelas mask • "
                 "Space = berikutnya • D = sebelumnya • Ctrl+Z = Undo"
                 if self.kontrol_label.get() == "mudah" else
                 "PgUp = merah • PgDn = biru • Shift+PgUp/PgDn ubah kelas "
                 "mask • ←/→ = frame • Ctrl+Z = Undo") + alat)

    def ganti_kontrol_label(self):
        if hasattr(self, "label_bantuan_kontrol"):
            self.label_bantuan_kontrol.configure(text=self.teks_kontrol_label())
        self.simpan_preferensi()
        self.status.set("Kontrol label disimpan: " + ("Mudah (A/S/Space)." if self.kontrol_label.get() == "mudah" else "Klasik (PgUp/PgDn/panah)."))

    def pilih_mode_keyboard(self, nama: str):
        """Pilih warna mask secara sadar sebelum mengedit/navigasi."""
        self.mode_label.set(nama)
        self.kanvas.mode = nama
        self._mode_label_dipilih = True
        indeks = self.kanvas.aktif_indeks.get(nama)
        if indeks is not None:
            self.nomor_mask.set(indeks + 1)
        self.kanvas.focus_set()
        self.kanvas.render()
        warna = {"objek": "merah / bidang tegak", "acuan": "biru / tapakan"}[nama]
        self.status.set(f"Mode {warna} aktif. Titik warna lain tidak akan ikut terseleksi.")
        return "break"

    def pindah_frame_keyboard(self, arah: int, event=None):
        """Pindah frame dari papan ketik, tanpa menumpuk penekanan auto-repeat.

        Menahan Space mengirim ~30 penekanan per detik, sedangkan membuka satu
        frame lebih lambat dari itu. Penekanan menumpuk di antrean dan terus
        diproses setelah tombol dilepas, sehingga frame masih maju sendiri.
        Cap waktu event (jam server X, ms) dipetakan ke jam lokal; penekanan
        yang tiba selagi frame sebelumnya masih dimuat dibuang, bukan diantre.
        """
        t = getattr(event, "time", 0) if event is not None else 0
        if t:
            sekarang = time.perf_counter()
            if self._jam_x is None or t < self._jam_x[0]:
                self._jam_x = (t, sekarang)                 # sinkron pertama, atau jam X berputar ulang
            tiba = self._jam_x[1] + (t - self._jam_x[0]) / 1000.0
            if tiba > sekarang + 0.05:                      # jam bergeser: sinkron ulang
                self._jam_x, tiba = (t, sekarang), sekarang
            if tiba < self._navigasi_selesai:
                return "break"
        self.pindah_frame_label(arah)
        self._navigasi_selesai = time.perf_counter()
        return "break"

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
            self.kanvas.poligon = {nama: [] for nama in MODE_MASK}
            self.kanvas.catat_riwayat(); self.kanvas.render(); self.hitung_ukuran()
            self.status.set("Semua mask rekomendasi dibuang. Mask baru siap dibuat.")

    def jadwalkan_autosave(self):
        if not self.label_path or self.kanvas.rgb is None:
            return
        if self._autosave_setelah is not None:
            self.after_cancel(self._autosave_setelah)
        self._autosave_setelah = self.after(450, self.simpan_draft_label)

    # ------------------------------------------------------- periksa manual
    def status_periksa(self, frame=None) -> bool:
        """Apakah frame ini sudah ditandai diperiksa manusia."""
        f = frame or self.label_path
        if not f:
            return False
        return bool(baca_json(f / "label_draft.json", {}).get("diperiksa_manual", False))

    def tandai_diperiksa(self, nilai: bool = True, senyap: bool = False) -> None:
        """Tandai frame sudah diperiksa manusia dan cabut penanda otomatis.

        Penanda `otomatis` dan `diperiksa_manual` sengaja saling meniadakan.
        Selama keduanya bisa bernilai benar bersamaan, tidak ada cara membedakan
        label yang sudah diverifikasi manusia dari label yang masih usulan
        mesin -- dan pembedaan itu menentukan frame mana yang sah dipakai
        sebagai acuan pengujian.
        """
        if not self.label_path:
            if not senyap:
                messagebox.showinfo("Pilih frame", "Pilih frame lebih dahulu.", parent=self)
            return
        j = self.label_path / "label_draft.json"
        d = baca_json(j, {})
        baru = not d
        if baru:
            if not nilai:
                return
            # Frame tanpa draft, misalnya lantai atau bordes tanpa tangga yang
            # tidak mendapat usulan. Draft dibuat supaya penanda periksa punya
            # tempat dan frame dapat disimpan sebagai contoh latar.
            d = {"versi": 2, "disimpan_iso": datetime.now().isoformat(timespec="seconds"),
                 "poligon": self.kanvas.poligon}
        d["diperiksa_manual"] = bool(nilai)
        d["otomatis"] = not bool(nilai)
        d["diperiksa_iso"] = datetime.now().isoformat(timespec="seconds") if nilai else None
        tulis_json(j, d)
        # Frame tanpa poligon: mask kosong ditulis atau dicabut mengikuti penanda.
        # Frame berpoligon yang ditandai diperiksa: mask PNG dan label YOLO ditulis ulang
        # dari poligon. Dulu hanya penanda di label_draft.json yang berubah, sehingga label
        # usulan yang tersimpan tanpa mask PNG (mis. label Claude 5 Oktober) tetap tidak
        # punya mask dan diam-diam tidak ikut dilatih (15 frame terperiksa di rekaman ✔).
        if baru or not self._ada_poligon() or (nilai and self.kanvas.rgb is not None):
            self.simpan_label(senyap=True)
        self.perbarui_lencana_periksa()
        if not senyap:
            self.status.set("Frame ditandai SUDAH DIPERIKSA manual." if nilai
                            else "Penanda periksa manual dicabut; frame kembali berstatus usulan otomatis.")

    def _periksa_lalu_lanjut(self) -> None:
        """Sama dengan Enter di kanvas: tandai frame ini diperiksa lalu buka frame berikutnya."""
        if not self.label_path:
            return
        nama = self.label_path.name
        self.tandai_diperiksa(True, senyap=True)
        self.pindah_frame_label(1)
        self.status.set(f"\u2714 {nama} ditandai diperiksa manual, lanjut ke frame berikutnya.")

    def toggle_periksa(self) -> None:
        self.tandai_diperiksa(not self.status_periksa())

    def perbarui_lencana_periksa(self) -> None:
        lbl = getattr(self, "lencana_periksa", None)
        if lbl is None:
            return
        if not self.label_path:
            lbl.config(text="—  belum ada frame dipilih", bg=PANEL, fg=MUTED); return
        if self.status_periksa():
            if not self._ada_poligon():
                teks = "✔  DIPERIKSA • tanpa tangga (contoh latar)"
            else:
                teks = "✔  SUDAH DIPERIKSA MANUAL"
            lbl.config(text=teks, bg="#DCEFD8", fg="#1E5B2A")
        else:
            draf = baca_json(self.label_path / "label_draft.json", {})
            if draf.get("oleh_claude") and draf.get("perlu_dicek"):
                lbl.config(text="\U0001f916\u26a0  dilabel Claude, PERLU DICEK: " + str(draf.get("catatan_claude", ""))[:70],
                           bg="#FDE7E5", fg="#B42318")
            elif draf.get("oleh_claude") and not any(draf.get("poligon", {}).values()):
                lbl.config(text="\U0001f916\u2205  Claude: tidak ada tangga di frame ini, belum diperiksa", bg="#EEF1F5", fg="#475569")
            elif draf.get("oleh_claude"):
                lbl.config(text="\U0001f916  dilabel Claude, belum diperiksa", bg="#EFE7FD", fg="#5B21B6")
            else:
                lbl.config(text="⚠  masih usulan otomatis", bg="#FBEEDA", fg="#8A5A12")
        self._segarkan_item_frame()

    BERKAS_LABEL = ("label_draft.json", "label_yolo_seg.txt", "mask_objek.png", "mask_acuan.png")
    MAKS_RIWAYAT_FRAME = 10

    # Riwayat Ctrl+Z lintas frame. Dulu riwayat kanvas direset setiap frame
    # dibuka, jadi usulan auto-label atau suntingan di frame yang sudah
    # ditinggalkan tidak bisa dibatalkan lagi. Sekarang riwayat 10 frame
    # terakhir yang disunting disimpan di memori: kembali ke frame itu,
    # Ctrl+Z melanjutkan riwayatnya; Ctrl+Z di frame yang tidak punya langkah
    # lagi membuka frame yang terakhir disunting lalu membatalkan langkahnya.
    def _simpan_riwayat_frame(self) -> None:
        if self.label_path is None or not hasattr(self, "_riwayat_frame"):
            return
        k = self.kanvas
        kunci = str(self.label_path)
        self._riwayat_frame.pop(kunci, None)
        if len(getattr(k, "_riwayat", [])) <= 1:
            return
        self._riwayat_frame[kunci] = {"riwayat": list(k._riwayat), "pos": k._riwayat_pos,
                                      "usulan": getattr(self, "_usulan_aktif", None)}
        while len(self._riwayat_frame) > self.MAKS_RIWAYAT_FRAME:
            self._riwayat_frame.popitem(last=False)

    def _pulihkan_riwayat_frame(self, p: Path) -> None:
        r = getattr(self, "_riwayat_frame", {}).get(str(p))
        if r is not None and r["riwayat"][r["pos"]] != self.kanvas._salin_poligon():
            self._riwayat_frame.pop(str(p), None)          # label berubah di luar kanvas: riwayatnya basi
            r = None
        if r is None:
            # Potret saat dibuka: Ctrl+Z sampai keadaan ini memulihkan berkas label
            # persis (status ✔/otomatis/Claude), bukan menyimpannya sebagai suntingan.
            self._usulan_aktif = {"frame": p, "sebelum": self.kanvas._salin_poligon(),
                                  "berkas": self._potret_berkas_label(p), "usulan": None}
            return
        self.kanvas._riwayat, self.kanvas._riwayat_pos = list(r["riwayat"]), r["pos"]
        self._usulan_aktif = r["usulan"]

    def _undo_frame_lain(self) -> None:
        """Ctrl+Z tanpa langkah di frame ini: buka frame yang terakhir disunting dan batalkan langkahnya."""
        kini = str(self.label_path) if self.label_path else None
        posisi = {str(q): i for i, q in enumerate(self.frame_paths)}
        for kunci in reversed(list(getattr(self, "_riwayat_frame", {}))):
            r = self._riwayat_frame[kunci]
            if kunci == kini or r["pos"] <= 0 or kunci not in posisi:
                continue
            self._pilih_indeks_frame(posisi[kunci])
            if str(self.label_path) == kunci and self.kanvas._riwayat_pos > 0:
                self.kanvas.undo_riwayat()
                self.status.set(f"Ctrl+Z: kembali ke {Path(kunci).name} dan membatalkan langkah terakhirnya.")
            return
        self.status.set("Tidak ada lagi yang bisa dibatalkan (riwayat menyimpan 10 frame terakhir yang disunting).")

    def _potret_berkas_label(self, frame: Path | None) -> dict:
        """Isi berkas label sebuah frame (None = berkas belum ada), untuk dipulihkan persis."""
        if frame is None:
            return {}
        return {n: ((frame / n).read_bytes() if (frame / n).exists() else None) for n in self.BERKAS_LABEL}

    @staticmethod
    def _pulihkan_berkas_label(frame: Path, potret: dict) -> None:
        for nama, isi in potret.items():
            if isi is None:
                (frame / nama).unlink(missing_ok=True)     # dibuat oleh usulan yang dibatalkan
            else:
                (frame / nama).write_bytes(isi)

    def simpan_draft_label(self):
        self._autosave_setelah = None
        if not self.label_path or self.kanvas.rgb is None:
            return
        u = getattr(self, "_usulan_aktif", None)
        if u is not None and u["frame"] != self.label_path:
            u = None
        if u is not None:
            kini = self.kanvas._salin_poligon()
            # Hanya Ctrl+Z yang boleh memulihkan berkas sebelum usulan. Dulu
            # syaratnya cukup "poligon sama dengan sebelum usulan"; pada frame
            # baru keadaan itu = kosong, sehingga menghapus SEMUA mask usulan
            # (blok Ctrl+tarik / Hapus mask) dianggap membatalkan usulan:
            # berkas label dihapus, frame kembali "belum berlabel", dan saat
            # dibuka lagi auto-segmentasi memunculkan mask yang sama lagi.
            if kini == u["sebelum"] and getattr(self.kanvas, "dari_undo", False):
                # Ctrl+Z sampai sebelum usulan: berkas label kembali persis, termasuk statusnya.
                self._pulihkan_berkas_label(self.label_path, u["berkas"])
                self.perbarui_lencana_periksa()
                self.status.set(f"{'Suntingan' if u['usulan'] is None else 'Usulan'} dibatalkan; "
                                f"label sebelumnya dipulihkan: {self.label_path.name}")
                return
            if kini == u["usulan"]:
                # Usulan mesin yang belum disentuh bukan label diperiksa.
                tulis_json(self.label_path / "label_draft.json", {
                    "versi": 3, "otomatis": True, "diperiksa_manual": False, "diperiksa_iso": None,
                    "disimpan_iso": datetime.now().isoformat(timespec="seconds"), "poligon": self.kanvas.poligon,
                })
                self.perbarui_lencana_periksa()
                if self._ada_poligon():
                    self.simpan_label(senyap=True)
                if self._tahan_status:
                    self._tahan_status = False
                else:
                    self.status.set(f"Usulan disimpan sebagai belum diperiksa (Ctrl+Z membatalkan): {self.label_path.name}")
                return
        # Menyunting poligon berarti manusia sudah memeriksanya. Penanda
        # otomatis karena itu dicabut sendiri, bukan menunggu tombol ditekan:
        # kalau menunggu, sebagian besar frame akan tetap bertanda otomatis
        # padahal sudah dikoreksi, dan penandanya kehilangan makna.
        lama = baca_json(self.label_path / "label_draft.json", {})
        # Dibandingkan dalam bentuk JSON: berkas berisi list [x, y], kanvas berisi
        # tuple (x, y), dan [1, 2] != (1, 2). Dulu perbandingan langsung selalu
        # "berbeda", sehingga Ctrl+Z kembali ke label semula pun mencabut status
        # otomatis dan menandai frame "diperiksa manual" padahal tidak disentuh.
        kini_json = json.loads(json.dumps(self.kanvas.poligon))
        disunting = bool(lama.get("poligon")) and lama.get("poligon") != kini_json
        # Usulan yang diubah sebelum sempat tersimpan (autosave 450 ms) juga
        # suntingan manusia. Tanpa ini, menghapus semua mask usulan dengan cepat
        # menulis draft kosong bertanda "otomatis", yang _punya_label anggap
        # belum berlabel -- auto-segmentasi lalu berjalan ulang saat dibuka lagi.
        if u is not None and u.get("usulan") is not None and self.kanvas._salin_poligon() != u["usulan"]:
            disunting = True
        diperiksa = bool(lama.get("diperiksa_manual", False)) or disunting
        tulis_json(self.label_path / "label_draft.json", {
            "versi": 3,
            "otomatis": not diperiksa,
            "diperiksa_manual": diperiksa,
            "diperiksa_iso": (lama.get("diperiksa_iso")
                              or (datetime.now().isoformat(timespec="seconds") if disunting else None)),
            "disimpan_iso": datetime.now().isoformat(timespec="seconds"),
            "poligon": self.kanvas.poligon,
        })
        self.perbarui_lencana_periksa()
        # Frame diperiksa yang poligonnya dihapus semua ikut disinkronkan.
        # Tanpa ini mask lama yang berisi tertinggal dan tetap dipakai pelatihan.
        if diperiksa or self._ada_poligon():
            self.simpan_label(senyap=True)
        if self._tahan_status:
            self._tahan_status = False
        else:
            self.status.set(f"Draft mask otomatis disimpan: {self.label_path.name}")

    def _ada_poligon(self) -> bool:
        return any(len(poly) >= 3 for daftar in self.kanvas.poligon.values() for poly in daftar)

    def _lapor_blok(self, n_poly: int, n_titik: int) -> None:
        bagian = []
        if n_poly:
            bagian.append(f"{n_poly} mask")
        if n_titik:
            bagian.append(f"{n_titik} titik")
        self.status.set("Blok hapus: " + (" dan ".join(bagian) + " dibuang. Ctrl+Z membatalkan."
                                          if bagian else "tidak ada yang terkena."))

    def salin_nama_frame(self, penuh: bool = False) -> None:
        """Salin nama frame yang sedang dibuka ke papan klip.

        Ctrl+C menyalin namanya saja, Ctrl+Shift+C menyalin jalur lengkap.
        Nama frame cukup panjang untuk salah ketik ketika dilaporkan, dan
        jalur lengkap diperlukan saat frame itu hendak dibuka dari terminal.
        """
        if not self.label_path:
            self.status.set("Belum ada frame yang dibuka."); return
        teks = str(self.label_path) if penuh else self.label_path.name
        self.clipboard_clear(); self.clipboard_append(teks); self.update_idletasks()
        self.status.set(f"Disalin ke papan klip: {teks}")

    def _salin_teks(self, teks: str) -> str:
        self.clipboard_clear(); self.clipboard_append(teks); self.update_idletasks()
        self.status.set(f"Disalin ke papan klip: {teks}")
        return "break"

    def _salin_rekaman(self, penuh: bool = False) -> str:
        pilih = [self._map_sesi[i] for i in self.list_sesi.curselection() if i < len(self._map_sesi)]
        if not pilih:
            return "break"
        return self._salin_teks("\n".join(str(p) if penuh else p.name for p in pilih))

    def _salin_rekaman_kursor(self, e) -> str:
        i = self.list_sesi.nearest(e.y)
        if 0 <= i < len(self._map_sesi):
            return self._salin_teks(self._map_sesi[i].name)
        return "break"

    def _salin_frame_terpilih(self, penuh: bool = False) -> str:
        pilih = [self.frame_paths[i] for i in self.list_frame.curselection() if i < len(self.frame_paths)]
        if not pilih:
            return "break"
        return self._salin_teks("\n".join(str(p) if penuh else p.name for p in pilih))

    def _salin_dari_kursor(self, e) -> str:
        """Salin nama frame yang berada tepat di bawah kursor pada daftar (tanpa ikon status)."""
        i = self.list_frame.nearest(e.y)
        if i < 0 or i >= len(self.frame_paths):
            return "break"
        nama = self.frame_paths[i].name
        self.clipboard_clear(); self.clipboard_append(nama); self.update_idletasks()
        self.status.set(f"Disalin ke papan klip: {nama}")
        return "break"

    def _ganti_tampilan_sampah_frame(self):
        """Di tampilan sampah tombol yang sama memulihkan; labelnya ikut berganti agar tidak menyesatkan."""
        self.btn_sampah_frame.configure(text="♻ Pulihkan" if self.tampil_sampah_frame.get() else "🗑 Sampahkan")
        self.muat_frame()

    def toggle_sampah_frame(self):
        pilihan = [self.frame_paths[i] for i in self.list_frame.curselection() if i < len(self.frame_paths)]
        # Frame yang dipilih di daftar diutamakan: setelah tampilan diganti, label_path
        # masih bisa menunjuk frame dari tampilan sebelumnya.
        target = pilihan or ([self.label_path] if self.label_path else [])
        if not target:
            messagebox.showinfo("Pilih frame", "Pilih frame yang ingin dipindahkan atau dipulihkan.", parent=self)
            return
        ke_sampah = not self.tampil_sampah_frame.get()
        if len(target) > 1 and not messagebox.askyesno(
                "Sampah frame", f"{'Buang' if ke_sampah else 'Pulihkan'} {len(target)} frame terpilih"
                f"{' ke sampah' if ke_sampah else ' dari sampah'}?", parent=self):
            return
        # Edit yang masih menunggu autosave disimpan dulu; setelah label_path
        # dikosongkan di bawah, autosave itu tidak lagi tahu frame tujuannya.
        if self._autosave_setelah is not None:
            self.after_cancel(self._autosave_setelah)
            self._autosave_setelah = None
            self.simpan_draft_label()
        for f in target:
            state = baca_json(f / "frame_state.json", {"di_sampah": False})
            state["di_sampah"] = ke_sampah
            state["waktu_ubah_iso"] = datetime.now().isoformat(timespec="seconds")
            tulis_json(f / "frame_state.json", state)
        # Hanya frame INI yang ditandai - folder dan isinya tidak dipindah,
        # jadi frame lain, label, dan ekspor yang bisa dilanjutkan tidak
        # terganggu. Pemulihan tinggal membalik tanda lewat tampilan sampah.
        jumlah = f"{len(target)} frame" if len(target) > 1 else "Frame"
        pesan = (f"{jumlah} dibuang ke sampah, lanjut ke frame berikutnya. Centang 'Tampilkan sampah frame' untuk memulihkannya."
                 if state["di_sampah"] else
                 f"{jumlah} dipulihkan dari sampah, lanjut ke frame berikutnya.")
        # Indeks frame yang baru saja ditandai dicatat SEBELUM daftar dimuat
        # ulang. Setelah pemuatan ulang frame itu tidak lagi ada, sehingga
        # indeks yang sama menunjuk frame berikutnya.
        i = min((self.frame_paths.index(f) for f in target if f in self.frame_paths), default=0)
        self.label_path = None
        self.muat_frame(lanjut_ke=i)
        self.status.set(pesan)

    def hapus_frame_permanen(self):
        """Hapus hanya satu paket turunan ekspor; RAW sesi tetap tidak tersentuh."""
        p = self.label_path
        if not p:
            messagebox.showinfo("Pilih frame", "Pilih frame ekspor yang akan dihapus permanen.", parent=self); return
        if len(self.list_frame.curselection()) > 1:
            # Hapus permanen tidak dapat dibatalkan, jadi sengaja satu per satu;
            # untuk banyak frame pakai Sampahkan, yang bisa dipulihkan.
            messagebox.showinfo("Pilih satu frame", "Hapus permanen hanya untuk satu frame.\n"
                                "Untuk banyak frame gunakan Sampahkan (bisa dipulihkan).", parent=self); return
        exports = (self.sesi / "exports").resolve() if self.sesi else None
        if exports is None or exports not in p.resolve().parents:
            messagebox.showerror("Target tidak aman", "Frame bukan bagian dari exports sesi aktif.", parent=self); return
        if not messagebox.askyesno("Hapus frame", f"Hapus {p.name} dari daftar?\n\nFoldernya dipindah ke Tempat sampah Studio "
                                   "(kartu Rekaman, tab Tinjau) dan masih bisa dipulihkan sampai dikosongkan.", parent=self):
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
            TS.buang(self.root_data, p, "frame")
            self.label_path = None; self.label_info = None
            self.kanvas.rgb = self.kanvas.depth = None; self.kanvas.delete("all")
            self.muat_frame()
            tujuan = berikut if berikut in self.frame_paths else sebelum if sebelum in self.frame_paths else None
            if tujuan is not None:
                # Memakai pembantu yang sama dengan jalur sampah agar daftar
                # ikut tergulir ke frame terpilih, bukan hanya menyorotinya di
                # luar layar.
                self._pilih_indeks_frame(self.frame_paths.index(tujuan))
                self.status.set(f"{p.name} dipindah ke Tempat sampah Studio (bisa dipulihkan). Lanjut ke {tujuan.name}.")
            else:
                self.status.set(f"{p.name} dipindah ke Tempat sampah Studio (bisa dipulihkan).")
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
        if not self._pastikan_model_gpu():
            return
        info = self.label_info or {}
        if "intrinsics_rgb_native" not in info:
            messagebox.showinfo("Intrinsics tidak ada",
                                "frame.json frame ini tidak memuat intrinsics.", parent=self); return
        kategori = info.get("kategori", self.sesi.parent.name if self.sesi else "tangga_naik")
        try:
            # Bobot aktif hanya dilatih untuk kelas tangga. Batu/ramp tetap
            # mendapat proposal depth, bukan dipaksa ke model yang salah kelas.
            if kategori == "tangga_naik":
                # Tampilan kanvas RGB; YOLO harus menerima BGR agar sama dengan
                # loader training yang membaca color_raw.png melalui OpenCV.
                # ConvNeXt RGB-D menerima kedalaman sebagai KANAL MASUKAN,
                # berbeda dari pengusul lama yang bekerja dari warna lalu
                # memverifikasi dengan kedalaman sesudahnya. Lantai dan tapakan
                # sama-sama bidang mendatar bertekstur mirip; yang membedakannya
                # letak dalam ruang, dan itu hanya ada pada kedalaman.
                # Kanvas menyimpan depth Z16 mentah; intrinsics membawa
                # depth_scale agar model menerima kedalaman dalam meter.
                hasil = usulkan_convnext_depth(cv2.cvtColor(self.kanvas.rgb, cv2.COLOR_RGB2BGR),
                                               self.kanvas.depth, self._intrinsics(info))
                # Tanpa SAM 2; alasannya di _rekomendasi_tangga_data.
            else:
                hasil = usulkan_segmentasi(self.kanvas.depth, self._intrinsics(info))
                hasil["sumber"] = "depth (bobot YOLO tangga tidak dipakai untuk kategori ini)"
        except Exception as e:                                  # noqa: BLE001
            # Untuk tangga jangan diam-diam mengganti hasil YOLO dengan depth:
            # pengguna harus tahu bila bobot/runtime YOLO bermasalah.
            if kategori == "tangga_naik":
                messagebox.showerror("Model ConvNeXt tidak tersedia",
                                     f"Rekomendasi tangga memakai model RGB-D ConvNeXt.\n\n{e}", parent=self)
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
        # Simpanan tertunda dituntaskan dulu, lalu berkas label sebelum usulan dipotret:
        # Ctrl+Z sampai ke poligon sebelum usulan memulihkan berkas itu persis
        # (status ✔ diperiksa / 🤖 Claude / perlu dicek ikut kembali).
        if self._autosave_setelah is not None:
            self.after_cancel(self._autosave_setelah)
            self.simpan_draft_label()
        self._usulan_aktif = {"frame": self.label_path, "sebelum": self.kanvas._salin_poligon(),
                              "berkas": self._potret_berkas_label(self.label_path)}
        self.kanvas.poligon["acuan"] = [list(map(tuple, p)) for p in urut_bawah(tapakan)]
        self.kanvas.poligon["objek"] = [list(map(tuple, p)) for p in urut_bawah(hasil["bidang_tegak"])]
        self._usulan_aktif["usulan"] = self.kanvas._salin_poligon()
        self.kanvas.aktif_indeks = {nama: (len(daftar) - 1 if daftar else None)
                                    for nama, daftar in self.kanvas.poligon.items()}
        # Satu langkah riwayat, bukan reset: Ctrl+Z mengembalikan label sebelum usulan.
        self.kanvas.catat_riwayat()
        self._aktif_mask_berubah(self.mode_label.get(), self.kanvas.aktif_indeks[self.mode_label.get()])
        self.kanvas.render(); self.kanvas.on_change()
        urut = hasil.get("tapakan_terurut", [])
        tinggi = [t["tinggi_dari_terbawah_cm"] for t in urut if t["tinggi_dari_terbawah_cm"] is not None]
        beda = [round(b - a, 1) for a, b in zip(tinggi, tinggi[1:])] if len(tinggi) > 1 else []
        self.ukur_status.set(
            f"Rekomendasi {hasil.get('sumber', 'depth')}: {len(hasil['tapakan'])} tapakan (biru), {len(hasil['bidang_tegak'])} riser (merah). "
            + (f"Beda tinggi antar tapakan: {beda} cm." if beda else "")
            + " Periksa lalu koreksi sebelum menyimpan. Ctrl+Z membatalkan usulan ini.")

    def ganti_mode(self):
        self.kanvas.mode = self.mode_label.get()
        self._mode_label_dipilih = True
        self.kanvas.render()
    def ganti_depth(self): self.kanvas.depth_alpha=float(self.depth_alpha.get()); self.kanvas.render()
    def ganti_mode_depth(self, label: str):
        self.depth_mode.set(next((k for k, v in visual_depth.MODE.items() if v == label), "bidang"))
        self.kanvas.depth_mode = self.depth_mode.get()
        if self.kanvas.depth_alpha <= 0:                  # memilih tampilan depth berarti ingin melihatnya
            self.depth_alpha.set(0.6); self.kanvas.depth_alpha = 0.6
        self.kanvas.render(); self.simpan_preferensi()
        self.after_idle(self.kanvas.focus_set)
    def ganti_latar_ir(self):
        self.kanvas.pakai_ir = bool(self.pakai_ir.get())
        self.kanvas.render(); self.simpan_preferensi(); self._fokus_kanvas_label()
        if self.kanvas.pakai_ir and self.label_path and not (self.label_path / "ir_left_raw.png").exists():
            self.status.set("Frame ini tidak punya berkas inframerah; latar tetap RGB.")
    def ganti_garis_3d(self):
        self.kanvas.garis_3d = bool(self.garis_3d.get())
        self.kanvas.render(); self.simpan_preferensi(); self._fokus_kanvas_label()
        if self.kanvas.garis_3d:
            self.status.set("Garis lipatan dari BENTUK 3-D (depth): bayangan dan noda tidak memunculkan garis. "
                            "Kuning = ujung/bibir tread, biru = pangkal riser. Tidak muncul bila depth kosong/jauh.")

    def ganti_warna_bidang(self):
        self.kanvas.warna_bidang = float(self.warna_bidang.get())
        self.kanvas.render(); self.simpan_preferensi()
    def _fokus_kanvas_label(self, _e=None):
        if _e is not None and not self._di_jendela_utama(_e):
            return
        if self.tabs.select() == str(self.tab_label):
            self.after_idle(self.kanvas.focus_set)
    def ganti_kecerahan(self):
        self.kanvas.kecerahan = float(self.kecerahan.get())
        self.kanvas.render()
        self.simpan_preferensi()
    def ganti_ketajaman(self):
        self.kanvas.ketajaman = float(self.ketajaman.get())
        self.kanvas.render()
        self.simpan_preferensi()
    def ganti_bantu_arah(self):
        self.kanvas.bantu_arah = float(self.bantu_arah.get())
        self.kanvas.render()
        self.simpan_preferensi()
    def ganti_opasitas_mask(self): self.kanvas.mask_alpha=float(self.mask_alpha.get()); self.kanvas.render()
    def ganti_magnet_titik(self): self.kanvas.magnet_aktif=bool(self.magnet_titik.get())
    def rapikan_magnet_semua(self):
        if not any(self.kanvas.poligon.values()):
            messagebox.showinfo("Belum ada mask", "Buka frame yang memiliki mask terlebih dahulu.", parent=self)
            return
        if not messagebox.askyesno("Rapikan semua titik", "Satukan semua vertex yang jaraknya sangat dekat. Perubahan disimpan otomatis.", parent=self):
            return
        grup, titik = self.kanvas.rapikan_semua_magnet()
        self.status.set(f"Magnet merapikan {titik} titik dalam {grup} sambungan." if grup else
                        "Tidak ada titik yang cukup dekat untuk disatukan.")

    def _mask(self, nama):
        """Masker gabungan SEMUA instance kelas ini (untuk pengukuran)."""
        if self.kanvas.rgb is None: return None
        m = np.zeros(self.kanvas.rgb.shape[:2], np.uint8)
        for poly in self.kanvas.poligon[nama]:
            if len(poly) >= 3:
                isi_poligon(m, poly, 1)
        return m

    def simpan_label(self, senyap: bool = False):
        """Tulis label YOLO-seg: SATU BARIS PER INSTANCE, tiga kelas semantik.

        Poligon biru (acuan) ikut jadi label - pada dataset lama 'tapakan'
        memang kelas 0 yang dilatih, bukan sekadar alat bantu ukur.
        """
        if not self.label_path or self.kanvas.rgb is None:
            if not senyap:
                messagebox.showinfo("Pilih frame", "Pilih satu frame ekspor dahulu.", parent=self)
            return
        kategori = (self.label_info or {}).get("kategori", "tangga_naik")
        h, w = self.kanvas.rgb.shape[:2]
        baris, jumlah = baris_label_yolo(self.kanvas.poligon, kategori, w, h)
        if not baris:
            # Frame tanpa poligon. Mask kosong sah sebagai contoh latar (lantai
            # atau bordes tanpa tangga) hanya bila manusia sudah menandainya
            # diperiksa. Tanpa penanda itu tidak ada yang ditulis, supaya frame
            # yang belum dilabeli tidak diam-diam menjadi data latih latar:
            # pelatihan memakai setiap frame yang mask PNG-nya lengkap.
            if self.status_periksa():
                (self.label_path/"label_yolo_seg.txt").write_text("", encoding="utf-8")
                kosong = np.zeros((h, w), np.uint8)
                cv2.imwrite(str(self.label_path/"mask_objek.png"), kosong)
                cv2.imwrite(str(self.label_path/"mask_acuan.png"), kosong)
                if not senyap:
                    self.status.set(f"{self.label_path.name}: tanpa tangga, disimpan sebagai contoh latar (mask kosong).")
            else:
                self._cabut_mask_kosong()
                if not senyap:
                    messagebox.showwarning(
                        "Poligon belum cukup",
                        "Belum ada poligon dengan minimal tiga titik.\n\n"
                        "Bila frame ini memang tanpa tangga (misalnya hanya lantai atau bordes), "
                        "tandai frame sudah diperiksa manual agar tersimpan sebagai contoh latar.",
                        parent=self)
            return
        (self.label_path/"label_yolo_seg.txt").write_text("\n".join(baris) + "\n", encoding="utf-8")
        # PNG SELALU ditulis, termasuk ketika kelasnya kosong. Versi sebelumnya
        # melewatkan penulisan bila mask kosong, sehingga menghapus seluruh
        # poligon satu kelas meninggalkan PNG LAMA yang masih berisi. Layar
        # menunjukkan kosong, berkas menunjukkan isi, dan pelatihan memakai
        # berkas -- sehingga koreksi pemakainya hilang tanpa jejak. Terukur
        # sebelum perbaikan: 4 frame dengan IoU draft-lawan-PNG serendah 0,46.
        obj = self._mask("objek"); ref = self._mask("acuan")
        h, w = self.kanvas.rgb.shape[:2]
        kosong = np.zeros((h, w), np.uint8)
        cv2.imwrite(str(self.label_path/"mask_objek.png"),
                    (obj if obj is not None else kosong) * 255)
        cv2.imwrite(str(self.label_path/"mask_acuan.png"),
                    (ref if ref is not None else kosong) * 255)
        rinci = ", ".join(f"{k}={v}" for k, v in jumlah.items())
        if not senyap:
            self.status.set(f"Label disimpan untuk {self.label_path.name}: {len(baris)} instance ({rinci}).")

    def _cabut_mask_kosong(self) -> None:
        """Hapus mask kosong setelah penanda periksa frame tanpa tangga dicabut.

        Hanya dijalankan bila label YOLO kosong dan setiap mask PNG seluruhnya
        nol, sehingga label berisi tidak pernah hilang lewat jalur ini.
        """
        p = self.label_path
        label = p / "label_yolo_seg.txt"
        masker = [p / "mask_objek.png", p / "mask_acuan.png"]
        # mask_lantai.png peninggalan kelas lantai yang sudah ditarik: ikut
        # dibuang bila ada, tetapi isinya tidak lagi menahan pembersihan.
        lama_lantai = p / "mask_lantai.png"
        if label.exists() and label.read_text(encoding="utf-8").strip():
            return
        for f in masker:
            if f.exists():
                m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if m is None or m.any():
                    return
        for f in masker + [label, lama_lantai]:
            f.unlink(missing_ok=True)

    def bangun_yolo(self):
        """Buat turunan siap Ultralytics tanpa memindahkan/menghapus raw frame.

        Satu contoh tangga wajib memuat tapakan DAN bidang tegak. Contoh yang
        hanya punya satu sisi tidak dibuang dari Studio/RAW, hanya tidak ikut
        menjadi data latih karena terbukti membuat model menganggap lantai atau
        dinding sebagai tapakan.
        """
        if not self.sesi:
            messagebox.showinfo("Pilih sesi", "Pilih sesi pada tab Tinjau dahulu.", parent=self)
            return
        if not CR.baca(self.sesi).get("selesai"):
            messagebox.showinfo("Rekaman belum ✔", "Rekaman ini belum ditandai ✔ (sudah benar semua) di tab Tinjau, "
                                "jadi tidak dipakai untuk melatih. Tandai dulu bila labelnya sudah benar.", parent=self)
            return
        meta_sesi = baca_json(self.sesi / "source" / "session.json")
        split = meta_sesi.get("split", "train")
        root = self.root_data / "dataset_yolo_seg"
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        peta = KELAS_MODE.get(meta_sesi.get("kategori", "tangga_naik"), KELAS_MODE["tangga_naik"])
        wajib = {KELAS_YOLO[peta["acuan"]], KELAS_YOLO[peta["objek"]]}
        count = 0; dikeluarkan_tidak_lengkap = 0
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
            # Label disusun ulang dari poligon label_draft.json (yang terlihat di
            # kanvas), bukan disalin: 15 berkas lama masih memuat baris kelas 4
            # "lantai" yang sudah ditarik dan tidak ada di dataset_yolo_seg.yaml.
            draft = baca_json(frame / "label_draft.json", {})
            img = cv2.imread(str(frame / "color_raw.png"), cv2.IMREAD_GRAYSCALE)
            if draft.get("poligon") and img is not None:
                isi, _ = baris_label_yolo(draft["poligon"], info_frame.get("kategori", "tangga_naik"),
                                          img.shape[1], img.shape[0])
            else:
                try:
                    isi = [b for b in label.read_text(encoding="utf-8").splitlines()
                           if b.strip() and int(b.split(maxsplit=1)[0]) in KELAS_YOLO.values()]
                except (OSError, ValueError):
                    dikeluarkan_tidak_lengkap += 1
                    continue
            kelas = {int(b.split(maxsplit=1)[0]) for b in isi}
            if not wajib.issubset(kelas):
                dikeluarkan_tidak_lengkap += 1
                continue
            stem = f"{self.sesi.name}_{frame.parent.parent.name}_{frame.name}"
            shutil.copy2(frame / "color_raw.png", root / "images" / split / f"{stem}.png")
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(isi) + "\n", encoding="utf-8")
            count += 1
        tulis_json(root / "dataset_yolo_seg.yaml", {"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test", "names": {str(v): k for k, v in sorted(KELAS_YOLO.items(), key=lambda x: x[1])}})
        self.status.set(f"Dataset YOLO diperbarui: {count} frame lengkap masuk split {split}; "
                        f"{dikeluarkan_tidak_lengkap} label tidak lengkap tidak dipakai. Raw tetap ada di sesi.")

    def hitung_ukuran(self):
        """Tinggi tiap riser terhadap tread tepat di bawahnya (per anak tangga).

        Dulu SEMUA tread digabung menjadi satu bidang; pada tangga, tread berbeda
        ketinggian sehingga bidang gabungan tidak pernah datar dan pengukuran
        hampir selalu gagal. Kini tiap riser dipasangkan dengan tread yang
        menempel di bawahnya. Riser yang tingginya menyimpang > 2 cm dari median
        anak tangga lain ditandai: biasanya tanda label atau depth bermasalah.
        """
        self.jadwalkan_autosave()
        if not self.label_path or not self.label_info:return
        if "intrinsics_rgb_native" not in self.label_info:
            self.ukur_status.set("frame.json frame ini tidak memuat intrinsics kamera. "
                                 "Ekspor ulang frame ini agar metadata kamera ikut tertulis.")
            return
        bentuk = self.kanvas.rgb.shape[:2] if self.kanvas.rgb is not None else None
        if bentuk is None:
            return
        def masker(poly):
            m = np.zeros(bentuk, np.uint8)
            isi_poligon(m, poly, 1)
            return m
        riser = [masker(p) for p in self.kanvas.poligon["objek"] if len(p) >= 3]
        tread = [masker(p) for p in self.kanvas.poligon["acuan"] if len(p) >= 3]
        if not riser or not tread:
            return
        try:
            # Depth digeser ke posisi RGB (tempat label digambar) sebelum RANSAC.
            with self._kunci_cache:
                data_ukur = self._cache_frame.get(self.label_path)
            depth = self._depth_rgb(self.label_path, data_ukur)
            k = self._intrinsics(self.label_info)
            hasil_semua = []
            for i, mr in enumerate(riser, 1):
                # Tread di bawah riser: yang paling banyak beririsan dengan pita
                # 12 px tepat di bawah tepi bawah riser.
                turun = np.zeros_like(mr); turun[12:] = mr[:-12]
                pita = (turun > 0) & (mr == 0)
                irisan = [int((pita & (mt > 0)).sum()) for mt in tread]
                j = int(np.argmax(irisan))
                if irisan[j] > 0:
                    h = ukur(mr, tread[j], depth, k)
                    hasil_semua.append((i, h, None) if h.get("ok") else (i, None, h.get("alasan", "depth kurang")))
                    continue
                # Riser terbawah berdiri di lantai (bukan tread): pita lantai 40 px
                # tepat di bawahnya, tanpa area berlabel, menjadi bidang acuan
                # (sama dengan pelacak). Diterima hanya bila tingginya wajar.
                semua_label = np.zeros_like(mr)
                for m_ in riser + tread:
                    semua_label |= m_
                lantai = np.zeros_like(mr)
                for dy_ in range(1, 41):
                    lantai[dy_:] |= mr[:-dy_]
                lantai = (lantai > 0) & (semua_label == 0)
                h = ukur(mr, lantai.astype(np.uint8), depth, k) if lantai.sum() > 500 else {"ok": False}
                if h.get("ok") and 8 <= h["tinggi_cm"] <= 30:
                    h["acuan"] = "lantai"
                    hasil_semua.append((i, h, None))
                else:
                    hasil_semua.append((i, None, "tidak ada tread di bawahnya, lantai tak terukur"))
            tinggi = [h["tinggi_cm"] for _, h, _ in hasil_semua if h]
            median = float(np.median(tinggi)) if tinggi else None
            baris = []
            for i, h, alasan in hasil_semua:
                if h is None:
                    baris.append(f"Riser {i}: gagal ({alasan})")
                    continue
                tanda = "  ⚠ menyimpang" if median is not None and len(tinggi) >= 3 and abs(h["tinggi_cm"] - median) > 2 else ""
                acuan = ", acuan lantai" if h.get("acuan") == "lantai" else ""
                baris.append(f"Riser {i}: {h['tinggi_cm']:.1f} cm (jarak {h['jarak_median_objek_m']:.1f} m{acuan}){tanda}")
            if median is not None:
                baris.append(f"Median tinggi riser: {median:.1f} cm dari {len(tinggi)} anak tangga")
            self.ukur_status.set("\n".join(baris))
            tulis_json(self.label_path / "hasil_ukur_depth.json",
                       {"per_riser": [{"riser": i, **(h or {"ok": False, "alasan": alasan})} for i, h, alasan in hasil_semua],
                        "median_tinggi_cm": median})
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
                elif k=="progres_rekaman":
                    if v[0] == self.sesi:
                        self.info_progres.config(text=v[1])
                elif k=="model_siap":
                    if not self._model_gpu_aktif:          # dimatikan selagi pemanasan berjalan
                        threading.Thread(target=lepas_convnext, daemon=True).start()
                    else:
                        self.status.set(f"Pengusul label siap di GPU: {v}")
                        # Handle cuDNN milik thread; panaskan juga thread UI sekarang
                        # agar auto-label frame pertama tidak membeku ~0,35 s.
                        self.after(100, lambda: self._model_gpu_aktif and panaskan_utas_ini())
                    self._perbarui_info_model_gpu()
                elif k=="model_dilepas":
                    self._perbarui_info_model_gpu()
                    self.status.set("Model GPU dilepas. Auto-label, Rekomendasi, dan peta model Bantu arah mati; "
                                    "Bantu arah memakai arah normal depth (CPU).")
                elif k=="model_gagal":
                    self.status.set(f"Pengusul label belum siap: {v}")
                elif k=="info_frame":
                    if v == self.label_path:
                        self._perbarui_info_rekaman(v)
                elif k=="sinkron_selesai":
                    sesi_s, r = v
                    self._sinkron_jalan = None
                    if "galat" in r:
                        self.status.set(f"Cek sinkron {sesi_s.name} gagal: {r['galat']}")
                    else:
                        self.status.set(f"{sesi_s.name}: {sinkron.teks_ringkas(r)}")
                    if sesi_s == self.sesi:
                        self._segarkan_info_tinjau()
                    if self.label_path is not None and self.label_path.parent.parent.parent == sesi_s:
                        self._perbarui_info_rekaman(self.label_path)
                elif k=="batch_auto_selesai":
                    jadi, gagal = v; self.muat_frame()
                    self.status.set(f"Batch auto-label selesai: {jadi} frame dibuat, {gagal} gagal. Periksa dan rapikan hasilnya.")
                elif k=="error":
                    self.preview_diminta = False
                    self.btn_preview_kamera.configure(text="Aktifkan preview kamera", state="normal")
                    self.status.set(str(v)); messagebox.showerror("RGB-D Labelling Studio",str(v),parent=self)
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
        sibuk = [nama for nama, atr in (("ekspor frame", "_ekspor_thread"), ("batch auto-label", "_batch_thread"))
                 if getattr(self, atr, None) is not None and self._ekspor_aktif(atr)]
        if sibuk and not messagebox.askyesno("Proses masih berjalan", f"{', '.join(sibuk)} masih berjalan dan akan "
                                             "terhenti di tengah. Tetap tutup?", parent=self):
            return
        # Simpan semua yang tertunda dulu, karena proses diakhiri paksa di bawah.
        try:
            if self._autosave_setelah is not None:
                self.after_cancel(self._autosave_setelah); self._autosave_setelah = None
                self.simpan_draft_label()
            self._simpan_catatan_sekarang()
        except Exception:                                       # noqa: BLE001
            pass
        self.simpan_preferensi()
        for langkah in (self._tutup_video, self._hentikan_render, self.cam.hentikan):
            try:
                langkah()
            except Exception:                                   # noqa: BLE001
                pass
        self.destroy()
        # Thread latar yang tersangkut di librealsense (kode C) tidak dapat
        # dihentikan dan dulu membuat aplikasi tidak bisa ditutup. Data sudah
        # tersimpan di atas, jadi proses diakhiri langsung.
        if getattr(self, "_buka_ulang", False):
            # Ganti tema: Studio baru dengan perintah yang sama; jeda agar kamera
            # sempat dilepas proses ini dulu.
            import subprocess
            subprocess.Popen(["/bin/sh", "-c", 'sleep 1.5; exec "$@"', "studio", *sys.orig_argv],
                             cwd=os.getcwd(), start_new_session=True)
        os._exit(0)

    def _ekspor_aktif(self, atr: str) -> bool:
        t = getattr(self, atr, None)
        return bool(t is not None and t.is_alive())


def main():
    ap=argparse.ArgumentParser(description="Studio rekaman RGB-D D435 non-destruktif untuk dataset YOLO segmentation")
    ap.add_argument("--keluar", default=str(Path(__file__).resolve().parents[2] / "dataset" / "studio_rgbd"), help="folder utama dataset studio")
    ap.add_argument("--lebar",type=int,default=848);ap.add_argument("--tinggi",type=int,default=480);ap.add_argument("--fps",type=int,default=30)
    # Daftar pilihan diambil dari lapisan kamera. Dulu daftar di sini punya
    # nama sendiri ("high_accuracy"/"high_density") yang tidak dikenal
    # terapkan_preset(), sehingga keduanya diam-diam memasang preset Default.
    ap.add_argument("--preset",default="jangan",choices=PRESET_PILIHAN);ap.add_argument("--batas-frame",type=int,default=8000)
    ap.add_argument("--exposure",type=int,default=0,help="exposure RGB dalam µs (0=auto). Rekomendasi 4000-8000 untuk anti-blur saat berjalan.")
    # 20 Hz terukur nol lonjakan event-loop >33 ms pada mesin ini. Naikkan bila
    # mesin Anda lebih kuat, turunkan di Jetson kalau preview mulai tersendat.
    ap.add_argument("--fps-preview",type=int,default=20,help="laju render preview (Hz)")
    Studio(ap.parse_args()).mainloop()


if __name__=="__main__":main()
