"""Citra inframerah D435 yang diselaraskan ke koordinat RGB, untuk melabel frame gelap.

Pada frame yang gelap total, RGB hanya berisi beberapa tingkat kecerahan
(211803 frame_002493: rerata 5 dari 255), sehingga detail tidak dapat
dipulihkan dengan pencerahan apa pun. Kamera IR kiri D435 tetap melihat
(rerata ~100) karena diterangi proyektor laser, tetapi (1) tertutup pola titik
proyektor dan (2) berada di posisi kamera yang berbeda dari RGB.

(1) Pola titik (bintik terang 2-4 px) dideteksi lewat top-hat dan hanya
    piksel titik yang ditambal (lihat bersihkan_titik); piksel lain tetap asli.
(2) Setiap piksel RGB dipetakan ke IR lewat depth selaras-RGB dan kalibrasi:
    X_ir = R (X_rgb - t), dengan R dan t dari ``extrinsics_depth_ke_rgb``
    frame.json apa adanya. Konvensi ini diperiksa terhadap ``depth_raw`` (yang
    berada di koordinat IR): 96% titik cocok dalam 5 mm, median 1,5-1,7 mm;
    konvensi transpos hanya 32-40%.
Piksel RGB tanpa depth tidak punya padanan IR dan diisi dari tetangganya bila
lubangnya kecil, selain itu hitam.
(3) Sisa geser akibat rolling shutter RGB saat kamera bergerak dikoreksi per
    frame (geser_ke_rgb), agar tepi di IR jatuh di tempat yang sama dengan RGB.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def bersihkan_titik(ir: np.ndarray, ukuran: int = 7) -> np.ndarray:
    """Buang pola titik proyektor tanpa mengaburkan piksel lain.

    Titik laser menutupi 41-47% piksel IR (211803, 105840). Opening abu-abu
    yang dulu dipakai mengubah SEMUA piksel sehingga detail ikut kabur dan
    tersisa tekstur belang. Kini titik dideteksi satu per satu lewat top-hat
    (terang lokal yang lebih kecil dari elemen ``ukuran``), hanya piksel titik
    yang ditambal dari tetangganya (inpainting), lalu derau halus diredam
    bilateral yang menjaga tepi.
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran, ukuran))
    tophat = cv2.subtract(ir, cv2.morphologyEx(ir, cv2.MORPH_OPEN, ker))
    ambang = max(12.0, float(np.percentile(tophat, 75)) * 1.5)
    titik = cv2.dilate((tophat > ambang).astype(np.uint8), np.ones((3, 3), np.uint8))
    tambal = cv2.inpaint(ir, titik, 3, cv2.INPAINT_TELEA)
    return cv2.bilateralFilter(tambal, 5, 18, 3)


def selaraskan(folder: Path, depth_selaras: np.ndarray | None = None, koreksi: bool = True) -> np.ndarray | None:
    """IR kiri (tanpa pola titik) dalam koordinat citra RGB, uint8 abu-abu; None bila tidak ada IR."""
    folder = Path(folder)
    f_ir = folder / "ir_left_raw.png"
    if not f_ir.exists():
        return None
    ir = cv2.imread(str(f_ir), cv2.IMREAD_UNCHANGED)
    if ir is None:
        return None
    if ir.dtype != np.uint8:
        ir = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    info = json.loads((folder / "frame.json").read_text())
    kc, kd = info["intrinsics_rgb_native"], info["intrinsics_depth_native"]
    ek = info["extrinsics_depth_ke_rgb"]
    R = np.array(ek["rotation_row_major"], np.float32).reshape(3, 3)
    t = np.array(ek["translation_meter"], np.float32)
    if depth_selaras is None:
        depth_selaras = np.load(folder / "depth_aligned_to_color.npy")
    z = depth_selaras.astype(np.float32) * float(info.get("depth_scale", 0.001))
    bersih = bersihkan_titik(ir)
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    Xc = np.dstack([(u - kc["ppx"]) * z / kc["fx"], (v - kc["ppy"]) * z / kc["fy"], z])
    Xd = (Xc - t) @ R.T                                      # = R (X_rgb - t) per piksel
    zd = np.maximum(Xd[..., 2], 1e-6)
    ud = (Xd[..., 0] / zd * kd["fx"] + kd["ppx"]).astype(np.float32)
    vd = (Xd[..., 1] / zd * kd["fy"] + kd["ppy"]).astype(np.float32)
    out = cv2.remap(bersih, ud, vd, cv2.INTER_LINEAR, borderValue=0)
    lubang = (z <= 0).astype(np.uint8)
    if lubang.any():
        # Lubang depth kecil diisi dari tetangga; area tanpa depth yang luas tetap hitam.
        kecil = lubang & (cv2.erode(lubang, np.ones((9, 9), np.uint8)) == 0)
        out[lubang > 0] = 0
        out = cv2.inpaint(out, kecil.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    out = cv2.createCLAHE(2.0, (8, 8)).apply(out)
    if koreksi:
        rgb = cv2.imread(str(folder / "color_raw.png"), cv2.IMREAD_GRAYSCALE)
        if rgb is not None:
            dx, dy = geser_ke_rgb(out, rgb)
            if dx or dy:
                out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (out.shape[1], out.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderValue=0)
    return out


def _gradien(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (m.mean() + 1e-6)


def geser_ke_rgb(ir: np.ndarray, rgb_abu: np.ndarray, ry: int = 10, rx: int = 4) -> tuple[int, int]:
    """Geser (dx, dy) piksel yang membuat IR selaras paling cocok dengan RGB.

    Sensor RGB D435 membaca baris demi baris (rolling shutter) sedangkan IR
    memotret sekaligus; saat kamera naik-turun waktu berjalan, tepi anak tangga
    di IR bisa bergeser beberapa piksel dari RGB (diukur: 12% frame >= 3 px
    vertikal, maks. 8 px, selalu ke atas). Pergeseran dicari dengan mencocokkan
    peta gradien (matchTemplate, setengah resolusi, +-2*ry baris dan +-2*rx kolom)
    dan hanya dipakai bila RGB cukup terang dan kecocokannya jelas; pada frame
    gelap RGB tidak punya tepi sehingga IR dibiarkan apa adanya.
    """
    if float(rgb_abu.mean()) < 35:
        return 0, 0
    a = _gradien(cv2.resize(rgb_abu, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    b = _gradien(cv2.resize(ir, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    pola = b[ry:-ry, rx:-rx]
    hasil = cv2.matchTemplate(a, pola, cv2.TM_CCOEFF_NORMED)
    _, maks, _, (x, y) = cv2.minMaxLoc(hasil)
    nol = float(hasil[ry, rx])
    if maks < 0.3 or maks - nol < 0.02:
        return 0, 0
    return 2 * (x - rx), 2 * (y - ry)
