"""Tampilan depth untuk melabel: menampakkan BENTUK anak tangga, bukan hanya jarak.

Warna jarak (turbo) membuat tangga tampak sebagai gradasi halus, karena jarak
pada tangga naik bertambah pelan dari bawah ke atas. Yang membedakan tread
dan riser adalah ARAH permukaannya (tread menghadap atas, riser menghadap
kamera), dan itu terbaca dari depth tanpa bergantung pada cahaya.

Mode:
- ``bidang``: tinggi tiap piksel terhadap arah "atas" (dari normal bidang
  datar yang dominan). Bidang datar (tread, lantai, bordes) diwarnai menurut
  tingginya dengan warna berulang tiap 0,5 m, sehingga tiap anak tangga
  menjadi satu blok warna rata; permukaan tegak (riser) abu-abu berbayang;
  batas tajam antar-bidang digaris hitam.
- ``relief``: permukaan disinari cahaya buatan dari atas-depan (normal 3-D
  dari depth + intrinsik). Tread terang, riser lebih gelap, tepi anak tangga
  tegas, seperti foto yang diterangi dari atas.
- ``normal``: peta normal berwarna (tiap arah permukaan satu warna).
- ``kontras``: depth dikurangi versi halusnya; tonjolan dan tepi anak tangga
  terlihat sebagai garis terang/gelap.
- ``jarak``: warna jarak (turbo), tampilan lama.
Piksel tanpa depth (lubang, terlalu dekat/jauh) digambar hitam.
"""
from __future__ import annotations

import cv2
import numpy as np

MODE = {"bidang": "Bidang (warna per ketinggian anak tangga)", "relief": "Relief 3-D (bentuk anak tangga)",
        "normal": "Arah permukaan (warna normal)", "jarak": "Jarak (warna turbo)",
        "ir": "Inframerah selaras (frame gelap)",
        "ir_bidang": "Inframerah + warna bidang (frame gelap)"}
BUTUH_BERKAS = ("ir", "ir_bidang")        # butuh berkas IR frame, dihitung Studio (hitung_dari_folder)
# "ir" tidak dihitung di sini karena butuh berkas IR frame; lihat ir_selaras.py.
# "kontras" tetap tersedia untuk eksperimen, tetapi tidak ditawarkan di Studio:
# pada frame 211803/105840 hasilnya berderau dan tepinya kurang terbaca.
CAHAYA = np.array([0.25, -0.9, -0.35], np.float32)       # kamera: x kanan, y bawah, z maju -> dari atas-depan
CAHAYA /= np.linalg.norm(CAHAYA)


def _meter_halus(depth: np.ndarray, skala: float) -> tuple[np.ndarray, np.ndarray]:
    d = depth.astype(np.float32) * skala
    sah = (d > 0.15) & (d < 8.0)
    if not sah.any():
        return d, sah
    # Lubang kecil diisi dari tetangga agar normal tidak berderau di tepinya.
    isi = d.copy()
    isi[~sah] = 0
    bobot = cv2.GaussianBlur(sah.astype(np.float32), (0, 0), 2.0)
    rata = cv2.GaussianBlur(isi, (0, 0), 2.0) / np.maximum(bobot, 1e-3)
    d = np.where(sah, d, rata)
    d = cv2.bilateralFilter(d, 7, 0.03, 3)                  # halus di bidang, tepi anak tangga tetap tajam
    return d, sah


def _normal(d: np.ndarray, fx: float, fy: float, cx: float, cy: float, L: int = 3) -> np.ndarray:
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    P = np.dstack([(u - cx) * d / fx, (v - cy) * d / fy, d])
    n = np.cross(np.roll(P, -L, 1) - np.roll(P, L, 1), np.roll(P, -L, 0) - np.roll(P, L, 0))
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    n[(n * P).sum(2) > 0] *= -1                             # menghadap kamera
    return n


def gambar(depth: np.ndarray, mode: str, intrinsik: dict | None) -> np.ndarray:
    """Citra RGB uint8 seukuran depth untuk mode yang diminta."""
    k = intrinsik or {}
    skala = float(k.get("depth_scale", 0.001))
    if mode == "jarak" or not {"fx", "fy", "cx", "cy"} <= set(k):
        d = depth.astype(np.float32)
        sah = d > 0
        out = np.zeros(d.shape + (3,), np.uint8)
        if sah.any():
            lo, hi = np.percentile(d[sah], (3, 97))
            vis = np.clip((d - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
            out = cv2.cvtColor(cv2.applyColorMap(vis, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            out[~sah] = 0
        return out
    d, sah = _meter_halus(depth, skala)
    if mode == "kontras":
        detail = d - cv2.GaussianBlur(d, (0, 0), 9.0)
        s = np.percentile(np.abs(detail[sah]), 98) if sah.any() else 1.0
        g = np.clip(128 + detail / max(s, 1e-4) * 127, 0, 255).astype(np.uint8)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        out = np.dstack([g, g, g])
    else:
        # Derau depth D435 (~1% jarak) membuat bidang datar berbintik bila normal
        # dihitung terlalu lokal. Bilateral kedua + jangkauan beda 5 px + normal
        # dihaluskan: lantai/tread mulus, tepi anak tangga (~15-20 cm) tetap tegas.
        d = cv2.bilateralFilter(d, 9, 0.03, 7)
        n = _normal(d, k["fx"], k["fy"], k["cx"], k["cy"], L=5)
        n = cv2.GaussianBlur(n, (0, 0), 3.0)
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
        if mode == "bidang":
            out = _bidang(d, n, sah, k)
            out[~sah] = 0
            return out
        if mode == "normal":
            out = ((n * np.array([1, -1, -1], np.float32) * 0.5 + 0.5) * 255).astype(np.uint8)
        else:
            terang = np.clip(n @ CAHAYA, 0, 1)
            g = (40 + 215 * terang ** 0.8).astype(np.uint8)
            out = np.dstack([g, g, (g * 0.94).astype(np.uint8)])   # sedikit hangat agar tidak tertukar dengan mask biru
    out[~sah] = 0
    return out


def _bidang(d: np.ndarray, n: np.ndarray, sah: np.ndarray, k: dict, periode: float = 0.5) -> np.ndarray:
    """Warna per ketinggian untuk bidang datar; lihat docstring modul."""
    atas = np.array([0, -1, 0], np.float32)
    for ambang in (0.5, 0.9, 0.95):                        # normal bidang datar dominan, diperketat bertahap
        c = n[sah & ((n @ atas) > ambang)]
        if len(c) > 500:
            atas = c.mean(0)
            atas /= np.linalg.norm(atas)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    P = np.dstack([(u - k["cx"]) * d / k["fx"], (v - k["cy"]) * d / k["fy"], d])
    tinggi = P @ atas
    tinggi -= np.percentile(tinggi[sah], 2) if sah.any() else 0
    # Kemiringan dihaluskan sebelum diberi ambang: tanpa ini derau depth
    # membuat bercak "tidak datar" di tengah lantai/tread (tampak abu-abu).
    kos = cv2.GaussianBlur((n @ atas).astype(np.float32), (0, 0), 4.0)
    datar, tegak = kos > 0.85, np.abs(kos) < 0.45
    hue = ((tinggi / periode) % 1.0 * 179).astype(np.uint8)
    sat = np.where(datar, 200, np.where(tegak, 30, 80)).astype(np.uint8)
    val = (70 + 185 * np.where(datar, 1.0, np.clip(n @ CAHAYA, 0, 1))).astype(np.uint8)
    out = cv2.cvtColor(np.dstack([hue, sat, val]), cv2.COLOR_HSV2RGB)
    # Batas antar-bidang: perubahan normal yang tajam. Potongan pendek (derau)
    # dibuang agar yang tersisa garis tepi anak tangga, bukan bintik.
    gx = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
    tepi = (np.sqrt((gx ** 2 + gy ** 2).sum(2)) > 0.9).astype(np.uint8)
    nk, lab, stat, _ = cv2.connectedComponentsWithStats(tepi, 8)
    panjang = np.maximum(stat[:, cv2.CC_STAT_WIDTH], stat[:, cv2.CC_STAT_HEIGHT])
    simpan = np.zeros(nk, bool); simpan[1:] = panjang[1:] >= 60
    out[simpan[lab]] = (25, 25, 25)
    return out


def gabung_ir(bidang_rgb: np.ndarray, ir: np.ndarray, bobot_ir: float = 0.75) -> np.ndarray:
    """Warna (hue, saturasi) dari mode Bidang, terang-gelap dan tekstur dari IR selaras.

    Garis tepi hitam Bidang tidak ikut: di atas tekstur IR ia tampak seperti
    coretan, sedangkan tepi anak tangga sudah terlihat dari IR itu sendiri.
    """
    hsv = cv2.cvtColor(bidang_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    tepi = (bidang_rgb.max(2) < 40) & (bidang_rgb.sum(2) > 0)
    hsv[..., 1][tepi] = 0
    hsv[..., 2] = np.clip((1 - bobot_ir) * np.where(tepi, 150, hsv[..., 2]) + bobot_ir * ir.astype(np.float32), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    out[(bidang_rgb.sum(2) == 0) & (ir == 0)] = 0
    return out


def hitung_dari_folder(folder, depth: np.ndarray, intrinsik: dict, mode: str) -> np.ndarray | None:
    """Tampilan bantu untuk satu folder frame, termasuk mode yang butuh berkas IR."""
    if mode in BUTUH_BERKAS:
        from . import ir_selaras
        # IR diproyeksikan ulang dengan depth berkasnya sendiri (bukan ``depth`` yang
        # mungkin sudah digeser ke RGB); koreksi geser ke RGB dilakukan di dalamnya.
        ir = ir_selaras.selaraskan(folder)
        if mode == "ir" or ir is None:
            return ir if mode == "ir" else gambar(depth, "bidang", intrinsik)
        return gabung_ir(gambar(depth, "bidang", intrinsik), ir)
    return gambar(depth, mode, intrinsik)


def warnai(latar_rgb: np.ndarray, bidang_rgb: np.ndarray, kekuatan: float) -> np.ndarray:
    """Oleskan warna ketinggian Bidang pada latar (RGB atau IR) tanpa menutup teksturnya.

    Hue dan saturasi dari Bidang, terang-gelap dari latar; ``kekuatan`` 0..1
    mencampur hasilnya dengan latar asli. Garis tepi hitam Bidang tidak ikut.
    """
    if kekuatan <= 0 or bidang_rgb is None:
        return latar_rgb
    hb = cv2.cvtColor(bidang_rgb, cv2.COLOR_RGB2HSV)
    hl = cv2.cvtColor(latar_rgb, cv2.COLOR_RGB2HSV)
    tepi = (bidang_rgb.max(2) < 40) & (bidang_rgb.sum(2) > 0)
    s = hb[..., 1].copy(); s[tepi] = 0
    # Latar gelap diangkat sedikit agar warna tetap terbaca pada frame gelap.
    v = np.maximum(hl[..., 2], 60).astype(np.uint8)
    warna = cv2.cvtColor(np.dstack([hb[..., 0], s, v]), cv2.COLOR_HSV2RGB)
    campur = cv2.addWeighted(latar_rgb, 1 - kekuatan, warna, kekuatan, 0)
    # Tanpa data depth (tepi hasil geser, lubang): latar apa adanya, bukan pita pucat.
    return np.where((bidang_rgb.max(2) > 0)[..., None], campur, latar_rgb)
