"""Geometri 3-D minimal untuk pengukuran objek Studio."""
from __future__ import annotations

import cv2
import numpy as np

Z_MIN, Z_MAX = 0.25, 4.0
EROSI_PIKSEL, MIN_TITIK = 7, 300
TEBAL_BIDANG, RASIO_INLIER_MIN, RMS_MAKS = 0.008, 0.60, 0.010


def titik_dari_masker(masker: np.ndarray, depth: np.ndarray, k: dict, langkah: int = 1) -> np.ndarray:
    m = cv2.erode(masker.astype(np.uint8), np.ones((EROSI_PIKSEL, EROSI_PIKSEL), np.uint8), iterations=1)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.empty((0, 3))
    ys, xs = ys[::langkah], xs[::langkah]
    z = depth[ys, xs].astype(np.float32) * float(k["depth_scale"])
    ok = (z > Z_MIN) & (z < Z_MAX)
    if not ok.any():
        return np.empty((0, 3))
    xs, ys, z = xs[ok], ys[ok], z[ok]
    return np.stack([(xs - k["cx"]) * z / k["fx"], (ys - k["cy"]) * z / k["fy"], z], axis=1).astype(np.float64)


def pasang_bidang(titik: np.ndarray, rng: np.random.Generator) -> tuple | None:
    if len(titik) < MIN_TITIK:
        return None
    terbaik, jumlah = None, 0
    for tri in rng.integers(0, len(titik), size=(220, 3)):
        a, b, c = titik[tri]
        normal = np.cross(b - a, c - a)
        panjang = float(np.linalg.norm(normal))
        if panjang < 1e-9:
            continue
        normal /= panjang
        inlier = np.abs(titik @ normal - normal @ a) < TEBAL_BIDANG
        if int(inlier.sum()) > jumlah:
            terbaik, jumlah = inlier, int(inlier.sum())
    if terbaik is None or jumlah < MIN_TITIK or jumlah / len(titik) < RASIO_INLIER_MIN:
        return None
    q = titik[terbaik]
    pusat = q.mean(axis=0)
    normal = np.linalg.svd(q - pusat, full_matrices=False)[2][-1]
    normal /= np.linalg.norm(normal)
    rms = float(np.sqrt(np.mean(((q - pusat) @ normal) ** 2)))
    return None if rms > RMS_MAKS else (normal, q, pusat, rms)
