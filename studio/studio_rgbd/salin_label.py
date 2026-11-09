"""Membawa label dari frame tetangga (sebelum atau sesudahnya) ke frame aktif.

Kamera bergerak antar-frame, jadi poligon dipindahkan lewat homografi dari pencocokan
fitur ORB dua citra. Homografi hanya kira-kira (tangga tidak satu bidang), karena itu
hasilnya selalu berupa usulan yang disunting pemakai, bukan label jadi.
"""
from __future__ import annotations

import cv2
import numpy as np


def homografi(lama: np.ndarray, baru: np.ndarray) -> np.ndarray | None:
    """Homografi citra lama -> baru (RGB), atau None bila pencocokannya tidak bisa dipercaya."""
    clahe = cv2.createCLAHE(2.0, (8, 8))                  # frame gelap: fitur ORB muncul setelah kontras diratakan
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(clahe.apply(cv2.cvtColor(lama, cv2.COLOR_RGB2GRAY)), None)
    k2, d2 = orb.detectAndCompute(clahe.apply(cv2.cvtColor(baru, cv2.COLOR_RGB2GRAY)), None)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None
    pasangan = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    baik = [p[0] for p in pasangan if len(p) == 2 and p[0].distance < .75 * p[1].distance]
    if len(baik) < 25:
        return None
    a = np.float32([k1[m.queryIdx].pt for m in baik])
    b = np.float32([k2[m.trainIdx].pt for m in baik])
    H, inlier = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    if H is None or inlier is None or int(inlier.sum()) < 20 or inlier.mean() < .3:
        return None
    # Tolak hasil yang melipat atau melempar citra jauh: gerak antar-frame selalu kecil.
    h, w = lama.shape[:2]
    ujung = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    jadi = cv2.perspectiveTransform(ujung, H)
    rasio = abs(cv2.contourArea(jadi)) / float(w * h)
    if not cv2.isContourConvex(jadi) or not .4 < rasio < 2.5 or np.linalg.norm(jadi - ujung, axis=2).max() > .5 * np.hypot(w, h):
        return None
    return H


def _potong_kotak(titik: list, w: int, h: int) -> list:
    """Sutherland-Hodgman terhadap kotak citra: bagian yang keluar frame terpotong di tepinya."""
    def di_x(c):
        return lambda p, q: (c, p[1] + (q[1] - p[1]) * (c - p[0]) / (q[0] - p[0]))

    def di_y(c):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (c - p[1]) / (q[1] - p[1]), c)
    sisi = ((lambda p: p[0] >= 0, di_x(0)), (lambda p: p[0] <= w - 1, di_x(w - 1)),
            (lambda p: p[1] >= 0, di_y(0)), (lambda p: p[1] <= h - 1, di_y(h - 1)))
    for masuk, silang in sisi:
        hasil = []
        for i, q in enumerate(titik):
            p = titik[i - 1]
            if masuk(q):
                if not masuk(p):
                    hasil.append(silang(p, q))
                hasil.append(q)
            elif masuk(p):
                hasil.append(silang(p, q))
        titik = hasil
        if not titik:
            break
    return titik


def pindahkan(poligon: dict, H: np.ndarray | None, w: int, h: int, luas_min: float = 150.0) -> dict:
    """Poligon {kelas: [[(x, y), ...], ...]} di frame baru; H None = di tempat yang sama. Urutan mask dipertahankan."""
    hasil = {}
    for nama, daftar in poligon.items():
        baru = []
        for poly in daftar:
            if len(poly) < 3:
                continue
            a = np.float32(poly).reshape(-1, 1, 2)
            titik = _potong_kotak((cv2.perspectiveTransform(a, H) if H is not None else a).reshape(-1, 2).tolist(), w, h)
            if len(titik) >= 3 and abs(cv2.contourArea(np.float32(titik))) >= luas_min:   # sudah keluar frame
                baru.append([(round(float(x), 1), round(float(y), 1)) for x, y in titik])
        hasil[nama] = baru
    return hasil
