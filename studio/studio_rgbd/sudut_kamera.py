"""Sudut tunduk kamera dari depth: bidang mendatar terbesar (lantai atau tread) di bagian bawah citra.

D435 pemakai tidak punya IMU, jadi arah "atas" diambil dari permukaan mendatar yang terlihat. Bidang
dicari dengan RANSAC pada titik 3-D bagian bawah citra; hanya bidang yang normalnya cukup tegak
(permukaan mendatar, bukan dinding/riser) yang diterima. Bidang tread sejajar lantai, jadi tangga
yang terlihat tidak mengganggu.

Sudut tunduk: 0 = kamera mendatar, 90 = menghadap lurus ke bawah. Sudut samping: kemiringan kiri-kanan
(roll), + = sisi kanan citra lebih rendah. Pada rekaman bermeteran galat ukur terkecil di 20-30 derajat
(model/bukti/final_d435/k_dinding_20261112/sudut.txt); saran 25-35 derajat.
"""
from __future__ import annotations

import math

import numpy as np

SARAN_DERAJAT = (25.0, 35.0)


def sudut_tunduk(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                 porsi_bawah: float = 0.6, maks_titik: int = 2500, iterasi: int = 80,
                 ambang_m: float = 0.02, rng: np.random.Generator | None = None) -> dict | None:
    """{'tunduk': derajat, 'samping': derajat, 'porsi': fraksi inlier} atau None bila tidak ada bidang mendatar."""
    rng = rng or np.random.default_rng(0)
    h, w = depth_m.shape
    y0 = int(h * (1 - porsi_bawah))
    z = depth_m[y0:].astype(np.float32)
    v, u = np.mgrid[y0:h, 0:w]
    sah = (z > 0.2) & (z < 5.0)
    if sah.sum() < 200:
        return None
    zs = z[sah]
    p = np.stack([(u[sah] - cx) * zs / fx, (v[sah] - cy) * zs / fy, zs], 1)
    if len(p) > maks_titik:
        p = p[rng.choice(len(p), maks_titik, replace=False)]
    i = rng.integers(0, len(p), (iterasi, 3))
    a, b, c = p[i[:, 0]], p[i[:, 1]], p[i[:, 2]]
    n = np.cross(b - a, c - a)
    pj = np.linalg.norm(n, axis=1)
    ok = pj > 1e-6
    n[ok] /= pj[ok, None]
    ok &= np.abs(n[:, 1]) >= 0.5                       # mendatar: normal dekat sumbu atas-bawah kamera (tunduk <= 60)
    if not ok.any():
        return None
    n, a = n[ok], a[ok]
    d = np.einsum("ij,ij->i", n, a)
    jumlah = (np.abs(p @ n.T - d) < ambang_m).sum(0)
    # Riser juga lolos saringan bila tunduk >= 30 derajat (normalnya |n_y| = sin tunduk). Bobot n_y^2
    # mengutamakan bidang mendatar selama tunduk < 45 derajat (|n_y| = cos tunduk > sin tunduk).
    k = int(np.argmax(jumlah * n[:, 1] ** 2))
    inlier = np.abs(p @ n[k] - d[k]) < ambang_m
    if inlier.sum() < 100:
        return None
    q = p[inlier]
    _, _, vt = np.linalg.svd(q - q.mean(0), full_matrices=False)
    nn = vt[2]
    if nn[1] > 0:                                       # arahkan ke atas (Y kamera menunjuk ke bawah)
        nn = -nn
    tunduk = math.degrees(math.atan2(abs(nn[2]), math.hypot(nn[0], nn[1])))
    samping = math.degrees(math.atan2(nn[0], -nn[1]))
    return {"tunduk": tunduk, "samping": samping, "porsi": float(inlier.mean())}


def teks(hasil: dict | None) -> str:
    """Teks pita preview."""
    if hasil is None:
        return "Sudut kamera: lantai/tangga belum terlihat"
    lo, hi = SARAN_DERAJAT
    tanda = "OK" if lo <= hasil["tunduk"] <= hi else ("tundukkan" if hasil["tunduk"] < lo else "angkat")
    return (f"Sudut tunduk {int(round(hasil['tunduk']))} deg ({tanda}; saran {lo:.0f}-{hi:.0f})  "
            f"samping {int(round(hasil['samping'])):+d} deg")


def ringkas(riwayat: list[tuple[float, float, float]]) -> dict | None:
    """Ringkasan untuk session.json dari [(waktu, tunduk, samping), ...]."""
    if not riwayat:
        return None
    t = np.array([r[1] for r in riwayat]); s = np.array([r[2] for r in riwayat])
    lo, hi = SARAN_DERAJAT
    return {"tunduk_median": round(float(np.median(t)), 1), "tunduk_p10": round(float(np.percentile(t, 10)), 1),
            "tunduk_p90": round(float(np.percentile(t, 90)), 1), "samping_median": round(float(np.median(s)), 1),
            "porsi_dalam_saran": round(float(((t >= lo) & (t <= hi)).mean()), 3), "n": len(riwayat),
            "cara": "bidang mendatar terbesar dari depth (D435 tanpa IMU), sudut_kamera.py"}
