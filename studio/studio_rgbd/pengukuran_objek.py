"""Pengukuran tinggi objek terhadap bidang acuan pada data RGB-D Studio."""
from __future__ import annotations

import numpy as np

from .geometri import pasang_bidang, titik_dari_masker


def _titik_objek(mask: np.ndarray, depth: np.ndarray, k: dict, langkah: int) -> np.ndarray:
    ys, xs = np.nonzero(mask.astype(bool))
    ys, xs = ys[::langkah], xs[::langkah]
    z = depth[ys, xs].astype(np.float32) * float(k["depth_scale"])
    sah = (z > 0.15) & (z < 6.0)
    xs, ys, z = xs[sah], ys[sah], z[sah]
    return np.column_stack([(xs - k["cx"]) * z / k["fx"], (ys - k["cy"]) * z / k["fy"], z]).astype(np.float64)


def ukur(mask_objek: np.ndarray, mask_acuan: np.ndarray, depth_aligned: np.ndarray, intrinsics: dict, subsample: int = 2) -> dict:
    objek = _titik_objek(mask_objek, depth_aligned, intrinsics, max(1, subsample))
    acuan = titik_dari_masker(mask_acuan, depth_aligned, intrinsics, max(1, subsample))
    if len(objek) < 100:
        return {"ok": False, "alasan": "Titik depth objek terlalu sedikit."}
    if len(acuan) < 300:
        return {"ok": False, "alasan": "Titik depth bidang acuan terlalu sedikit."}
    bidang = pasang_bidang(acuan, np.random.default_rng(7))
    if bidang is None:
        return {"ok": False, "alasan": "Bidang acuan tidak cukup datar atau depth rusak."}
    normal, titik_acuan, pusat, rms = bidang
    jarak = np.abs((objek - pusat) @ normal)
    tinggi = float(np.quantile(jarak, 0.95))
    return {"ok": True, "tinggi_m": tinggi, "tinggi_cm": tinggi * 100, "elevasi_p05_cm": float(np.quantile(jarak, 0.05) * 100), "elevasi_p95_cm": tinggi * 100, "jarak_median_objek_m": float(np.median(objek[:, 2])), "titik_objek": int(len(objek)), "titik_bidang_acuan": int(len(titik_acuan)), "rms_bidang_acuan_mm": rms * 1000, "rumus": "persentil 95 jarak titik objek ke bidang acuan 3-D"}
