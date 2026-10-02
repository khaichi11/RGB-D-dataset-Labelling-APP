"""Pemeriksaan kualitas label otomatis: label yang tidak cocok dengan geometri depth.

Ini DAFTAR PERIKSA untuk ditinjau manusia, bukan vonis: depth D435 berderau dan
poligon tipis/jauh sulit dinilai, jadi sebagian tanda adalah alarm palsu. Yang
dinilai (depth digeser ke RGB lebih dulu, lihat ir_selaras.geser_frame):

- tread yang tidak datar: < 60% piksel tread berlabel (dekat, <= 3 m, cukup
  besar) yang datar menurut normal depth -> label tread kemungkinan meleset ke
  riser/lantai/dinding.
- riser yang tidak tegak: < 35% piksel riser berlabel yang tegak (ambang
  longgar karena riser tipis mudah terbaca miring setelah penghalusan).
- tinggi riser menyimpang dari median frame lebih dari max(2,5 cm, 15%).
- poligon sangat kecil (< 600 px): sering klik nyasar, kadang potongan sah.

Diuji pada 1.114 frame terperiksa (bukti/.../auto_label/qa_label.py): contoh
yang benar ditemukan antara lain poligon nyasar di frame lantai (191513
frame_001910).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import ir_selaras as IR, visual_depth as V
from .pengukuran_objek import ukur


def _kos_atas(dep: np.ndarray, k: dict) -> tuple[np.ndarray, np.ndarray]:
    d, sah = V._meter_halus(dep, k["depth_scale"])
    d = cv2.bilateralFilter(d, 9, 0.03, 7)
    n = V._normal(d, k["fx"], k["fy"], k["cx"], k["cy"], L=5)
    n = cv2.GaussianBlur(n, (0, 0), 3.0)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    atas = np.array([0, -1, 0], np.float32)
    for a in (0.5, 0.9, 0.95):
        c = n[sah & ((n @ atas) > a)]
        if len(c) > 500:
            atas = c.mean(0)
            atas /= np.linalg.norm(atas)
    return cv2.GaussianBlur((n @ atas).astype(np.float32), (0, 0), 4.0), sah


def _masker(p, bentuk) -> np.ndarray:
    m = np.zeros(bentuk, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32)], 1)
    return m


def periksa_frame(d: Path) -> dict | None:
    """Skor kecurigaan dan alasan untuk satu frame berlabel; None bila tidak ada label."""
    d = Path(d)
    j = json.loads((d / "label_draft.json").read_text()) if (d / "label_draft.json").exists() else {}
    pol = j.get("poligon") or {}
    rp = [p for p in pol.get("objek", []) if len(p) >= 3]
    tp = [p for p in pol.get("acuan", []) if len(p) >= 3]
    if not rp and not tp:
        return None
    info = json.loads((d / "frame.json").read_text())
    i = info["intrinsics_rgb_native"]
    k = {"depth_scale": float(info["depth_scale"]), "fx": i["fx"], "fy": i["fy"], "cx": i["ppx"], "cy": i["ppy"]}
    dep = np.load(d / "depth_aligned_to_color.npy")
    g = IR.geser_frame(d, dep) if (d / "ir_left_raw.png").exists() else (0, 0)
    dep = IR.geser(dep, *g, terdekat=True)
    kos, sah = _kos_atas(dep, k)
    z = dep.astype(np.float32) * k["depth_scale"]
    b = dep.shape
    riser = [_masker(p, b) for p in rp]
    tread = [_masker(p, b) for p in tp]
    semua_riser = np.any(riser, 0) if riser else np.zeros(b, bool)
    erosi = lambda m: cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    alasan, skor = [], 0.0
    for idx, m in enumerate(tread, 1):
        e = erosi(m & ~semua_riser) & sah
        if e.sum() >= 1500 and np.median(z[e]) <= 3.0:
            f = float((kos > 0.85)[e].mean())
            if f < 0.6:
                alasan.append(f"tread T{idx} hanya {f:.0%} datar menurut depth")
                skor += 1 - f
    for idx, m in enumerate(riser, 1):
        e = erosi(m) & sah
        if e.sum() >= 800 and np.median(z[e]) <= 3.0:
            f = float((np.abs(kos) < 0.45)[e].mean())
            if f < 0.35:
                alasan.append(f"riser R{idx} hanya {f:.0%} tegak menurut depth")
                skor += 0.5 * (1 - f)
    for nm, daftar in (("R", riser), ("T", tread)):
        for idx, m in enumerate(daftar, 1):
            if m.sum() < 600:
                alasan.append(f"poligon {nm}{idx} sangat kecil ({int(m.sum())} px): klik nyasar?")
                skor += 0.6
    tinggi = []
    for mr in riser:
        e = erosi(mr) & sah
        if e.sum() < 300 or np.median(z[e]) > 3.0:
            tinggi.append(None)
            continue
        turun = np.zeros_like(mr); turun[12:] = mr[:-12]
        pita = (turun > 0) & (mr == 0)
        irisan = [int((pita & (mt > 0)).sum()) for mt in tread] or [0]
        jj = int(np.argmax(irisan))
        h = ukur(mr, tread[jj], dep, k) if irisan[jj] > 0 else {"ok": False}
        tinggi.append(h["tinggi_cm"] if h.get("ok") else None)
    ok = [t for t in tinggi if t is not None]
    median = float(np.median(ok)) if ok else None
    if median is not None and len(ok) >= 3:
        for idx, t in enumerate(tinggi, 1):
            if t is not None and abs(t - median) > max(2.5, 0.15 * median):
                alasan.append(f"tinggi R{idx} {t:.1f} cm vs median {median:.1f} cm")
                skor += 0.4
    return {"skor": round(skor, 3), "alasan": alasan, "geser": list(g),
            "median_riser_cm": None if median is None else round(median, 1)}
