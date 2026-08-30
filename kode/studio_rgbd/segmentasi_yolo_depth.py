"""Rekomendasi instance mask dari YOLO yang dirapikan memakai depth RGB-D."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .segmentasi_otomatis import usulkan as usulkan_depth

_MODEL = None
_MODEL_PATH = None


def bobot_tangga() -> Path:
    return Path(__file__).resolve().parents[2] / "bobot" / "aktif" / "tangga_yolo26s_seg_512_best.pt"


def _model(path: Path):
    global _MODEL, _MODEL_PATH
    if _MODEL is None or _MODEL_PATH != path:
        from ultralytics import YOLO
        _MODEL = YOLO(str(path))
        _MODEL_PATH = path
    return _MODEL


def _poligon(mask: np.ndarray) -> list[tuple[int, int]] | None:
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return None
    c = max(kontur, key=cv2.contourArea)
    if cv2.contourArea(c) < 250:
        return None
    eps = max(2.0, 0.008 * cv2.arcLength(c, True))
    p = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in p] if len(p) >= 3 else None


def usulkan(rgb: np.ndarray, depth: np.ndarray, k: dict, confidence: float = .28) -> dict:
    """YOLO instance-seg + pengecekan orientasi depth.

    YOLO menentukan instance dan batas global. Depth hanya mengikis piksel
    yang jelas bertentangan dengan orientasi permukaan; bila depth berlubang,
    mask YOLO dipertahankan agar tidak pecah seperti metode depth-only.
    """
    path = bobot_tangga()
    if not path.exists():
        raise FileNotFoundError(f"Bobot tangga tidak ditemukan: {path}")
    dasar = usulkan_depth(depth, k)
    hasil = _model(path).predict(rgb, conf=confidence, verbose=False, retina_masks=True)[0]
    out = {"tapakan": [], "bidang_tegak": [], "sumber": "YOLO + depth", "jumlah_yolo": 0}
    if hasil.masks is None or hasil.boxes is None:
        return out
    h, w = depth.shape[:2]
    kelas = {0: ("tapakan", dasar["mask_datar"]), 1: ("bidang_tegak", dasar["mask_tegak"])}
    kernel = np.ones((13, 13), np.uint8)
    for poly, cls in zip(hasil.masks.xy, hasil.boxes.cls.cpu().numpy().astype(int)):
        if cls not in kelas:
            continue
        nama, orientasi = kelas[cls]
        mask = np.zeros((h, w), np.uint8)
        p = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [p], 1)
        # Depth D435 kerap berlubang di tepi. Dilation kecil membuat fusi
        # toleran terhadap lubang, namun tetap membuang bagian yang jelas salah.
        cocok = cv2.dilate(orientasi, kernel)
        irisan = cv2.bitwise_and(mask, cocok)
        final = irisan if int(irisan.sum()) >= .30 * int(mask.sum()) else mask
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
        sederhana = _poligon(final)
        if sederhana:
            out[nama].append(sederhana)
            out["jumlah_yolo"] += 1
    # Nomor 1 = objek/tapakan paling bawah dalam citra.
    for nama in ("tapakan", "bidang_tegak"):
        out[nama].sort(key=lambda p: -sum(y for _, y in p) / len(p))
    return out
