"""Rekomendasi instance mask YOLO; batas akhir dapat dirapikan SAM 2."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

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
    """YOLO sebagai kandidat awal; depth tidak dipakai memotong mask."""
    path = bobot_tangga()
    if not path.exists():
        raise FileNotFoundError(f"Bobot tangga tidak ditemukan: {path}")
    hasil = _model(path).predict(rgb, conf=confidence, verbose=False, retina_masks=True)[0]
    out = {"tapakan": [], "bidang_tegak": [], "sumber": "YOLO", "jumlah_yolo": 0}
    if hasil.masks is None or hasil.boxes is None:
        return out
    h, w = depth.shape[:2]
    kelas = {0: "tapakan", 1: "bidang_tegak"}
    kernel = np.ones((7, 7), np.uint8)
    for poly, cls in zip(hasil.masks.xy, hasil.boxes.cls.cpu().numpy().astype(int)):
        if cls not in kelas:
            continue
        nama = kelas[cls]
        mask = np.zeros((h, w), np.uint8)
        p = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [p], 1)
        final = mask
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
        sederhana = _poligon(final)
        if sederhana:
            out[nama].append(sederhana)
            out["jumlah_yolo"] += 1
    # Nomor 1 = objek/tapakan paling bawah dalam citra.
    for nama in ("tapakan", "bidang_tegak"):
        out[nama].sort(key=lambda p: -sum(y for _, y in p) / len(p))
    return out
