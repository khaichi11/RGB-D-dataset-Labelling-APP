"""Penyempurnaan mask kandidat dengan SAM 2.1 Tiny di GPU bila tersedia."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_PREDICTOR = None


def bobot_sam2() -> Path:
    return Path(__file__).resolve().parents[2] / "bobot" / "aktif" / "sam2.1_hiera_tiny.pt"


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        path = bobot_sam2()
        if not path.exists() or path.stat().st_size < 100_000_000:
            raise FileNotFoundError("Bobot SAM 2.1 Tiny belum selesai diunduh.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA tidak terdeteksi oleh Python aplikasi; SAM 2 tidak dijalankan pada CPU.")
        model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(path), device="cuda")
        _PREDICTOR = SAM2ImagePredictor(model)
    return _PREDICTOR


def _poligon(mask: np.ndarray) -> list[tuple[int, int]] | None:
    kontur, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return None
    c = max(kontur, key=cv2.contourArea)
    if cv2.contourArea(c) < 250:
        return None
    p = cv2.approxPolyDP(c, max(2.0, .006 * cv2.arcLength(c, True)), True).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in p] if len(p) >= 3 else None


def rapikan_kelompok(rgb: np.ndarray, kelompok: dict[str, list[list[tuple[int, int]]]]) -> dict[str, list[list[tuple[int, int]]]]:
    """Rapikan semua kandidat YOLO dengan satu embedding gambar SAM 2."""
    if not any(kelompok.values()):
        return kelompok
    pred = _predictor()
    pred.set_image(rgb)
    hasil: dict[str, list[list[tuple[int, int]]]] = {}
    for nama, kandidat in kelompok.items():
        hasil[nama] = []
        for poly in kandidat:
            pts = np.asarray(poly, dtype=np.float32)
            x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
            # Sedikit margin agar batas mask tidak dipotong tepat di kotak YOLO.
            h, w = rgb.shape[:2]
            box = np.array([max(0, x0 - 8), max(0, y0 - 8), min(w - 1, x1 + 8), min(h - 1, y1 + 8)])
            masks, scores, _ = pred.predict(box=box, multimask_output=True)
            mask = masks[int(np.argmax(scores))]
            baru = _poligon(mask)
            hasil[nama].append(baru or poly)
    return hasil
