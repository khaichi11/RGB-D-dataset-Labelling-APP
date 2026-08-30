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


def _prompt_dari_mask(mask: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Satu titik positif objek dan satu negatif pada bidang tetangga."""
    dalam = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(np.argmax(dalam)), dalam.shape)
    x0, y0, x1, y1 = np.round(box).astype(int)
    area = np.zeros_like(mask)
    area[max(0, y0):min(mask.shape[0], y1 + 1), max(0, x0):min(mask.shape[1], x1 + 1)] = 1
    luar = area & (1 - cv2.dilate(mask, np.ones((7, 7), np.uint8)))
    if luar.any():
        jarak = cv2.distanceTransform(luar, cv2.DIST_L2, 5)
        ny, nx = np.unravel_index(int(np.argmax(jarak)), jarak.shape)
        return np.array([[px, py], [nx, ny]], np.float32), np.array([1, 0], np.int32)
    return np.array([[px, py]], np.float32), np.array([1], np.int32)


def rapikan_kelompok(rgb: np.ndarray, kelompok: dict[str, list[list[tuple[int, int]]]],
                     petunjuk_depth: dict[str, np.ndarray] | None = None) -> dict[str, list[list[tuple[int, int]]]]:
    """Rapikan kandidat YOLO memakai prompt titik dan petunjuk depth lunak."""
    if not any(kelompok.values()):
        return kelompok
    pred = _predictor()
    pred.set_image(rgb)
    hasil: dict[str, list[list[tuple[int, int]]]] = {}
    for nama, kandidat in kelompok.items():
        hasil[nama] = []
        for poly in kandidat:
            dasar = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillPoly(dasar, [np.asarray(poly, np.int32)], 1)
            pts = np.asarray(poly, dtype=np.float32)
            x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
            # Sedikit margin agar batas mask tidak dipotong tepat di kotak YOLO.
            h, w = rgb.shape[:2]
            box = np.array([max(0, x0 - 8), max(0, y0 - 8), min(w - 1, x1 + 8), min(h - 1, y1 + 8)])
            titik, label_titik = _prompt_dari_mask(dasar, box)
            masks, scores, _ = pred.predict(point_coords=titik, point_labels=label_titik,
                                             box=box, multimask_output=True)
            # SAM dapat memilih lantai/dinding besar di dalam box. Pilih mask
            # yang paling setia pada kandidat YOLO, dengan orientasi depth
            # sebagai bonus kecil saja agar depth berderau tidak merusak batas.
            nilai = []
            hint = None if petunjuk_depth is None else petunjuk_depth.get(nama)
            for m, skor_sam in zip(masks, scores):
                biner = m.astype(np.uint8)
                gabung = int(np.logical_and(biner, dasar).sum())
                union = int(np.logical_or(biner, dasar).sum())
                iou = gabung / max(1, union)
                cocok_depth = 0.0 if hint is None else float(np.logical_and(biner, hint > 0).sum()) / max(1, int(biner.sum()))
                nilai.append(.50 * float(skor_sam) + .42 * iou + .08 * cocok_depth)
            mask = masks[int(np.argmax(nilai))]
            baru = _poligon(mask)
            hasil[nama].append(baru or poly)
    return hasil
