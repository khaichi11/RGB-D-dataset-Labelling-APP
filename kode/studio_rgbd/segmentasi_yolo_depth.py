"""Rekomendasi instance mask YOLO dengan koreksi depth yang tahan-noise."""
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


def _depth_mulus(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalisasi depth valid, isi lubang kecil, lalu haluskan tepi noise.

    Nilai 0 pada D435 adalah lubang, bukan jarak nol. Karena itu lubang diisi
    hanya pada citra panduan; mask YOLO tidak dipaksa mengikuti lubang tersebut.
    """
    valid = depth > 0
    if not valid.any():
        return np.zeros(depth.shape, np.uint8), valid
    nilai = depth[valid].astype(np.float32)
    lo, hi = np.percentile(nilai, (2, 98))
    normal = np.clip((depth.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    lubang = (~valid).astype(np.uint8) * 255
    normal = cv2.inpaint(normal, lubang, 3, cv2.INPAINT_TELEA)
    return cv2.bilateralFilter(normal, 7, 28, 9), valid


def usulkan(rgb: np.ndarray, depth: np.ndarray, k: dict, confidence: float = .28) -> dict:
    """YOLO sebagai bentuk utama, depth ternormalisasi hanya merapikan noise."""
    path = bobot_tangga()
    if not path.exists():
        raise FileNotFoundError(f"Bobot tangga tidak ditemukan: {path}")
    hasil = _model(path).predict(rgb, conf=confidence, verbose=False, retina_masks=True)[0]
    out = {"tapakan": [], "bidang_tegak": [], "sumber": "YOLO + depth ternormalisasi", "jumlah_yolo": 0}
    if hasil.masks is None or hasil.boxes is None:
        return out
    h, w = depth.shape[:2]
    kelas = {0: "tapakan", 1: "bidang_tegak"}
    halus, valid = _depth_mulus(depth)
    kernel = np.ones((7, 7), np.uint8)
    for poly, cls in zip(hasil.masks.xy, hasil.boxes.cls.cpu().numpy().astype(int)):
        if cls not in kelas:
            continue
        nama = kelas[cls]
        mask = np.zeros((h, w), np.uint8)
        p = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [p], 1)
        # YOLO tetap otoritas batas. Depth yang sudah normal hanya membuang
        # outlier ekstrem di DALAM mask; jika terlalu banyak terbuang, gunakan
        # mask YOLO utuh agar tidak muncul garis/belang akibat noise D435.
        nilai = halus[(mask > 0) & valid]
        final = mask
        if len(nilai) >= 40:
            tengah = float(np.median(nilai))
            mad = float(np.median(np.abs(nilai.astype(np.float32) - tengah)))
            toleransi = max(18.0, 4.0 * mad)
            konsisten = ((np.abs(halus.astype(np.float32) - tengah) <= toleransi) | ~valid).astype(np.uint8)
            kandidat = cv2.bitwise_and(mask, konsisten)
            if int(kandidat.sum()) >= .78 * int(mask.sum()):
                final = kandidat
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
        sederhana = _poligon(final)
        if sederhana:
            out[nama].append(sederhana)
            out["jumlah_yolo"] += 1
    # Nomor 1 = objek/tapakan paling bawah dalam citra.
    for nama in ("tapakan", "bidang_tegak"):
        out[nama].sort(key=lambda p: -sum(y for _, y in p) / len(p))
    return out
