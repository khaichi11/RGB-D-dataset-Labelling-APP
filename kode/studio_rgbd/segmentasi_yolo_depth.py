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


def verifikasi_depth(kelompok: dict[str, list[list[tuple[int, int]]]], depth: np.ndarray) -> dict[str, list[list[tuple[int, int]]]]:
    """Depth sebagai pemeriksa akhir SAM, bukan pemotong batas utama.

    D435 sering berlubang di tepi, maka piksel valid dilebarkan sedikit dan
    hasil depth hanya dipakai jika tetap mempertahankan >=90% mask SAM.
    """
    valid = (depth > 0).astype(np.uint8)
    dukung = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dukung = cv2.dilate(dukung, np.ones((5, 5), np.uint8))
    h, w = depth.shape[:2]
    hasil: dict[str, list[list[tuple[int, int]]]] = {}
    for nama, daftar in kelompok.items():
        hasil[nama] = []
        for poly in daftar:
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 1)
            kandidat = cv2.bitwise_and(mask, dukung)
            akhir = kandidat if int(kandidat.sum()) >= .90 * int(mask.sum()) else mask
            akhir = cv2.morphologyEx(akhir, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            hasil[nama].append(_poligon(akhir) or poly)
    return hasil


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
