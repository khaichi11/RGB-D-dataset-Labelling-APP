"""Rekomendasi instance mask dari RF-DETR Seg; batas akhir dirapikan SAM 2."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODEL = None
_MODEL_PATH = None
RESOLUSI_RFDETR = 312  # kelipatan 12, sesuai RF-DETR Seg Nano


def bobot_rfdetr() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "bobot" / "kandidat" / "rfdetr_seg_nano_dualclass" / "fold_0_312_eval5" / "checkpoint_best_regular.pth"


def _model(path: Path):
    global _MODEL, _MODEL_PATH
    if _MODEL is None or _MODEL_PATH != path:
        from rfdetr import RFDETR
        m = RFDETR.from_checkpoint(str(path), trust_checkpoint=True)
        m.inference(compile=False, batch_size=1, dtype="float16", inplace=True)
        _MODEL = m
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
    """Depth sebagai pemeriksa akhir, tidak dipakai memotong batas utama."""
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
            akhir = kandidat if int(kandidat.sum()) >= 0.90 * int(mask.sum()) else mask
            akhir = cv2.morphologyEx(akhir, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            hasil[nama].append(_poligon(akhir) or poly)
    return hasil


def usulkan(rgb_bgr: np.ndarray, depth: np.ndarray, k: dict,
            confidence: float = 0.28) -> dict:
    """RF-DETR Seg sebagai kandidat awal; depth tidak dipakai memotong mask."""
    path = bobot_rfdetr()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint RF-DETR tidak ditemukan: {path}")

    # RF-DETR mengharapkan RGB; Studio meneruskan BGR dari OpenCV
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    deteksi = _model(path).predict(
        rgb, threshold=confidence,
        shape=(RESOLUSI_RFDETR, RESOLUSI_RFDETR),
        include_source_image=False,
    )

    out: dict = {"tapakan": [], "bidang_tegak": [], "sumber": "RF-DETR", "jumlah_rfdetr": 0}
    masks_raw = getattr(deteksi, "mask", None)
    classes_raw = getattr(deteksi, "class_id", None)
    confidences_raw = getattr(deteksi, "confidence", None)
    if masks_raw is None or classes_raw is None or confidences_raw is None:
        return out

    h, w = depth.shape[:2]
    kelas = {0: "tapakan", 1: "bidang_tegak"}
    kernel = np.ones((7, 7), np.uint8)
    for raw_mask, cls, conf in zip(masks_raw, classes_raw, confidences_raw):
        if int(cls) not in kelas or float(conf) < confidence:
            continue
        nama = kelas[int(cls)]
        # RF-DETR mengembalikan mask pada resolusi aslinya; sejajarkan ke frame
        binary = (np.asarray(raw_mask) > 0.5).astype(np.uint8)
        if binary.shape[:2] != (h, w):
            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        final = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        sederhana = _poligon(final)
        if sederhana:
            out[nama].append(sederhana)
            out["jumlah_rfdetr"] += 1

    for nama in ("tapakan", "bidang_tegak"):
        out[nama].sort(key=lambda p: -sum(y for _, y in p) / max(1, len(p)))
    return out
