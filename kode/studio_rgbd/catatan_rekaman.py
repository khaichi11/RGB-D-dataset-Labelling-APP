"""Catatan per rekaman: warna stabilo, keterangan/lokasi, dan scene di dalamnya.

Disimpan di ``<rekaman>/catatan.json``, di luar ``source/``, sehingga rekaman
mentah tetap tidak tersentuh::

    {"warna": "kuning",
     "catatan": "Gedung F lt. 2, tangga darurat; bagian akhir gelap",
     "lux": "12",                # diukur dengan lux meter di lokasi (bukan dari gambar)
     "scene": [{"nama": "Tangga A", "awal": 0, "akhir": 1450},
               {"nama": "Bordes + tangga B", "awal": 1451, "akhir": 3000}]}

``awal``/``akhir`` adalah indeks frame mentah, sama dengan angka pada nama
folder ekspor ``frame_000123`` dan slider potongan di tab Tinjau.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WARNA = {                     # nama -> latar daftar (pastel, teks tetap terbaca)
    "kuning": "#FFF0A0",
    "hijau": "#CDEBC5",
    "biru": "#C9DDF2",
    "merah": "#F6C4BF",
    "ungu": "#E2D2F2",
}
NAMA_BERKAS = "catatan.json"


def baca(sesi: Path) -> dict:
    p = Path(sesi) / NAMA_BERKAS
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        d = {}
    scene = [s for s in d.get("scene") or [] if isinstance(s, dict) and "awal" in s and "akhir" in s]
    return {"warna": d.get("warna") if d.get("warna") in WARNA else None,
            "catatan": str(d.get("catatan") or ""),
            "lux": str(d.get("lux") or ""),
            "scene": sorted(scene, key=lambda s: int(s["awal"]))}


def tulis(sesi: Path, data: dict) -> None:
    p = Path(sesi) / NAMA_BERKAS
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def indeks_frame(nama_frame: str) -> int | None:
    """``frame_000123`` -> 123."""
    try:
        return int(nama_frame.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def scene_untuk(data: dict, indeks: int | None) -> dict | None:
    if indeks is None:
        return None
    for s in data.get("scene", []):
        if int(s["awal"]) <= indeks <= int(s["akhir"]):
            return s
    return None


def ringkas(data: dict, panjang: int = 60) -> str:
    """Satu baris untuk daftar: catatan dipotong."""
    t = " ".join(data.get("catatan", "").split())
    return t if len(t) <= panjang else t[:panjang - 1] + "…"


def kecerahan(rgb: np.ndarray) -> tuple[float, str]:
    """Rerata kecerahan citra (L* 0-255) dan kategorinya.

    Ini deskripsi GAMBAR, bukan cahaya ruangan: kamera memakai auto-exposure
    dan rekaman tidak menyimpan waktu pencahayaan/gain, sehingga lux tidak dapat
    diturunkan dari piksel. Ambang kategori: < 30 gelap, 30-80 redup, > 80 terang.
    """
    import cv2
    l = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].mean())
    return l, ("gelap" if l < 30 else "redup" if l < 80 else "terang")
