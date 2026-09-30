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
            "lux_sumber": str(d.get("lux_sumber") or ""),
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


# ------------------------------------------------------------------ waktu pengambilan
HARI = ("Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min")
BULAN = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")


def periode(jam: int) -> str:
    """Pagi 05-10, siang 10-15, sore 15-18, malam 18-24, dini hari 00-05."""
    return ("dini hari" if jam < 5 else "pagi" if jam < 10 else "siang" if jam < 15
            else "sore" if jam < 18 else "malam")


def format_waktu(t, dengan_detik: bool = True) -> str:
    """datetime -> 'Sen 14 Sep 2026, 21.19.28 WIB (malam)'."""
    jam = t.strftime("%H.%M.%S" if dengan_detik else "%H.%M")
    zona = t.astimezone().tzname() or ""
    return f"{HARI[t.weekday()]} {t.day} {BULAN[t.month - 1]} {t.year}, {jam} {zona} ({periode(t.hour)})"


def waktu_rekaman(sesi: Path) -> tuple | None:
    """(mulai, selesai) dari source/session.json, datetime lokal; None bila tidak ada."""
    from datetime import datetime
    f = Path(sesi) / "source" / "session.json"
    try:
        d = json.loads(f.read_text()) if f.exists() else {}
        mulai = datetime.fromisoformat(d["mulai_iso"])
        selesai = datetime.fromisoformat(d["selesai_iso"]) if d.get("selesai_iso") else None
        return mulai, selesai
    except (OSError, ValueError, KeyError):
        return None


def waktu_frame(info: dict):
    """Waktu pengambilan satu frame dari frame.json (cap waktu kamera, ms sejak epoch), datetime lokal."""
    from datetime import datetime
    t = info.get("timestamp_kamera_ms")
    try:
        return datetime.fromtimestamp(float(t) / 1000.0) if t else None
    except (TypeError, ValueError, OSError):
        return None


# ------------------------------------------------------------------ lux dari foto HP
def lux_dari_foto(path) -> tuple[float, str]:
    """Perkiraan lux dari EXIF foto HP: E = c * N^2 / (t * ISO), c = 250.

    Rumus fotometri yang sama dengan ``colour_hdri.average_illuminance``
    (kalibrasi meter cahaya datang, c = 250). Kamera HP mengukur cahaya PANTULAN
    dan menganggap permukaan abu-abu sedang, jadi hasilnya perkiraan kasar
    (sekitar +-30-50%). Tidak berlaku untuk frame D435: rekamannya tidak
    menyimpan waktu pencahayaan dan gain.
    """
    from PIL import Image
    with Image.open(path) as im:
        exif = im.getexif()
        ifd = exif.get_ifd(0x8769) if exif else {}
    ambil = lambda k: ifd.get(k) or (exif.get(k) if exif else None)
    N, t = ambil(33437), ambil(33434)                      # FNumber, ExposureTime
    S = ambil(34855) or ambil(34867)                       # ISOSpeedRatings / PhotographicSensitivity
    if isinstance(S, (tuple, list)):
        S = S[0]
    if not (N and t and S):
        raise ValueError("Foto tidak memuat EXIF bukaan lensa, waktu pencahayaan, dan ISO "
                         "(pastikan foto asli dari kamera HP, bukan tangkapan layar/hasil kirim WhatsApp).")
    N, t, S = float(N), float(t), float(S)
    lux = 250.0 * N * N / (t * S)
    ket = f"foto HP (f/{N:g}, {('1/' + str(round(1 / t))) if t < 1 else f'{t:g}'} s, ISO {S:g}); perkiraan +-30-50%"
    return lux, ket
