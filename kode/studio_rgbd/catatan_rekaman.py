"""Catatan per rekaman: warna stabilo, keterangan/lokasi, dan scene di dalamnya.

Disimpan di ``<rekaman>/catatan.json``, di luar ``source/``, sehingga rekaman
mentah tetap tidak tersentuh::

    {"warna": "kuning",
     "catatan": "Gedung F lt. 2, tangga darurat; bagian akhir gelap",
     "lux": "12",                # diukur dengan lux meter di lokasi (bukan dari gambar)
     "tangga": "T01",           # ID tangga fisik (tangga.json); scene boleh punya tangga sendiri
     "scene": [{"nama": "Tangga A", "awal": 0, "akhir": 1450},
               {"nama": "Bordes + tangga B", "awal": 1451, "akhir": 3000, "tangga": "T02"}]}

``awal``/``akhir`` adalah indeks frame mentah, sama dengan angka pada nama
folder ekspor ``frame_000123`` dan slider potongan di tab Tinjau.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

WARNA = {                     # nama -> latar daftar (pastel, teks tetap terbaca)
    "kuning": "#FFF0A0",
    "hijau": "#CDEBC5",
    "biru": "#C9DDF2",
    "merah": "#F6C4BF",
    "ungu": "#E2D2F2",
    "oranye": "#FFD9B0",
    "toska": "#BFEDE6",
    "pink": "#FAD2E6",
    "zaitun": "#E2E8B8",
    "abu": "#DDDAD6",
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
            "tangga": str(d.get("tangga") or ""),
            "selesai": bool(d.get("selesai", False)),
            "selesai_iso": str(d.get("selesai_iso") or ""),
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


# ------------------------------------------------------------------ tangga fisik
# <dataset>/studio_rgbd/tangga.json: {"T01": {"nama": "Rusunawa tangga 1", "lokasi": ""}, ...}
# Rekaman (atau scene) yang merekam tangga fisik yang sama diberi ID yang sama,
# sehingga pembagian train/val/test dapat dilakukan per tangga: tangga yang sama
# di train dan test membuat angka uji terlalu optimistis.
BERKAS_TANGGA = "tangga.json"


def baca_tangga(akar_data: Path) -> dict[str, dict]:
    f = Path(akar_data) / BERKAS_TANGGA
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (OSError, ValueError):
        d = {}
    return {k: {"nama": str(v.get("nama", "")), "lokasi": str(v.get("lokasi", "")),
                "warna": v.get("warna") if v.get("warna") in WARNA else None}
            for k, v in sorted(d.items()) if isinstance(v, dict)}


def tulis_tangga(akar_data: Path, data: dict) -> None:
    f = Path(akar_data) / BERKAS_TANGGA
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(sorted(data.items())), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(f)


def id_tangga_baru(data: dict) -> str:
    angka = [int(k[1:]) for k in data if k[:1] == "T" and k[1:].isdigit()]
    return f"T{(max(angka) + 1) if angka else 1:02d}"


def label_tangga(tid: str | None, daftar: dict) -> str:
    if not tid:
        return "belum ada tangga"
    nama = (daftar.get(tid) or {}).get("nama", "")
    return f"{tid} · {nama}" if nama else tid


def tangga_untuk(data: dict, indeks: int | None) -> str | None:
    """ID tangga sebuah frame: tangga scene-nya bila ada, selain itu tangga rekaman."""
    sc = scene_untuk(data, indeks)
    return (sc or {}).get("tangga") or data.get("tangga") or None



def warna_tangga_baru(daftar: dict) -> str:
    """Warna palet pertama yang belum dipakai tangga lain (berulang bila palet habis)."""
    dipakai = [v.get("warna") for v in daftar.values()]
    for nama in WARNA:
        if nama not in dipakai:
            return nama
    return list(WARNA)[len(daftar) % len(WARNA)]


def warna_rekaman(data: dict, daftar_tangga: dict) -> str | None:
    """Warna stabilo rekaman: warna TANGGA-nya bila ada, selain itu stabilo rekaman sendiri."""
    t = data.get("tangga")
    if t and (daftar_tangga.get(t) or {}).get("warna"):
        return daftar_tangga[t]["warna"]
    return data.get("warna")


def warna_untuk_tangga(tid: str | None, daftar_tangga: dict) -> str | None:
    return (daftar_tangga.get(tid) or {}).get("warna") if tid else None


# ------------------------------------------------------------------ ukuran berkas
BAGIAN = {"source": "mentah", "derived": "turunan", "exports": "ekspor"}


def ukuran_rekaman(sesi: Path) -> dict:
    """Byte dan jumlah berkas per bagian rekaman.

    mentah = source/ (rekaman .db3/.bag dari kamera), turunan = derived/
    (pratinjau MP4, cache depth, indeks frame; bisa dibuat ulang dari mentah),
    ekspor = exports/ (frame dataset + label), lain = edit/, catatan.json, dst.
    Hasil: {"mentah": [byte, berkas], ..., "total": [byte, berkas], "frame_ekspor": n}.
    """
    hasil = {k: [0, 0] for k in ("mentah", "turunan", "ekspor", "lain")}

    def tambah(e, kunci):
        try:
            if e.is_dir(follow_symlinks=False):
                with os.scandir(e.path) as it:
                    for anak in it:
                        tambah(anak, kunci)
            elif e.is_file(follow_symlinks=False):
                hasil[kunci][0] += e.stat(follow_symlinks=False).st_size
                hasil[kunci][1] += 1
        except OSError:
            pass

    try:
        with os.scandir(sesi) as it:
            for e in it:
                tambah(e, BAGIAN.get(e.name, "lain"))
    except OSError:
        pass
    hasil["total"] = [sum(v[0] for v in hasil.values()), sum(v[1] for v in hasil.values())]
    fr = Path(sesi) / "exports" / "frames"
    try:
        hasil["frame_ekspor"] = sum(1 for d in os.scandir(fr) if d.name.startswith("frame_") and d.is_dir())
    except OSError:
        hasil["frame_ekspor"] = 0
    return hasil


def format_ukuran(b: float) -> str:
    """Byte -> '8,9 GB' / '867 MB' / '94 KB' (satuan desimal, sama dengan pengelola berkas Ubuntu)."""
    for satuan, f in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= f:
            v = b / f
            return (f"{v:.1f}" if v < 100 else f"{v:.0f}").replace(".", ",") + " " + satuan
    return f"{int(b)} B"


def ribuan(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def teks_ukuran(u: dict) -> str:
    """Dua baris untuk kartu rekaman."""
    t = u["total"]
    baris = [f"💾 Total {format_ukuran(t[0])} · {ribuan(t[1])} berkas"]
    rinci = [f"{nama} {format_ukuran(u[k][0])}" for k, nama in (("mentah", "mentah"), ("turunan", "turunan"),
                                                                ("ekspor", "ekspor")) if u[k][1]]
    n = u.get("frame_ekspor", 0)
    if n and u["ekspor"][1]:
        rinci[-1] += f" ({ribuan(n)} frame, ≈{format_ukuran(u['ekspor'][0] / n)}/frame)"
    if u["lain"][0] >= 1e6:                       # mis. folder hasil uji lama di dalam rekaman
        rinci.append(f"lain {format_ukuran(u['lain'][0])}")
    if rinci:
        baris.append("     " + " · ".join(rinci))
    return "\n".join(baris)
