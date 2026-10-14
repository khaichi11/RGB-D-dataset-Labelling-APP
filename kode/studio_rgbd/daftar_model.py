"""Daftar checkpoint model tangga di laptop ini, terbaru dulu, untuk pilihan di tab Uji sistem / Uji berkas.

Yang didaftar: ``aplikasi_tangga/runs/<run>/deployment_penuh.pt`` dan
``runs/<kelompok>/<run>/deployment_penuh.pt``, ditambah checkpoint rujukan lama
``bobot/final_d435``. Model lipatan cross-validation (folder ``lipat*``) tidak
didaftar: masing-masing sengaja tidak dilatih sepertiga rekaman, jadi bukan model
pakai. Waktu = waktu berkas checkpoint ditulis (akhir pelatihan; di laptop lain
waktu unduhan). Checkpoint di ``runs/`` tidak masuk Git, jadi isi daftar
tergantung laptop.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

try:
    from . import segmentasi_convnext_depth as SC
except ImportError:
    import segmentasi_convnext_depth as SC

BULAN = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")
RUJUKAN = "bobot/final_d435/cnx_atto_in1k_384.pt"
# Folder relatif runs/ -> keterangan singkat. Folder lain tetap didaftar tanpa keterangan.
KETERANGAN = {
    "cv_robust/akhir_kuat": "augmentasi kuat, rekaman ✔ + publik (CV belum tuntas)",
    "cv_publik_gabung/akhir_baru": "rekaman ✔ + publik",
    "pengusul_20261013_selesai": "rekaman ✔ saja",
    "pengusul_20261004_semua": "semua label terperiksa",
    "pengusul_20261003_uji": "latih ulang 1.303 frame",
    "eksperimen_semua_data_20260925": "eksperimen seluruh data",
}


@dataclass(frozen=True)
class Model:
    jalur: Path
    nama: str            # folder relatif runs/, mis. "cv_robust/akhir_kuat"
    waktu: dt.datetime
    keterangan: str

    def label(self, dipakai_studio: Path | None = None) -> str:
        w = self.waktu
        teks = f"{w.day:02d} {BULAN[w.month - 1]} {w.year} {w:%H:%M} · {self.nama}"
        if self.keterangan:
            teks += f" · {self.keterangan}"
        if dipakai_studio is not None and self.jalur.resolve() == dipakai_studio.resolve():
            teks += "  [dipakai Studio]"
        return teks


def daftar_model(akar: Path | None = None) -> list[Model]:
    """Semua checkpoint pakai yang ada, terbaru dulu."""
    akar = Path(akar) if akar is not None else SC.akar_aplikasi()
    runs = akar / "runs"
    calon = list(runs.glob("*/deployment_penuh.pt")) + list(runs.glob("*/*/deployment_penuh.pt"))
    hasil = []
    for p in calon:
        rel = p.parent.relative_to(runs).as_posix()
        if any(b.startswith("lipat") for b in rel.split("/")):
            continue
        hasil.append(Model(p, rel, dt.datetime.fromtimestamp(p.stat().st_mtime), KETERANGAN.get(rel, "")))
    rujukan = akar / RUJUKAN
    if rujukan.exists():
        hasil.append(Model(rujukan, "rujukan final_d435", dt.datetime.fromtimestamp(rujukan.stat().st_mtime),
                           "model lama, 5 rekaman latih"))
    return sorted(hasil, key=lambda m: m.waktu, reverse=True)


def model_studio() -> Path | None:
    """Checkpoint pengusul label Studio (satu berkas), atau None bila ensembel / tidak ada."""
    try:
        jalur = SC.pilih_bobot_usulan()[1]
    except (FileNotFoundError, OSError):
        return None
    return jalur[0] if len(jalur) == 1 else None


def label_dan_jalur(akar: Path | None = None) -> tuple[list[str], dict[str, Path], str]:
    """(label urut terbaru dulu, label -> jalur, label bawaan = model Studio atau yang terbaru)."""
    studio = model_studio()
    daftar = daftar_model(akar)
    peta = {m.label(studio): m.jalur for m in daftar}
    label = list(peta)
    bawaan = next((lb for lb, j in peta.items() if studio is not None and j.resolve() == studio.resolve()),
                  label[0] if label else "")
    return label, peta, bawaan
