"""Usulan label otomatis dari model RGB-D ConvNeXt yang dilatih pada proyek ini.

Model menerima kedalaman sebagai KANAL MASUKAN, bukan hanya untuk memverifikasi
sesudahnya. Itu penting karena lantai dan tapakan sama-sama bidang mendatar
bertekstur mirip; yang membedakannya adalah letak dalam ruang, dan informasi itu
hanya ada pada kedalaman.

Antarmuka `usulkan` sama dengan pengusul lain di Studio, sehingga penghalusan
SAM 2 dan verifikasi kedalaman yang sudah ada tetap dipakai tanpa perubahan.

Perubahan 15 September 2026:

- **Model.** Pengusul kini memakai checkpoint rujukan aplikasi aktif
  (`Train-RGB-D-Model/aplikasi_tangga/bobot/final_d435/cnx_atto_in1k_384.pt`,
  ConvNeXt V2 Atto RGB-D, fine-tune D435, masukan letterbox 384) lewat
  `rgbd_convnext.konfigurasi_utama`. Sebelumnya dipakai bobot pra-latih publik
  saja (Femto 512). Peta kelas diperbesar dari logit secara bilinear, sama
  dengan renderer dan evaluasi aplikasi.
- **Satuan kedalaman.** Studio mengirim depth Z16 mentah. Versi lama
  memperlakukannya sebagai meter, sehingga setelah dinormalisasi ke 0,2–4,0 m
  semua piksel sah bernilai 1 dan kanal kedalaman praktis tidak terpakai. Kini
  Z16 dikonversi ke meter dengan `depth_scale`.

Diukur pada 15 frame rekaman 20260914_212135 dan _211803 yang sudah diperiksa
manual (Dice rerata riser dan tread, peta kelas sebelum SAM 2): pengusul lama
0,22; bobot lama dengan kedalaman benar 0,76; pengusul baru 0,80.

Catatan metodologis: model rujukan dilatih pada label D435 yang ada. Usulannya
dapat mewarisi kebiasaan label itu, jadi setiap usulan tetap wajib diperiksa
dan ditandai `diperiksa_manual` sebelum dipakai melatih.

`cari_bobot()` dan `KANDIDAT_BOBOT` dipertahankan untuk alat uji berkas
(`uji_berkas.py`), yang memuat checkpoint dengan kode model lama.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_MODEL = None

# Bobot lama untuk uji_berkas.py (kode model stair_fusion_atto). Tidak dipakai pengusul label.
KANDIDAT_BOBOT = [
    ('ConvNeXt Femto (pra-latih publik)', 'bobot/kandidat/banding4/convnext_femto/pra/best.pt'),
    ('ConvNeXt Atto ImageNet (pra-latih publik)', 'bobot/kandidat/banding5kecil/cnx_atto_in1k/pra/best.pt'),
    ('ConvNeXt Atto (pra-latih publik)', 'bobot/kandidat/banding4/convnext_atto/pra/best.pt'),
]
NAMA_BOBOT_USULAN = 'ConvNeXt V2 Atto RGB-D 384 (fine-tune D435, aplikasi_tangga)'
SKALA_DEPTH_D435 = 0.001            # meter per satuan Z16 bila frame.json tidak menyertakannya
RISER, TREAD = 1, 2


def akar_proyek() -> Path:
    """Akar paket_ubuntu_zenexo, dihitung dari letak berkas ini."""
    return Path(__file__).resolve().parents[2]


def akar_aplikasi() -> Path:
    """Folder aplikasi aktif yang memuat paket `rgbd_convnext` dan checkpoint rujukan."""
    return akar_proyek() / 'Train-RGB-D-Model' / 'aplikasi_tangga'


def cari_bobot() -> tuple[str, Path]:
    """Bobot lama untuk uji_berkas.py; bukan bobot pengusul label."""
    akar = akar_proyek()
    for nama, rel in KANDIDAT_BOBOT:
        p = akar / rel
        if p.exists():
            return nama, p
    raise FileNotFoundError('Bobot ConvNeXt lama tidak ditemukan. Yang dicari, berurutan:\n  '
                            + '\n  '.join(rel for _, rel in KANDIDAT_BOBOT))


def bobot_usulan() -> Path:
    """Checkpoint rujukan aplikasi yang dipakai pengusul label."""
    p = akar_aplikasi() / 'bobot' / 'final_d435' / 'cnx_atto_in1k_384.pt'
    if not p.exists():
        raise FileNotFoundError(f'Checkpoint pengusul label tidak ditemukan:\n  {p}')
    return p


def _muat():
    """Muat model sekali lalu simpan; memuat ulang tiap frame terlalu lambat."""
    global _MODEL
    if _MODEL is None:
        jalur = bobot_usulan()
        akar = str(akar_aplikasi())
        if akar not in sys.path:
            sys.path.insert(0, akar)
        from rgbd_convnext.konfigurasi_utama import muat_model_utama
        _MODEL = muat_model_utama(jalur)
    return _MODEL


def hangatkan() -> str:
    """Muat model lebih awal agar klik pertama tidak terasa lambat."""
    _muat()
    return NAMA_BOBOT_USULAN


def kedalaman_meter(depth: np.ndarray, k: dict | None = None, skala_depth: float | None = None) -> np.ndarray:
    """Ubah kedalaman ke meter.

    Urutan penentu skala: `skala_depth` bila diberikan, lalu `k['depth_scale']`,
    lalu anggapan Z16 D435 (0,001 m) bila larik bertipe bilangan bulat. Larik
    pecahan tanpa skala dianggap sudah dalam meter.
    """
    if skala_depth is None and k is not None and 'depth_scale' in k:
        skala_depth = float(k['depth_scale'])
    if skala_depth is None:
        skala_depth = SKALA_DEPTH_D435 if np.issubdtype(depth.dtype, np.integer) else 1.0
    return depth.astype(np.float32) * float(skala_depth)


def _bersihkan(biner: np.ndarray, kernel: int = 7) -> np.ndarray:
    """Buang bercak tipis lalu tutup lubang kecil di dalam permukaan.

    Pembukaan morfologis lebih dahulu, penutupan sesudahnya. Urutan itu penting:
    penutupan lebih dulu akan menyambungkan bercak ke permukaan besar di
    dekatnya dan justru mengabadikannya, bukan membuangnya.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    b = cv2.morphologyEx(biner.astype(np.uint8), cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)


def _poligon(biner: np.ndarray, luas_min: int = 2500, rasio_min: float = 0.08,
             epsilon: float = 1.5):
    """Kontur luar tiap komponen, disederhanakan agar mudah disunting tangan.

    epsilon 1,5 piksel menyeimbangkan ketepatan poligon dan jumlah titik yang
    harus disunting. Ambang mutlak 2500 piksel membuang bercak kecil; ambang
    nisbi 8% dari komponen terbesar sekelas menangani frame jarak jauh, tempat
    seluruh permukaan mengecil.
    """
    biner = _bersihkan(biner)
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    luas = [cv2.contourArea(k) for k in kontur]
    if not luas:
        return []
    ambang = max(luas_min, max(luas) * rasio_min)
    keluar = []
    for k, a in zip(kontur, luas):
        if a < ambang:
            continue
        k = cv2.approxPolyDP(k, epsilon, True).reshape(-1, 2)
        if len(k) >= 3:
            keluar.append([(int(x), int(y)) for x, y in k])
    return keluar


def usulkan(rgb_bgr: np.ndarray, depth: np.ndarray, k: dict | None = None,
            skala_depth: float | None = None) -> dict:
    """Usulkan poligon tapakan dan bidang tegak dari citra dan kedalaman.

    rgb_bgr : citra BGR resolusi kamera (848 × 480 pada D435)
    depth   : kedalaman selaras warna, resolusi sama; Z16 mentah atau meter
              (lihat :func:`kedalaman_meter`)
    k       : intrinsik frame dari Studio; kunci `depth_scale` dipakai bila ada
    """
    mu = _muat()
    akar = str(akar_aplikasi())
    if akar not in sys.path:
        sys.path.insert(0, akar)
    from rgbd_convnext.konfigurasi_utama import prediksi_kelas

    meter = kedalaman_meter(depth, k, skala_depth)
    peta_kelas = prediksi_kelas(mu, rgb_bgr, meter)
    return {
        'tapakan': _poligon(peta_kelas == TREAD),
        'bidang_tegak': _poligon(peta_kelas == RISER),
        'sumber': f'{NAMA_BOBOT_USULAN} (kedalaman sebagai masukan model)',
        'peta_kelas': peta_kelas,
    }
