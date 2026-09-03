"""Usulan label dari model RGB-D ConvNeXt yang dilatih pada proyek ini.

Menggantikan pengusul berbasis RF-DETR. Perbedaannya bukan sekadar berganti
model: pengusul lama bekerja dari citra warna lalu memakai kedalaman untuk
memverifikasi sesudahnya, sedangkan model ini menerima kedalaman sebagai KANAL
MASUKAN. Perbedaan letak itu penting karena lantai dan tapakan sama-sama bidang
mendatar bertekstur mirip; yang membedakannya adalah letak dalam ruang, dan
informasi itu hanya ada pada kedalaman.

Antarmukanya dibuat sama persis dengan `segmentasi_rfdetr_depth.usulkan`
sehingga alur penghalusan SAM 2 dan verifikasi kedalaman yang sudah ada tetap
dipakai tanpa perubahan.

Bobot yang dipakai adalah hasil pra-latih dataset publik saja, tanpa fine-tune
pada rekaman D435. Alasannya bukan teknis melainkan metodologis: memakai model
yang dilatih pada label yang sedang diperiksa untuk mengusulkan label baru
membuat model mengukuhkan kesalahannya sendiri, dan kesalahan itu menjadi makin
sulit terlihat karena usulan dan acuan berasal dari sumber yang sama.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODEL = None
_DEV = None

# Urutan pencarian bobot. Yang dipakai adalah bobot PRA-LATIH DATASET PUBLIK
# saja, bukan yang sudah di-fine-tune pada rekaman D435. Ini pilihan sadar
# pemilik proyek: label D435 masih dalam proses pemeriksaan, dan memakai model
# yang dilatih pada label itu untuk mengusulkan label baru berarti model
# mengukuhkan kesalahannya sendiri.
#
# Harga pilihan ini terukur pada rekaman 105348 yang tidak pernah dilatih:
#
#   pra-latih publik saja        Dice 0,7234
#   setelah fine-tune D435       Dice 0,8402  (0,9358 pada frame tervalidasi)
#
# Jadi usulan akan lebih kasar dan perlu lebih banyak koreksi tangan. Setelah
# pemeriksaan label selesai, tukar ke jalur ft/best.pt untuk mengembalikan
# selisih itu.
#
# Femto didahulukan karena Dice-nya tertinggi di antara bobot pra-latih
# (0,7234 melawan 0,7126 dan 0,7118). Dice yang dipakai sebagai penentu, bukan
# F1 garis, sebab pelabel membentuk poligon dari peta kelas dan tidak memakai
# keluaran kepala garis sama sekali.
KANDIDAT_BOBOT = [
    ('ConvNeXt Femto (pra-latih publik)', 'bobot/kandidat/banding4/convnext_femto/pra/best.pt'),
    ('ConvNeXt Atto ImageNet (pra-latih publik)', 'bobot/kandidat/banding5kecil/cnx_atto_in1k/pra/best.pt'),
    ('ConvNeXt Atto (pra-latih publik)', 'bobot/kandidat/banding4/convnext_atto/pra/best.pt'),
]
MIN_M, MAKS_M = 0.2, 4.0
UKURAN = 512
BG, RISER, TREAD = 0, 1, 2


def akar_proyek() -> Path:
    """Akar paket_ubuntu_zenexo, dihitung dari letak berkas ini."""
    return Path(__file__).resolve().parents[2]


def cari_bobot() -> tuple[str, Path]:
    akar = akar_proyek()
    for nama, rel in KANDIDAT_BOBOT:
        p = akar / rel
        if p.exists():
            return nama, p
    raise FileNotFoundError(
        'Bobot ConvNeXt belum tersedia. Yang dicari, berurutan:\n  '
        + '\n  '.join(rel for _, rel in KANDIDAT_BOBOT)
        + '\n\nVarian ImageNet sedang dilatih; sampai selesai, pengusul ini '
          'belum dapat dipakai.')


def _muat():
    """Muat model sekali lalu simpan; memuat ulang tiap frame terlalu lambat."""
    global _MODEL, _DEV
    if _MODEL is not None:
        return _MODEL, _DEV
    import torch
    import sys
    akar = akar_proyek() / 'kode'
    if str(akar) not in sys.path:
        sys.path.insert(0, str(akar))
    from stair_fusion_atto.model_kandidat import (StairFusionAttoKandidat,
                                                  stride_dari_checkpoint,
                                                  varian_dari_checkpoint)
    nama, jalur = cari_bobot()
    ckpt = torch.load(jalur, map_location='cpu', weights_only=False)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = StairFusionAttoKandidat(line_kernel=tuple(ckpt.get('kernel', (5, 5))),
                                varian=varian_dari_checkpoint(ckpt),
                                semantic_stride=stride_dari_checkpoint(ckpt),
                                timm_pretrained=False).to(dev).eval()
    m.load_state_dict(ckpt['model'])
    _MODEL, _DEV = (m, nama), dev
    return _MODEL, _DEV


def hangatkan() -> str:
    """Muat model lebih awal agar klik pertama tidak terasa lambat."""
    (m, nama), _ = _muat()
    return nama


def _letterbox(a: np.ndarray, interp: int) -> tuple[np.ndarray, float, int, int]:
    h, w = a.shape[:2]
    s = UKURAN / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    kecil = cv2.resize(a, (nw, nh), interpolation=interp)
    dx, dy = (UKURAN - nw) // 2, (UKURAN - nh) // 2
    out = np.zeros((UKURAN, UKURAN) + a.shape[2:], a.dtype)
    out[dy:dy + nh, dx:dx + nw] = kecil
    return out, s, dx, dy


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

    epsilon 1,5 piksel dipilih dari pengukuran pertukaran, bukan dari kebiasaan.
    Poligonisasi tidak pernah mewakili peta kelas dengan sempurna: lubang di
    dalam permukaan ikut tertutup dan batas bergerigi diluruskan. Terukur pada
    delapan frame, IoU poligon terhadap peta kelas asli adalah 0,9718 pada
    epsilon 0,5 dengan 95 titik per poligon, dan 0,9485 pada epsilon 3,0 dengan
    14 titik. Nilai 1,5 memberi 0,9640 dengan 40 titik -- memulihkan sebagian
    ketepatan yang hilang pada 2,0 tanpa membuat penyuntingan tangan berat.

    Dua penyaring dipakai bersama, karena masing-masing sendirian tidak cukup.
    Ambang mutlak 2500 piksel membuang bercak yang jelas terlalu kecil untuk
    menjadi permukaan anak tangga; terukur pada 84 frame, ambang lama 400
    piksel meloloskan 56 komponen di bawah 5000 piksel yang tampak sebagai
    bercak di dalam permukaan besar. Ambang nisbi 8% dari komponen terbesar
    sekelas menangani frame jarak jauh, tempat seluruh permukaan mengecil
    sehingga ambang mutlak saja akan membuang anak tangga yang sah.
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
            skala_depth: float = 1.0) -> dict:
    """Usulkan poligon tapakan dan bidang tegak dari citra dan kedalaman.

    rgb_bgr : citra BGR resolusi kamera
    depth   : kedalaman resolusi sama; satuan meter bila skala_depth = 1,
              atau Z16 mentah bila skala_depth diisi depth_scale kamera
    """
    import torch
    (model, nama_bobot), dev = _muat()
    h, w = rgb_bgr.shape[:2]
    dm = depth.astype(np.float32) * float(skala_depth)

    rgb_l, s, dx, dy = _letterbox(rgb_bgr, cv2.INTER_LINEAR)
    norm = np.clip((dm - MIN_M) / (MAKS_M - MIN_M), 0, 1).astype(np.float32)
    sah = (dm > 0).astype(np.float32)
    norm_l, _, _, _ = _letterbox(norm, cv2.INTER_NEAREST)
    sah_l, _, _, _ = _letterbox(sah, cv2.INTER_NEAREST)

    x_rgb = cv2.cvtColor(rgb_l, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
    x_dep = np.stack([norm_l * 2 - 1, sah_l]).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(x_rgb)[None].to(dev),
                    torch.from_numpy(x_dep)[None].to(dev))
    sem = out['semantic'].argmax(1)[0].cpu().numpy().astype(np.uint8)

    # Buka letterbox lalu kembalikan ke resolusi kamera, bukan sebaliknya:
    # menskalakan poligon setelah dibentuk pada 512 menumpuk galat pembulatan.
    nh, nw = int(round(h * s)), int(round(w * s))
    inti = sem[dy:dy + nh, dx:dx + nw]
    sem_penuh = cv2.resize(inti, (w, h), interpolation=cv2.INTER_NEAREST)

    return {
        'tapakan': _poligon(sem_penuh == TREAD),
        'bidang_tegak': _poligon(sem_penuh == RISER),
        'sumber': f'{nama_bobot} (RGB-D, kedalaman sebagai masukan model)',
        'peta_kelas': sem_penuh,
    }
