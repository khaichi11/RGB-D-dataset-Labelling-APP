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

Bobot yang dipakai adalah varian yang pelatihannya sudah selesai dan angkanya
tervalidasi pada rekaman uji, bukan checkpoint yang masih berubah. Memakai
checkpoint yang belum selesai membuat usulan label tidak dapat diulang, karena
frame yang dilabeli pada waktu berbeda akan memakai bobot berbeda tanpa
terlihat oleh pemakainya.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODEL = None
_DEV = None

# Urutan pencarian bobot. Varian dari bobot acak didahulukan karena
# pelatihannya SUDAH SELESAI dan angkanya tervalidasi pada rekaman uji:
# Dice 0,9086 dan F1 garis 0,8254 pada rekaman 105348 yang tidak pernah
# dilatih. Varian ImageNet masih dalam tahap fine-tune, dan memakai checkpoint
# yang masih berubah membuat usulan label tidak dapat diulang: frame yang
# dilabeli hari ini dan besok akan memakai bobot berbeda tanpa terlihat.
#
# Setelah fine-tune varian ImageNet selesai dan angkanya terukur, tukar urutan
# dua baris pertama di bawah ini.
KANDIDAT_BOBOT = [
    ('ConvNeXt Atto', 'bobot/kandidat/banding4/convnext_atto/ft/best.pt'),
    ('ConvNeXt Femto', 'bobot/kandidat/banding4/convnext_femto/ft/best.pt'),
    ('ConvNeXt Atto ImageNet', 'bobot/kandidat/banding5kecil/cnx_atto_in1k/ft/best.pt'),
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


def _poligon(biner: np.ndarray, luas_min: int = 400, epsilon: float = 2.0):
    """Kontur luar tiap komponen, disederhanakan agar mudah disunting tangan.

    epsilon 2 piksel dipilih supaya jumlah titik masuk akal untuk disunting;
    di bawah itu poligon berisi ratusan titik dan tidak praktis digeser satu
    per satu di kanvas.
    """
    kontur, _ = cv2.findContours(biner.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    keluar = []
    for k in kontur:
        if cv2.contourArea(k) < luas_min:
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
