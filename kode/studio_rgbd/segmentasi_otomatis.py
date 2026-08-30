"""Usulan segmentasi tapakan/bidang tegak dari depth, untuk mempercepat pelabelan.

Kenapa ini ada: satu frame tangga berisi sekitar lima anak tangga, dan tiap
anak tangga perlu dua poligon (tapakan datar + bidang tegak). Menggambar
sepuluh poligon per frame dengan klik-per-klik untuk ribuan frame tidak
masuk akal. Padahal informasinya sudah ada di depth: tapakan dan bidang
tegak berbeda ARAH NORMAL permukaannya, dan itu bisa dihitung.

Yang dihasilkan modul ini adalah USULAN, bukan kebenaran. Pemakainya tetap
memeriksa dan mengoreksi di kanvas label. Karena itu ambangnya sengaja
longgar: lebih baik mengusulkan sedikit berlebih lalu dihapus, daripada
melewatkan anak tangga dan harus digambar dari nol.

Alur:
  1. depth -> awan titik 3-D memakai intrinsics
  2. normal per piksel dari beda tetangga
  3. cari arah "atas" - arah normal yang paling dominan di adegan; pada
     tangga, tapakan selalu menang jumlah
  4. piksel dengan normal sejajar "atas"     -> tapakan
     piksel dengan normal tegak lurus "atas" -> bidang tegak
  5. pisahkan jadi komponen tersambung, buang yang terlalu kecil
  6. urutkan tapakan berdasarkan tinggi terhadap tapakan terbawah, sehingga
     nomor anak tangga keluar sendiri tanpa perlu dilabeli tangan
"""
from __future__ import annotations

import cv2
import numpy as np

Z_MIN, Z_MAX = 0.25, 4.0
# Ambang sudut. Longgar karena depth D435 berderau dan tapakan sering miring
# sedikit terhadap kamera.
SUDUT_DATAR = 30.0          # normal <= 30 deg dari "atas"  -> permukaan datar
SUDUT_TEGAK = 60.0          # normal >= 60 deg dari "atas"  -> bidang tegak
LUAS_MIN = 1200             # piksel; komponen lebih kecil dianggap derau
# Dua tapakan yang bedanya di bawah ini dianggap SATU anak tangga yang pecah.
# Riser tangga bangunan praktis tidak pernah di bawah 5 cm.
BEDA_ANAK_TANGGA_MIN = 5.0  # cm
EPS_POLIGON = 0.012         # penyederhanaan kontur, relatif terhadap kelilingnya


def awan_titik(depth: np.ndarray, k: dict) -> np.ndarray:
    """depth Z16 -> peta titik 3-D (H,W,3) dalam meter. Titik tak sah = NaN."""
    z = depth.astype(np.float32) * float(k["depth_scale"])
    z[(z < Z_MIN) | (z > Z_MAX)] = np.nan
    h, w = z.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return np.dstack([(xs - k["cx"]) * z / k["fx"],
                      (ys - k["cy"]) * z / k["fy"], z])


def normal_permukaan(P: np.ndarray, langkah: int = 4) -> np.ndarray:
    """Normal satuan per piksel dari beda tetangga. Titik tak sah = NaN.

    langkah dibuat >1 supaya bedanya diambil dari tetangga yang cukup jauh;
    pada jarak 1-2 m, beda antar piksel bersebelahan lebih kecil daripada
    derau depth itu sendiri, dan normalnya jadi acak.
    """
    n = np.full_like(P, np.nan)
    du = P[:, 2 * langkah:] - P[:, :-2 * langkah]
    dv = P[2 * langkah:, :] - P[:-2 * langkah, :]
    du = du[langkah:-langkah, :]
    dv = dv[:, langkah:-langkah]
    v = np.cross(du, dv)
    panjang = np.linalg.norm(v, axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = v / panjang
    v[panjang[:, :, 0] < 1e-9] = np.nan
    n[langkah:-langkah, langkah:-langkah] = v
    return n


def arah_atas(normal: np.ndarray, ulang: int = 4) -> np.ndarray:
    """Arah normal paling dominan di adegan; pada tangga itu arah tapakan.

    Dimulai dari tebakan -Y kamera (kamera dipegang kurang lebih tegak), lalu
    dirata-ratakan ulang hanya atas normal yang dekat dengan tebakan itu.
    Beberapa iterasi cukup untuk mengunci ke arah tapakan yang sebenarnya,
    tanpa perlu tahu orientasi kamera.
    """
    n = normal.reshape(-1, 3)
    n = n[np.isfinite(n).all(axis=1)]
    if len(n) < 100:
        return np.array([0., -1., 0.])
    # Normal boleh menghadap dua arah; samakan tandanya dulu.
    atas = np.array([0., -1., 0.])
    for _ in range(ulang):
        d = n @ atas
        m = n * np.sign(d)[:, None]
        dekat = m[np.abs(d) > np.cos(np.deg2rad(45))]
        if len(dekat) < 100:
            break
        baru = dekat.mean(axis=0)
        panjang = np.linalg.norm(baru)
        if panjang < 1e-9:
            break
        atas = baru / panjang
    return atas


def _poligon(masker: np.ndarray, luas_min: int) -> list[list[tuple[int, int]]]:
    """Komponen tersambung -> daftar poligon sederhana."""
    masker = cv2.morphologyEx(masker.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((5, 5), np.uint8))
    masker = cv2.morphologyEx(masker, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    jumlah, label = cv2.connectedComponents(masker)
    hasil = []
    for i in range(1, jumlah):
        komponen = (label == i).astype(np.uint8)
        if int(komponen.sum()) < luas_min:
            continue
        kontur, _ = cv2.findContours(komponen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not kontur:
            continue
        c = max(kontur, key=cv2.contourArea)
        eps = EPS_POLIGON * cv2.arcLength(c, True)
        p = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(p) >= 3:
            hasil.append([(int(x), int(y)) for x, y in p])
    return hasil


def usulkan(depth: np.ndarray, k: dict, luas_min: int = LUAS_MIN) -> dict:
    """-> {"tapakan": [poligon...], "bidang_tegak": [poligon...], ...info}

    k butuh kunci: depth_scale, fx, fy, cx, cy.
    """
    P = awan_titik(depth, k)
    n = normal_permukaan(P)
    atas = arah_atas(n)
    kos = np.abs(np.einsum("hwc,c->hw", n, atas))
    sah = np.isfinite(kos)
    datar = sah & (kos >= np.cos(np.deg2rad(SUDUT_DATAR)))
    tegak = sah & (kos <= np.cos(np.deg2rad(SUDUT_TEGAK)))

    poli_datar = _poligon(datar, luas_min)
    poli_tegak = _poligon(tegak, luas_min)

    # Nomor anak tangga = urutan tinggi tapakan terhadap tapakan TERBAWAH.
    # Dihitung, bukan dilabeli tangan.
    tinggi = []
    for poly in poli_datar:
        m = np.zeros(depth.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
        titik = P[(m > 0) & np.isfinite(P).all(axis=2)]
        tinggi.append(float(np.median(titik @ atas)) if len(titik) else np.nan)
    # Satu tapakan kerap pecah jadi beberapa komponen - terhalang pegangan
    # tangga, atau lubang depth di tengahnya. Kalau tiap pecahan dihitung
    # sebagai anak tangga sendiri, nomornya melenceng: pernah terukur beda
    # tinggi [0.2, 12.7, 12.1, 2.0, ...] yang jelas 0,2 dan 2,0 cm itu pecahan,
    # bukan anak tangga baru. Karena itu penomoran dikelompokkan per TINGGI,
    # bukan per komponen. Poligonnya sendiri tetap dipisah - untuk YOLO-seg,
    # dua pecahan yang tidak bersambung memang dua instance.
    urut = np.argsort([t if np.isfinite(t) else 1e9 for t in tinggi])
    dasar = tinggi[urut[0]] if len(urut) and np.isfinite(tinggi[urut[0]]) else 0.0
    tangga, nomor, acuan_tinggi = [], -1, None
    for i in urut:
        t = tinggi[i]
        if not np.isfinite(t):
            tangga.append({"poligon": poli_datar[i], "urutan": None,
                           "tinggi_dari_terbawah_cm": None}); continue
        if acuan_tinggi is None or (t - acuan_tinggi) * 100 > BEDA_ANAK_TANGGA_MIN:
            nomor += 1; acuan_tinggi = t
        tangga.append({"poligon": poli_datar[i], "urutan": nomor,
                       "tinggi_dari_terbawah_cm": (t - dasar) * 100})
    return {"tapakan": poli_datar, "bidang_tegak": poli_tegak,
            "tapakan_terurut": tangga, "arah_atas": atas.tolist(),
            # Mask orientasi dipakai tahap fusi YOLO+depth. Tidak dipakai
            # langsung sebagai label karena normal depth bisa berderau.
            "mask_datar": datar.astype(np.uint8), "mask_tegak": tegak.astype(np.uint8),
            "piksel_datar": int(datar.sum()), "piksel_tegak": int(tegak.sum()),
            "piksel_sah": int(sah.sum())}
