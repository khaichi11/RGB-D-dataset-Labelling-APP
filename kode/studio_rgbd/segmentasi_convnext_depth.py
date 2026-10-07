"""Usulan label otomatis dari model RGB-D ConvNeXt yang dilatih pada proyek ini.

Model menerima kedalaman sebagai KANAL MASUKAN, bukan hanya untuk memverifikasi
sesudahnya. Itu penting karena lantai dan tapakan sama-sama bidang mendatar
bertekstur mirip; yang membedakannya adalah letak dalam ruang, dan informasi itu
hanya ada pada kedalaman.

Antarmuka `usulkan` sama dengan pengusul lain di Studio. Poligon diambil
langsung dari peta kelas; SAM 2 tidak lagi dipakai (lihat
``_rekomendasi_tangga_data`` di studio_dataset_rgbd.py).

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

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

_MODEL = None
_KUNCI = threading.RLock()

# Bobot lama untuk uji_berkas.py (kode model stair_fusion_atto). Tidak dipakai pengusul label.
KANDIDAT_BOBOT = [
    ('ConvNeXt Femto (pra-latih publik)', 'bobot/kandidat/banding4/convnext_femto/pra/best.pt'),
    ('ConvNeXt Atto ImageNet (pra-latih publik)', 'bobot/kandidat/banding5kecil/cnx_atto_in1k/pra/best.pt'),
    ('ConvNeXt Atto (pra-latih publik)', 'bobot/kandidat/banding4/convnext_atto/pra/best.pt'),
]
NAMA_BOBOT_USULAN = 'ConvNeXt V2 Atto RGB-D 384'   # diganti nama checkpoint yang benar-benar dimuat
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


# Pengusul label = ENSEMBEL checkpoint (rerata softmax) yang tersedia, berurutan:
# model eksperimen seluruh data (2026-09-25), model latih ulang 2026-10-03
# (1.303 frame terperiksa, tanpa rekaman 20261001_* dan 20260927_203*), dan
# checkpoint rujukan. Diuji pada 115 frame terperiksa dari 8 rekaman yang belum
# pernah dilihat model mana pun (auto_label/uji_ensembel_pengusul.py): Dice
# rerata 0,695 (model 2026-09-25 saja) -> 0,708, median 0,915 -> 0,941; lebih
# baik pada 26% frame, lebih buruk pada 17%; jumlah poligon dan titik per
# poligon tidak bertambah. Model latih ulang SENDIRIAN memang rerata 0,717,
# tetapi lebih buruk pada 32% frame, jadi tidak dipakai sendirian. Tangga kecil
# dan jauh (20261001_054922) tetap gagal di semua varian: masih wajib dikoreksi.
# Checkpoint di runs/ tidak masuk Git; di laptop lain yang tersedia saja yang
# dipakai (minimal rujukan). STUDIO_BOBOT_USULAN (satu berkas) mengalahkan semua.
BOBOT_USULAN = [
    ('eksperimen seluruh data 2026-09-25', 'runs/eksperimen_semua_data_20260925/deployment_penuh.pt'),
    ('latih ulang 2026-10-03', 'runs/pengusul_20261003_uji/deployment_penuh.pt'),
    ('rujukan final_d435', 'bobot/final_d435/cnx_atto_in1k_384.pt'),
]
# 4 Oktober 2026: pengusul dilatih dengan SEMUA 1.574 label terperiksa (41 rekaman,
# termasuk 20261001_* dan 20260927_203*). Pada 55 frame terperiksa yang ditahan dari
# rekaman-rekaman baru, Dice riser/tread 0,905/0,899 lawan 0,671/0,591 ensembel di
# atas; frame jelek (< 0,7) 23 -> 5; menggabungkannya dengan ensembel lama justru
# menurunkan (0,890/0,857). Bila berkasnya ada, model ini dipakai SENDIRIAN.
# Bukti: Train-RGB-D-Model/bukti/final_d435/label_claude_20261004/.
BOBOT_USULAN_UTAMA = ('latih semua label 2026-10-04', 'runs/pengusul_20261004_semua/deployment_penuh.pt')


def pilih_bobot_usulan() -> tuple[str, list[Path]]:
    """Nama dan jalur checkpoint pengusul label (satu atau beberapa untuk ensembel)."""
    env = os.environ.get('STUDIO_BOBOT_USULAN', '').strip()
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(f'STUDIO_BOBOT_USULAN menunjuk berkas yang tidak ada:\n  {p}')
        return f'ConvNeXt RGB-D ({p.name}, dari STUDIO_BOBOT_USULAN)', [p]
    utama = akar_aplikasi() / BOBOT_USULAN_UTAMA[1]
    if utama.exists():
        return f'ConvNeXt V2 Atto RGB-D 384 ({BOBOT_USULAN_UTAMA[0]})', [utama]
    ada = [(nama, akar_aplikasi() / rel) for nama, rel in BOBOT_USULAN if (akar_aplikasi() / rel).exists()]
    if not ada:
        raise FileNotFoundError('Checkpoint pengusul label tidak ditemukan. Yang dicari:\n  '
                                + '\n  '.join(str(akar_aplikasi() / rel) for _, rel in BOBOT_USULAN))
    if len(ada) == 1:
        return f'ConvNeXt V2 Atto RGB-D 384 ({ada[0][0]})', [ada[0][1]]
    return (f'Ensembel {len(ada)} ConvNeXt V2 Atto RGB-D 384 (' + ', '.join(n for n, _ in ada) + ')',
            [j for _, j in ada])


def bobot_usulan() -> list[Path]:
    """Checkpoint yang dipakai pengusul label (lihat :data:`BOBOT_USULAN`)."""
    return pilih_bobot_usulan()[1]


def _muat():
    """Muat model (semua anggota ensembel) sekali lalu simpan; memuat ulang tiap frame terlalu lambat.

    Kunci mencegah pemuatan ganda: pemanasan di thread latar dan auto-label
    frame pertama di thread UI bisa memanggil fungsi ini bersamaan.
    """
    global _MODEL, NAMA_BOBOT_USULAN
    with _KUNCI:
        if _MODEL is None:
            NAMA_BOBOT_USULAN, jalur = pilih_bobot_usulan()
            akar = str(akar_aplikasi())
            if akar not in sys.path:
                sys.path.insert(0, akar)
            from rgbd_convnext.konfigurasi_utama import muat_model_utama
            _MODEL = [muat_model_utama(j) for j in jalur]
    return _MODEL


def _prediksi(mus: list, bgr: np.ndarray, meter: np.ndarray) -> np.ndarray:
    """Peta kelas dari rerata softmax semua anggota (satu anggota = prediksi biasa)."""
    from rgbd_convnext.konfigurasi_utama import prediksi_kelas, tensor_masukan, peta_kelas_dari_logit
    if len(mus) == 1:
        return prediksi_kelas(mus[0], bgr, meter)
    import torch
    with torch.inference_mode():
        prob = None
        for mu in mus:
            r, d = tensor_masukan(mu, bgr, meter)
            p = mu.model(r, d)['semantic'].float().softmax(1)
            prob = p if prob is None else prob + p
    return peta_kelas_dari_logit(prob / len(mus), mus[0].ukuran, bgr.shape[1], bgr.shape[0])


def hangatkan() -> str:
    """Muat model dan jalankan satu inferensi kosong di thread pemanggil."""
    _muat()
    panaskan_utas_ini()
    return NAMA_BOBOT_USULAN


def panaskan_utas_ini() -> None:
    """Satu inferensi kosong di thread pemanggil.

    Inferensi pertama tiap thread menanggung pembuatan handle cuDNN/cuBLAS
    (~0,35 s terukur di Studio, lalu 10-17 ms per frame). Handle itu milik
    thread, jadi pemanasan di thread latar tidak menolong thread UI; Studio
    memanggil fungsi ini sekali di thread UI begitu model siap, sebelum
    auto-label frame pertama.
    """
    mus = _muat()
    with _KUNCI:
        _prediksi(mus, np.zeros((480, 848, 3), np.uint8), np.ones((480, 848), np.float32))


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


def _luruskan(kontur: np.ndarray, titik: np.ndarray, sudut_min: float = 20.0,
              geser_maks: float = 15.0) -> np.ndarray:
    """Pindahkan tiap sudut ke perpotongan dua sisi lurus di kiri-kanannya.

    Titik hasil approxPolyDP selalu jatuh pada kontur mask yang bergerigi, jadi
    sudutnya ikut meleset beberapa piksel. Tiap sisi di sini diganti garis hasil
    fit (Huber) ke titik kontur di antara dua sudutnya, lalu sudut baru adalah
    perpotongan dua garis bertetangga. Sudut yang hampir lurus (< sudut_min
    derajat) atau perpotongan yang bergeser lebih dari geser_maks piksel tetap
    memakai titik lama, karena perpotongannya tidak stabil.
    """
    idx = [int(np.argmin(((kontur - v) ** 2).sum(1))) for v in titik]
    garis = []
    for i in range(len(idx)):
        a, b = idx[i], idx[(i + 1) % len(idx)]
        seg = kontur[a:b + 1] if b >= a else np.vstack([kontur[a:], kontur[:b + 1]])
        if len(seg) < 4:
            garis.append(None)
            continue
        vx, vy, x0, y0 = cv2.fitLine(seg.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        garis.append((np.array([x0, y0]), np.array([vx, vy])))
    keluar = []
    for i, v in enumerate(titik.astype(float)):
        g1, g2 = garis[i - 1], garis[i]
        if g1 is not None and g2 is not None:
            (p1, d1), (p2, d2) = g1, g2
            if abs(d1[0] * d2[1] - d1[1] * d2[0]) > np.sin(np.radians(sudut_min)):
                t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
                q = p1 + t[0] * d1
                if np.hypot(*(q - v)) < geser_maks:
                    v = q
        keluar.append(v)
    return np.round(np.array(keluar)).astype(np.int32)


def _poligon(biner: np.ndarray, luas_min: int = 800, rasio_min: float = 0.02,
             epsilon_nisbi: float = 0.005):
    """Kontur luar tiap komponen, dengan titik sesedikit mungkin untuk disunting.

    Permukaan tangga pada citra pada dasarnya segi empat, jadi yang perlu
    disimpan hanya sudut-sudutnya. Toleransi approxPolyDP 0,5% keliling
    membuang gerigi tepi mask model, lalu :func:`_luruskan` memindahkan sudut ke
    perpotongan sisi lurus. Pada 52 frame terperiksa 211803, titik per poligon
    turun dari median 26 (toleransi tetap 1,5 px) menjadi 6 dengan Dice sama
    (0,893). Ambang mutlak 2500 piksel membuang bercak kecil; ambang nisbi 8%
    dari komponen terbesar sekelas menangani frame jarak jauh, tempat seluruh
    permukaan mengecil.
    """
    biner = _bersihkan(biner)
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    luas = [cv2.contourArea(k) for k in kontur]
    if not luas:
        return []
    ambang = max(luas_min, max(luas) * rasio_min)
    keluar = []
    for k, a in zip(kontur, luas):
        if a < ambang:
            continue
        p = cv2.approxPolyDP(k, epsilon_nisbi * cv2.arcLength(k, True), True).reshape(-1, 2)
        if len(p) >= 3:
            p = _luruskan(k.reshape(-1, 2), p)
            keluar.append(_segi_empat(p, biner.shape))
    return keluar


def _segi_empat(p: np.ndarray, bentuk, ambang: float = 0.96) -> list[tuple[int, int]]:
    """Sederhanakan ke 4 titik (atau 5-6) selama IoU dengan poligon asal >= ambang.

    Label pemakai berupa pita segi empat (6.800 poligon terperiksa: median 4 titik,
    96% <= 6 titik). Pada 55 frame uji, segi empat naik 38% -> 68% dan Dice tetap
    (0,9047 -> 0,9038). Bentuk yang memang bukan segi empat disederhanakan bertahap
    (toleransi 1-3% keliling) dengan IoU >= ambang - 0,02.
    """
    h, w = bentuk[:2]
    P = np.asarray(p, np.float32).reshape(-1, 1, 2)
    if len(P) <= 4:
        return [(int(x), int(y)) for x, y in P.reshape(-1, 2)]
    asal = np.zeros((h, w), np.uint8); cv2.fillPoly(asal, [np.round(P).astype(np.int32)], 1)
    def iou(q):
        m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 1)
        return float((m & asal).sum()) / max(float((m | asal).sum()), 1.0)
    for n in (4, 5, 6):
        if len(P) <= n:
            break
        try:
            q = cv2.approxPolyN(P, n, ensure_convex=False)
        except cv2.error:
            continue
        if len(q) >= 3 and iou(q) >= ambang:
            return [(int(round(x)), int(round(y))) for x, y in q.reshape(-1, 2)]
    terbaik = P
    keliling = cv2.arcLength(P, True)
    for e in (0.01, 0.015, 0.02, 0.03):
        q = cv2.approxPolyDP(P, e * keliling, True)
        if len(q) >= 3 and iou(q) >= ambang - 0.02:
            terbaik = q
    return [(int(round(x)), int(round(y))) for x, y in terbaik.reshape(-1, 2)]


def _potong_ke_citra(p, w: int, h: int) -> list[tuple[int, int]]:
    """Potong poligon dengan bingkai citra (Sutherland-Hodgman).

    approxPolyN dan penyatuan sudut dapat menaruh sudut di perpotongan garis di LUAR
    gambar (terukur sampai 470 px), sehingga titik tampak melayang di luar citra saat
    disunting. Titik itu diganti titik potong di tepi; area di dalam gambar tetap.
    """
    P = [tuple(map(float, q)) for q in p]
    for a, b, c in ((1, 0, 0.0), (-1, 0, -(w - 1.0)), (0, 1, 0.0), (0, -1, -(h - 1.0))):
        keluar = []
        for i in range(len(P)):
            s_, e = P[i - 1], P[i]
            ds, de = a * s_[0] + b * s_[1] - c, a * e[0] + b * e[1] - c
            if de >= 0:
                if ds < 0:
                    t = ds / (ds - de); keluar.append((s_[0] + t * (e[0] - s_[0]), s_[1] + t * (e[1] - s_[1])))
                keluar.append(e)
            elif ds >= 0:
                t = ds / (ds - de); keluar.append((s_[0] + t * (e[0] - s_[0]), s_[1] + t * (e[1] - s_[1])))
        P = keluar
        if not P:
            return []
    hasil: list[tuple[int, int]] = []
    for x, y in P:
        q = (int(round(x)), int(round(y)))
        if not hasil or q != hasil[-1]:
            hasil.append(q)
    if len(hasil) > 1 and hasil[0] == hasil[-1]:
        hasil.pop()
    return hasil if len(hasil) >= 3 else []


def satukan_sudut(tapakan: list, tegak: list, jarak: float = 10.0) -> tuple[list, list]:
    """Sudut riser dan tread yang berdekatan (< jarak px) disatukan ke titik tengahnya.

    Pita bertetangga lalu berbagi tepi yang sama, seperti label pemakai; pada 55 frame
    uji celah antar-poligon median 110 -> 15 px per frame (label pemakai 11), Dice tetap.
    """
    R = [np.asarray(p, float).copy() for p in tegak]
    T = [np.asarray(p, float).copy() for p in tapakan]
    pasangan = []
    for i, P in enumerate(R):
        for a, v in enumerate(P):
            for j, Q in enumerate(T):
                d = np.hypot(*(Q - v).T)
                b = int(np.argmin(d))
                if d[b] <= jarak:
                    pasangan.append((float(d[b]), i, a, j, b))
    pakai_r, pakai_t = set(), set()
    for _, i, a, j, b in sorted(pasangan):
        if (i, a) in pakai_r or (j, b) in pakai_t:
            continue
        m = (R[i][a] + T[j][b]) / 2
        R[i][a] = m; T[j][b] = m
        pakai_r.add((i, a)); pakai_t.add((j, b))
    def bulat(L):
        return [q for q in (_potong_ke_citra(p, 848, 480) for p in L) if q]
    return bulat(T), bulat(R)


def peta_kelas(rgb_bgr: np.ndarray, depth: np.ndarray, k: dict | None = None,
               skala_depth: float | None = None) -> np.ndarray:
    """Peta kelas per piksel (0 latar, 1 riser, 2 tread) dari model pengusul."""
    mus = _muat()
    meter = kedalaman_meter(depth, k, skala_depth)
    with _KUNCI:
        return _prediksi(mus, rgb_bgr, meter)


def model_siap() -> bool:
    """Model sudah dimuat; thread latar boleh memakainya tanpa memicu pemuatan."""
    return _MODEL is not None


def lepas() -> bool:
    """Lepaskan model dari memori GPU; dimuat ulang otomatis saat dipakai lagi. -> ada yang dilepas?

    Konteks CUDA proses (beberapa ratus MB) tetap ada sampai Studio ditutup;
    yang dikembalikan adalah bobot dan cache alokator torch.
    """
    global _MODEL
    with _KUNCI:                                 # tunggu inferensi yang sedang berjalan selesai
        if _MODEL is None:
            return False
        _MODEL = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return True


def usulkan(rgb_bgr: np.ndarray, depth: np.ndarray, k: dict | None = None,
            skala_depth: float | None = None) -> dict:
    """Usulkan poligon tapakan dan bidang tegak dari citra dan kedalaman.

    rgb_bgr : citra BGR resolusi kamera (848 × 480 pada D435)
    depth   : kedalaman selaras warna, resolusi sama; Z16 mentah atau meter
              (lihat :func:`kedalaman_meter`)
    k       : intrinsik frame dari Studio; kunci `depth_scale` dipakai bila ada
    """
    peta_kelas_ = peta_kelas(rgb_bgr, depth, k, skala_depth)
    tapakan, tegak = satukan_sudut(_poligon(peta_kelas_ == TREAD), _poligon(peta_kelas_ == RISER))
    return {
        'tapakan': tapakan,
        'bidang_tegak': tegak,
        'sumber': f'{NAMA_BOBOT_USULAN} (kedalaman sebagai masukan model)',
        'peta_kelas': peta_kelas_,
    }
