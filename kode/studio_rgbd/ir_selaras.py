"""Citra inframerah D435 yang diselaraskan ke koordinat RGB, untuk melabel frame gelap.

Pada frame yang gelap total, RGB hanya berisi beberapa tingkat kecerahan
(211803 frame_002493: rerata 5 dari 255), sehingga detail tidak dapat
dipulihkan dengan pencerahan apa pun. Kamera IR kiri D435 tetap melihat
(rerata ~100) karena diterangi proyektor laser, tetapi (1) tertutup pola titik
proyektor dan (2) berada di posisi kamera yang berbeda dari RGB.

(1) Pola titik (bintik terang 2-4 px) dideteksi lewat top-hat dan hanya
    piksel titik yang ditambal (lihat bersihkan_titik); piksel lain tetap asli.
(2) Setiap piksel RGB dipetakan ke IR lewat depth selaras-RGB dan kalibrasi:
    X_ir = R (X_rgb - t), dengan R dan t dari ``extrinsics_depth_ke_rgb``
    frame.json apa adanya. Konvensi ini diperiksa terhadap ``depth_raw`` (yang
    berada di koordinat IR): 96% titik cocok dalam 5 mm, median 1,5-1,7 mm;
    konvensi transpos hanya 32-40%.
Piksel RGB tanpa depth (bayangan oklusi, permukaan gelap/mengilap, terlalu
dekat) tetap terlihat oleh kamera IR; untuk proyeksinya depth diperkirakan dari
tetangga (lihat _isi_lubang), sehingga IR selaras tampil penuh.
(3) Sisa geser akibat rolling shutter RGB saat kamera bergerak dikoreksi per
    frame (ukur_geser + terapkan), agar tepi di IR dan depth jatuh di tempat
    yang sama dengan RGB.

Sinkronisasi waktu sudah diperiksa (sinkron.py): RGB, depth, dan IR tiap frame
ekspor berasal dari frameset yang sama, cap waktu IR = depth, RGB-depth
0,018 ms. Sisa ketidakcocokan karena itu bukan salah pasang frame, melainkan
rolling shutter: baris RGB dibaca berurutan dari atas ke bawah, sedangkan IR
dan depth memotret semua baris sekaligus.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def bersihkan_titik(ir: np.ndarray, ukuran: int = 7) -> np.ndarray:
    """Buang pola titik proyektor tanpa mengaburkan piksel lain.

    Titik laser menutupi 41-47% piksel IR (211803, 105840). Opening abu-abu
    yang dulu dipakai mengubah SEMUA piksel sehingga detail ikut kabur dan
    tersisa tekstur belang. Kini titik dideteksi satu per satu lewat top-hat
    (terang lokal yang lebih kecil dari elemen ``ukuran``), hanya piksel titik
    yang ditambal dari tetangganya (inpainting), lalu derau halus diredam
    bilateral yang menjaga tepi.
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran, ukuran))
    tophat = cv2.subtract(ir, cv2.morphologyEx(ir, cv2.MORPH_OPEN, ker))
    ambang = max(12.0, float(np.percentile(tophat, 75)) * 1.5)
    titik = cv2.dilate((tophat > ambang).astype(np.uint8), np.ones((3, 3), np.uint8))
    tambal = cv2.inpaint(ir, titik, 3, cv2.INPAINT_TELEA)
    return cv2.bilateralFilter(tambal, 5, 18, 3)


def selaraskan(folder: Path, depth_selaras: np.ndarray | None = None, koreksi: bool = True) -> np.ndarray | None:
    """IR kiri (tanpa pola titik) dalam koordinat citra RGB, uint8 abu-abu; None bila tidak ada IR."""
    folder = Path(folder)
    f_ir = folder / "ir_left_raw.png"
    if not f_ir.exists():
        return None
    ir = cv2.imread(str(f_ir), cv2.IMREAD_UNCHANGED)
    if ir is None:
        return None
    if ir.dtype != np.uint8:
        ir = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    info = json.loads((folder / "frame.json").read_text())
    from .sinkron import ir_sinkron
    if not ir_sinkron(folder, info):
        return None                    # IR dari saat lain (frameset basi): menyesatkan bila ditampilkan
    kc, kd = info["intrinsics_rgb_native"], info["intrinsics_depth_native"]
    ek = info["extrinsics_depth_ke_rgb"]
    R = np.array(ek["rotation_row_major"], np.float32).reshape(3, 3)
    t = np.array(ek["translation_meter"], np.float32)
    if depth_selaras is None:
        depth_selaras = np.load(folder / "depth_aligned_to_color.npy")
    z = _isi_lubang(depth_selaras.astype(np.float32) * float(info.get("depth_scale", 0.001)))
    bersih = bersihkan_titik(ir)
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    Xc = np.dstack([(u - kc["ppx"]) * z / kc["fx"], (v - kc["ppy"]) * z / kc["fy"], z])
    Xd = (Xc - t) @ R.T                                      # = R (X_rgb - t) per piksel
    zd = np.maximum(Xd[..., 2], 1e-6)
    ud = (Xd[..., 0] / zd * kd["fx"] + kd["ppx"]).astype(np.float32)
    vd = (Xd[..., 1] / zd * kd["fy"] + kd["ppy"]).astype(np.float32)
    out = cv2.remap(bersih, ud, vd, cv2.INTER_LINEAR, borderValue=0)
    hi, wi = bersih.shape[:2]
    luar = (ud < 0) | (ud > wi - 1) | (vd < 0) | (vd > hi - 1)
    out = cv2.createCLAHE(2.0, (8, 8)).apply(out)
    # Di luar pandangan kamera IR tidak ada data: 0 (dulu CLAHE mengubah 0 menjadi
    # abu-abu gelap sehingga lubang tampak seperti bercak, bukan "tanpa data").
    out[luar] = 0
    if koreksi:
        rgb = cv2.imread(str(folder / "color_raw.png"), cv2.IMREAD_GRAYSCALE)
        if rgb is not None:
            out = terapkan(out, ukur_geser(out, rgb), isi_tepi=True)
    return out


def _isi_lubang(z: np.ndarray) -> np.ndarray:
    """Depth (meter) dengan lubang diisi rerata tertimbang tetangga, dari dekat ke jauh; HANYA untuk proyeksi IR.

    Paralaks IR kiri - RGB D435 hanya 15 mm (extrinsics), jadi depth perkiraan
    cukup: Z 1,0 m terbaca 1,3 m menggeser IR ~1,5 px; 0,5 terbaca 1,0 m ~3 px.
    """
    sah = z > 0
    if sah.all():
        return z
    if not sah.any():
        return np.full_like(z, 1.5)
    h, w = z.shape
    hasil, kurang = z.copy(), ~sah
    nilai, bobot = np.where(sah, z, 0).astype(np.float32), sah.astype(np.float32)
    for f in (1, 4, 16):
        if f > 1:
            n_ = cv2.resize(nilai, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
            b_ = cv2.resize(bobot, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        else:
            n_, b_ = nilai, bobot
        bn, bb = cv2.GaussianBlur(n_, (0, 0), 3.0), cv2.GaussianBlur(b_, (0, 0), 3.0)
        isi, ada = bn / np.maximum(bb, 1e-6), (bb > 1e-3).astype(np.uint8)
        if f > 1:
            isi = cv2.resize(isi, (w, h), interpolation=cv2.INTER_LINEAR)
            ada = cv2.resize(ada, (w, h), interpolation=cv2.INTER_NEAREST)
        pakai = kurang & (ada > 0)
        hasil[pakai] = isi[pakai]
        kurang &= ~pakai
        if not kurang.any():
            break
    hasil[kurang] = float(np.median(z[sah]))
    return hasil


def geser(citra: np.ndarray, dx: int, dy: int, terdekat: bool = False) -> np.ndarray:
    """Geser citra/depth sejauh (dx, dy) piksel; tepi yang kosong diisi 0."""
    if not dx and not dy:
        return citra
    return cv2.warpAffine(citra, np.float32([[1, 0, dx], [0, 1, dy]]), (citra.shape[1], citra.shape[0]),
                          flags=cv2.INTER_NEAREST if terdekat else cv2.INTER_LINEAR, borderValue=0)


def geser_frame(folder: Path, depth_selaras: np.ndarray | None = None,
                ir: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """Model geser depth/IR -> RGB untuk satu frame (lihat ukur_geser); NOL bila tak dapat diukur.

    Depth D435 dihitung dari kamera IR pada saat yang sama dengan IR, jadi model
    yang sama berlaku untuk depth selaras: semua lapisan dari depth (Bidang,
    relief, RANSAC/ukur) harus diterapkan model ini agar jatuh di tempat yang
    sama dengan RGB dan label. ``ir`` = hasil selaraskan(koreksi=False) bila
    sudah ada, agar tidak dihitung dua kali.
    """
    if ir is None:
        ir = selaraskan(folder, depth_selaras, koreksi=False)
    rgb = cv2.imread(str(Path(folder) / "color_raw.png"), cv2.IMREAD_GRAYSCALE)
    return ukur_geser(ir, rgb) if ir is not None and rgb is not None else NOL


def _gradien(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (m.mean() + 1e-6)


NOL = (0.0, 0.0, 0.0, 0.0)


def _rgb_untuk_cocok(rgb_abu: np.ndarray) -> np.ndarray:
    """Frame gelap: rentangkan kecerahan, CLAHE, lalu haluskan derau sensor sebelum mencari tepi."""
    if float(rgb_abu.mean()) >= 35:
        return rgb_abu
    lo, hi = np.percentile(rgb_abu, (1, 99.5))
    g = np.clip((rgb_abu.astype(np.float32) - lo) / max(float(hi - lo), 4.0), 0, 1) ** 0.5
    return cv2.GaussianBlur(cv2.createCLAHE(3.0, (8, 8)).apply((g * 255).astype(np.uint8)), (0, 0), 2.0)


def _subpiksel(kiri: float, tengah: float, kanan: float) -> float:
    d = kiri - 2 * tengah + kanan
    return 0.0 if d >= 0 else float(np.clip(0.5 * (kiri - kanan) / d, -0.5, 0.5))


def ukur_geser(ir: np.ndarray, rgb_abu: np.ndarray, ry: int = 10, rx: int = 4,
               tinggi_pita: int = 60, n_pita: int = 5) -> tuple[float, float, float, float]:
    """Model geser rolling shutter depth/IR -> RGB: (dx_atas, dy_atas, dx_bawah, dy_bawah).

    Nilai berlaku di baris pertama dan terakhir, linear di antaranya. Satu geser
    untuk seluruh citra (geser_ke_rgb, cara lama) tidak cukup: baris RGB dibaca
    dari atas ke bawah, jadi saat kamera bergerak baris bawah tertinggal lebih
    jauh dari IR daripada baris atas. Terukur pada 604 frame terang berlabel:
    geser vertikal pita atas/tengah/bawah rerata 0,7 / 2,0 / 3,9 px; menurut
    model ini baris bawah median 5 px lebih jauh daripada baris atas (bukti:
    Train-RGB-D-Model/bukti/final_d435/eksperimen_semua_data_20260925/auto_label/
    hasil_uji_geser_per_baris.txt).

    Cara ukur: peta gradien RGB dan IR (setengah resolusi) dicocokkan per pita
    mendatar (``n_pita`` pita setinggi ``tinggi_pita`` baris, saling tumpang,
    +-2*ry baris dan +-2*rx kolom, presisi subpiksel). Pita yang kecocokannya
    lemah (< 0,3) atau puncaknya di batas pencarian dibuang; garis lurus
    dipasang ke pita sisanya (berbobot kecocokan, pencilan > 3 px dibuang).
    Model ditolak (NOL) bila tidak lebih cocok daripada tanpa geser; kemiringan
    yang tidak masuk akal jatuh ke geser rata (satu nilai seluruh citra).
    Uji pada 1.083 frame berlabel: frame dengan pita yang meleset >= 3 px turun
    dari 58-62% (tanpa koreksi / cara lama) ke 6%.
    """
    if ir is None or rgb_abu is None or float(rgb_abu.mean()) < 3:   # hitam total: tak ada yang dicocokkan
        return NOL
    a = _gradien(cv2.resize(_rgb_untuk_cocok(rgb_abu), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    b = _gradien(cv2.resize(ir, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    H, W = a.shape
    hb = min(tinggi_pita, H - 2 * ry - 2)
    if hb < 16 or W <= 2 * rx + 16:
        return NOL
    titik = []                                          # (baris resolusi penuh, dx, dy, bobot)
    for yc in np.linspace(ry + hb / 2, H - ry - hb / 2, n_pita):
        t0 = int(round(yc - hb / 2))
        t1 = t0 + hb
        pola = b[t0:t1, rx:W - rx]
        if float(pola.std()) < 1e-3:
            continue
        h = cv2.matchTemplate(a[t0 - ry:t1 + ry], pola, cv2.TM_CCOEFF_NORMED)
        _, m, _, (x, y) = cv2.minMaxLoc(h)
        if m < 0.3 or x in (0, 2 * rx) or y in (0, 2 * ry):
            continue
        fx = x + _subpiksel(h[y, x - 1], h[y, x], h[y, x + 1])
        fy = y + _subpiksel(h[y - 1, x], h[y, x], h[y + 1, x])
        titik.append((t0 + t1, 2 * (fx - rx), 2 * (fy - ry), float(m)))
    if not titik:
        return NOL
    T = np.array(titik)
    Hp = 2 * H
    while True:
        yy, dx, dy, w = T.T
        rata = (np.average(dx, weights=w), np.average(dy, weights=w))
        if len(T) >= 2 and yy.max() - yy.min() >= 100:
            A = np.stack([np.ones_like(yy), yy], 1) * np.sqrt(w)[:, None]
            cx = np.linalg.lstsq(A, dx * np.sqrt(w), rcond=None)[0]
            cy = np.linalg.lstsq(A, dy * np.sqrt(w), rcond=None)[0]
        else:
            cx, cy = np.array([rata[0], 0.0]), np.array([rata[1], 0.0])
        sisa = np.hypot(cx[0] + cx[1] * yy - dx, cy[0] + cy[1] * yy - dy)
        if len(T) > 3 and sisa.max() > 3:
            T = np.delete(T, int(np.argmax(sisa)), 0)
            continue
        break
    g = (cx[0], cy[0], cx[0] + cx[1] * (Hp - 1), cy[0] + cy[1] * (Hp - 1))
    # Ekstrapolasi ke baris 0 dan terakhir dari pita yang berdekatan bisa
    # kebablasan jauh di luar jangkauan cari; itu tidak masuk akal, jadi pakai
    # geser rata.
    if (abs(g[3] - g[1]) > 4 * ry + 4 or abs(g[2] - g[0]) > 4 * rx + 4
            or max(abs(g[1]), abs(g[3])) > 2 * ry + 4 or max(abs(g[0]), abs(g[2])) > 2 * rx + 4):
        g = (rata[0], rata[1], rata[0], rata[1])
    g = tuple(round(float(v), 2) for v in g)
    if max(abs(v) for v in g) < 0.5:
        return NOL
    # Penjaga: model harus membuat IR lebih cocok dengan RGB daripada tanpa geser.
    kecil = (g[0] / 2, g[1] / 2, g[2] / 2, g[3] / 2)
    tepi = (slice(2 * ry, H - 2 * ry), slice(2 * rx, W - 2 * rx))
    ncc = lambda u, v: float(cv2.matchTemplate(u[tepi], v[tepi], cv2.TM_CCOEFF_NORMED)[0, 0])
    if ncc(a, terapkan(b, kecil)) < ncc(a, b):
        return NOL
    return g


def terapkan(citra: np.ndarray, g, terdekat: bool = False, isi_tepi: bool = False) -> np.ndarray:
    """Terapkan model geser ``g`` (ukur_geser) ke citra/depth seukuran RGB.

    ``terdekat``: tanpa interpolasi (depth: nilai Z16 tidak boleh dicampur).
    ``isi_tepi``: pita tepi yang tidak punya sumber diisi piksel tepi terdekat
    (untuk TAMPILAN; tanpa ini tampak garis/pita tanpa IR atau warna bidang di
    sisi gambar). Untuk pengukuran biarkan False: tepi bernilai 0 = tanpa data.
    """
    if g is None or len(g) != 4 or max(abs(float(v)) for v in g) < 0.25:
        return citra
    dx0, dy0, dx1, dy1 = map(float, g)
    h, w = citra.shape[:2]
    sy, sx = (dy1 - dy0) / max(h - 1, 1), (dx1 - dx0) / max(h - 1, 1)
    # Baris sumber ys masuk ke baris ys + dy(ys) di RGB; dibalik tepat untuk model linear.
    ys = (np.arange(h, dtype=np.float32) - dy0) / (1.0 + sy)
    peta_y = np.repeat(ys[:, None], w, 1).astype(np.float32)
    peta_x = (np.arange(w, dtype=np.float32)[None, :] - (dx0 + sx * ys)[:, None]).astype(np.float32)
    return cv2.remap(citra, peta_x, peta_y, cv2.INTER_NEAREST if terdekat else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE if isi_tepi else cv2.BORDER_CONSTANT, borderValue=0)


def teks_geser(g) -> str:
    """Ringkasan model geser untuk info frame; '' bila tidak ada geser berarti."""
    if g is None or len(g) != 4 or max(abs(float(v)) for v in g) < 1.0:
        return ""
    dx0, dy0, dx1, dy1 = g
    if abs(dy1 - dy0) < 1 and abs(dx1 - dx0) < 1:
        return f"{abs((dy0 + dy1) / 2):.0f} px vertikal, {abs((dx0 + dx1) / 2):.0f} px mendatar"
    return f"vertikal {abs(dy0):.0f} px di atas → {abs(dy1):.0f} px di bawah, mendatar {abs(dx0):.0f} → {abs(dx1):.0f} px"


def geser_ke_rgb(ir: np.ndarray, rgb_abu: np.ndarray, ry: int = 10, rx: int = 4) -> tuple[int, int]:
    """CARA LAMA (pembanding di bukti): satu geser (dx, dy) untuk seluruh citra; pakai ukur_geser.

    Sensor RGB D435 membaca baris demi baris (rolling shutter) sedangkan IR
    memotret sekaligus; saat kamera naik-turun waktu berjalan, tepi anak tangga
    di IR bisa bergeser beberapa piksel dari RGB (diukur: 12% frame >= 3 px
    vertikal, maks. 8 px, selalu ke atas). Pergeseran dicari dengan mencocokkan
    peta gradien (matchTemplate, setengah resolusi, +-2*ry baris dan +-2*rx kolom)
    dan hanya dipakai bila kecocokannya jelas. Frame gelap dicerahkan dulu;
    frame hitam total tidak dikoreksi.
    """
    terang = float(rgb_abu.mean())
    if terang < 3:                                   # hitam total: tidak ada yang bisa dicocokkan
        return 0, 0
    rgb_abu = _rgb_untuk_cocok(rgb_abu)          # frame gelap dicerahkan dulu (diaudit: 25% juga bergeser >= 3 px)
    a = _gradien(cv2.resize(rgb_abu, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    b = _gradien(cv2.resize(ir, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    pola = b[ry:-ry, rx:-rx]
    hasil = cv2.matchTemplate(a, pola, cv2.TM_CCOEFF_NORMED)
    _, maks, _, (x, y) = cv2.minMaxLoc(hasil)
    nol = float(hasil[ry, rx])
    if maks < 0.3 or maks - nol < 0.02:
        return 0, 0
    dx, dy = 2 * (x - rx), 2 * (y - ry)
    # Geser 2 px (satu langkah di setengah resolusi) sebagian besar derau
    # pengukuran: terhadap label, frame terang membaik 58% (>=2 px) lawan 63%
    # (>=4 px); frame gelap 100% pada keduanya. Hanya geser >= 4 px yang dipakai.
    return (dx, dy) if max(abs(dx), abs(dy)) >= 4 else (0, 0)
