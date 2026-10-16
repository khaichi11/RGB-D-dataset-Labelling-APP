"""Tampilan depth untuk melabel: menampakkan BENTUK anak tangga, bukan hanya jarak.

Warna jarak (turbo) membuat tangga tampak sebagai gradasi halus, karena jarak
pada tangga naik bertambah pelan dari bawah ke atas. Yang membedakan tread
dan riser adalah ARAH permukaannya (tread menghadap atas, riser menghadap
kamera), dan itu terbaca dari depth tanpa bergantung pada cahaya.

Mode:
- ``bidang``: tinggi tiap piksel terhadap arah "atas" (dari normal bidang
  datar yang dominan). Bidang datar (tread, lantai, bordes) diwarnai menurut
  tingginya dengan warna berulang tiap 0,5 m, sehingga tiap anak tangga
  menjadi satu blok warna rata; permukaan tegak (riser) abu-abu berbayang;
  batas tajam antar-bidang digaris hitam.
- ``relief``: permukaan disinari cahaya buatan dari atas-depan (normal 3-D
  dari depth + intrinsik). Tread terang, riser lebih gelap, tepi anak tangga
  tegas, seperti foto yang diterangi dari atas.
- ``normal``: peta normal berwarna (tiap arah permukaan satu warna).
- ``kontras``: depth dikurangi versi halusnya; tonjolan dan tepi anak tangga
  terlihat sebagai garis terang/gelap.
- ``jarak``: warna jarak (turbo), tampilan lama.
Piksel tanpa depth (lubang, terlalu dekat/jauh) digambar hitam.
"""
from __future__ import annotations

import cv2
import numpy as np

MODE = {"bidang": "Bidang (warna per ketinggian anak tangga)", "relief": "Relief 3-D (bentuk anak tangga)",
        "normal": "Arah permukaan (warna normal)", "jarak": "Jarak (warna turbo)",
        "ir": "Inframerah selaras (frame gelap)",
        "ir_bidang": "Inframerah + warna bidang (frame gelap)"}
BUTUH_BERKAS = ("ir", "ir_bidang")        # butuh berkas IR frame, dihitung Studio (hitung_dari_folder)
# "ir" tidak dihitung di sini karena butuh berkas IR frame; lihat ir_selaras.py.
# "kontras" tetap tersedia untuk eksperimen, tetapi tidak ditawarkan di Studio:
# pada frame 211803/105840 hasilnya berderau dan tepinya kurang terbaca.
CAHAYA = np.array([0.25, -0.9, -0.35], np.float32)       # kamera: x kanan, y bawah, z maju -> dari atas-depan
CAHAYA /= np.linalg.norm(CAHAYA)


def _meter_halus(depth: np.ndarray, skala: float) -> tuple[np.ndarray, np.ndarray]:
    d = depth.astype(np.float32) * skala
    sah = (d > 0.15) & (d < 8.0)
    if not sah.any():
        return d, sah
    # Lubang kecil diisi dari tetangga agar normal tidak berderau di tepinya.
    isi = d.copy()
    isi[~sah] = 0
    bobot = cv2.GaussianBlur(sah.astype(np.float32), (0, 0), 2.0)
    rata = cv2.GaussianBlur(isi, (0, 0), 2.0) / np.maximum(bobot, 1e-3)
    d = np.where(sah, d, rata)
    d = cv2.bilateralFilter(d, 7, 0.03, 3)                  # halus di bidang, tepi anak tangga tetap tajam
    return d, sah


def _normal(d: np.ndarray, fx: float, fy: float, cx: float, cy: float, L: int = 3) -> np.ndarray:
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    P = np.dstack([(u - cx) * d / fx, (v - cy) * d / fy, d])
    n = np.cross(np.roll(P, -L, 1) - np.roll(P, L, 1), np.roll(P, -L, 0) - np.roll(P, L, 0))
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    n[(n * P).sum(2) > 0] *= -1                             # menghadap kamera
    return n


def gambar(depth: np.ndarray, mode: str, intrinsik: dict | None) -> np.ndarray:
    """Citra RGB uint8 seukuran depth untuk mode yang diminta."""
    k = intrinsik or {}
    skala = float(k.get("depth_scale", 0.001))
    if mode == "jarak" or not {"fx", "fy", "cx", "cy"} <= set(k):
        d = depth.astype(np.float32)
        sah = d > 0
        out = np.zeros(d.shape + (3,), np.uint8)
        if sah.any():
            lo, hi = np.percentile(d[sah], (3, 97))
            vis = np.clip((d - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
            out = cv2.cvtColor(cv2.applyColorMap(vis, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            out[~sah] = 0
        return out
    d, sah = _meter_halus(depth, skala)
    if mode == "kontras":
        detail = d - cv2.GaussianBlur(d, (0, 0), 9.0)
        s = np.percentile(np.abs(detail[sah]), 98) if sah.any() else 1.0
        g = np.clip(128 + detail / max(s, 1e-4) * 127, 0, 255).astype(np.uint8)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        out = np.dstack([g, g, g])
    else:
        # Derau depth D435 (~1% jarak) membuat bidang datar berbintik bila normal
        # dihitung terlalu lokal. Bilateral kedua + jangkauan beda 5 px + normal
        # dihaluskan: lantai/tread mulus, tepi anak tangga (~15-20 cm) tetap tegas.
        d = cv2.bilateralFilter(d, 9, 0.03, 7)
        n = _normal(d, k["fx"], k["fy"], k["cx"], k["cy"], L=5)
        n = cv2.GaussianBlur(n, (0, 0), 3.0)
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
        if mode == "bidang":
            out = _bidang(d, n, sah, k)
            out[~_tambal_lubang(sah)] = 0
            return out
        if mode == "normal":
            out = ((n * np.array([1, -1, -1], np.float32) * 0.5 + 0.5) * 255).astype(np.uint8)
        else:
            terang = np.clip(n @ CAHAYA, 0, 1)
            g = (40 + 215 * terang ** 0.8).astype(np.uint8)
            out = np.dstack([g, g, (g * 0.94).astype(np.uint8)])   # sedikit hangat agar tidak tertukar dengan mask biru
    out[~sah] = 0
    return out


def _tambal_lubang(mask: np.ndarray, porsi_maks: float = 0.0015) -> np.ndarray:
    """Isi lubang kecil yang TERTUTUP mask (tidak menyentuh tepi citra, <= porsi_maks luas citra).

    Untuk tampilan Bidang: bercak derau depth di tengah lantai/tread membuat
    permukaan tampak "bolong" (abu-abu atau tanpa warna). Hanya lubang kecil
    yang dikelilingi bidang sejenis yang diisi; riser tipis di antara dua tread
    tidak tertutup satu bidang sehingga tidak ikut terisi.
    """
    inv = (~mask).astype(np.uint8)
    nk, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = mask.shape
    x, y, ww, hh, luas = (st[:, i] for i in range(5))
    isi = (luas <= porsi_maks * mask.size) & (x > 0) & (y > 0) & (x + ww < w) & (y + hh < h)
    isi[0] = False
    return mask | isi[lab]


def _histeresis(kuat: np.ndarray, lemah: np.ndarray) -> np.ndarray:
    """Piksel lemah diterima bila tersambung (8-tetangga) ke komponen yang memuat piksel kuat."""
    nk, lab = cv2.connectedComponents((kuat | lemah).astype(np.uint8), connectivity=8)
    ada = np.zeros(nk, bool)
    ada[np.unique(lab[kuat])] = True
    ada[0] = False
    return ada[lab]


def _datar_tegak(kos: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Topeng bidang datar dan tegak dari kosinus normal terhadap "atas" (lihat _bidang)."""
    datar = _tambal_lubang(_histeresis(kos > 0.85, (kos > 0.75) & (d < 2.0)))
    tegak = _tambal_lubang(_histeresis((np.abs(kos) < 0.45) & ~datar, (np.abs(kos) < 0.55) & ~datar)) & ~datar
    return datar, tegak


def _bidang(d: np.ndarray, n: np.ndarray, sah: np.ndarray, k: dict, periode: float = 0.5) -> np.ndarray:
    """Warna per ketinggian untuk bidang datar; lihat docstring modul."""
    atas = np.array([0, -1, 0], np.float32)
    for ambang in (0.5, 0.9, 0.95):                        # normal bidang datar dominan, diperketat bertahap
        c = n[sah & ((n @ atas) > ambang)]
        if len(c) > 500:
            atas = c.mean(0)
            atas /= np.linalg.norm(atas)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    P = np.dstack([(u - k["cx"]) * d / k["fx"], (v - k["cy"]) * d / k["fy"], d])
    tinggi = P @ atas
    tinggi -= np.percentile(tinggi[sah], 2) if sah.any() else 0
    # Kemiringan dihaluskan sebelum diberi ambang: tanpa ini derau depth
    # membuat bercak "tidak datar" di tengah lantai/tread (tampak abu-abu).
    kos = cv2.GaussianBlur((n @ atas).astype(np.float32), (0, 0), 4.0)
    # Ambang ganda (histeresis): piksel "agak datar" (> 0,75) atau "agak tegak"
    # (< 0,55) ikut bila tersambung ke bidang yang jelas datar/tegak, sehingga
    # bercak derau di tengah permukaan tidak lagi tampak bolong. Datar lemah
    # hanya < 2 m: lebih jauh ia menelan riser tipis (salah warna riser jauh
    # 6,7% -> 14%). Uji 216 frame terperiksa (auto_label/uji_bidang_bolong.py):
    # tread/riser dekat terwarnai benar 92->96% / 89->94%, riser jauh 59->70%,
    # salah warna riser jauh tetap 6,8%.
    datar, tegak = _datar_tegak(kos, d)
    hue = ((tinggi / periode) % 1.0 * 179).astype(np.uint8)
    sat = np.where(datar, 200, np.where(tegak, 30, 80)).astype(np.uint8)
    val = (70 + 185 * np.where(datar, 1.0, np.clip(n @ CAHAYA, 0, 1))).astype(np.uint8)
    out = cv2.cvtColor(np.dstack([hue, sat, val]), cv2.COLOR_HSV2RGB)
    # Batas antar-bidang: perubahan normal yang tajam. Potongan pendek (derau)
    # dibuang agar yang tersisa garis tepi anak tangga, bukan bintik.
    gx = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
    tepi = (np.sqrt((gx ** 2 + gy ** 2).sum(2)) > 0.9).astype(np.uint8)
    nk, lab, stat, _ = cv2.connectedComponentsWithStats(tepi, 8)
    panjang = np.maximum(stat[:, cv2.CC_STAT_WIDTH], stat[:, cv2.CC_STAT_HEIGHT])
    # Garis yang hanya dikelilingi SATU jenis bidang (lingkar bercak derau di
    # tengah lantai/tread, yang kini sudah terwarnai lewat _datar_tegak) bukan
    # batas bidang: dibuang. Batas datar|tegak (tepi anak tangga) tetap.
    sekitar = np.ones((9, 9), np.uint8)
    batas = (cv2.dilate(datar.astype(np.uint8), sekitar) > 0) & (cv2.dilate(tegak.astype(np.uint8), sekitar) > 0)
    porsi_batas = np.bincount(lab.ravel(), weights=batas.ravel().astype(np.float64), minlength=nk) / np.maximum(stat[:, cv2.CC_STAT_AREA], 1)
    simpan = np.zeros(nk, bool); simpan[1:] = (panjang[1:] >= 60) & (porsi_batas[1:] >= 0.3)
    out[simpan[lab]] = (25, 25, 25)
    return out


UJUNG, PANGKAL = 1, 2
WARNA_GARIS = {UJUNG: (255, 210, 0), PANGKAL: (0, 225, 255)}     # RGB: kuning = ujung, biru muda = pangkal


def garis_tepi(depth: np.ndarray, intrinsik: dict, z_maks: float = 4.0, jangkau: int = 10,
               beda_tinggi: float = 0.015, panjang_min: int = 40, hadap_min: float = 0.4) -> np.ndarray:
    """Garis lipatan anak tangga dari BENTUK 3-D: 1 = ujung (bibir tread), 2 = pangkal riser, 0 = bukan.

    Bayangan dan noda hanya mengubah terang-gelap, tidak mengubah depth, jadi
    tidak pernah memunculkan garis di sini; penajaman (Laplace/unsharp) tidak
    bisa membedakannya karena menajamkan semua perubahan terang-gelap.

    Tepi tiap bidang datar yang berbatasan dengan bidang tegak MENGHADAP kamera
    (riser; dinding samping tidak) dinilai dari tinggi titik riser di
    seberangnya: riser lebih rendah daripada tread = ujung/bibir (lipatan
    cembung), lebih tinggi = pangkal (sudut dalam). Potongan < ``panjang_min``
    px dibuang. Uji pada 268 frame terperiksa (auto_label/uji_tepi_depth.py):
    jenis garis hampir tidak pernah tertukar (0-0,1%), jarak ke tepi label
    median 2-3 px; 56-64% piksel garis dalam 4 px dari tepi label, 7-18% muncul
    di tengah permukaan berlabel; garis tidak ada di tempat depth kosong/jauh.
    """
    k = intrinsik
    d, sah = _meter_halus(depth, float(k.get("depth_scale", 0.001)))
    d = cv2.bilateralFilter(d, 9, 0.03, 7)
    n = _normal(d, k["fx"], k["fy"], k["cx"], k["cy"], L=5)
    n = cv2.GaussianBlur(n, (0, 0), 3.0)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    atas = np.array([0, -1, 0], np.float32)
    for ambang in (0.5, 0.9, 0.95):
        c = n[sah & ((n @ atas) > ambang)]
        if len(c) > 500:
            atas = c.mean(0)
            atas /= np.linalg.norm(atas)
    kos = cv2.GaussianBlur((n @ atas).astype(np.float32), (0, 0), 2.0)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    P = np.dstack([(u - k["cx"]) * d / k["fx"], (v - k["cy"]) * d / k["fy"], d])
    tinggi = P @ atas
    dekat = sah & (d < z_maks)
    lihat = P / (np.linalg.norm(P, axis=2, keepdims=True) + 1e-9)
    datar = cv2.morphologyEx(((kos > 0.8) & dekat).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tegak = (np.abs(kos) < 0.45) & dekat & (np.abs((n * lihat).sum(2)) > hadap_min)
    tegak = cv2.morphologyEx(tegak.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    batas = np.array([w - 1, h - 1])
    mentah = np.zeros((h, w), np.uint8)
    kontur, _ = cv2.findContours(datar, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for kt in kontur:
        if len(kt) < panjang_min:
            continue
        p = kt.reshape(-1, 2).astype(np.float32)
        t = np.roll(p, -3, 0) - np.roll(p, 3, 0)
        nn = np.stack([t[:, 1], -t[:, 0]], 1)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-6
        uji = np.clip((p + nn * 3).round().astype(int), 0, batas)
        if datar[uji[:, 1], uji[:, 0]].mean() > 0.5:        # normal kontur harus mengarah ke LUAR bidang datar
            nn = -nn
        luar = np.clip((p + nn * jangkau).round().astype(int), 0, batas)
        dalam = np.clip((p - nn * 4).round().astype(int), 0, batas)
        dh = tinggi[luar[:, 1], luar[:, 0]] - tinggi[dalam[:, 1], dalam[:, 0]]
        jenis = np.where(dh < -beda_tinggi, UJUNG, np.where(dh > beda_tinggi, PANGKAL, 0))
        jenis = jenis * tegak[luar[:, 1], luar[:, 0]]
        x, y = p[:, 0].astype(int), p[:, 1].astype(int)
        for j in (UJUNG, PANGKAL):
            m = jenis == j
            if m.sum() >= 3:
                mentah[y[m], x[m]] = j
    hasil = np.zeros_like(mentah)
    for j in (UJUNG, PANGKAL):
        b = cv2.dilate((mentah == j).astype(np.uint8), np.ones((3, 3), np.uint8))
        nk, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
        simpan = np.zeros(nk, bool)
        simpan[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= panjang_min
        hasil[(mentah == j) & simpan[lab]] = j
    return hasil


def gambar_garis(rgb: np.ndarray, peta: np.ndarray | None) -> np.ndarray:
    """Gambar garis_tepi di atas citra (tebal 3 px); salinan, citra asli tidak diubah."""
    if peta is None or not peta.any():
        return rgb
    out = rgb.copy()
    for j, warna in WARNA_GARIS.items():
        out[cv2.dilate((peta == j).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = warna
    return out


def gabung_ir(bidang_rgb: np.ndarray, ir: np.ndarray, bobot_ir: float = 0.75) -> np.ndarray:
    """Warna (hue, saturasi) dari mode Bidang, terang-gelap dan tekstur dari IR selaras.

    Garis tepi hitam Bidang tidak ikut: di atas tekstur IR ia tampak seperti
    coretan, sedangkan tepi anak tangga sudah terlihat dari IR itu sendiri.
    """
    hsv = cv2.cvtColor(bidang_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    tepi = (bidang_rgb.max(2) < 40) & (bidang_rgb.sum(2) > 0)
    hsv[..., 1][tepi] = 0
    hsv[..., 2] = np.clip((1 - bobot_ir) * np.where(tepi, 150, hsv[..., 2]) + bobot_ir * ir.astype(np.float32), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    out[(bidang_rgb.sum(2) == 0) & (ir == 0)] = 0
    return out


def hitung_dari_folder(folder, depth: np.ndarray, intrinsik: dict, mode: str,
                       geser=None, ir: np.ndarray | None = None) -> np.ndarray | None:
    """Tampilan bantu satu frame, termasuk mode yang butuh berkas IR.

    ``depth`` = depth selaras APA ADANYA (belum dikoreksi rolling shutter); IR
    diproyeksikan dengan depth yang sama (``ir`` = hasil
    ir_selaras.selaraskan(koreksi=False) bila sudah ada). Hasilnya baru
    dipindah ke posisi RGB dengan model ``geser`` (ir_selaras.ukur_geser), dan
    pita tepi yang tidak punya sumber diisi piksel tepi terdekat. Dulu depth
    digeser lebih dulu dengan tepi 0, sehingga tampak garis tanpa IR/warna
    bidang di sisi gambar.
    """
    from . import ir_selaras
    if mode == "garis_3d":
        return ir_selaras.terapkan(garis_tepi(depth, intrinsik), geser, terdekat=True)
    if mode in BUTUH_BERKAS:
        if ir is None:
            ir = ir_selaras.selaraskan(folder, depth, koreksi=False)
        if ir is None:
            vis = None if mode == "ir" else gambar(depth, "bidang", intrinsik)
        else:
            vis = ir if mode == "ir" else gabung_ir(gambar(depth, "bidang", intrinsik), ir)
    else:
        vis = gambar(depth, mode, intrinsik)
    return None if vis is None else ir_selaras.terapkan(vis, geser, isi_tepi=True)


def warnai(latar_rgb: np.ndarray, bidang_rgb: np.ndarray, kekuatan: float) -> np.ndarray:
    """Oleskan warna ketinggian Bidang pada latar (RGB atau IR) tanpa menutup teksturnya.

    Hue dan saturasi dari Bidang, terang-gelap dari latar; ``kekuatan`` 0..1
    mencampur hasilnya dengan latar asli. Garis tepi hitam Bidang tidak ikut.
    """
    if kekuatan <= 0 or bidang_rgb is None:
        return latar_rgb
    hb = cv2.cvtColor(bidang_rgb, cv2.COLOR_RGB2HSV)
    hl = cv2.cvtColor(latar_rgb, cv2.COLOR_RGB2HSV)
    tepi = (bidang_rgb.max(2) < 40) & (bidang_rgb.sum(2) > 0)
    s = hb[..., 1].copy(); s[tepi] = 0
    # Latar gelap diangkat sedikit agar warna tetap terbaca pada frame gelap.
    v = np.maximum(hl[..., 2], 60).astype(np.uint8)
    warna = cv2.cvtColor(np.dstack([hb[..., 0], s, v]), cv2.COLOR_HSV2RGB)
    campur = cv2.addWeighted(latar_rgb, 1 - kekuatan, warna, kekuatan, 0)
    # Tanpa data depth (tepi hasil geser, lubang): latar apa adanya, bukan pita pucat.
    return np.where((bidang_rgb.max(2) > 0)[..., None], campur, latar_rgb)
