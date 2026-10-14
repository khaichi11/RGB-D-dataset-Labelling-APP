"""Kurasi dataset latih: buang frame yang hampir kembar atau buram, tandai yang perlu dilihat.

Peran: dipakai tab Split (jendela "Kurasi dataset") dan skrip latih (``--kurasi``).

Frame diekspor tiap ~0,33 detik dari video, jadi saat kamera diam puluhan frame
berturut-turut hampir sama: menambah jumlah tanpa menambah variasi, dan membuat
rekaman panjang mendominasi pelatihan. Kurasi ini hanya MEMILIH; berkas frame
dan label tidak diubah. Hasilnya ``<dataset>/kurasi.json``:

    {"frame": {"REKAMAN/frame_x": {"pilih": bool, "alasan": [...], "info": {...}}},
     "manual": {"REKAMAN/frame_x": bool}}   # keputusan pemakai, mengalahkan aturan

Aturan (lihat ``AturanKurasi``):
- mirip: dibanding frame TERPILIH terakhir di rekaman yang sama, gambar abu-abu
  kecil berkorelasi >= ``min_korelasi`` DAN label riser/tread hampir sama
  (IoU >= ``min_iou_label``) -> dibuang;
- buram: varians Laplacian < ``rasio_buram`` x median rekaman -> dibuang;
- perlu dilihat (tetap dipilih, keputusan di tanganmu):
  * model CV out-of-fold sangat berbeda dengan label (Dice < ``maks_dice_curiga``);
  * berlabel tangga tetapi model CV hampir tidak melihat tangga, atau mask sangat
    kecil / ada potongan mask kecil terpisah -> mungkin terlabeli tak sengaja;
  * berlabel latar tetapi model CV melihat tangga cukup luas -> mungkin tangga
    yang lupa dilabel.
Frame gelap TIDAK dibuang: gelap adalah kondisi yang harus dikuasai model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

UKURAN_KECIL = (64, 36)


@dataclass(frozen=True)
class AturanKurasi:
    min_korelasi: float = 0.92
    min_korelasi_latar: float = 0.85      # dua frame latar (tanpa label) cukup mirip gambarnya saja
    min_terang_buram: float = 25.0        # frame lebih gelap dari ini tidak dinilai buram (gelap != buram)
    min_iou_label: float = 0.85
    rasio_buram: float = 0.35
    maks_dice_curiga: float = 0.5
    min_luas_mask: float = 1.0            # % citra; mask tangga lebih kecil dari ini dicurigai tak sengaja
    min_potongan: float = 0.15            # % citra; potongan mask terpisah sekecil ini dicurigai coretan
    maks_pred_kosong: float = 0.005       # model CV memprediksi < 0,5% citra = model tidak melihat tangga
    min_pred_latar: float = 0.02          # frame latar yang diberi tangga >= 2% citra oleh model CV


def kunci(d: Path) -> str:
    return f"{d.parent.parent.parent.name}/{d.name}"


def periode(jam: int) -> str:
    return "pagi" if 5 <= jam < 10 else "siang" if 10 <= jam < 15 else "sore" if 15 <= jam < 18 else "malam"


def fitur_frame(d: Path) -> dict | None:
    bgr = cv2.imread(str(d / "color_raw.png"))
    r = cv2.imread(str(d / "mask_objek.png"), 0)
    t = cv2.imread(str(d / "mask_acuan.png"), 0)
    if bgr is None or r is None or t is None:
        return None
    abu = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kecil = cv2.resize(abu, UKURAN_KECIL, interpolation=cv2.INTER_AREA).astype(np.float32)
    kecil = (kecil - kecil.mean()) / (kecil.std() + 1e-6)
    sem = np.zeros(r.shape, np.uint8)
    sem[t > 0] = 2
    sem[r > 0] = 1
    sem_kecil = cv2.resize(sem, UKURAN_KECIL, interpolation=cv2.INTER_NEAREST)
    sedang = cv2.resize(abu, (212, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Ketajaman dari citra yang kontrasnya dinormalkan: varians Laplacian mentah juga kecil pada
    # frame gelap atau lantai polos, padahal tidak buram.
    sedang = (sedang - sedang.mean()) / (sedang.std() + 1.0)
    valid = None
    try:
        valid = float((np.load(d / "depth_aligned_to_color.npy") > 0).mean())
    except (OSError, ValueError):
        pass
    potongan = []
    for kelas in (1, 2):
        n, _, st, _ = cv2.connectedComponentsWithStats((sem == kelas).astype(np.uint8), 8)
        potongan += [float(st[i, cv2.CC_STAT_AREA]) / sem.size * 100 for i in range(1, n)]
    return {"kecil": kecil, "sem": sem_kecil, "potongan_min": min(potongan) if potongan else None, "buram": float(cv2.Laplacian(sedang, cv2.CV_32F).var()),
            "terang": float(abu.mean()), "latar": not bool(sem.any()), "depth_valid": valid,
            "riser_persen": float((sem == 1).mean() * 100), "tread_persen": float((sem == 2).mean() * 100)}


def _iou_label(a: np.ndarray, b: np.ndarray) -> float:
    nilai = []
    for k in (1, 2):
        x, y = a == k, b == k
        u = (x | y).sum()
        if u:
            nilai.append((x & y).sum() / u)
    return float(np.mean(nilai)) if nilai else 1.0


def pilih_rekaman(frames: list[Path], fitur: dict, dice_cv: dict, aturan: AturanKurasi) -> dict:
    """Keputusan per frame satu rekaman (frame berurutan menurut nama)."""
    hasil = {}
    ada = [d for d in frames if fitur.get(d) is not None]
    if not ada:
        return hasil
    median_buram = float(np.median([fitur[d]["buram"] for d in ada]))
    jam = int(frames[0].parent.parent.parent.name[-6:-4]) if frames else 12
    terakhir = None
    for d in ada:
        f = fitur[d]
        alasan, pilih = [], True
        if f["terang"] >= aturan.min_terang_buram and f["buram"] < aturan.rasio_buram * median_buram:
            alasan.append("buram"); pilih = False
        if pilih and terakhir is not None:
            g = fitur[terakhir]
            kor = float((f["kecil"] * g["kecil"]).mean())
            batas = aturan.min_korelasi_latar if (f["latar"] and g["latar"]) else aturan.min_korelasi
            if kor >= batas and _iou_label(f["sem"], g["sem"]) >= aturan.min_iou_label:
                alasan.append(f"mirip {terakhir.name}"); pilih = False
        cv = dice_cv.get(kunci(d)) or {}
        dc, pred = cv.get("dice"), cv.get("luas_pred")
        luas = f["riser_persen"] + f["tread_persen"]
        if not f["latar"]:
            if dc is not None and dc < aturan.maks_dice_curiga:
                alasan.append(f"perlu dilihat: model CV Dice {dc:.2f}")
            if pred is not None and pred < aturan.maks_pred_kosong:
                alasan.append("perlu dilihat: berlabel tangga, model CV tidak melihat tangga")
            if luas < aturan.min_luas_mask:
                alasan.append(f"perlu dilihat: mask sangat kecil ({luas:.1f}% citra), tak sengaja?")
            elif f["potongan_min"] is not None and f["potongan_min"] < aturan.min_potongan:
                alasan.append(f"perlu dilihat: potongan mask kecil terpisah ({f['potongan_min']:.2f}%)")
        elif pred is not None and pred >= aturan.min_pred_latar:
            alasan.append(f"perlu dilihat: latar, model CV melihat tangga {100 * pred:.0f}% citra")
        if pilih:
            terakhir = d
        hasil[kunci(d)] = {"pilih": pilih, "alasan": alasan,
                           "info": {"waktu": periode(jam), "terang": round(f["terang"], 1),
                                    "buram": round(f["buram"], 3), "latar": f["latar"],
                                    "depth_valid": None if f["depth_valid"] is None else round(f["depth_valid"], 3),
                                    "riser_persen": round(f["riser_persen"], 1),
                                    "tread_persen": round(f["tread_persen"], 1), "dice_cv": dc,
                                    "pred_cv_persen": None if pred is None else round(100 * pred, 1)}}
    return hasil


def baca(akar_data: Path) -> dict:
    try:
        return json.loads((Path(akar_data) / "kurasi.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"frame": {}, "manual": {}}


def tulis(akar_data: Path, data: dict) -> None:
    f = Path(akar_data) / "kurasi.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=0, ensure_ascii=False), encoding="utf-8")
    tmp.replace(f)


def dipilih(data: dict, k: str) -> bool | None:
    """Keputusan akhir satu frame: manual mengalahkan aturan; None bila belum dikurasi."""
    if k in data.get("manual", {}):
        return bool(data["manual"][k])
    v = data.get("frame", {}).get(k)
    return None if v is None else bool(v["pilih"])


def kurasi(akar_data: Path, frames: list[Path], dice_cv: dict | None = None,
           aturan: AturanKurasi = AturanKurasi(), progres=None) -> dict:
    """Hitung keputusan untuk ``frames`` (frame siap latih) lalu simpan; keputusan manual dipertahankan."""
    dice_cv = dice_cv or {}
    per_rek: dict[str, list[Path]] = {}
    for d in frames:
        per_rek.setdefault(d.parent.parent.parent.name, []).append(d)
    fitur, n = {}, 0
    for r, daftar in per_rek.items():
        for d in sorted(daftar, key=lambda p: p.name):
            fitur[d] = fitur_frame(d)
            n += 1
            if progres is not None and n % 50 == 0:
                progres(n, len(frames))
    lama = baca(akar_data)
    data = {"versi": 1, "dibuat": datetime.now().isoformat(timespec="seconds"), "aturan": asdict(aturan),
            "frame": {}, "manual": lama.get("manual", {})}
    for r, daftar in per_rek.items():
        data["frame"].update(pilih_rekaman(sorted(daftar, key=lambda p: p.name), fitur, dice_cv, aturan))
    tulis(akar_data, data)
    return data


def dice_cv_dari(berkas: Path) -> dict:
    """Hasil CV out-of-fold per frame: Dice rerata riser/tread peta kelas dan luas tangga prediksi (fraksi)."""
    try:
        d = json.loads(Path(berkas).read_text())["cv"]["baru"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return {f"{x['rekaman']}/{x['frame']}": {"dice": float(np.mean(x["peta"])) if x.get("ada_tangga") else None,
                                             "luas_pred": float(x["usulan"][2]) if len(x.get("usulan", [])) > 2 else None}
            for x in d}
