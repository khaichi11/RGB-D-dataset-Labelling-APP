"""Kesesuaian WAKTU RGB, depth, dan IR per frame ekspor (dibaca ulang dari rekaman mentah).

Ekspor lama hanya menulis cap waktu depth ke frame.json (ekspor baru juga RGB
dan IR). Modul ini membaca ulang rekaman mentah dengan urutan frame yang SAMA
PERSIS dengan ``PembacaBag.iter_frame`` (salinan basi di awal dan frame kembar
dibuang), lalu untuk tiap frameset mencatat cap waktu dan nomor frame RGB,
depth, IR kiri, dan IR kanan. Tiap folder ekspor dicocokkan lewat cap waktu
depth (``timestamp_kamera_ms``), lalu ISINYA dibandingkan: color_raw.png,
depth_raw.png, ir_left_raw.png, dan ir_right_raw.png harus identik bit demi bit
dengan frame RGB, depth, dan IR dari frameset itu. Lolos keduanya berarti
keempat gambar di folder memang satu jepretan.

Hasil per rekaman: ``exports/sinkron_waktu.json`` (di luar folder frame, jadi
frame.json dan sinkronisasi Hugging Face tidak berubah).

Selisih waktu RGB-depth:
- <= 5 ms: serentak (D435 menyinkronkan RGB dan depth di perangkat);
- > 5 ms: RGB dan depth dari saat berbeda (frame terlewat saat merekam); saat
  kamera bergerak, isi RGB dan depth tidak lagi cocok.
IR dan depth berasal dari sensor yang sama; cap waktunya harus identik (nomor
frame IR selalu selisih tetap 1 dari depth, jadi yang dibandingkan cap waktu).

    cd kode && .venv/bin/python -m studio_rgbd.sinkron <folder rekaman/kategori atau satu rekaman>
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

AMBANG_SERENTAK_MS = 5.0
BERKAS = "sinkron_waktu.json"


def _sidik(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=12).hexdigest()


def pindai(bag: Path, lapor: Callable[[int], None] | None = None,
           cap_isi: set[float] | None = None) -> list[dict]:
    """Cap waktu tiap frameset, berurutan seperti PembacaBag.iter_frame (tanpa align, jadi cepat).

    Untuk frameset yang cap waktu depth-nya ada di ``cap_isi`` (dibulatkan 3
    desimal), sidik isi RGB/depth/IR ikut dicatat (``sidik_*``).
    """
    import pyrealsense2 as rs
    pipe, cfg = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(cfg, str(bag), repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)

    def paket(native):
        c, d = native.get_color_frame(), native.get_depth_frame()
        if not c or not d:
            return None
        hasil = {"ts_depth": float(d.get_timestamp()), "ts_rgb": float(c.get_timestamp()),
                 "fn_depth": int(d.get_frame_number()), "fn_rgb": int(c.get_frame_number())}
        isi = cap_isi is not None and round(hasil["ts_depth"], 3) in cap_isi
        if isi:
            hasil["sidik_rgb"] = _sidik(np.asanyarray(c.get_data()))
            hasil["sidik_depth"] = _sidik(np.asanyarray(d.get_data()))
        for idx in (1, 2):
            ir = native.get_infrared_frame(idx)
            if ir:
                hasil[f"ts_ir{idx}"] = float(ir.get_timestamp())
                hasil[f"fn_ir{idx}"] = int(ir.get_frame_number())
                if isi:
                    hasil[f"sidik_ir{idx}"] = _sidik(np.asanyarray(ir.get_data()))
        return hasil

    keluar: list[dict] = []
    tunda, mulai, terakhir = [], False, -1
    try:
        while True:
            try:
                native = pipe.wait_for_frames(5000)
            except RuntimeError:
                break
            p = paket(native)
            if p is None:
                continue
            nomor = p["fn_depth"]
            if mulai and nomor == terakhir:
                continue
            if not mulai:
                if tunda and nomor == tunda[-1][0]:
                    continue
                if tunda and nomor < tunda[-1][0]:
                    tunda.clear()
                tunda.append((nomor, p))
                if len(tunda) < 2:
                    continue
                mulai = True
                for nm, p0 in tunda:
                    terakhir = nm
                    keluar.append(p0)
                tunda.clear()
                continue
            terakhir = nomor
            keluar.append(p)
            if lapor and len(keluar) % 200 == 0:
                lapor(len(keluar))
        keluar.extend(p0 for _, p0 in tunda)
    finally:
        pipe.stop()
    for i, p in enumerate(keluar):
        p["i"] = i
    return keluar


BERKAS_ISI = (("rgb", "color_raw.png"), ("depth", "depth_raw.png"), ("ir1", "ir_left_raw.png"),
              ("ir2", "ir_right_raw.png"))


def periksa_sesi(sesi: Path, bag: Path, lapor: Callable[[int], None] | None = None) -> dict:
    """Cocokkan cap waktu DAN isi RGB/depth/IR ke tiap frame ekspor; tulis exports/sinkron_waktu.json."""
    import cv2
    sesi = Path(sesi)
    ekspor = []
    for d in sorted((sesi / "exports" / "frames").glob("frame_*")):
        try:
            info = json.loads((d / "frame.json").read_text())
            ekspor.append((d, info, round(float(info["timestamp_kamera_ms"]), 3)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    semua = pindai(bag, lapor, {t for _, _, t in ekspor})
    per_cap = {round(p["ts_depth"], 3): p for p in semua}
    hasil = {}
    for d, info, cap in ekspor:
        p = per_cap.get(cap)
        if p is None:
            hasil[d.name] = {"pemetaan_terverifikasi": False, "serentak": None, "ir_sama_depth": None,
                             "isi_cocok": None}
            continue
        selisih = p["ts_rgb"] - p["ts_depth"]
        isi = {}
        for k, nama in BERKAS_ISI:
            if f"sidik_{k}" not in p or not (d / nama).exists():
                continue
            g = cv2.imread(str(d / nama), cv2.IMREAD_UNCHANGED)
            isi[k] = g is not None and _sidik(g) == p[f"sidik_{k}"]
        hasil[d.name] = {
            "selisih_rgb_depth_ms": round(selisih, 3),
            "serentak": abs(selisih) <= AMBANG_SERENTAK_MS,
            # Penghitung nomor frame IR selisih tetap dari depth (terukur: selalu 1), jadi
            # yang dibandingkan cap waktunya: IR dan depth harus identik.
            "ir_sama_depth": all(abs(p.get(f"ts_ir{k}", p["ts_depth"]) - p["ts_depth"]) <= 1.0 for k in (1, 2)),
            "pemetaan_terverifikasi": True,
            "indeks_sama": p["i"] == int(info.get("index_bag", -1)),
            "isi_cocok": all(isi.values()) if isi else None,
            "isi": isi,
            **{k: v for k, v in p.items() if k != "i" and not k.startswith("sidik_")},
        }
    nilai = list(hasil.values())
    ringkas = {"frame": len(hasil),
               "pemetaan_terverifikasi": sum(bool(v["pemetaan_terverifikasi"]) for v in nilai),
               "pemetaan_gagal": sum(not v["pemetaan_terverifikasi"] for v in nilai),
               "tidak_serentak": sum(v["serentak"] is False for v in nilai),
               "ir_beda_depth": sum(v["ir_sama_depth"] is False for v in nilai),
               "isi_cocok": sum(v["isi_cocok"] is True for v in nilai),
               "isi_beda": sorted(k for k, v in hasil.items() if v["isi_cocok"] is False),
               "satu_jepretan": sum(v["serentak"] is True and v["ir_sama_depth"] is True and v["isi_cocok"] is True
                                    for v in nilai),
               "selisih_maks_ms": max((abs(v["selisih_rgb_depth_ms"]) for v in nilai if "selisih_rgb_depth_ms" in v),
                                      default=None),
               "frame_rekaman": len(semua)}
    data = {"versi": VERSI, "ringkasan": ringkas, "frame": hasil}
    (sesi / "exports").mkdir(parents=True, exist_ok=True)
    tmp = sesi / "exports" / (BERKAS + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(sesi / "exports" / BERKAS)
    return ringkas


def teks_ringkas(r: dict | None) -> str:
    """Satu baris untuk kartu rekaman."""
    if not r:
        return "⏱ Sinkron RGB–depth–IR belum diperiksa"
    if not r.get("frame"):
        return "⏱ Sinkron: tidak ada frame ekspor"
    ok = r.get("satu_jepretan", 0) == r["frame"]
    teks = (f"⏱ Sinkron RGB–depth–IR: {r.get('satu_jepretan', 0)}/{r['frame']} frame satu jepretan"
            + (f" (Δt RGB–depth maks {r['selisih_maks_ms']:.2f} ms)".replace(".", ",")
               if r.get("selisih_maks_ms") is not None else ""))
    if not ok:
        masalah = []
        for kunci, nama in (("pemetaan_gagal", "tidak ketemu di rekaman"), ("tidak_serentak", "RGB beda saat"),
                            ("ir_beda_depth", "IR beda saat"), ("isi_beda", "isi berkas beda")):
            n = len(r[kunci]) if isinstance(r.get(kunci), list) else r.get(kunci, 0)
            if n:
                masalah.append(f"{n} {nama}")
        teks += "  ⚠ " + ", ".join(masalah)
    return teks


def info_dari_frame_json(info: dict) -> str:
    """Baris info dari cap waktu yang ditulis ekspor baru (timestamp_rgb_ms, timestamp_ir_ms); '' bila tidak ada."""
    try:
        td, tr = float(info["timestamp_kamera_ms"]), float(info["timestamp_rgb_ms"])
    except (KeyError, TypeError, ValueError):
        return ""
    ir = [float(v) for v in (info.get("timestamp_ir_ms") or {}).values()]
    serentak, ir_ok = abs(tr - td) <= AMBANG_SERENTAK_MS, all(abs(v - td) <= 1.0 for v in ir)
    if serentak and ir_ok:
        return f"⏱ RGB, depth, dan IR satu jepretan (Δt RGB–depth {abs(tr - td):.3f} ms)".replace(".", ",")
    return "⚠ Sinkron: " + "; ".join(([f"RGB dan depth beda {abs(tr - td):.1f} ms".replace(".", ",")]
                                       if not serentak else []) + ([] if ir_ok else ["IR beda saat dengan depth"]))


def info_frame(data: dict, nama_frame: str) -> str:
    """Baris info tab Label untuk satu frame dari hasil periksa_sesi; '' bila belum diperiksa."""
    v = (data or {}).get("frame", {}).get(nama_frame)
    if not v:
        return ""
    if not v.get("pemetaan_terverifikasi"):
        return "⚠ Sinkron: frame ini tidak ditemukan saat rekaman mentah dibaca ulang"
    if v.get("serentak") and v.get("ir_sama_depth") and v.get("isi_cocok") is not False:
        return (f"⏱ RGB, depth, dan IR satu jepretan (Δt RGB–depth {abs(v['selisih_rgb_depth_ms']):.3f} ms"
                .replace(".", ",") + ("; isi berkas identik dengan rekaman)" if v.get("isi_cocok") else ")"))
    masalah = []
    if not v.get("serentak"):
        masalah.append(f"RGB dan depth beda {abs(v['selisih_rgb_depth_ms']):.1f} ms".replace(".", ","))
    if not v.get("ir_sama_depth"):
        beda = max((v.get(f"ts_ir{k}", v["ts_depth"]) - v["ts_depth"] for k in (1, 2)), key=abs)
        masalah.append(f"IR dari saat lain ({beda:+.0f} ms dari depth), lapisan IR tidak dipakai")
    if v.get("isi_cocok") is False:
        masalah.append("isi berkas " + ", ".join(k for k, ok in v.get("isi", {}).items() if not ok)
                       + " beda dari rekaman")
    return "⚠ Sinkron: " + "; ".join(masalah)


_CACHE: dict = {}


def _baca_cache(sesi: Path) -> dict:
    f = Path(sesi) / "exports" / BERKAS
    try:
        mt = f.stat().st_mtime
    except OSError:
        return {}
    if _CACHE.get(f, (None,))[0] != mt:
        _CACHE[f] = (mt, baca(sesi))
    return _CACHE[f][1]


def ir_sinkron(folder: Path, info: dict | None = None) -> bool:
    """False bila IR frame ini diketahui berasal dari saat lain daripada depth.

    Terukur: frameset pertama rekaman 105348 membawa IR basi 201 ms lebih awal
    (nomor frame IR 5, depth 0); isi berkasnya cocok dengan rekaman, jadi bukan
    salah ekspor, tetapi IR itu tidak menggambarkan saat yang sama dengan RGB
    dan depth. Sumber: cap waktu di frame.json (ekspor baru) atau
    exports/sinkron_waktu.json (Cek sinkron). True bila sinkron ATAU belum diketahui.
    """
    folder = Path(folder)
    ts = (info or {}).get("timestamp_ir_ms")
    if ts and "timestamp_kamera_ms" in info:
        return all(abs(float(v) - float(info["timestamp_kamera_ms"])) <= 1.0 for v in ts.values())
    v = _baca_cache(folder.parent.parent.parent).get("frame", {}).get(folder.name) or {}
    return v.get("ir_sama_depth") is not False


VERSI = 2          # versi 1 membandingkan nomor frame IR (selalu selisih 1) sehingga semua IR dianggap beda


def baca(sesi: Path) -> dict:
    """Isi exports/sinkron_waktu.json; {} bila belum ada, rusak, atau dari versi lama yang keliru."""
    f = Path(sesi) / "exports" / BERKAS
    try:
        data = json.loads(f.read_text()) if f.exists() else {}
    except (OSError, ValueError):
        return {}
    return data if data.get("versi", 1) >= VERSI else {}


if __name__ == "__main__":
    import sys
    import time
    from .kamera_rgbd import cari_rekaman
    from .studio_dataset_rgbd import periksa_rekaman
    akar = Path(sys.argv[1])
    daftar = [akar] if (akar / "exports").is_dir() else sorted(p for p in akar.iterdir() if (p / "exports" / "frames").is_dir())
    for sesi in daftar:
        bag = cari_rekaman(sesi / "source")
        alasan = periksa_rekaman(bag) if bag else "tidak ada rekaman mentah"
        if alasan:
            print(f"{sesi.name}: dilewati ({alasan})", flush=True)
            continue
        t0 = time.time()
        r = periksa_sesi(sesi, bag)
        print(f"{sesi.name}: {teks_ringkas(r)} | frame rekaman {r['frame_rekaman']} | {time.time() - t0:.0f} s", flush=True)
