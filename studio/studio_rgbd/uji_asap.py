"""Uji asap Studio: jalankan alur utama tiap tab secara otomatis dan kumpulkan SEMUA galat.

Galat ditangkap dari tiga jalur: callback Tk (report_callback_exception),
thread latar (threading.excepthook), dan pemanggilan langsung. Dialog
(messagebox/simpledialog/filedialog) diganti agar tidak menunggu klik; isinya
ikut dicatat. Ini pelengkap, bukan pengganti, uji manual.

Pakai SALINAN dataset (tombol yang diuji menulis catatan, label, dan status):

    cd kode && .venv/bin/python -m studio_rgbd.uji_asap /tmp/studio_uji
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
import types

GALAT: list[str] = []
DIALOG: list[str] = []


def _pasang_penangkap() -> None:
    from tkinter import filedialog, messagebox, simpledialog

    def catat(jenis, nilai):
        def f(*a, **k):
            DIALOG.append(f"{jenis}: {' | '.join(str(x) for x in a[:2])}"[:300])
            return nilai
        return f
    for nama in ("showinfo", "showwarning", "showerror"):
        setattr(messagebox, nama, catat(nama, "ok"))
    for nama in ("askyesno", "askokcancel", "askyesnocancel", "askretrycancel"):
        setattr(messagebox, nama, catat(nama, False))   # jawaban "tidak": jangan pernah menghapus apa pun
    messagebox.askquestion = catat("askquestion", "no")
    simpledialog.askstring = catat("askstring", None)
    simpledialog.askinteger = catat("askinteger", None)
    filedialog.askopenfilename = catat("askopenfilename", "")
    filedialog.askdirectory = catat("askdirectory", "")
    filedialog.asksaveasfilename = catat("asksaveasfilename", "")
    threading.excepthook = lambda a: GALAT.append(
        f"thread {a.thread.name if a.thread else '?'}: "
        + "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))


def jalankan(keluar: str) -> int:
    _pasang_penangkap()
    from . import studio_dataset_rgbd as S, visual_depth
    args = types.SimpleNamespace(keluar=keluar, lebar=848, tinggi=480, fps=30, preset="jangan",
                                 batas_frame=8000, exposure=0, fps_preview=20)
    app = S.Studio(args)
    app.report_callback_exception = lambda et, ev, tb: GALAT.append(
        "callback Tk: " + "".join(traceback.format_exception(et, ev, tb)))
    app.geometry("1366x740+0+0")

    def jalan(d: float) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < d:
            app.update()
            time.sleep(0.003)

    def langkah(nama: str, f, tunggu: float = 0.25) -> None:
        t0 = time.perf_counter()
        try:
            f()
        except Exception:                                   # noqa: BLE001
            GALAT.append(f"{nama}: " + traceback.format_exc())
        jalan(tunggu)
        dt = time.perf_counter() - t0 - tunggu
        if dt > 1.5:
            print(f"  lambat: {nama} {dt:.1f} s", flush=True)

    jalan(1.5)
    # ---------------------------------------------------------------- Tinjau
    app.tabs.select(app.tab_tinjau)
    langkah("muat daftar", app.muat_daftar, 0.5)
    for i in range(app.list_sesi.size()):
        def pilih(i=i):
            app.list_sesi.selection_clear(0, "end")
            app.list_sesi.selection_set(i)
            app.pilih_sesi()
        langkah(f"pilih sesi {i} ({app.list_sesi.get(i)[:40]})", pilih, 1.2)
        langkah("cek sinkron", app.cek_sinkron, 0.3)
    # Tunggu cek sinkron yang sedang jalan (rekaman dengan sumber asli).
    t0 = time.time()
    while getattr(app, "_sinkron_jalan", None) and time.time() - t0 < 180:
        jalan(0.5)
    print("  info rekaman:", app.info_progres.cget("text").replace("\n", " / ")[:300], flush=True)
    langkah("data test (dialog dijawab otomatis)", app.toggle_data_test, 0.3)
    langkah("jendela kelompok tangga", app.buka_kelompok_tangga, 1.5)
    jk = getattr(app, "_jendela_kelompok", None)
    if jk is not None and jk.winfo_exists():
        if jk.daftar.size():
            langkah("kelompok: klik tangga", lambda: (jk.daftar.selection_set(0), jk._klik_tangga()), 0.5)
        langkah("kelompok: tutup", jk.destroy, 0.3)
    # ---------------------------------------------------------------- Label
    app.tabs.select(app.tab_label)
    jalan(0.5)
    langkah("muat frame", app.muat_frame, 1.0)
    n = len(app.frame_paths)
    print(f"  frame di tab Label: {n}", flush=True)
    for j in range(min(n, 8)):
        langkah(f"buka frame {j}", lambda j=j: app._pilih_indeks_frame(j), 0.4)
    langkah("next", lambda: app.pindah_frame_label(1), 0.4)
    langkah("prev", lambda: app.pindah_frame_label(-1), 0.4)
    # Tampilan bantu: tiap mode, IR, warna ketinggian.
    for mode, label in visual_depth.MODE.items():
        langkah(f"mode depth {mode}", lambda label=label: app.ganti_mode_depth(label), 0.6)
        if app.kanvas.rgb is not None:
            out = app.kanvas.gambar_tampil()
            if out.shape[:2] != app.kanvas.rgb.shape[:2]:
                GALAT.append(f"mode {mode}: ukuran tampilan {out.shape} != RGB {app.kanvas.rgb.shape}")
    langkah("IR on", lambda: (app.pakai_ir.set(True), app.ganti_latar_ir()), 0.6)
    langkah("garis 3-D on", lambda: (app.garis_3d.set(True), app.ganti_garis_3d()), 0.6)
    langkah("warna bidang 0,7", lambda: (app.warna_bidang.set(0.7), app.ganti_warna_bidang()), 0.6)
    for j in range(min(n, 6)):
        langkah(f"IR+bidang frame {j}", lambda j=j: app._pilih_indeks_frame(j), 0.5)
        kv = app.kanvas
        if kv.rgb is not None:
            out = kv.gambar_tampil()
            # Pita tepi tanpa data (garis "transparan"): tepi 4 px harus sama berisinya dengan bagian dalam.
            g = app._geser_rgb(app.label_path, app._cache_frame.get(app.label_path))
            ir = app._depth_vis_kini("ir")
            from . import sinkron
            if ir is None and (app.label_path / "ir_left_raw.png").exists() and sinkron.ir_sinkron(app.label_path):
                GALAT.append(f"{app.label_path.name}: berkas IR ada dan sinkron, tetapi IR selaras tidak tersedia")
            if ir is not None and ir.ndim == 2:
                tepi = min(ir[:4].min(), ir[-4:].min(), ir[:, :4].min(), ir[:, -4:].min())
                if tepi == 0 and (ir[:4] == 0).mean() > 0.5:
                    GALAT.append(f"{app.label_path.name}: pita IR kosong di tepi (geser {g})")
    langkah("zoom masuk", lambda: app.kanvas.zoom_tengah(1.6), 0.4)
    langkah("geser layar", lambda: app.kanvas.geser_layar(120, 60), 0.4)
    langkah("muat pas", app.muat_pas, 0.4)
    langkah("IR off", lambda: (app.pakai_ir.set(False), app.ganti_latar_ir()), 0.3)
    langkah("garis 3-D off", lambda: (app.garis_3d.set(False), app.ganti_garis_3d()), 0.3)
    langkah("warna bidang 0", lambda: (app.warna_bidang.set(0.0), app.ganti_warna_bidang()), 0.3)
    langkah("depth off", lambda: (app.depth_alpha.set(0.0), app.ganti_depth()), 0.3)
    langkah("usulkan segmentasi", app.usulkan_segmentasi, 6.0)
    langkah("hitung ukuran", app.hitung_ukuran, 1.0)
    langkah("tandai diperiksa", lambda: app.tandai_diperiksa(True), 0.3)
    langkah("batal diperiksa", lambda: app.tandai_diperiksa(False), 0.3)
    langkah("simpan draft", app.simpan_draft_label, 0.3)
    for j, fp in enumerate(app.frame_paths[:3]):
        langkah(f"info frame {fp.name}", lambda j=j: app._pilih_indeks_frame(j), 1.0)
        print(f"  info {fp.name}:", app.info_rekaman.cget("text").replace("\n", " / ")[:400], flush=True)
    # Sesi lain: buka beberapa frame tiap sesi (tab Label menampilkan frame sesi terpilih).
    for i in range(app.list_sesi.size()):
        def pilih(i=i):
            app.list_sesi.selection_clear(0, "end")
            app.list_sesi.selection_set(i)
            app.pilih_sesi()
            app.muat_frame()
        langkah(f"Label sesi {i}", pilih, 0.8)
        for j in (0, len(app.frame_paths) // 2, len(app.frame_paths) - 1):
            if app.frame_paths:
                langkah(f"sesi {i} frame {j}", lambda j=j: app._pilih_indeks_frame(j), 0.4)
    # ---------------------------------------------------------------- Pilah & Split
    app.tabs.select(app.tab_split)
    jalan(2.0)
    ts = getattr(app, "panel_split", None)
    if ts is not None:
        t0 = time.time()
        while getattr(ts, "_sibuk", False) and time.time() - t0 < 120:
            jalan(0.3)
        for filt in ("semua", "perlu dicek", "saran buang", "dibuang", "latar (tanpa mask)", "semua"):
            langkah(f"pilah tampilkan {filt}", lambda filt=filt: (ts.filter.set(filt), ts._isi_daftar()), 0.3)
        print(f"  pilah: {len(ts.daftar)} frame, rekaman {len(ts.rekaman)}", flush=True)
        # Waktu tiap aksi (tanpa jeda): panah dan Del harus terasa seketika.
        def ukur(nama, f, ulang=10):
            t0 = time.perf_counter()
            for _ in range(ulang):
                ts._geser_terakhir = 0.0
                f()
                app.update_idletasks()
            print(f"  waktu {nama}: {(time.perf_counter() - t0) / ulang * 1000:.0f} ms", flush=True)
        if ts.daftar:
            ukur("panah kanan", lambda: ts.geser(1))
            ukur("panah kiri", lambda: ts.geser(-1))
            awal = next((j for j, f in enumerate(ts.daftar) if ts._status(f) == "pakai"), 0)
            f_uji = ts.daftar[awal]
            ukur("Del (buang lalu pulihkan)", lambda: (ts._fokus(awal), ts.buang_pulihkan(), ts._fokus(awal),
                                                       ts.buang_pulihkan()), ulang=4)
            import json as _json
            if f_uji.sampah or _json.loads((f_uji.d / "frame_state.json").read_text()).get("di_sampah"):
                GALAT.append(f"pilah: {f_uji.k} masih dibuang setelah buang lalu pulihkan")
        langkah("pilah K label benar", lambda: (ts.filter.set("perlu dicek"), ts._isi_daftar(), ts.tandai_ok()), 0.3)
        langkah("pilah buang semua yang tampil (dijawab tidak)", ts.buang_semua_tampil, 0.3)
        langkah("pilah bagi otomatis (dijawab tidak)", ts.bagi_otomatis, 0.5)
        langkah("pilah per tangga", lambda: (ts.per_tangga.set(True), ts._isi_tree()), 0.5)
        langkah("pilah per rekaman", lambda: (ts.per_tangga.set(False), ts._isi_tree()), 0.5)
        langkah("pilah muat ulang", ts.muat_ulang, 3.0)
        langkah("pilah ringkasan CSV", ts.ringkasan_csv, 0.5)
        t0 = time.time()
        while getattr(ts, "_sibuk", False) and time.time() - t0 < 120:
            jalan(0.5)
    else:
        print("  (objek tab Pilah & Split tidak ditemukan; dilewati)")
    langkah("buka tab Rekam", lambda: app.tabs.select(app.tab_rekam), 0.8)
    for mode in ("kamera", "berkas", "kamera"):
        langkah(f"buka tab Uji ({mode})", lambda mode=mode: (app.tabs.select(app.tab_uji_halaman), app.pilih_uji(mode)), 0.8)
    langkah("pemberitahuan dalam jendela", lambda: app.kabar("uji pemberitahuan", aksi=("↶ Batalkan", lambda: None)), 0.3)
    jalan(1.0)
    try:
        app.destroy()
    except Exception:                                       # noqa: BLE001
        GALAT.append("destroy: " + traceback.format_exc())
    print(f"\nDialog yang muncul ({len(DIALOG)}):")
    for d in DIALOG:
        print("  ", d)
    print(f"\nGalat: {len(GALAT)}")
    for g in GALAT:
        print("-" * 70)
        print(g.rstrip())
    return 1 if GALAT else 0


if __name__ == "__main__":
    sys.exit(jalankan(sys.argv[1] if len(sys.argv) > 1 else "/tmp/studio_uji"))
