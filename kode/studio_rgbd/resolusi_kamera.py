"""Pembukaan aliran RealSense yang tahan banting, dipakai skrip 01/02/04/05.

Dua masalah nyata yang ditangani modul ini, keduanya ditemukan lewat pengujian
pada D435 yang tersambung di USB 2:

1. 848x480@30 DITOLAK  ("Couldn't resolve requests"). Bandwidth USB 2 tidak
   cukup. Modul ini menyediakan --lebar/--tinggi/--fps dan, kalau ditolak,
   mencetak daftar profil yang benar-benar tersedia.

2. Preset "High Accuracy" MEMBUAT FRAME BERHENTI DATANG. Diuji berdampingan:
   640x480@30 tanpa preset -> frame datang < 4 detik.
   640x480@30 dengan High Accuracy -> tidak ada frame dalam 15 detik.
   Karena itu mode bawaan "auto" mencoba High Accuracy dulu, lalu MENYETEL
   BALIK ke preset Default kalau frame tidak datang. Menyetel balik, bukan
   sekadar berhenti menyentuh - kamera tetap memegang preset sebelumnya.
   Preset yang akhirnya terpakai selalu dicatat di meta.json.

Yang TIDAK dilakukan: menurunkan resolusi diam-diam. Ukuran citra memengaruhi
jumlah titik untuk RANSAC dan lebar bidang pandang, jadi penurunannya harus
keputusan sadar Anda dan tercatat di laporan.
"""
from __future__ import annotations

import sys

BAWAAN = (848, 480, 30)
PRESET_PILIHAN = ("auto", "akurasi", "bawaan", "jangan")


def tambah_argumen(ap) -> None:
    ap.add_argument("--lebar", type=int, default=BAWAAN[0],
                    help=f"lebar citra depth & warna (bawaan {BAWAAN[0]})")
    ap.add_argument("--tinggi", type=int, default=BAWAAN[1],
                    help=f"tinggi citra (bawaan {BAWAAN[1]})")
    ap.add_argument("--fps", type=int, default=BAWAAN[2],
                    help=f"frame per detik (bawaan {BAWAAN[2]})")
    ap.add_argument("--preset", default="auto", choices=PRESET_PILIHAN,
                    help="auto: coba High Accuracy, mundur sendiri kalau frame "
                         "tidak datang. akurasi: paksa High Accuracy. "
                         "bawaan: paksa preset Default. jangan: jangan sentuh preset.")
    ap.add_argument("--batas-frame", type=int, default=8000,
                    help="berapa milidetik menunggu satu frame sebelum menyerah")


def _profil(rs, dev, aliran, fmt) -> list[tuple[int, int, int]]:
    hasil = set()
    for sen in dev.query_sensors():
        for p in sen.get_stream_profiles():
            if p.stream_type() != aliran or p.format() != fmt:
                continue
            try:
                v = p.as_video_stream_profile()
            except Exception:                                   # noqa: BLE001
                continue
            hasil.add((v.width(), v.height(), p.fps()))
    return sorted(hasil, reverse=True)


def profil_depth(rs, dev) -> list[tuple[int, int, int]]:
    return _profil(rs, dev, rs.stream.depth, rs.format.z16)


def profil_warna(rs, dev) -> list[tuple[int, int, int]]:
    return _profil(rs, dev, rs.stream.color, rs.format.bgr8)


def pilih_warna(rs, lebar: int, tinggi: int, fps: int) -> tuple[int, int, int]:
    """Ukuran warna yang paling dekat dengan ukuran depth, DAN benar-benar ada.

    Sensor depth dan sensor RGB punya daftar resolusi yang BERBEDA. 480x270
    ada di depth tapi tidak ada di RGB, sehingga meminta keduanya 480x270
    ditolak mentah-mentah dengan "Couldn't resolve requests". Dulu modul ini
    memaksa keduanya sama - itu keliru.
    """
    try:
        dev = list(rs.context().query_devices())[0]
        tersedia = profil_warna(rs, dev)
    except Exception:                                           # noqa: BLE001
        return lebar, tinggi, fps
    if not tersedia:
        return lebar, tinggi, fps
    if (lebar, tinggi, fps) in tersedia:
        return lebar, tinggi, fps
    sama_fps = [t for t in tersedia if t[2] == fps] or tersedia
    # paling dekat luasnya, lalu paling dekat nisbah lebar:tinggi
    target_luas = lebar * tinggi
    target_nisbah = lebar / max(1, tinggi)
    pilih = min(sama_fps, key=lambda t: (abs(t[0] * t[1] - target_luas),
                                         abs(t[0] / max(1, t[1]) - target_nisbah)))
    return pilih


def jenis_usb(rs, dev) -> str | None:
    try:
        return dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:                                           # noqa: BLE001
        return None


def _sensor_depth(rs):
    try:
        return list(rs.context().query_devices())[0].first_depth_sensor()
    except Exception:                                           # noqa: BLE001
        return None


def terapkan_preset(rs, mode: str) -> str:
    """Setel visual_preset SEBELUM aliran dinyalakan. -> nama preset terpakai."""
    if mode == "jangan":
        return "tidak disentuh"
    ds = _sensor_depth(rs)
    if ds is None:
        return "tidak disentuh (sensor tak terbaca)"
    cari = "High Accuracy" if mode == "akurasi" else "Default"
    terpakai = "tidak diketahui"
    try:
        if ds.supports(rs.option.visual_preset):
            r = ds.get_option_range(rs.option.visual_preset)
            for i in range(int(r.min), int(r.max) + 1):
                nama = ds.get_option_value_description(rs.option.visual_preset, i)
                if cari in nama:
                    ds.set_option(rs.option.visual_preset, i)
                    terpakai = nama
                    break
        if ds.supports(rs.option.emitter_enabled):
            ds.set_option(rs.option.emitter_enabled, 1)
    except Exception as e:                                      # noqa: BLE001
        print(f"  (preset '{cari}' tidak bisa disetel: {e})")
        terpakai = "gagal disetel"
    return terpakai


def _jelaskan_tolakan(rs, lebar, tinggi, fps, e) -> None:
    print(f"\n  GAGAL menyalakan {lebar}x{tinggi}@{fps}: {e}\n")
    try:
        dev = list(rs.context().query_devices())[0]
    except Exception:                                           # noqa: BLE001
        sys.exit("  Kamera tidak terbaca sama sekali. "
                 "Jalankan: python cek_usb_realsense.py")
    ut = jenis_usb(rs, dev)
    if ut:
        print(f"  Kamera tersambung sebagai USB {ut}.")
        if ut.startswith("2"):
            print("  Pada USB 2.x, D435 tidak punya bandwidth untuk 848x480@30.")
            print("  Sebabnya hampir selalu kabel: kabel USB-C biasa sering")
            print("  tidak memasang jalur SuperSpeed sama sekali.")
    prof = profil_depth(rs, dev)
    if prof:
        print("\n  Profil depth yang tersedia pada sambungan ini:")
        for w, h, f in prof[:12]:
            print(f"      --lebar {w} --tinggi {h} --fps {f}")
    print("\n  Untuk menelusuri lebih lengkap:  python cek_usb_realsense.py")


def buka(rs, lebar: int, tinggi: int, fps: int, preset: str = "auto",
         hias_cfg=None, hangat: int = 30, batas_ms: int = 8000,
         infrared: bool = False):
    """Nyalakan depth+color dan pastikan frame benar-benar datang.

    hias_cfg: fungsi opsional untuk menambah pengaturan pada config sebelum
              start, mis. enable_record_to_file pada skrip perekam.

    infrared: bila True, ikut aktifkan IR kiri (index 1) dan IR kanan
    (index 2). Ini diperlukan saat ingin mengarsipkan stream native D435
    secara lengkap, termasuk dalam rekaman .bag.

    -> (pipe, profile, nama_preset_terpakai)
    """
    # Mundur harus MENYETEL BALIK ke Default, bukan sekadar "jangan disentuh".
    # Kamera tetap memegang preset dari percobaan sebelumnya, jadi kalau High
    # Accuracy yang membuat frame berhenti, tidak menyentuhnya sama sekali
    # berarti kamera masih dalam keadaan rusak itu juga di percobaan kedua.
    urutan = ["akurasi", "bawaan"] if preset == "auto" else [preset]
    for ke, mode in enumerate(urutan):
        terpakai = terapkan_preset(rs, mode)
        wl, wt, wf = pilih_warna(rs, lebar, tinggi, fps)
        if (wl, wt, wf) != (lebar, tinggi, fps) and ke == 0:
            print(f"  Sensor RGB tidak punya {lebar}x{tinggi}@{fps}; "
                  f"warna dipakai {wl}x{wt}@{wf}.")
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.depth, lebar, tinggi, rs.format.z16, fps)
        # Urutan API librealsense: width, height, FORMAT, fps.
        # Menaruh fps sebelum format membuat config gagal dibangun pada D435.
        cfg.enable_stream(rs.stream.color, wl, wt, rs.format.bgr8, wf)
        if infrared:
            cfg.enable_stream(rs.stream.infrared, 1, lebar, tinggi, rs.format.y8, fps)
            cfg.enable_stream(rs.stream.infrared, 2, lebar, tinggi, rs.format.y8, fps)
        if hias_cfg is not None:
            hias_cfg(cfg)
        try:
            profile = pipe.start(cfg)
        except RuntimeError as e:
            _jelaskan_tolakan(rs, lebar, tinggi, fps, e)
            sys.exit(1)
        n_frame = 0
        try:
            for _ in range(max(1, hangat)):     # auto-exposure perlu waktu stabil
                pipe.wait_for_frames(batas_ms)
                n_frame += 1
            if ke > 0:
                print(f"  Frame datang setelah preset dimundurkan "
                      f"-> preset terpakai: {terpakai}")
            return pipe, profile, terpakai
        except RuntimeError as e:
            try:
                pipe.stop()
            except Exception:                                   # noqa: BLE001
                pass
            if ke + 1 < len(urutan):
                print(f"\n  Preset '{terpakai}' dinyalakan tapi FRAME TIDAK DATANG"
                      f" dalam {batas_ms/1000:.0f} detik.")
                print("  Menyetel kamera kembali ke preset Default, lalu mengulang...")
                continue
            print(f"\n  Frame BERHENTI setelah {n_frame} frame "
                  f"(batas tunggu {batas_ms/1000:.0f} detik): {e}")
            if n_frame == 0:
                print("  Aliran menyala tapi TIDAK SATU frame pun sampai.")
            else:
                print(f"  {n_frame} frame sempat datang lalu terhenti - ini pola")
                print("  bandwidth USB 2 yang tidak sanggup menopang laju itu.")
            if fps > 15:
                print(f"\n  Coba fps lebih rendah:")
                print(f"      --lebar {lebar} --tinggi {tinggi} --fps 15")
                print(f"      --lebar 480 --tinggi 270 --fps 15")
            print("  Pastikan tidak ada aplikasi lain yang memegang kamera")
            print("  (realsense-viewer, Cheese, browser).")
            print("  Cabut-colok kamera lalu ulangi.")
            sys.exit(1)
    raise RuntimeError("tidak tercapai")


def catat(meta: dict, lebar: int, tinggi: int, fps: int,
          rs=None, dev=None, preset: str | None = None) -> None:
    """Simpan kondisi pengambilan ke meta, supaya laporan tahu apa adanya."""
    meta["lebar_diminta"] = lebar
    meta["tinggi_diminta"] = tinggi
    meta["fps_diminta"] = fps
    meta["di_bawah_rencana"] = (lebar, tinggi, fps) != BAWAAN
    if preset is not None:
        meta["preset_terpakai"] = preset
    if rs is not None and dev is not None:
        meta["usb_type"] = jenis_usb(rs, dev)


def peringatan(lebar: int, tinggi: int, fps: int) -> None:
    if (lebar, tinggi, fps) != BAWAAN:
        print(f"\n  !! Berjalan pada {lebar}x{tinggi}@{fps}, BUKAN "
              f"{BAWAAN[0]}x{BAWAAN[1]}@{BAWAAN[2]} seperti rancangan.")
        print("     Bidang pandang lebih sempit dan titik untuk RANSAC lebih sedikit.")
        print("     Catat ini di laporan; jangan diam-diam dibandingkan dengan")
        print("     hasil yang diambil pada resolusi penuh.\n")
