"""Pembukaan aliran RealSense yang tahan banting, dipakai skrip 01/02/04/05.

Empat masalah nyata yang ditangani modul ini, semuanya ditemukan lewat
pengujian pada D435 yang tersambung di USB 2:

1. 848x480@30 DITOLAK  ("Couldn't resolve requests"). Bandwidth USB 2 tidak
   cukup. Modul ini menyediakan --lebar/--tinggi/--fps dan, kalau ditolak,
   menyertakan daftar profil yang benar-benar tersedia di dalam galatnya.

2. Preset "High Accuracy" MEMBUAT FRAME BERHENTI DATANG. Diuji berdampingan:
   640x480@30 tanpa preset -> frame datang < 4 detik.
   640x480@30 dengan High Accuracy -> tidak ada frame dalam 15 detik.
   Karena itu mode bawaan "auto" mencoba High Accuracy dulu, lalu MENYETEL
   BALIK ke preset Default kalau frame tidak datang. Menyetel balik, bukan
   sekadar berhenti menyentuh - kamera tetap memegang preset sebelumnya.
   Preset yang akhirnya terpakai selalu dicatat di meta.json.

3. IR kiri+kanan MENAMBAH BEBAN BANDWIDTH hampir dua kali lipat. Pada USB 2
   depth+color saja masih lolos tapi ditambah IR langsung ditolak. Modul ini
   sekarang MUNDUR SENDIRI tanpa IR kalau konfigurasi ber-IR ditolak, dan
   mencatat kemunduran itu supaya rekaman tidak diam-diam kehilangan stream.

4. GALAT SESAAT saat kamera baru dicolok ("Failed to set power state",
   "Device or resource busy"). Percobaan kedua beberapa detik kemudian
   hampir selalu berhasil, jadi galat semacam ini diulang, bukan diserahkan
   ke pengguna sebagai kegagalan.

Yang TIDAK dilakukan: menurunkan resolusi diam-diam. Ukuran citra memengaruhi
jumlah titik untuk RANSAC dan lebar bidang pandang, jadi penurunannya harus
keputusan sadar Anda dan tercatat di laporan.

Catatan penting soal galat: fungsi di modul ini TIDAK PERNAH memanggil
sys.exit(). Dulu memanggilnya, dan itu bug serius untuk aplikasi Tk: SystemExit
turunan BaseException, jadi lolos dari "except Exception" milik pemanggil.
Di Studio hal itu membunuh thread pembuka kamera tanpa suara - tombol tinggal
"Menyalakan kamera..." selamanya dan tidak ada satu pun pesan galat muncul.
Sekarang kegagalan dilempar sebagai KameraGagal, dan str(e) sudah berisi
penjelasan lengkap yang siap ditampilkan di dialog.
"""
from __future__ import annotations

import time

BAWAAN = (848, 480, 30)
PRESET_PILIHAN = ("auto", "akurasi", "high_accuracy", "high_density",
                  "bawaan", "jangan")

# Nama preset librealsense yang sebenarnya dicari di daftar perangkat.
# Tanpa peta ini, "--preset high_accuracy" diam-diam memasang Default:
# pemeriksaannya dulu hanya `mode == "akurasi"`, sehingga setiap nama lain
# jatuh ke Default tanpa memberi tahu siapa pun.
ALIAS_PRESET = {
    "akurasi": "High Accuracy",
    "high_accuracy": "High Accuracy",
    "high_density": "High Density",
    "kepadatan": "High Density",
    "bawaan": "Default",
    "default": "Default",
}

# Galat yang hilang sendiri kalau dicoba lagi beberapa detik kemudian.
SEMENTARA = ("failed to set power state", "device or resource busy",
             "resource temporarily unavailable", "no data received",
             "xioctl", "error accessing")
# Galat yang berarti kombinasi aliran ini memang tidak bisa dipenuhi.
DITOLAK = ("couldn't resolve requests", "not supported", "invalid value",
           "unsupported", "resolve")
# Kamera yang BARU dicolok butuh beberapa detik sebelum uvc melepasnya.
# Diukur pada D435 di USB 3.2: percobaan ke-3..ke-6 yang akhirnya berhasil,
# kira-kira 9 detik setelah pencolokan. Jeda menaik supaya kasus itu lolos
# TANPA harus menurunkan konfigurasi.
PERCOBAAN_SEMENTARA = 4
JEDA_ULANG = (1.5, 3.0, 4.5)


class KameraGagal(RuntimeError):
    """Kamera tidak bisa dibuka. str(e) sudah berupa penjelasan siap tampil."""


def tambah_argumen(ap) -> None:
    ap.add_argument("--lebar", type=int, default=BAWAAN[0],
                    help=f"lebar citra depth & warna (bawaan {BAWAAN[0]})")
    ap.add_argument("--tinggi", type=int, default=BAWAAN[1],
                    help=f"tinggi citra (bawaan {BAWAAN[1]})")
    ap.add_argument("--fps", type=int, default=BAWAAN[2],
                    help=f"frame per detik (bawaan {BAWAAN[2]})")
    ap.add_argument("--preset", default="auto", choices=PRESET_PILIHAN,
                    help="auto: coba High Accuracy, mundur sendiri kalau frame "
                         "tidak datang. akurasi/high_accuracy: paksa High Accuracy. "
                         "high_density: paksa High Density. "
                         "bawaan: paksa preset Default. jangan: jangan sentuh preset.")
    ap.add_argument("--batas-frame", type=int, default=8000,
                    help="berapa milidetik menunggu satu frame sebelum menyerah")


# --------------------------------------------------------------------------
# Konteks tunggal
# --------------------------------------------------------------------------
# rs.context() yang dibuat lalu dibuang MELEPAS perangkatnya saat dikumpulkan
# sampah. Dulu modul ini membuat konteks baru di terapkan_preset(), sekali lagi
# di pilih_warna(), dan pipeline membuat konteks ketiga sendiri. Menyetel
# visual_preset lewat sensor milik konteks yang sesaat kemudian mati membuat
# D435 kerap masuk keadaan setengah-terbuka, dan pipe.start() berikutnya
# menjawab "Failed to set power state". Satu konteks dipakai bersama-sama.
_KONTEKS = None


def konteks(rs):
    """Konteks librealsense tunggal untuk seluruh proses."""
    global _KONTEKS
    if _KONTEKS is None:
        _KONTEKS = rs.context()
    return _KONTEKS


def perangkat(rs) -> list:
    """Daftar perangkat RealSense yang terbaca. Tidak pernah melempar."""
    try:
        return list(konteks(rs).query_devices())
    except Exception:                                           # noqa: BLE001
        return []


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
    dev = perangkat(rs)
    if not dev:
        return lebar, tinggi, fps
    try:
        tersedia = profil_warna(rs, dev[0])
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
    dev = perangkat(rs)
    if not dev:
        return None
    try:
        return dev[0].first_depth_sensor()
    except Exception:                                           # noqa: BLE001
        return None


def terapkan_preset(rs, mode: str) -> str:
    """Setel visual_preset SEBELUM aliran dinyalakan. -> nama preset terpakai."""
    if mode == "jangan":
        return "tidak disentuh"
    ds = _sensor_depth(rs)
    if ds is None:
        return "tidak disentuh (sensor tak terbaca)"
    cari = ALIAS_PRESET.get(mode)
    if cari is None:
        # Lebih baik berisik daripada memasang preset yang tidak diminta.
        print(f"  (preset '{mode}' tidak dikenal; memakai Default)")
        cari = "Default"
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


# --------------------------------------------------------------------------
# Penjelasan kegagalan
# --------------------------------------------------------------------------
def _pesan_tanpa_perangkat() -> str:
    return "\n".join([
        "Tidak ada kamera RealSense yang terbaca sama sekali.",
        "",
        "librealsense mengembalikan daftar perangkat KOSONG, jadi D435 belum",
        "sampai ke sistem - ini bukan soal resolusi, preset, atau kode.",
        "",
        "Periksa berurutan:",
        "  1. Kabel. Pakai kabel bawaan D435. Banyak kabel USB-C hanya untuk",
        "     mengisi daya atau hanya USB 2.0, dan dari luar tampak sama.",
        "  2. Colok LANGSUNG ke komputer, jangan lewat hub atau dock.",
        "  3. Coba port USB-A biru (bertanda SS) kalau ada.",
        "  4. Pastikan LED kecil di dekat lensa menyala.",
        "  5. Tutup aplikasi lain yang memegang kamera (realsense-viewer, Cheese).",
        "",
        "Telusuri lebih lengkap:  python alat/cek_usb_realsense.py",
    ])


def _diagnosa_tolakan(rs, lebar, tinggi, fps, e) -> str:
    """Susun penjelasan lengkap kenapa konfigurasi ini ditolak. -> teks."""
    baris = [f"GAGAL menyalakan {lebar}x{tinggi}@{fps}: {e}", ""]
    dev = perangkat(rs)
    if not dev:
        baris.append("Kamera tidak terbaca lagi saat galat diperiksa -")
        baris.append("kemungkinan sambungan terputus di tengah jalan.")
        baris.append("")
        baris.append("Telusuri:  python alat/cek_usb_realsense.py")
        return "\n".join(baris)
    ut = jenis_usb(rs, dev[0])
    if ut:
        baris.append(f"Kamera tersambung sebagai USB {ut}.")
        if ut.startswith("2"):
            baris.append("Pada USB 2.x, D435 tidak punya bandwidth untuk 848x480@30.")
            baris.append("Sebabnya hampir selalu kabel: kabel USB-C biasa sering")
            baris.append("tidak memasang jalur SuperSpeed sama sekali.")
        baris.append("")
    prof = profil_depth(rs, dev[0])
    if prof:
        baris.append("Profil depth yang tersedia pada sambungan ini:")
        for w, h, f in prof[:12]:
            baris.append(f"    --lebar {w} --tinggi {h} --fps {f}")
        baris.append("")
    baris.append("Telusuri lebih lengkap:  python alat/cek_usb_realsense.py")
    return "\n".join(baris)


def _pesan_sementara(pesan_asli: str) -> str:
    return "\n".join([
        "Kamera terbaca tapi menolak dibuka: perangkat masih dipegang pihak lain.",
        "",
        f"Pesan asli: {pesan_asli.splitlines()[0]}",
        "",
        f"Sudah dicoba ulang {PERCOBAAN_SEMENTARA} kali dan tetap sama.",
        "",
        "Konfigurasi TIDAK diturunkan diam-diam - lebih baik gagal terang-terangan",
        "daripada merekam tanpa IR tanpa Anda sadari.",
        "",
        "Yang biasanya menyelesaikan:",
        "  1. Tutup aplikasi lain yang memegang kamera (realsense-viewer, Cheese,",
        "     browser, atau Studio yang masih terbuka di jendela lain).",
        "  2. Kalau kamera baru saja dicolok, tunggu ~10 detik lalu coba lagi.",
        "  3. Cabut-colok kamera, lalu coba lagi.",
    ])


def _diagnosa_frame_mati(rs, lebar, tinggi, fps, n_frame, batas_ms, e) -> str:
    baris = [f"Frame BERHENTI setelah {n_frame} frame "
             f"(batas tunggu {batas_ms/1000:.0f} detik): {e}", ""]
    if n_frame == 0:
        baris.append("Aliran menyala tapi TIDAK SATU frame pun sampai.")
    else:
        baris.append(f"{n_frame} frame sempat datang lalu terhenti - ini pola")
        baris.append("bandwidth USB 2 yang tidak sanggup menopang laju itu.")
    baris.append("")
    if fps > 15:
        baris.append("Coba fps lebih rendah:")
        baris.append(f"    --lebar {lebar} --tinggi {tinggi} --fps 15")
        baris.append("    --lebar 480 --tinggi 270 --fps 15")
        baris.append("")
    baris.append("Pastikan tidak ada aplikasi lain yang memegang kamera")
    baris.append("(realsense-viewer, Cheese, browser).")
    baris.append("Cabut-colok kamera lalu ulangi.")
    return "\n".join(baris)


def _cocok(pesan: str, pola) -> bool:
    rendah = pesan.lower()
    return any(p in rendah for p in pola)


def _nyalakan(rs, lebar, tinggi, fps, wl, wt, wf, ir, hias_cfg):
    """Bangun config lalu start. -> (pipe, profile). Melempar RuntimeError asli."""
    # Pipeline memakai konteks bersama supaya perangkat yang sudah dienumerasi
    # tidak dilepas lalu diambil ulang di tengah proses.
    pipe, cfg = rs.pipeline(konteks(rs)), rs.config()
    cfg.enable_stream(rs.stream.depth, lebar, tinggi, rs.format.z16, fps)
    # Urutan API librealsense: width, height, FORMAT, fps.
    # Menaruh fps sebelum format membuat config gagal dibangun pada D435.
    cfg.enable_stream(rs.stream.color, wl, wt, rs.format.bgr8, wf)
    if ir:
        cfg.enable_stream(rs.stream.infrared, 1, lebar, tinggi, rs.format.y8, fps)
        cfg.enable_stream(rs.stream.infrared, 2, lebar, tinggi, rs.format.y8, fps)
    if hias_cfg is not None:
        hias_cfg(cfg)
    return pipe, pipe.start(cfg)


def buka(rs, lebar: int, tinggi: int, fps: int, preset: str = "auto",
         hias_cfg=None, hangat: int = 30, batas_ms: int = 8000,
         infrared: bool = False, catatan: dict | None = None):
    """Nyalakan depth+color dan pastikan frame benar-benar datang.

    hias_cfg: fungsi opsional untuk menambah pengaturan pada config sebelum
              start, mis. enable_record_to_file pada skrip perekam.

    infrared: bila True, ikut aktifkan IR kiri (index 1) dan IR kanan
    (index 2). Ini diperlukan saat ingin mengarsipkan stream native D435
    secara lengkap, termasuk dalam rekaman .bag. Kalau bandwidth tidak
    sanggup, IR dilepas sendiri dan catatan["infrared"] menjadi False.

    catatan: dict opsional yang diisi kondisi pembukaan sebenarnya
             (preset, infrared, ukuran warna, jumlah percobaan).

    -> (pipe, profile, nama_preset_terpakai)
    Melempar KameraGagal - TIDAK memanggil sys.exit.
    """
    if catatan is None:
        catatan = {}

    # Diperiksa lebih dulu supaya "kamera tidak dicolok" tidak menyamar
    # sebagai "resolusi ditolak" dan mengirim orang mengejar setelan yang salah.
    if not perangkat(rs):
        raise KameraGagal(_pesan_tanpa_perangkat())

    # Mundur harus MENYETEL BALIK ke Default, bukan sekadar "jangan disentuh".
    # Kamera tetap memegang preset dari percobaan sebelumnya, jadi kalau High
    # Accuracy yang membuat frame berhenti, tidak menyentuhnya sama sekali
    # berarti kamera masih dalam keadaan rusak itu juga di percobaan kedua.
    urutan = ["akurasi", "bawaan"] if preset == "auto" else [preset]
    variasi_ir = [True, False] if infrared else [False]
    rencana = [(m, ir) for m in urutan for ir in variasi_ir]

    total_percobaan = 0
    galat_akhir = f"{lebar}x{tinggi}@{fps} tidak dapat dinyalakan."
    for ke, (mode, ir) in enumerate(rencana):
        terpakai = terapkan_preset(rs, mode)
        wl, wt, wf = pilih_warna(rs, lebar, tinggi, fps)
        if (wl, wt, wf) != (lebar, tinggi, fps) and ke == 0:
            print(f"  Sensor RGB tidak punya {lebar}x{tinggi}@{fps}; "
                  f"warna dipakai {wl}x{wt}@{wf}.")

        # ---- start, dengan pengulangan untuk galat sesaat ----
        # Galat SEMENTARA diulang pada konfigurasi YANG SAMA. Ia tidak boleh
        # memicu kemunduran ke rencana berikutnya: rencana berikutnya lebih
        # miskin (IR dilepas), padahal sebabnya cuma perangkat sedang sibuk.
        # Bug itu pernah nyata - depth+color+IR 848x480@30 terbukti jalan di
        # USB 3.2, tapi "Device or resource busy" sesaat setelah pencolokan
        # membuat rekaman diam-diam kehilangan kedua stream IR.
        pipe = profile = None
        for percobaan in range(1, PERCOBAAN_SEMENTARA + 1):
            total_percobaan += 1
            try:
                pipe, profile = _nyalakan(rs, lebar, tinggi, fps, wl, wt, wf,
                                          ir, hias_cfg)
                break
            except RuntimeError as e:
                pesan = str(e)
                if _cocok(pesan, SEMENTARA):
                    if percobaan < PERCOBAAN_SEMENTARA:
                        jeda = JEDA_ULANG[min(percobaan - 1, len(JEDA_ULANG) - 1)]
                        print(f"  Perangkat sedang sibuk "
                              f"({pesan.splitlines()[0][:60]}); mencoba lagi "
                              f"{percobaan}/{PERCOBAAN_SEMENTARA - 1} "
                              f"dalam {jeda:.1f} detik...")
                        time.sleep(jeda)
                        continue
                    raise KameraGagal(_pesan_sementara(pesan)) from e
                galat_akhir = _diagnosa_tolakan(rs, lebar, tinggi, fps, e)
                if ke + 1 < len(rencana):
                    if ir and _cocok(pesan, DITOLAK):
                        print("  Konfigurasi dengan IR ditolak; mencoba tanpa IR.")
                    break                       # lanjut ke rencana berikutnya
                raise KameraGagal(galat_akhir) from e
        if pipe is None:
            continue                            # rencana ini ditolak

        # ---- pastikan frame benar-benar mengalir ----
        n_frame = 0
        try:
            for _ in range(max(1, hangat)):     # auto-exposure perlu waktu stabil
                pipe.wait_for_frames(batas_ms)
                n_frame += 1
        except RuntimeError as e:
            try:
                pipe.stop()
            except Exception:                   # noqa: BLE001
                pass
            galat_akhir = _diagnosa_frame_mati(rs, lebar, tinggi, fps,
                                               n_frame, batas_ms, e)
            if ke + 1 < len(rencana):
                print(f"\n  Preset '{terpakai}'"
                      f"{' dengan IR' if ir else ''} dinyalakan tapi FRAME TIDAK"
                      f" DATANG dalam {batas_ms/1000:.0f} detik.")
                print("  Mengulang dengan konfigurasi berikutnya...")
                continue
            raise KameraGagal(galat_akhir) from e

        if ke > 0:
            print(f"  Frame datang setelah konfigurasi dimundurkan "
                  f"-> preset terpakai: {terpakai}"
                  f"{'' if ir else ', IR dimatikan'}")
        catatan.update({"preset_terpakai": terpakai, "infrared": ir,
                        "warna_terpakai": [wl, wt, wf],
                        "percobaan_pembukaan": total_percobaan})
        return pipe, profile, terpakai

    raise KameraGagal(galat_akhir)


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
