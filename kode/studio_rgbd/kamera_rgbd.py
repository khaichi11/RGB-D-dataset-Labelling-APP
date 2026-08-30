"""Akses kamera Intel RealSense D435 khusus aplikasi Studio."""
from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 belum tersedia. Jalankan dari .venv proyek.")

from .resolusi_kamera import (ALIAS_PRESET, KameraGagal, buka,
                              catat as catat_resolusi)

__all__ = ["EKSTENSI_REKAM", "KameraGagal", "KameraRGBD", "NAMA_REKAMAN",
           "cari_rekaman", "nama_aman", "nama_rekaman", "tulis_json"]

NAMA_AMAN = re.compile(r"[^A-Za-z0-9_-]+")


def _versi_rs() -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(rs.__version__).split(".")[:3])
    except Exception:                                           # noqa: BLE001
        return ()


# librealsense modern merekam ke SQLite (.db3); yang lama memakai rosbag
# (.bag). Ini BUKAN sekadar soal nama: pipe.start() menolak mentah-mentah
# ekstensi yang salah dengan "Output file must have .db3 extension", dan
# dulu penolakan itu membuat tombol "Mulai rekam" gagal total di
# pyrealsense2 2.58 - tidak satu pun rekaman bisa dibuat.
# Versi hanya dipakai sebagai tebakan awal; yang menentukan tetap jawaban
# pipe.start(), lihat KameraRGBD.mulai().
EKSTENSI_REKAM = ".db3" if _versi_rs() >= (2, 56) else ".bag"
EKSTENSI_REKAM_LAIN = ".bag" if EKSTENSI_REKAM == ".db3" else ".db3"
# Urutan pencarian rekaman lama di disk; sesi yang direkam pyrealsense2
# versi sebelumnya tetap harus bisa dibuka.
NAMA_REKAMAN = ("raw" + EKSTENSI_REKAM, "raw" + EKSTENSI_REKAM_LAIN)


def nama_rekaman() -> str:
    """Nama berkas rekaman primer untuk versi pyrealsense2 yang terpasang."""
    return NAMA_REKAMAN[0]


def cari_rekaman(folder_source: Path) -> Path | None:
    """Rekaman primer yang benar-benar ada di folder ini, apa pun ekstensinya."""
    for nama in NAMA_REKAMAN:
        p = folder_source / nama
        if p.exists():
            return p
    return None


def nama_aman(teks: str) -> str:
    """Normalisasi nama folder tanpa spasi dan karakter shell."""
    return NAMA_AMAN.sub("_", teks.strip()).strip("_-")


def tulis_json(path: Path, isi: dict) -> None:
    path.write_text(json.dumps(isi, ensure_ascii=False, indent=2), encoding="utf-8")


class KameraRGBD:
    """Pemilik tunggal pipeline D435 agar Studio tidak membuka kamera ganda."""

    def __init__(self, lebar: int, tinggi: int, fps: int, preset: str, batas_frame: int) -> None:
        self.lebar, self.tinggi, self.fps = lebar, tinggi, fps
        self.preset, self.batas_frame = preset, batas_frame
        self.pipe = self.profile = self.preset_terpakai = None
        self.catatan_pembukaan: dict = {}
        # --- pompa frame ---
        # Pipeline HARUS dikuras pada laju penuh. Kalau frame hanya diambil
        # ~5x/detik dari thread UI, antrean librealsense penuh terus sehingga
        # frame yang tampil basi, dan tiap pengambilan memblokir Tk - itulah
        # yang terasa sebagai jendela patah-patah.
        self._pompa: threading.Thread | None = None
        self._henti = threading.Event()
        self._kunci = threading.Lock()
        self._frameset = None
        self.n_terkuras = 0
        self.n_putus = 0
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        self.record_path: Path | None = None

    @property
    def hidup(self) -> bool:
        return self.pipe is not None

    @property
    def sedang_rekam(self) -> bool:
        return self.record_path is not None

    def mulai(self, record_path: Path | None = None) -> None:
        """Nyalakan kamera. Kalau librealsense menolak ekstensi berkas rekaman,
        ulangi sekali dengan ekstensi pasangannya - dengan begitu aplikasi ini
        jalan baik di pyrealsense2 lama (.bag) maupun baru (.db3)."""
        try:
            self._mulai(record_path)
            return
        except KameraGagal as e:
            lain = self._ekstensi_pasangan(record_path, e)
            if lain is None:
                raise
            print(f"  librealsense menolak '{record_path.suffix}'; "
                  f"merekam ke '{lain.suffix}' sebagai gantinya.")
        self._mulai(lain)

    @staticmethod
    def _ekstensi_pasangan(record_path: Path | None, e: Exception) -> Path | None:
        """-> path dengan ekstensi pasangan, bila galatnya memang soal ekstensi."""
        if record_path is None:
            return None
        pesan = str(e).lower()
        if "extension" not in pesan:
            return None
        for ext in (".db3", ".bag"):
            if ext in pesan and record_path.suffix != ext:
                return record_path.with_suffix(ext)
        return None

    def _mulai(self, record_path: Path | None = None) -> None:
        if self.pipe is not None:
            self.hentikan()
        decorator = None if record_path is None else lambda cfg: cfg.enable_record_to_file(str(record_path))
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
        catatan: dict = {}
        try:
            self.pipe, self.profile, self.preset_terpakai = buka(
                rs, self.lebar, self.tinggi, self.fps, self.preset,
                hias_cfg=decorator, batas_ms=self.batas_frame, infrared=True,
                catatan=catatan)
        except BaseException:
            # Termasuk KameraGagal. Kalau start gagal, jangan tinggalkan objek
            # ini seolah-olah kamera hidup - hidup/sedang_rekam harus tetap
            # False supaya UI bisa menyalakan ulang tanpa restart aplikasi.
            self.pipe = self.profile = None
            self.record_path = None
            raise
        self.catatan_pembukaan = catatan
        self.record_path = record_path

    def hentikan(self) -> None:
        # Pompa harus mati SEBELUM pipe.stop(), kalau tidak thread pompa masih
        # menunggu frame pada pipeline yang sudah dibongkar.
        self.hentikan_pompa()
        if self.pipe is not None:
            try:
                self.pipe.stop()
            finally:
                self.pipe = self.profile = self.record_path = None
                self.preset_terpakai = None

    # ---------------- pompa frame ----------------
    @property
    def pompa_hidup(self) -> bool:
        return self._pompa is not None and self._pompa.is_alive()

    def mulai_pompa(self) -> None:
        """Kuras pipeline terus-menerus di thread sendiri.

        Ini yang menjaga dua hal sekaligus: preview tidak lagi memblokir Tk,
        dan aliran tetap mengalir penuh sehingga perekaman tidak tersendat
        karena konsumen yang lambat.
        """
        if self.pipe is None:
            raise RuntimeError("Kamera belum dinyalakan.")
        if self.pompa_hidup:
            return
        self._henti.clear()
        self.n_terkuras = self.n_putus = 0
        self._pompa = threading.Thread(target=self._putar_pompa, daemon=True)
        self._pompa.start()

    def hentikan_pompa(self) -> None:
        self._henti.set()
        t = self._pompa
        if t is not None and t.is_alive():
            # wait_for_frames memblokir; beri waktu satu batas frame + margin.
            t.join(timeout=self.batas_frame / 1000 + 1.0)
        self._pompa = None
        with self._kunci:
            self._frameset = None

    def _putar_pompa(self) -> None:
        while not self._henti.is_set():
            pipe = self.pipe
            if pipe is None:
                break
            try:
                fs = pipe.wait_for_frames(self.batas_frame)
            except RuntimeError:
                # Kamera dihentikan atau frame telat. Bukan alasan mematikan
                # pompa - hentikan_pompa() yang berwenang menghentikannya.
                self.n_putus += 1
                continue
            with self._kunci:
                self._frameset = fs
                self.n_terkuras += 1

    def frameset_terbaru(self):
        """Frameset paling akhir yang dikuras pompa, atau None."""
        with self._kunci:
            return self._frameset

    # ---------------- preset saat kamera hidup ----------------
    def daftar_preset(self) -> list[str]:
        """Nama visual preset yang benar-benar ditawarkan perangkat ini."""
        if self.profile is None:
            return []
        try:
            ds = self.profile.get_device().first_depth_sensor()
            if not ds.supports(rs.option.visual_preset):
                return []
            r = ds.get_option_range(rs.option.visual_preset)
            return [ds.get_option_value_description(rs.option.visual_preset, i)
                    for i in range(int(r.min), int(r.max) + 1)]
        except Exception:                                       # noqa: BLE001
            return []

    def preset_sekarang(self) -> str | None:
        if self.profile is None:
            return None
        try:
            ds = self.profile.get_device().first_depth_sensor()
            if not ds.supports(rs.option.visual_preset):
                return None
            return ds.get_option_value_description(
                rs.option.visual_preset, int(ds.get_option(rs.option.visual_preset)))
        except Exception:                                       # noqa: BLE001
            return None

    def ganti_preset(self, nama: str) -> str:
        """Ganti visual preset pada kamera yang sedang hidup. -> nama terpakai.

        Tidak perlu menyalakan ulang aliran: visual_preset boleh disetel saat
        streaming. Nama diambil dari daftar perangkat, bukan ditebak.
        """
        if self.profile is None:
            raise RuntimeError("Kamera belum dinyalakan.")
        ds = self.profile.get_device().first_depth_sensor()
        if not ds.supports(rs.option.visual_preset):
            raise RuntimeError("Perangkat ini tidak punya visual preset.")
        r = ds.get_option_range(rs.option.visual_preset)
        for i in range(int(r.min), int(r.max) + 1):
            if ds.get_option_value_description(rs.option.visual_preset, i) == nama:
                ds.set_option(rs.option.visual_preset, i)
                self.preset_terpakai = nama
                self.catatan_pembukaan["preset_terpakai"] = nama
                return nama
        raise RuntimeError(f"Preset {nama!r} tidak ada pada perangkat ini.")

    def ambil(self):
        if self.pipe is None:
            raise RuntimeError("Kamera belum dinyalakan.")
        # Kalau pompa hidup, pakai frame terakhir darinya. Dua pembaca yang
        # sama-sama memanggil wait_for_frames akan saling mencuri frame.
        native = self.frameset_terbaru() if self.pompa_hidup else None
        if native is None:
            native = self.pipe.wait_for_frames(self.batas_frame)
        color, depth = native.get_color_frame(), native.get_depth_frame()
        aligned = self.align.process(native)
        color_aligned, depth_aligned = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth or not color_aligned or not depth_aligned:
            raise RuntimeError("Frame RGB atau depth kosong.")
        return native, color, depth, color_aligned, depth_aligned

    def info(self, color_frame, depth_frame) -> dict:
        if self.profile is None:
            raise RuntimeError("Profil kamera belum tersedia.")
        dev = self.profile.get_device()
        warna = color_frame.profile.as_video_stream_profile()
        depth = depth_frame.profile.as_video_stream_profile()
        intr, intr_depth = warna.intrinsics, depth.intrinsics
        extr = depth.get_extrinsics_to(warna)
        info = {
            "kamera": dev.get_info(rs.camera_info.name),
            "nomor_seri": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "depth_scale": float(dev.first_depth_sensor().get_depth_scale()),
            "lebar": intr.width, "tinggi": intr.height, "fps": self.fps,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "model_distorsi": str(intr.model), "koef_distorsi": list(intr.coeffs),
            "intrinsics_rgb_native": {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs)},
            "intrinsics_depth_native": {"width": intr_depth.width, "height": intr_depth.height, "fx": intr_depth.fx, "fy": intr_depth.fy, "ppx": intr_depth.ppx, "ppy": intr_depth.ppy, "model": str(intr_depth.model), "coeffs": list(intr_depth.coeffs)},
            "extrinsics_depth_ke_rgb": {"rotation_row_major": list(extr.rotation), "translation_meter": list(extr.translation)},
            "preset_terpakai": self.preset_terpakai,
        }
        catat_resolusi(info, self.lebar, self.tinggi, self.fps, rs, dev, self.preset_terpakai)
        # Kalau IR terpaksa dilepas atau warna diturunkan, itu harus terbaca di
        # meta - rekaman yang kehilangan stream tidak boleh diam-diam.
        info["pembukaan"] = dict(self.catatan_pembukaan)
        return info

    def simpan_ply(self, depth_frame, color_frame, tujuan: Path) -> None:
        self.pc.map_to(color_frame)
        self.pc.calculate(depth_frame).export_to_ply(str(tujuan), color_frame)
