"""Akses kamera Intel RealSense D435 khusus aplikasi Studio."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 belum tersedia. Jalankan dari .venv proyek.")

from .resolusi_kamera import buka, catat as catat_resolusi

NAMA_AMAN = re.compile(r"[^A-Za-z0-9_-]+")


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
        if self.pipe is not None:
            self.hentikan()
        decorator = None if record_path is None else lambda cfg: cfg.enable_record_to_file(str(record_path))
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
        self.pipe, self.profile, self.preset_terpakai = buka(
            rs, self.lebar, self.tinggi, self.fps, self.preset,
            hias_cfg=decorator, batas_ms=self.batas_frame, infrared=True)
        self.record_path = record_path

    def hentikan(self) -> None:
        if self.pipe is not None:
            try:
                self.pipe.stop()
            finally:
                self.pipe = self.profile = self.record_path = None

    def ambil(self):
        if self.pipe is None:
            raise RuntimeError("Kamera belum dinyalakan.")
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
        return info

    def simpan_ply(self, depth_frame, color_frame, tujuan: Path) -> None:
        self.pc.map_to(color_frame)
        self.pc.calculate(depth_frame).export_to_ply(str(tujuan), color_frame)
