"""Ekspor point cloud berwarna dari satu paket frame Studio RGB-D.

Depth yang dipakai adalah ``depth_aligned_to_color.npy``. Karena sudah
disejajarkan ke RGB, proyeksinya wajib memakai intrinsics RGB pada
``frame.json``. Output PLY binary little-endian dapat dibuka di MeshLab atau
CloudCompare dan koordinatnya dinyatakan dalam meter.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np


PLY_HEADER = """ply
format binary_little_endian 1.0
comment RGB-D Studio ZenExo; coordinates in metres
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""


def point_cloud(frame_dir: Path, stride: int, min_depth_m: float,
                max_depth_m: float) -> np.ndarray:
    """Kembalikan titik ``x,y,z,r,g,b`` dari satu paket frame yang tervalidasi."""
    metadata = json.loads((frame_dir / "frame.json").read_text(encoding="utf-8"))
    depth = np.load(frame_dir / "depth_aligned_to_color.npy")
    color = cv2.imread(str(frame_dir / "color_raw.png"), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"RGB tidak dapat dibaca: {frame_dir / 'color_raw.png'}")
    if depth.shape[:2] != color.shape[:2]:
        raise RuntimeError(f"RGB/depth tidak sejajar: {color.shape[:2]} vs {depth.shape[:2]}")

    k = metadata["intrinsics_rgb_native"]
    scale = float(metadata["depth_scale"])
    ys, xs = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[ys, xs].astype(np.float32) * scale
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    bgr = color[ys, xs]
    x = (xs.astype(np.float32) - float(k["ppx"])) * z / float(k["fx"])
    y = (ys.astype(np.float32) - float(k["ppy"])) * z / float(k["fy"])
    return np.column_stack((x, y, z, bgr[:, 2], bgr[:, 1], bgr[:, 0]))


def write_ply(path: Path, points: np.ndarray) -> None:
    """Tulis PLY binary agar tetap ringkas untuk point cloud berwarna."""
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"] = points[:, 3].astype(np.uint8)
    data["green"] = points[:, 4].astype(np.uint8)
    data["blue"] = points[:, 5].astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(PLY_HEADER.format(count=len(data)).encode("ascii"))
        handle.write(data.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ekspor point cloud PLY dari frame Studio RGB-D")
    parser.add_argument("--frame", required=True, type=Path, help="folder frame_XXXXXX hasil ekspor")
    parser.add_argument("--out", required=True, type=Path, help="berkas .ply tujuan")
    parser.add_argument("--stride", type=int, default=2, help="ambil satu titik tiap N piksel")
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride harus >= 1")
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        raise SystemExit("rentang depth tidak valid")
    points = point_cloud(args.frame, args.stride, args.min_depth_m, args.max_depth_m)
    if not len(points):
        raise SystemExit("Tidak ada titik depth valid dalam rentang yang dipilih")
    write_ply(args.out, points)
    print(f"Selesai: {args.out.resolve()} | {len(points):,} titik | stride={args.stride}")


if __name__ == "__main__":
    main()