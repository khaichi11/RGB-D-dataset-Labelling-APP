"""Buat video berdampingan FPN dan U-Net pada rekaman D435 yang sama.

Video ini hanya untuk inspeksi visual yang adil: kedua model menerima frame,
normalisasi kedalaman, dan *letterbox* identik. Tidak ada penghalusan temporal,
pelacak garis, atau pemasangan bidang karena ketiganya akan menyamarkan
perbedaan yang benar-benar berasal dari model segmentasi.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kode.stair_fusion_atto.model_kandidat import StairFusionAttoKandidat
from kode.stair_fusion_atto.stabil.infer import PALETTE, letterbox, unletterbox_map
from model_unet import StairFusionAttoUNet


def muat_fpn(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    dilation = checkpoint.get("dilation") or (1, 1)
    model = StairFusionAttoKandidat(
        varian=checkpoint.get("varian", "cnx_atto_in1k"),
        line_kernel=tuple(checkpoint.get("kernel", (5, 5))),
        line_dilation=tuple(dilation),
        semantic_stride=int(checkpoint.get("semantic_stride", 2)),
        timm_pretrained=False,
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


def muat_unet(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("decoder") != "unet_light":
        raise ValueError("--unet harus checkpoint keluaran unet-new, bukan FPN.")
    model = StairFusionAttoUNet(
        varian=checkpoint.get("varian", "cnx_atto_in1k"),
        pretrained=False,
        line_kernel=tuple(checkpoint.get("kernel", (5, 5))),
        line_dilation=tuple(checkpoint.get("dilation", (1, 1))),
        semantic_stride=int(checkpoint.get("semantic_stride", 2)),
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


def siapkan_input(rgb: np.ndarray, z16: np.ndarray, size: int, depth_scale: float,
                   min_m: float, max_m: float) -> tuple[torch.Tensor, torch.Tensor, tuple[float, int, int]]:
    valid = z16 > 0
    metre = z16.astype(np.float32) * depth_scale
    normal = np.clip((metre - min_m) / (max_m - min_m), 0, 1)
    rgb_pad, scale, dx, dy = letterbox(rgb, size, cv2.INTER_LINEAR)
    normal_pad, _, _, _ = letterbox(normal, size, cv2.INTER_NEAREST)
    valid_pad, _, _, _ = letterbox(valid.astype(np.float32), size, cv2.INTER_NEAREST)
    rgb_tensor = torch.from_numpy(rgb_pad.transpose(2, 0, 1)).float()[None] / 127.5 - 1
    depth_tensor = torch.from_numpy(np.stack((normal_pad * 2 - 1, valid_pad))[None]).float()
    return rgb_tensor, depth_tensor, (scale, dx, dy)


def prediksi(model: torch.nn.Module, rgb: torch.Tensor, depth: torch.Tensor,
             letterbox_info: tuple[float, int, int], width: int, height: int) -> np.ndarray:
    scale, dx, dy = letterbox_info
    with torch.inference_mode():
        probabilitas = model(rgb, depth)["semantic"].softmax(1)[0].float().cpu().numpy()
    return unletterbox_map(probabilitas, scale, dx, dy, width, height).argmax(0).astype(np.uint8)


def panel(rgb: np.ndarray, kelas: np.ndarray, judul: str) -> np.ndarray:
    overlay = cv2.addWeighted(rgb, 0.64, PALETTE[kelas], 0.36, 0)
    out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (25, 25, 25), -1)
    cv2.putText(out, judul, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def tambah_keterangan(frame: np.ndarray) -> np.ndarray:
    tinggi, lebar = frame.shape[:2]
    bar = np.full((52, lebar, 3), 245, np.uint8)
    info = [("latar", (0, 0, 0)), ("riser", tuple(map(int, PALETTE[1][::-1]))),
            ("tapakan", tuple(map(int, PALETTE[2][::-1])))]
    x = 18
    for nama, warna in info:
        cv2.rectangle(bar, (x, 16), (x + 18, 34), warna, -1)
        cv2.putText(bar, nama, (x + 26, 31), cv2.FONT_HERSHEY_SIMPLEX, .52, (30, 30, 30), 1, cv2.LINE_AA)
        x += 110
    cv2.putText(bar, "Frame dan normalisasi depth identik; tanpa smoothing temporal", (max(x + 8, lebar // 2), 31),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (55, 55, 55), 1, cv2.LINE_AA)
    return np.vstack((frame, bar))


def main() -> None:
    parser = argparse.ArgumentParser(description="Video perbandingan FPN dan U-Net pada satu RAW D435")
    parser.add_argument("--bag", required=True, help="raw.db3 atau raw.bag D435")
    parser.add_argument("--fpn", required=True, help="checkpoint baseline FPN")
    parser.add_argument("--unet", required=True, help="checkpoint U-Net fine-tune")
    parser.add_argument("--output", required=True, help="video .mp4 keluaran")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--min-depth-m", type=float, default=.2)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 berarti seluruh rekaman")
    args = parser.parse_args()
    if args.max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m harus lebih besar daripada --min-depth-m")

    import pyrealsense2 as rs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fpn, unet = muat_fpn(Path(args.fpn), device), muat_unet(Path(args.unet), device)
    pipeline, config = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(config, str(Path(args.bag)), repeat_playback=False)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    fps = float(color_profile.fps()) or 30.0
    align = rs.align(rs.stream.color)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    writer, jumlah = None, 0
    try:
        while True:
            try:
                frames = align.process(pipeline.wait_for_frames())
            except RuntimeError:
                break
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            rgb = np.asanyarray(color.get_data())
            z16 = np.asanyarray(depth.get_data())
            rgb_t, depth_t, info = siapkan_input(rgb, z16, args.size, depth_scale,
                                                  args.min_depth_m, args.max_depth_m)
            width, height = rgb.shape[1], rgb.shape[0]
            kelas_fpn = prediksi(fpn, rgb_t.to(device), depth_t.to(device), info, width, height)
            kelas_unet = prediksi(unet, rgb_t.to(device), depth_t.to(device), info, width, height)
            view = tambah_keterangan(np.hstack((panel(rgb, kelas_fpn, "FPN baseline (fine-tuned)"),
                                                panel(rgb, kelas_unet, "U-Net (fine-tuned)"))))
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (view.shape[1], view.shape[0]))
            writer.write(view)
            jumlah += 1
            if jumlah % 100 == 0:
                print(f"{jumlah} frame", flush=True)
            if args.max_frames and jumlah >= args.max_frames:
                break
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
    meta = {
        "rekaman": str(Path(args.bag).resolve()), "checkpoint_fpn": str(Path(args.fpn).resolve()),
        "checkpoint_unet": str(Path(args.unet).resolve()), "frames": jumlah, "fps": fps,
        "input": {"size": args.size, "min_depth_m": args.min_depth_m, "max_depth_m": args.max_depth_m},
        "metode": "Perbandingan prediksi mentah berdampingan; frame, letterbox, dan normalisasi depth sama; tanpa smoothing temporal.",
        "batasan": "Video inspeksi visual. Metrik kuantitatif harus dihitung pada set evaluasi terpisah."
    }
    Path(f"{output}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
