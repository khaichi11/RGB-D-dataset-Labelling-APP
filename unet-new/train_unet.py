"""Latih varian decoder U-Net tanpa mengubah baseline FPN.

Contoh:
python unet-new/train_unet.py --data dataset/rgbd_stair --output bobot/kandidat/unet_atto
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kode.stair_fusion_atto.dataset_kandidat import RGBDStairCandidateDataset, collate_candidate
from kode.stair_fusion_atto.train_kandidat import EMA, run_epoch
from model_unet import StairFusionAttoUNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih ConvNeXt RGB-D dengan decoder U-Net ringan")
    parser.add_argument("--data", required=True, help="folder yang memuat train dan val")
    parser.add_argument("--output", required=True, help="folder checkpoint baru")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--line-weight", type=float, default=1.0)
    parser.add_argument("--semantic-stride", choices=(1, 2), type=int, default=2,
                        help="resolusi kepala semantik; 2 dipakai agar setara baseline FPN")
    parser.add_argument("--kernel", choices=("3x3", "3x5", "3x9", "5x5"), default="3x3")
    parser.add_argument("--dilation", default="1x1")
    parser.add_argument("--varian", choices=("cnx_atto_in1k", "cnx_atto_in1k_acak", "cnx_femto_in1k"),
                        default="cnx_atto_in1k")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="hanya untuk ablasi atau uji lokal tanpa unduhan bobot")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--depth-lama", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"device": str(device), "decoder": "unet_light"}
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_set = RGBDStairCandidateDataset(root / "train", augment=not args.no_augment,
                                          depth_lama=args.depth_lama, hflip_prob=args.hflip_prob)
    val_set = RGBDStairCandidateDataset(root / "val", depth_lama=args.depth_lama)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=pin, persistent_workers=args.workers > 0,
                              collate_fn=collate_candidate)
    val_loader = DataLoader(val_set, args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=pin, persistent_workers=args.workers > 0,
                            collate_fn=collate_candidate)

    kernel = tuple(map(int, args.kernel.split("x")))
    dilation = tuple(map(int, args.dilation.split("x")))
    model = StairFusionAttoUNet(varian=args.varian, pretrained=not args.no_pretrained,
                                line_kernel=kernel, line_dilation=dilation,
                                semantic_stride=args.semantic_stride).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr / 30)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    names = ["loss", "semantic_loss", "line_detect_loss", "line_class_loss", "dice_riser", "dice_tread",
             "iou_riser", "iou_tread", "mean_dice", "miou", "boundary_precision", "boundary_recall",
             "boundary_f1", "boundary_threshold", "boundary_f1_at_half", "line_class_accuracy",
             "lokalisasi_median_px", "lokalisasi_p90_px", "lokalisasi_n", "lokalisasi_cakupan"]
    best = -float("inf")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "lr"] +
                                [f"{split}_{name}" for split in ("train", "val") for name in names])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, optimizer, device, args.dice_weight,
                              args.line_weight, ema=ema)
            raw = {key: value.detach().clone() for key, value in model.state_dict().items()} if ema else None
            if ema:
                model.load_state_dict(ema.salin_ke(model))
            with torch.inference_mode():
                valid = run_epoch(model, val_loader, None, device, args.dice_weight, args.line_weight)
            saved_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if ema:
                model.load_state_dict(raw)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row |= {f"train_{name}": train.get(name, "") for name in names}
            row |= {f"val_{name}": valid.get(name, "") for name in names}
            writer.writerow(row)
            handle.flush()
            score = valid["mean_dice"] + valid["boundary_f1"]
            checkpoint = {"epoch": epoch, "model": saved_model, "metrics": row, "varian": args.varian,
                          "decoder": "unet_light", "kernel": kernel, "dilation": dilation,
                          "semantic_stride": args.semantic_stride,
                          "best_score": max(best, score)}
            torch.save(checkpoint, output / "last.pt")
            if score > best:
                best = score
                torch.save(checkpoint, output / "best.pt")
            scheduler.step()
            print(f"epoch {epoch:03d}/{args.epochs} | Dice {valid['mean_dice']:.4f} | "
                  f"Boundary F1 {valid['boundary_f1']:.4f}@{valid['boundary_threshold']:.2f}", flush=True)


if __name__ == "__main__":
    main()
