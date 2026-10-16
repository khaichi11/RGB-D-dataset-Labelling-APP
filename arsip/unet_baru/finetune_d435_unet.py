"""Fine-tune decoder U-Net pada frame D435 tanpa mencampur data evaluasi.

Skrip ini memakai pembaca frame D435 dan metrik yang sama dengan jalur FPN,
tetapi hanya dapat memuat checkpoint yang dibuat oleh ``train_unet.py``.
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

from kode.stair_fusion_atto.finetune_d435.dataset import FrameD435, frame_bersih
from kode.stair_fusion_atto.finetune_d435.train import EMA, bagi_rekaman, jalankan
from model_unet import StairFusionAttoUNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt U-Net pada data D435")
    parser.add_argument("--akar", default="dataset/studio_rgbd/rekaman/tangga_naik")
    parser.add_argument("--latih", nargs="+", default=["105633", "105802", "110530"])
    parser.add_argument("--validasi", nargs="+", default=["105348"])
    parser.add_argument("--bagi", nargs="*", default=["013859"],
                        help="rekaman yang sebagian framenya ditahan sebagai validasi")
    parser.add_argument("--bagi-n-val", type=int, default=10)
    parser.add_argument("--init", required=True, help="best.pt keluaran train_unet.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--line-weight", type=float, default=1.0)
    parser.add_argument("--bg-weight", type=float, default=2.0)
    parser.add_argument("--tread-weight", type=float, default=1.6)
    parser.add_argument("--depth-dropout", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hanya-diperiksa", action="store_true",
                        help="pakai hanya label yang sudah diperiksa manual")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    akar, output = Path(args.akar), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    f_latih = frame_bersih(akar, args.latih, hanya_diperiksa=args.hanya_diperiksa)
    f_val = frame_bersih(akar, args.validasi, hanya_diperiksa=args.hanya_diperiksa)
    if {path.parent.parent.parent.name for path in f_latih} & {path.parent.parent.parent.name for path in f_val}:
        raise SystemExit("rekaman latih dan validasi tumpang tindih")
    for nama in args.bagi:
        train_part, val_part = bagi_rekaman(akar, [nama], args.bagi_n_val, args.seed)
        if set(train_part) & set(val_part):
            raise SystemExit(f"{nama}: frame latih dan validasi tumpang tindih")
        f_latih.extend(train_part)
        f_val.extend(val_part)
    if not f_latih or not f_val:
        raise SystemExit("frame latih atau validasi kosong; periksa nama rekaman dan label")

    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    if checkpoint.get("decoder") != "unet_light":
        raise SystemExit("--init harus checkpoint dari unet-new/train_unet.py, bukan baseline FPN")
    kernel = tuple(checkpoint.get("kernel", (5, 5)))
    dilation = tuple(checkpoint.get("dilation", (1, 1)))
    stride = int(checkpoint.get("semantic_stride", 2))
    varian = checkpoint.get("varian", "cnx_atto_in1k")
    model = StairFusionAttoUNet(varian=varian, pretrained=False, line_kernel=kernel,
                                line_dilation=dilation, semantic_stride=stride).to(device)
    model.load_state_dict(checkpoint["model"])

    pin = device.type == "cuda"
    train_loader = DataLoader(FrameD435(f_latih, augment=True, depth_dropout=args.depth_dropout),
                              args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=pin, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(FrameD435(f_val, augment=False), args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin,
                            persistent_workers=args.workers > 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr / 20)
    ema = EMA(model, args.ema_decay)
    names = ["loss", "semantic_loss", "line_detect_loss", "line_class_loss", "dice_riser", "dice_tread",
             "iou_riser", "iou_tread", "mean_dice", "miou", "boundary_precision", "boundary_recall",
             "boundary_f1", "boundary_threshold", "boundary_f1_at_half", "line_class_accuracy"]
    settings = vars(args) | {"device": str(device), "n_latih": len(f_latih), "n_validasi": len(f_val),
                             "varian": varian, "kernel": kernel, "dilation": dilation,
                             "semantic_stride": stride, "decoder": "unet_light"}
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"latih {len(f_latih)} frame | validasi {len(f_val)} frame | init {args.init}", flush=True)

    best, since_best = -float("inf"), 0
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "lr"] +
                                [f"{split}_{name}" for split in ("train", "val") for name in names])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train = jalankan(model, train_loader, optimizer, device, args.dice_weight,
                              args.line_weight, args.tread_weight, ema=ema, bg_weight=args.bg_weight)
            raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
            model.load_state_dict(ema.salin_ke(model))
            with torch.inference_mode():
                valid = jalankan(model, val_loader, None, device, args.dice_weight,
                                  args.line_weight, args.tread_weight, bg_weight=args.bg_weight)
            saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
            model.load_state_dict(raw)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row |= {f"train_{name}": train.get(name, "") for name in names}
            row |= {f"val_{name}": valid.get(name, "") for name in names}
            writer.writerow(row)
            handle.flush()
            score = valid["mean_dice"] + valid["boundary_f1"]
            saved_checkpoint = {"epoch": epoch, "model": saved, "metrics": row, "varian": varian,
                                "decoder": "unet_light", "kernel": kernel, "dilation": dilation,
                                "semantic_stride": stride, "best_score": max(best, score)}
            torch.save(saved_checkpoint, output / "last.pt")
            if score > best:
                best, since_best = score, 0
                torch.save(saved_checkpoint, output / "best.pt")
            else:
                since_best += 1
            scheduler.step()
            print(f"epoch {epoch:03d}/{args.epochs} | val Dice {valid['mean_dice']:.4f} | "
                  f"Boundary F1 {valid['boundary_f1']:.4f} | sejak-baik {since_best}", flush=True)
            if since_best >= args.patience:
                print("berhenti awal: validasi tidak membaik", flush=True)
                break


if __name__ == "__main__":
    main()
