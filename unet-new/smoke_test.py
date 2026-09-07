"""Uji bentuk keluaran tanpa mengunduh bobot ImageNet atau melatih model."""
from __future__ import annotations

import torch

from model_unet import StairFusionAttoUNet


def main() -> None:
    model = StairFusionAttoUNet(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 2, 64, 64))
    assert output["semantic"].shape == (1, 3, 64, 64), output["semantic"].shape
    assert output["line"].shape == (1, 2, 32, 32), output["line"].shape
    print("OK", {key: tuple(value.shape) for key, value in output.items()})


if __name__ == "__main__":
    main()
