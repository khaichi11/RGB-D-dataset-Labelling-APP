"""Varian ConvNeXt RGB-D dengan decoder U-Net ringan.

Modul ini sengaja dipisahkan dari ``kode/stair_fusion_atto`` agar baseline FPN
tetap utuh.  Encoder RGB, encoder kedalaman, fusi bergerbang, dan kepala garis
menggunakan komponen proyek yang sama; satu-satunya perubahan eksperimen adalah
decoder: penggabungan lateral FPN diganti dengan koneksi-lewati U-Net.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# Script dapat dijalankan dari folder proyek maupun dari folder ``unet-new``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kode.stair_fusion_atto.encoder_timm import buat_encoder
from kode.stair_fusion_atto.kandidat_garis import CabangGaris
from kode.stair_fusion_atto.model import GatedFusion


def _groups(channels: int) -> int:
    """Pilih jumlah grup yang selalu membagi jumlah kanal."""
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class DuaKonvolusi(nn.Module):
    """Dua konvolusi 3x3 ringan setelah penggabungan skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BlokDekoderUNet(nn.Module):
    """Naikkan fitur kasar, gabungkan fitur encoder, lalu rapikan fitur gabungan.

    Penggabungan dilakukan dengan ``concatenate`` seperti U-Net klasik, bukan
    penjumlahan lateral seperti FPN.  Karena itu decoder menerima detail spasial
    dari tingkat encoder yang sama sebelum memprediksi mask resolusi lebih besar.
    """

    def __init__(self, decoder_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(skip_channels, decoder_channels, 1, bias=False)
        self.mix = DuaKonvolusi(2 * decoder_channels, decoder_channels)

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear",
                                align_corners=False)
        skip = self.skip(skip)
        return self.mix(torch.cat((decoder, skip), dim=1))


class StairFusionAttoUNet(nn.Module):
    """ConvNeXt V2 Atto RGB-D dengan decoder U-Net dan dua keluaran.

    Keluaran ``semantic`` adalah logit tiga kelas: latar, riser, dan tapakan.
    Keluaran ``line`` adalah logit deteksi serta kelas garis cembung/cekung.
    Antarmuka ini sengaja sama dengan ``StairFusionAttoKandidat`` agar prosedur
    latih dan metrik baseline dapat dipakai tanpa perubahan.
    """

    def __init__(self, *, varian: str = "cnx_atto_in1k", pretrained: bool = True,
                 classes: int = 3, decoder_channels: int = 96,
                 line_kernel: tuple[int, int] = (3, 3),
                 line_dilation: tuple[int, int] = (1, 1),
                 semantic_stride: int = 2) -> None:
        super().__init__()
        if semantic_stride not in (1, 2):
            raise ValueError("semantic_stride harus 1 atau 2")
        self.varian = varian
        self.semantic_stride = semantic_stride
        self.rgb, self.depth = buat_encoder(varian, pretrained=pretrained)
        rgb_widths, depth_widths = self.rgb.widths, self.depth.widths

        # Fusi dibuat identik dengan baseline; ini menjaga decoder sebagai
        # satu-satunya variabel arsitektur yang dibandingkan.
        self.fusions = nn.ModuleList(
            GatedFusion(rgb_channels, depth_channels)
            for rgb_channels, depth_channels in zip(rgb_widths, depth_widths)
        )

        self.bridge = nn.Sequential(
            nn.Conv2d(rgb_widths[-1], decoder_channels, 1, bias=False),
            nn.GroupNorm(_groups(decoder_channels), decoder_channels), nn.GELU(),
        )
        self.decoder = nn.ModuleList(
            BlokDekoderUNet(decoder_channels, channels)
            for channels in reversed(rgb_widths[:-1])
        )

        self.semantic = nn.Sequential(
            DuaKonvolusi(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, classes, 1),
        )
        self.line = CabangGaris(decoder_channels=decoder_channels,
                                kernel=line_kernel, dilation=line_dilation)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb_features = self.rgb(rgb)
        depth_features = self.depth(depth)
        fused = [module(rgb_feature, depth_feature)
                 for module, rgb_feature, depth_feature
                 in zip(self.fusions, rgb_features, depth_features)]

        decoder = self.bridge(fused[-1])
        for block, skip in zip(self.decoder, reversed(fused[:-1])):
            decoder = block(decoder, skip)

        # Baseline pembanding dapat menghitung kepala semantik pada stride 2.
        # Opsi ini menjaga perbandingan U-Net vs FPN tetap hanya menguji decoder.
        full_size = rgb.shape[-2:]
        semantic_size = (full_size[0] // self.semantic_stride,
                         full_size[1] // self.semantic_stride)
        full = F.interpolate(decoder, size=semantic_size, mode="bilinear",
                             align_corners=False)
        semantic = self.semantic(full)
        if self.semantic_stride > 1:
            semantic = F.interpolate(semantic, size=full_size, mode="bilinear",
                                     align_corners=False)
        return {
            "semantic": semantic,
            "line": self.line(torch.cat((rgb, depth), dim=1), decoder),
        }
