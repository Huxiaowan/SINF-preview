"""models.modules

This file intentionally contains **only the blocks that are used** by the current
entrypoints in this repository (e.g., `train_models/sRGB_train.py`).

Kept public APIs:
  - BSINF
  - DBSNl
  - ITE
  - TSGM

Everything else from the older, fragmented implementation has been removed to
reduce file count and prevent unused code from accumulating.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

__all__ = [
    "BSINF",
    "DBSNl",
    "ITE",
    "TSGM",
]


# -----------------------------------------------------------------------------
# Basic ops
# -----------------------------------------------------------------------------


class CentralMaskedConv2d(nn.Conv2d):
    """Conv2d with the center weight masked out (blind-spot)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class DCl(nn.Module):
    """Dilated conv block.

    Note: this matches the **effective** DCl that was used after the previous
    consolidation (the last definition won). It is kept to avoid changing runtime
    behavior.
    """

    def __init__(self, stride: int, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=1,
                padding=stride,
                dilation=stride,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ITE(nn.Module):
    """A small dilation stack used as a feature adapter / head."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, num_res: int = 9):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            *[DCl(2, mid_channels) for _ in range(num_res)],
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        ]
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


# -----------------------------------------------------------------------------
# DBSN variants
# -----------------------------------------------------------------------------


class DC_branchl(nn.Module):
    def __init__(self, stride: int, in_ch: int, num_module: int):
        super().__init__()
        ly = [
            CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            *[DCl(stride, in_ch) for _ in range(num_module)],
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        self.body = nn.Sequential(*ly)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DBSNl(nn.Module):
    """Dilated Blind-Spot Network (light version)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128, num_module: int = 9):
        super().__init__()
        assert base_ch % 2 == 0, "base channel should be divided with 2"

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.branch1(x)
        return self.tail(x)


class BSINF(nn.Module):
    """DBSN + INR-style coordinate conditioning (as used by SINF)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128, num_module: int = 9, stride: int = 2):
        super().__init__()
        assert base_ch % 2 == 0, "base channel should be divisible by 2"

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        class _CentralMaskedConv2d(nn.Conv2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer("mask", self.weight.data.clone())
                _, _, kH, kW = self.weight.size()
                self.mask.fill_(1)
                self.mask[:, :, kH // 2, kW // 2] = 0

            def forward(self, x):
                self.weight.data *= self.mask
                return super().forward(x)

        class _DilatedConvBlock(nn.Module):
            def __init__(self, stride_: int, in_ch_: int):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv2d(in_ch_, in_ch_, kernel_size=3, stride=1, padding=stride_, dilation=stride_),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch_, in_ch_, kernel_size=1),
                )

            def forward(self, x):
                return x + self.body(x)

        branch_layers = [
            _CentralMaskedConv2d(base_ch, base_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            *[_DilatedConvBlock(stride, base_ch) for _ in range(num_module)],
            nn.Conv2d(base_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        self.branch = nn.Sequential(*branch_layers)

        # INR feature enhancement layer
        self.inr_mlp = nn.Sequential(
            nn.Linear(base_ch + 2, base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch // 2, base_ch),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.head(x)
        feat = self.branch(x)  # (B, C, H, W)

        # normalized coordinate grid: (H*W, 2)
        y_coords = torch.linspace(-1, 1, H, device=x.device)
        x_coords = torch.linspace(-1, 1, W, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (H*W, 2)

        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        coords_expand = coords.unsqueeze(0).expand(B, -1, -1)
        inr_input = torch.cat([feat_flat, coords_expand], dim=-1)
        inr_out = self.inr_mlp(inr_input)

        feat_enhanced = feat_flat + inr_out
        feat_enhanced = feat_enhanced.view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.tail(feat_enhanced)


# -----------------------------------------------------------------------------
# SRFE + (windowed) graph attention
# -----------------------------------------------------------------------------


class Downsample(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        d2 = self.dilation ** 2
        x = rearrange(x, "b c (hd h) (wd w) -> b (c hd wd) h w", h=d2, w=d2)
        x = rearrange(x, "b c (hn hh) (wn ww) -> b c (hn wn) hh ww", hh=self.dilation, ww=self.dilation)
        x = rearrange(x, "b (c hd wd) cc hh ww-> b (c cc) (hd hh) (wd ww)", hd=H // d2, wd=W // d2)
        return x


class Upsample(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        d2 = self.dilation ** 2
        x = rearrange(x, "b (c cc) (hd hh) (wd ww) -> b (c hd wd) cc hh ww", cc=d2, hh=self.dilation, ww=self.dilation)
        x = rearrange(x, "b c (hn wn) hh ww -> b c (hn hh) (wn ww)", hn=self.dilation, wn=self.dilation)
        x = rearrange(x, "b (c hd wd) h w -> b c (hd h) (wd w)", hd=H // self.dilation, wd=W // self.dilation)
        return x


class Encoder(nn.Module):
    """Minimal encoder used by TSGM.

    In the current codebase, TSGM only uses `encoder.shuffler` to obtain
    downsampled features. We keep this class minimal to match actual usage.
    """

    def __init__(self, in_channels: int = 3, depth: int = 1):
        super().__init__()
        # NOTE: The old implementation used a fixed factor of 2.
        self.shuffler = Downsample(2)


class Decoder(nn.Module):
    def __init__(self, depth: int = 1):
        super().__init__()
        self.shuffler = Upsample(2)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.shuffler(feats)


class INRTimeEmbed(nn.Module):
    def __init__(self, embed_dim: int = 64, freq: int = 8):
        super().__init__()
        self.freq = freq
        self.embed_dim = embed_dim
        self.linear = nn.Linear(2 * freq, embed_dim)

    def forward(self, t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """t: [B] in [0,1]. Returns [B, embed_dim, H, W]."""
        B = t.shape[0]
        freq_bands = torch.linspace(1, self.freq, self.freq, device=t.device)
        angles = t.unsqueeze(-1) * freq_bands * math.pi
        embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        feat = self.linear(embed).unsqueeze(-1).unsqueeze(-1)
        return feat.expand(B, self.embed_dim, H, W)


class WindowedGraphAttention(nn.Module):
    def __init__(self, channels: int, window_size: int = 8, reduction: int = 8):
        super().__init__()
        self.window_size = window_size
        self.query_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x1.shape
        ws = self.window_size
        N = ws * ws

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        H_pad, W_pad = H + pad_h, W + pad_w

        x1_pad = F.pad(x1, (0, pad_w, 0, pad_h), mode="reflect")
        x2_pad = F.pad(x2, (0, pad_w, 0, pad_h), mode="reflect")

        def window_partition(x: torch.Tensor) -> torch.Tensor:
            return x.unfold(2, ws, ws).unfold(3, ws, ws)  # B, C, nw_h, nw_w, ws, ws

        def window_reverse(windows: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
            B_, C_, nw_h, nw_w, ws1, ws2 = windows.shape
            out = windows.contiguous().view(B_, C_, nw_h * ws1, nw_w * ws2)
            return out[:, :, :H_out, :W_out]

        q = window_partition(self.query_conv(x1_pad))
        k = window_partition(self.key_conv(x2_pad))
        v = window_partition(self.value_conv(x2_pad))

        B_, Cq, nw_h, nw_w, _, _ = q.shape

        q = q.permute(0, 2, 3, 4, 5, 1).contiguous().view(B_, nw_h, nw_w, N, Cq)
        k = k.permute(0, 2, 3, 1, 4, 5).contiguous().view(B_, nw_h, nw_w, Cq, N)
        v = v.permute(0, 2, 3, 4, 5, 1).contiguous().view(B_, nw_h, nw_w, N, C)

        attn = torch.matmul(q, k) / (Cq ** 0.5)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)

        out = out.permute(0, 4, 1, 2, 3).contiguous().view(B_, C, nw_h, nw_w, ws, ws)
        out = window_reverse(out, H_pad, W_pad)
        out = out[:, :, :H, :W]
        out = self.proj(out)
        return x1 + self.gamma * out


class TSGM(nn.Module):
    def __init__(self, in_channels: int = 3, depth: int = 1):
        super().__init__()
        self.padder_size = 4 ** depth
        embed_channels = in_channels * (4 ** depth)

        self.encoder = Encoder(in_channels=in_channels, depth=depth)
        self.decoder = Decoder(depth=depth)

        self.time_embed = INRTimeEmbed(embed_dim=embed_channels)
        self.graph_attn = WindowedGraphAttention(channels=embed_channels)

    def check_image_size(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x1.shape == x2.shape
        _, _, h, w = x1.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x1 = F.pad(x1, (0, mod_pad_w, 0, mod_pad_h))
        x2 = F.pad(x2, (0, mod_pad_w, 0, mod_pad_h))
        return x1, x2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t1: float = 0.0, t2: float = 1.0) -> torch.Tensor:
        _, _, H, W = x1.shape
        x1, x2 = self.check_image_size(x1, x2)

        # NOTE: preserves current repo behavior: directly use downsampled features.
        f1 = self.encoder.shuffler(x1)
        f2 = self.encoder.shuffler(x2)

        t_feat1 = self.time_embed(torch.tensor([t1], device=f1.device), f1.size(2), f1.size(3))
        t_feat2 = self.time_embed(torch.tensor([t2], device=f2.device), f2.size(2), f2.size(3))
        f1 = f1 + t_feat1
        f2 = f2 + t_feat2

        fused = self.graph_attn(f1, f2)
        out = self.decoder(fused)
        return out[:, :, :H, :W]
