"""ATM (Adaptive Thinking Mapper) encoder for EEG.

Reference:  Li et al., "Visual Decoding and Reconstruction via EEG Embeddings
            with Guided Diffusion", NeurIPS 2024.  arXiv:2403.07721

Three stages:
  1) Channel-wise self-attention transformer
  2) Temporal-spatial conv (ShallowNet-style PatchEmbed)
  3) Residual MLP projector + classification head

This is a clean rewrite of old_analysis/encoder/atm_encoder.py with no
external conf-class dependency.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class _ChannelTransformer(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 128, n_heads: int = 4,
                 d_ff: int = 256, n_layers: int = 1, dropout: float = 0.25) -> None:
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.pos_enc = nn.Parameter(_sinusoidal_pe(128, d_model),
                                     requires_grad=False)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                          dim_feedforward=d_ff,
                                          dropout=dropout, activation="gelu",
                                          batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        h = self.input_proj(x)
        h = h + self.pos_enc[:, :C, :]
        h = self.encoder(h)
        h = self.norm(h)
        return self.output_proj(h)


class _PatchEmbed(nn.Module):
    """ShallowNet-style temporal+spatial conv collapsing channels."""
    def __init__(self, n_channels: int, seq_len: int,
                 emb_size: int = 40, dropout: float = 0.5) -> None:
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_channels, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv2d(40, emb_size, (1, 1))
        with torch.no_grad():
            d = torch.zeros(1, 1, n_channels, seq_len)
            self.flat_dim = int(self.proj(self.tsconv(d)).numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.proj(self.tsconv(x))
        return x.flatten(1)


class _ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.res = nn.Sequential(nn.GELU(),
                                  nn.Linear(out_dim, out_dim),
                                  nn.Dropout(dropout))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.lin(x)
        return self.norm(h + self.res(h))


class ATMEncoder(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 d_model: int = 128, n_heads: int = 4, d_ff: int = 256,
                 e_layers: int = 1, feat_dim: int = 256,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.ch_transformer = _ChannelTransformer(
            n_samples, d_model, n_heads, d_ff, e_layers, dropout)
        self.patch_embed = _PatchEmbed(n_channels, n_samples, dropout=dropout)
        self.proj = _ResidualMLP(self.patch_embed.flat_dim, feat_dim, dropout)
        self.feat_dim = feat_dim
        self.cls_head = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ch_transformer(x)
        h = self.patch_embed(h)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.forward_features(x))
