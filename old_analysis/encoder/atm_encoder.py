"""
Adaptive Thinking Mapper (ATM) Encoder — minimal single-file implementation.

Reference: "Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion"
           Li et al., NeurIPS 2024.  https://arxiv.org/abs/2403.07721

Architecture:
  1) Channel-wise Transformer  — self-attention across EEG channels
  2) Temporal-Spatial Conv      — PatchEmbedding (ShallowNet-style)
  3) Residual MLP Projector     — feature → logits
"""

import math
import torch
import torch.nn as nn
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


# ── 1) Channel-wise Transformer ─────────────────────────────────────────────

class ChannelTransformer(nn.Module):
    """Treat each EEG channel as a token; apply self-attention across channels.

    Input:  (B, C, T)
    Output: (B, C, T)   — same shape, but with cross-channel information mixed.
    """
    def __init__(self, seq_len, d_model=128, n_heads=4, d_ff=256,
                 n_layers=1, dropout=0.25):
        super().__init__()
        # project T → d_model (token embedding)
        self.input_proj = nn.Linear(seq_len, d_model)
        # sinusoidal positional encoding for C channels
        self.pos_enc = nn.Parameter(self._sinusoidal_pe(128, d_model), requires_grad=False)
        # transformer encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # project back d_model → T
        self.output_proj = nn.Linear(d_model, seq_len)

    @staticmethod
    def _sinusoidal_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        h = self.input_proj(x)                      # (B, C, d_model)
        h = h + self.pos_enc[:, :C, :]              # add positional encoding
        h = self.encoder(h)                          # (B, C, d_model)
        h = self.norm(h)
        out = self.output_proj(h)                    # (B, C, T)
        return out


# ── 2) Temporal-Spatial Convolution (PatchEmbedding) ─────────────────────────

class PatchEmbedding(nn.Module):
    """ShallowNet-style temporal → spatial convolution.

    Input:  (B, C, T)
    Output: (B, flat_dim)
    """
    def __init__(self, num_channels, seq_len, emb_size=40, dropout=0.5):
        super().__init__()
        self.tsconv = nn.Sequential(
            # temporal conv: (B, 1, C, T) → (B, 40, C, T')
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # spatial conv: collapse channel dim
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

        # compute output flat dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, seq_len)
            out = self.tsconv(dummy)
            out = self.projection(out)              # (1, emb, 1, T')
            self.flat_dim = out.numel()

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)                         # (B, 1, C, T)
        x = self.tsconv(x)
        x = self.projection(x)                     # (B, emb, 1, T')
        x = x.contiguous().view(x.size(0), -1)     # (B, flat_dim)
        return x


# ── 3) Residual MLP Projector ────────────────────────────────────────────────

class Projector(nn.Module):
    """MLP with one residual block + LayerNorm."""
    def __init__(self, in_dim, proj_dim, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── ATM Encoder (main) ──────────────────────────────────────────────────────

class ATM_Encoder(nn.Module):
    """Adaptive Thinking Mapper for EEG multi-label classification.

    Args (read from args):
        enc_in:      number of EEG channels  (default 33)
        t_len:       temporal length          (default 500)
        num_classes: output classes           (default 79)
        dropout:     dropout rate             (default 0.3)
        n_heads:     transformer heads        (default 4)
        d_model:     transformer hidden dim   (default 128)
        d_ff:        feedforward dim           (default 256)
        e_layers:    transformer layers        (default 1)
    """
    def __init__(self, args):
        super().__init__()
        C = args.enc_in
        T = args.t_len
        n_cls = args.num_classes
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'n_heads', 4)
        d_ff = getattr(args, 'd_ff', 256)
        n_layers = getattr(args, 'e_layers', 3)
        dropout = getattr(args, 'dropout', 0.3)
        emb_size = 40
        proj_dim = getattr(args, 'feat_dim', 256)

        # stage 1: channel-wise transformer
        self.ch_transformer = ChannelTransformer(
            seq_len=T, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, n_layers=n_layers, dropout=dropout,
        )

        # stage 2: temporal-spatial conv
        self.patch_embed = PatchEmbedding(
            num_channels=C, seq_len=T, emb_size=emb_size, dropout=dropout,
        )

        # stage 3: projector → feature
        flat_dim = self.patch_embed.flat_dim
        self.proj = Projector(flat_dim, proj_dim, drop=dropout)

        # classification head
        self.cls_head = nn.Linear(proj_dim, n_cls)

    def forward_features(self, x, padding_mask=None, enc_self_mask=None,
                         dec_self_mask=None):
        # x: (B, C, T)
        h = self.ch_transformer(x)     # channel-wise attention
        h = self.patch_embed(h)         # temporal-spatial conv → flat
        feat = self.proj(h)             # residual MLP → (B, proj_dim)
        return feat

    def forward(self, x, padding_mask=None, enc_self_mask=None,
                dec_self_mask=None):
        feat = self.forward_features(x)
        return self.cls_head(feat)      # (B, num_classes)

    def forward_all(self, x, padding_mask=None, enc_self_mask=None,
                    dec_self_mask=None):
        feat = self.forward_features(x)
        logits = self.cls_head(feat)
        return feat, logits


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.enc_in = 33
            self.t_len = 500
            self.num_classes = 79
            self.d_model = 128
            self.n_heads = 4
            self.d_ff = 256
            self.e_layers = 1
            self.dropout = 0.3
            self.feat_dim = 256

    args = Args()
    model = ATM_Encoder(args)
    x = torch.randn(4, 33, 500)
    logits = model(x)
    print(f"Input: {x.shape} → Output: {logits.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
