"""Three baseline classifiers for attention decoding + patch-level fusion:

  1) ImgOnlyClassifier   — predict HINT from image embedding only (visual prior)
  2) EEGOnlyClassifier   — predict HINT from EEG features only
  3) EEGImgFusionClassifier — concatenated features + cross-attention (global)
  4) EEGImgPatchFusion   — cross-attention over **patch tokens**; exposes
       attention weights (B, P) for EEG→pic attention visualisation.

The fusion architecture is intentionally simple so the gain from adding EEG
can be attributed to information content, not extra capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────── image-only ────────────────────

class ImgOnlyClassifier(nn.Module):
    def __init__(self, img_dim: int, n_classes: int,
                 hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        return self.net(img_feat)


# ──────────────────── EEG-only ────────────────────

class EEGOnlyClassifier(nn.Module):
    """Wrap any EEG encoder (EEGNet / ATMEncoder)."""
    def __init__(self, eeg_encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = eeg_encoder

    def forward(self, x_eeg: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_eeg)


# ──────────────────── EEG + Img global fusion ────────────────────

class CrossAttentionFusion(nn.Module):
    def __init__(self, eeg_dim: int, img_dim: int, d_model: int = 256,
                 n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_proj = nn.Linear(eeg_dim, d_model)
        self.kv_proj = nn.Linear(img_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, eeg_feat, img_feat):
        q = self.q_proj(eeg_feat).unsqueeze(1)
        kv = self.kv_proj(img_feat).unsqueeze(1)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm(out.squeeze(1) + q.squeeze(1))


class EEGImgFusionClassifier(nn.Module):
    """EEG encoder + image MLP joined by cross-attention (single token)."""
    def __init__(self, eeg_encoder: nn.Module, eeg_dim: int,
                 img_dim: int, n_classes: int,
                 d_model: int = 256, n_heads: int = 4, dropout: float = 0.3,
                 freeze_img: bool = True) -> None:
        super().__init__()
        self.eeg_encoder = eeg_encoder
        if hasattr(self.eeg_encoder, "cls_head"):
            self.eeg_encoder.cls_head = nn.Identity()
        self.fuse = CrossAttentionFusion(eeg_dim, img_dim, d_model, n_heads,
                                          dropout=dropout)
        self.img_mlp = nn.Sequential(
            nn.Linear(img_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(d_model * 2, n_classes)
        self.freeze_img = freeze_img

    def forward(self, x_eeg, img_feat, img_patches=None):
        if self.freeze_img:
            img_feat = img_feat.detach()
        h_eeg = self.eeg_encoder.forward_features(x_eeg)
        h_fused = self.fuse(h_eeg, img_feat)
        h_img = self.img_mlp(img_feat)
        h = torch.cat([h_fused, h_img], dim=-1)
        return self.cls(h)


# ──────────────────── EEG + Img patch fusion (★ EEG→pic attn) ────────────────────

class EEGImgPatchFusion(nn.Module):
    """EEG encoder + image patch tokens, joined by multi-head cross-attention.

    Query  = projected EEG feature (B, 1, d_model)  → also gets a learnable
             token, so we can readout a fused vector.
    Key/V  = projected patch tokens (B, P, d_model)

    During inference, `forward_with_attn(x_eeg, patches)` additionally returns
    the per-head attention weights averaged across heads → (B, P) — this is
    the EEG→pic attention map used to overlay onto the stimulus image.
    """
    def __init__(self, eeg_encoder: nn.Module, eeg_dim: int,
                 patch_dim: int, img_dim: int, n_classes: int,
                 d_model: int = 256, n_heads: int = 4, dropout: float = 0.3,
                 freeze_img: bool = True) -> None:
        super().__init__()
        self.eeg_encoder = eeg_encoder
        if hasattr(self.eeg_encoder, "cls_head"):
            self.eeg_encoder.cls_head = nn.Identity()
        self.q_proj = nn.Linear(eeg_dim, d_model)
        self.kv_proj = nn.Linear(patch_dim, d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        # Also keep the global image vector path for stability:
        self.img_mlp = nn.Sequential(
            nn.Linear(img_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(d_model * 2, n_classes)
        self.freeze_img = freeze_img
        self.n_heads = n_heads
        self.d_model = d_model

    def _encode(self, x_eeg, img_feat, img_patches):
        if self.freeze_img:
            img_feat = img_feat.detach()
            img_patches = img_patches.detach()
        h_eeg = self.eeg_encoder.forward_features(x_eeg)         # (B, eeg_dim)
        q = self.q_proj(h_eeg).unsqueeze(1)                       # (B, 1, D)
        kv = self.kv_norm(self.kv_proj(img_patches))              # (B, P, D)
        return h_eeg, q, kv

    def forward(self, x_eeg, img_feat, img_patches):
        _, q, kv = self._encode(x_eeg, img_feat, img_patches)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        fused = self.attn_norm(out.squeeze(1) + q.squeeze(1))     # (B, D)
        h_img = self.img_mlp(img_feat.detach() if self.freeze_img else img_feat)
        return self.cls(torch.cat([fused, h_img], dim=-1))

    @torch.no_grad()
    def forward_with_attn(self, x_eeg, img_feat, img_patches):
        _, q, kv = self._encode(x_eeg, img_feat, img_patches)
        out, attn_w = self.attn(q, kv, kv,
                                 need_weights=True, average_attn_weights=True)
        # attn_w: (B, 1, P) — squeeze
        attn = attn_w.squeeze(1)                                  # (B, P)
        fused = self.attn_norm(out.squeeze(1) + q.squeeze(1))
        h_img = self.img_mlp(img_feat)
        logits = self.cls(torch.cat([fused, h_img], dim=-1))
        return logits, attn
