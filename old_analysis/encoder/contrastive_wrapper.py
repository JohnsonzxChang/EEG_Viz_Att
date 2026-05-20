"""
EncoderWithProjector
====================
Wraps any backbone encoder with a dedicated MLP projection head for
contrastive learning (Circle Loss), decoupled from the classification head.

Motivation (SimCLR, Chen et al. 2020):
  The contrastive objective should operate on a *separate* projection z,
  not on the backbone feature h.  This prevents the metric-learning and
  classification losses from competing on the same representation.

Data flow:
  x ──► backbone.forward_features(x) ──► h  (feat_dim)
                                          ├─► backbone.cls_head(h) ──► logits  → ASL
                                          └─► MLP projector(h)     ──► z       → Circle Loss

  forward_all(x) returns (z, logits)  — matching Exp_ClassificationCircle's interface.
  forward(x)     returns logits only  — for inference (projector is discarded).
"""

import torch
import torch.nn as nn


class EncoderWithProjector(nn.Module):
    """Wrap a backbone encoder with an MLP contrastive projection head.

    Args:
        backbone:  any encoder with forward_features() and cls_head attributes
        feat_dim:  backbone embedding dimension (input to projector)
        proj_dim:  projector output dimension (default 128)
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone

        # MLP projection head: Linear → BN → GELU → Linear
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x, padding_mask=None, enc_self_mask=None,
                dec_self_mask=None):
        """Standard forward — returns logits only (for inference)."""
        return self.backbone(x, padding_mask, enc_self_mask, dec_self_mask)

    def forward_features(self, x, padding_mask=None, enc_self_mask=None,
                         dec_self_mask=None):
        """Return backbone embedding h (before cls_head)."""
        return self.backbone.forward_features(x, padding_mask, enc_self_mask,
                                              dec_self_mask)

    def forward_all(self, x, padding_mask=None, enc_self_mask=None,
                    dec_self_mask=None):
        """Return (z, logits) where z is the MLP projection for Circle Loss."""
        h = self.backbone.forward_features(x, padding_mask, enc_self_mask,
                                           dec_self_mask)
        logits = self.backbone.cls_head(h)
        z = self.projector(h)
        return z, logits
