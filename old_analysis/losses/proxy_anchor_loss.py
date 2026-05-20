"""
Multi-Label Proxy-Anchor Loss
==============================
Reference: "Proxy Anchor Loss for Deep Metric Learning"
           Kim et al., CVPR 2020.  https://arxiv.org/abs/2003.13911

Each of the C classes has a learnable proxy vector (nn.Parameter).
Unlike pair-based Circle Loss, proxies are always available regardless
of batch composition — no "missing pair" problem for rare classes.

Multi-label adaptation: a sample has multiple positive proxies
(one per active label in its multi-hot vector).

Usage:
    loss_fn = ProxyAnchorLoss(num_classes=79, embedding_dim=128)
    loss = loss_fn(feats, labels)  # feats: (B,D), labels: (B,C) multi-hot
    # Remember to add loss_fn.parameters() to the optimizer!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor Loss with learnable per-class proxy vectors.

    Args:
        num_classes:    number of classes (79).
        embedding_dim:  embedding dimension (e.g. 128).
        scale:          temperature / scale factor. Default 32.
        delta:          margin. Default 0.1.
    """

    def __init__(self, num_classes: int, embedding_dim: int,
                 scale: float = 32.0, delta: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.delta = delta

        # Learnable proxy matrix: (C, D)
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats:  (B, D) — raw embeddings (L2-normalized inside).
            labels: (B, C) — multi-hot float labels.
        Returns:
            Scalar proxy-anchor loss.
        """
        z = F.normalize(feats, dim=-1)            # (B, D)
        p = F.normalize(self.proxies, dim=-1)     # (C, D)

        sim = z @ p.T                             # (B, C) cosine similarity

        is_pos = labels > 0                       # (B, C)
        is_neg = ~is_pos

        # ── Positive term: per proxy, aggregate over positive samples ──
        pos_exp = torch.exp(-self.scale * (sim - self.delta))
        pos_exp = pos_exp * is_pos.float()

        has_pos = is_pos.any(dim=0)                # (C,) — proxies with >=1 pos sample
        n_pos_proxies = has_pos.sum().clamp(min=1).float()

        pos_loss = torch.log(1.0 + pos_exp.sum(dim=0))    # (C,)
        pos_loss = (pos_loss * has_pos.float()).sum() / n_pos_proxies

        # ── Negative term: per proxy, aggregate over negative samples ──
        neg_exp = torch.exp(self.scale * (sim + self.delta))
        neg_exp = neg_exp * is_neg.float()

        neg_loss = torch.log(1.0 + neg_exp.sum(dim=0))    # (C,)
        neg_loss = neg_loss.sum() / self.num_classes

        return pos_loss + neg_loss
