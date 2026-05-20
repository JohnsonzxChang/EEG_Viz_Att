"""
Multi-label Circle Loss
=======================
Reference: "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
           Sun et al., CVPR 2020.  https://arxiv.org/abs/2002.10857

Adaptation for multi-label classification
------------------------------------------
For anchor i, given L2-normalized embeddings z and multi-hot label vectors y:

  Positive pairs  P(i) = {j ≠ i : dot(y_i, y_j) > 0}   — share ≥1 COCO label
  Negative pairs  N(i) = {j ≠ i : dot(y_i, y_j) = 0}   — share no label

Per-pair adaptive weights (stop-gradient):
  α_p = max(0, O_p − s_p),   O_p = 1 + m     ← how far below optimal pos sim
  α_n = max(0, s_n − O_n),   O_n = −m         ← how far above optimal neg sim

Decision boundaries:
  Δ_p = 1 − m  (positives should exceed this)
  Δ_n = m      (negatives should stay below this)

Loss per anchor i:
  L_i = softplus(
            logsumexp_n [ γ · α_n · (s_n − Δ_n) ]
          + logsumexp_p [−γ · α_p · (s_p − Δ_p) ]
        )

Optional Jaccard soft-weight
-----------------------------
When use_jaccard=True the α_p for each positive pair (i,j) is further
multiplied by Jaccard(y_i, y_j) ∈ (0,1].  Pairs that share more labels
get a proportionally stronger attraction signal.

Usage
-----
    loss_fn = MultiLabelCircleLoss(gamma=64, margin=0.25, use_jaccard=True)
    feat  = F.normalize(encoder(x), dim=-1)   # (B, D)
    loss  = loss_fn(feat, labels)             # labels: (B, C) float multi-hot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_NEG_INF = -1e9   # mask value for logsumexp padding


class MultiLabelCircleLoss(nn.Module):
    """Circle Loss adapted for multi-label positive/negative assignment.

    Args:
        gamma (float): Scale factor γ.  Default: 64.
        margin (float): Margin m.  Δ_p = 1-m, Δ_n = m.  Default: 0.25.
        use_jaccard (bool): If True, weight positive logits by Jaccard
            similarity of label vectors.  Provides a smooth signal
            proportional to label overlap degree.  Default: True.
    """

    def __init__(self, gamma: float = 64, margin: float = 0.25, use_jaccard: bool = True):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.use_jaccard = use_jaccard

        # Pre-compute constants
        self.O_p = 1.0 + margin   # optimal positive similarity
        self.O_n = -margin        # optimal negative similarity
        self.delta_p = 1.0 - margin   # pos decision boundary
        self.delta_n = margin         # neg decision boundary

    # ------------------------------------------------------------------
    def forward(self, feats: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            feats:  (B, D)  L2-normalized embeddings.
            labels: (B, C)  Multi-hot label vectors (float, 0/1).

        Returns:
            Scalar loss (mean over valid anchors).
        """
        B = feats.size(0)
        device = feats.device

        # ── Pairwise cosine similarity (feats already L2-normalized) ──
        sim = feats @ feats.T                         # (B, B)

        # ── Multi-label pos / neg masks ───────────────────────────────
        # label_overlap[i,j] = number of shared labels
        label_overlap = labels @ labels.T             # (B, B)  float
        eye = torch.eye(B, dtype=torch.bool, device=device)

        is_pos = (label_overlap > 0) & ~eye           # (B, B) bool
        is_neg = (label_overlap == 0) & ~eye          # (B, B) bool

        # ── Per-pair adaptive weights (detached — no gradient) ────────
        alpha_pos = (self.O_p - sim).clamp(min=0).detach()   # (B, B)
        alpha_neg = (sim - self.O_n).clamp(min=0).detach()   # (B, B)

        # ── Optional Jaccard soft-weight and margin scaling for positives ───────
        if self.use_jaccard:
            # Jaccard(i,j) = |y_i ∩ y_j| / |y_i ∪ y_j|
            label_counts = labels.sum(dim=1, keepdim=True)         # (B,1)
            union = label_counts + label_counts.T - label_overlap  # (B,B)
            jaccard = (label_overlap / union.clamp(min=1e-8)).clamp(0, 1)  # (B,B)
            
            # Dynamic targets: instead of targeting identical anchors (sim=1.0)
            # We target a similarity score proportional to the Jaccard overlap.
            O_p_dynamic = jaccard + self.margin
            delta_p_dynamic = jaccard - self.margin 
            
            # Reconstruct alpha_pos relative to the dynamic optimum
            alpha_pos = (O_p_dynamic - sim).clamp(min=0).detach()
            
            # Scale the alpha weight by jaccard to soften the gradient for partial matches
            alpha_pos = alpha_pos * jaccard.detach()
            
            # Recalculate pos_logit with dynamic boundary
            pos_logit = -self.gamma * alpha_pos * (sim - delta_p_dynamic)
        else:
            pos_logit = -self.gamma * alpha_pos * (sim - self.delta_p)

        pos_logit = pos_logit.masked_fill(~is_pos, _NEG_INF)

        # Negative term: logsumexp_n[ γ · α_n · (s_n - Δ_n) ]
        neg_logit = self.gamma * alpha_neg * (sim - self.delta_n)
        neg_logit = neg_logit.masked_fill(~is_neg, _NEG_INF)

        log_pos = torch.logsumexp(pos_logit, dim=1)   # (B,)
        log_neg = torch.logsumexp(neg_logit, dim=1)   # (B,)

        # ── Keep only anchors that have both a positive and a negative ─
        has_both = is_pos.any(dim=1) & is_neg.any(dim=1)  # (B,)
        if not has_both.any():
            return feats.sum() * 0.0    # keep in graph, return 0

        per_anchor_loss = F.softplus(log_neg[has_both] + log_pos[has_both])
        return per_anchor_loss.mean()
