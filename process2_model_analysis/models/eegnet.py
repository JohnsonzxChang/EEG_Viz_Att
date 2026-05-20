"""EEGNet (Lawhern et al. 2018, J. Neural Eng.) — adapted from old_analysis/encoder/eegnet_encoder.

Input  : (B, C, T)  raw EEG in µV.
Output : (B, feat_dim) features (forward_features) or (B, n_classes) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_t: int = 64, kernel_t2: int = 16,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_t), padding=(0, kernel_t // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3),
            _Conv2dWithConstraint(F1, F1 * D, (n_channels, 1), max_norm=1.0,
                                   groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, kernel_t2),
                      padding=(0, kernel_t2 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            f = self.block2(self.block1(dummy))
            self.feat_dim = int(f.numel())
        self.cls_head = nn.Linear(self.feat_dim, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.forward_features(x))
