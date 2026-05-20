"""
EEG → CLIP cross-modal retrieval via symmetric InfoNCE (CLIP-style contrastive loss).

Pipeline:
    EEG (B, C, T)
        → encoder.forward_features()   → (B, feat_dim)
        → EEGProjectionHead             → (B, proj_dim=768)  L2-normalized

    CLIP img_emb (B, 768) — pre-cached, L2-normalized at runtime
    CLIP cap_emb (B, 768) — pre-cached, L2-normalized at runtime

    Loss = alpha * InfoNCE(eeg, img) + (1-alpha) * InfoNCE(eeg, cap)

Evaluation: Recall@K  (EEG query → image gallery)
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from conf import BaseConfig
from utils import LoggerFile
from utils.data_factory import get_data_loader_cutt
from .task_base import Exp_Basic

warnings.filterwarnings('ignore')


# ── Contrastive Loss ─────────────────────────────────────────────────────────

class CLIPContrastiveLoss(nn.Module):
    """Symmetric InfoNCE with a learnable (log-)temperature parameter."""

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature))
        )

    def forward(self, eeg_emb: torch.Tensor, clip_emb: torch.Tensor) -> torch.Tensor:
        """
        eeg_emb  : (B, D)  L2-normalized
        clip_emb : (B, D)  L2-normalized
        """
        B = eeg_emb.size(0)
        temp = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        sim = eeg_emb @ clip_emb.T / temp          # (B, B)
        labels = torch.arange(B, device=eeg_emb.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss


# ── Projection Head ───────────────────────────────────────────────────────────

class EEGProjectionHead(nn.Module):
    """Two-layer MLP that maps EEG features into CLIP embedding space."""

    def __init__(self, in_dim: int, proj_dim: int = 768, drop: float = 0.3):
        super().__init__()
        hidden = proj_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.norm(x)
        return F.normalize(x, dim=-1)   # L2-normalize → unit hypersphere


# ── Retrieval Experiment ──────────────────────────────────────────────────────

class Exp_Retrieval(Exp_Basic):
    """EEG-to-CLIP cross-modal retrieval experiment."""

    def __init__(self, args: BaseConfig, logger: LoggerFile = None):
        super().__init__(args, logger)

        feat_dim    = getattr(args, 'feat_dim', 256)
        proj_dim    = getattr(args, 'proj_dim', 768)
        drop        = getattr(args, 'dropout', 0.3)
        temperature = getattr(args, 'temperature', 0.07)

        self.proj     = EEGProjectionHead(feat_dim, proj_dim, drop=drop).to(self.device)
        self.loss_fn  = CLIPContrastiveLoss(init_temperature=temperature).to(self.device)
        self.alpha    = getattr(args, 'alpha', 0.5)
        self.retrieval_ks = getattr(args, 'retrieval_ks', [1, 5, 10, 50])
        self.best_val_rk  = 0.0

    # ── abstract methods ──────────────────────────────────────────────────────

    def _build_model(self):
        return self.model_dict[self.args.model](self.args)

    def _get_data(self):
        return get_data_loader_cutt(self.args)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, proj_dim) L2-normalized."""
        feat = self.model.forward_features(x)   # (B, feat_dim)
        return self.proj(feat)                  # (B, proj_dim)

    def _select_optimizer(self):
        params = (
            list(self.model.parameters())
            + list(self.proj.parameters())
            + [self.loss_fn.log_temperature]
        )
        lr    = self.args.learning_rate
        wd    = getattr(self.args, 'weight_decay', 1e-5)
        betas = getattr(self.args, 'betas', (0.9, 0.98))
        if getattr(self.args, 'optimizer', 'adamw') == 'adam':
            return optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)

    # ── validation ────────────────────────────────────────────────────────────

    def vali(self, valid_loader, epoch: int):
        self.model.eval()
        self.proj.eval()
        all_eeg, all_img, losses = [], [], []

        with torch.no_grad():
            for batch in valid_loader:
                x   = batch['data'].to(self.device, non_blocking=True)
                img = batch['img_emb'].to(self.device, non_blocking=True)
                cap = batch['cap_emb'].to(self.device, non_blocking=True)

                eeg_emb = self._encode(x)
                img_n   = F.normalize(img, dim=-1)
                cap_n   = F.normalize(cap, dim=-1)

                loss = (self.alpha       * self.loss_fn(eeg_emb, img_n)
                        + (1-self.alpha) * self.loss_fn(eeg_emb, cap_n))
                losses.append(loss.item())
                all_eeg.append(eeg_emb.cpu())
                all_img.append(img_n.cpu())

        all_eeg = torch.cat(all_eeg, dim=0)   # (N, D)
        all_img = torch.cat(all_img, dim=0)   # (N, D)
        sim     = all_eeg @ all_img.T         # (N, N)
        N       = all_eeg.size(0)
        labels  = torch.arange(N)

        recalls = {}
        for k in self.retrieval_ks:
            topk_idx = sim.topk(k, dim=1).indices       # (N, k)
            hit = (topk_idx == labels.unsqueeze(1)).any(dim=1)
            recalls[f'R@{k}'] = hit.float().mean().item()

        # Save checkpoint on best Recall@10 (or largest K available)
        ck_key = 'R@10' if 'R@10' in recalls else f'R@{self.retrieval_ks[-1]}'
        if recalls[ck_key] > self.best_val_rk:
            self.best_val_rk = recalls[ck_key]
            torch.save({
                "epoch": epoch,
                "model": self.model.state_dict(),
                "proj":  self.proj.state_dict(),
                "recalls": recalls,
            }, f"{self.logger.writer.log_dir}/checkpoint.pth")

        self.model.train()
        self.proj.train()
        return float(np.mean(losses)), recalls

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self, setting: str):
        train_data, train_loader, valid_data, valid_loader = self._get_data()
        path = os.path.join(self.args.loggerdir, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.epoch, eta_min=1e-6
        )

        self.model.train()
        self.proj.train()

        for epoch in range(self.args.epoch):
            epoch_losses = []
            t0 = time.time()

            for batch in train_loader:
                x   = batch['data'].to(self.device, non_blocking=True)
                img = batch['img_emb'].to(self.device, non_blocking=True)
                cap = batch['cap_emb'].to(self.device, non_blocking=True)

                model_optim.zero_grad()
                eeg_emb = self._encode(x)
                img_n   = F.normalize(img, dim=-1)
                cap_n   = F.normalize(cap, dim=-1)

                loss = (self.alpha       * self.loss_fn(eeg_emb, img_n)
                        + (1-self.alpha) * self.loss_fn(eeg_emb, cap_n))
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.proj.parameters()),
                    max_norm=5.0,
                )
                model_optim.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            train_loss = float(np.mean(epoch_losses))
            vali_loss, recalls = self.vali(valid_loader, epoch + 1)
            temp = self.loss_fn.log_temperature.exp().item()

            # TensorBoard logging
            self.logger.log_scalar('train_loss',  train_loss,  epoch + 1)
            self.logger.log_scalar('vali_loss',   vali_loss,   epoch + 1)
            self.logger.log_scalar('temperature', temp,        epoch + 1)
            self.logger.log_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)
            for k, v in recalls.items():
                self.logger.log_scalar(k, v, epoch + 1)

            recall_str = '  '.join(f'{k}={v:.4f}' for k, v in recalls.items())
            print(
                f"Epoch {epoch+1:3d}  {time.time()-t0:.1f}s  "
                f"TrnLoss={train_loss:.4f}  ValLoss={vali_loss:.4f}  "
                f"temp={temp:.4f}  {recall_str}"
            )

        return self.model
