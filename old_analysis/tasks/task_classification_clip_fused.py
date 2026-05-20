"""
CLIP-Fused EEG Classification with ERP Averaging.

Architecture (inspired by EEG-CLIP / BraVL / ATM):
    EEG (B, C, T) → encoder.forward_features() → eeg_feat (B, feat_dim)
    CLIP img_emb   (B, 768) → clip_proj          → clip_feat (B, feat_dim)

    Fusion: GatedFusion(eeg_feat, clip_feat) → fused (B, feat_dim)
    Classification: cls_head(fused) → logits (B, 80)

Training strategy:
    - Stage 1 (optional warm-up): train EEG encoder alone with ASL loss
    - Stage 2: joint training with CLIP fusion + ASL + optional contrastive alignment

The model also supports inference WITHOUT CLIP features (test-time), where
the gating mechanism learns to rely on EEG features alone.
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import multilabel_confusion_matrix

from conf import BaseConfig
from utils import LoggerFile
from .task_base import Exp_Basic
from utils.tools import EarlyStoppingCla, adjust_learning_rate

warnings.filterwarnings('ignore')


# ── Metrics (reuse from multilabel task) ─────────────────────────────────────

def topk_multilabel_accuracy_torch(logits, targets, k=3):
    _, topk_idx = torch.topk(logits, k=k, dim=1)
    B = targets.size(0)
    row_idx = torch.arange(B, device=logits.device).unsqueeze(1)
    hits = targets[row_idx, topk_idx] > 0
    sample_hit = hits.any(dim=1).float()
    return sample_hit.mean().item()


def compute_map(logits, targets):
    probs = torch.sigmoid(logits).detach().numpy()
    y_true = targets.detach().numpy().astype(int)
    ap_per_class = average_precision_score(y_true, probs, average=None)
    valid_classes = (np.sum(y_true, axis=0) > 0)
    if np.sum(valid_classes) == 0:
        mAP = 0.0
    else:
        mAP = np.mean(ap_per_class[valid_classes])
    return ap_per_class, mAP


def compute_f1_micro_macro(logits, targets, threshold=0.3):
    probs = torch.sigmoid(logits).detach().numpy()
    y_true = targets.detach().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    valid_classes = (np.sum(y_true, axis=0) > 0)
    if np.sum(valid_classes) == 0:
        macro_f1 = 0.0
    else:
        macro_f1 = np.mean(f1_per_class[valid_classes])
    return micro_f1, macro_f1


def compute_hamming_loss(logits, targets, threshold=0.3):
    probs = torch.sigmoid(logits).detach().numpy()
    y_true = targets.detach().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)
    return np.mean(y_true != y_pred)


def compute_subset_accuracy(logits, targets, threshold=0.3):
    probs = torch.sigmoid(logits).detach().numpy()
    y_true = targets.detach().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)
    return np.mean(np.all(y_pred == y_true, axis=1))


# ── Loss ─────────────────────────────────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    """ASL for multi-label (ICCV 2021)."""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss *= torch.pow(1.0 - pt, gamma)
        return -loss.sum() / logits.size(0)


# ── Gated Fusion Module ─────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """Gated cross-modal fusion for EEG and CLIP features.

    Reference: "Gated Multimodal Units for Information Fusion" (Arevalo et al., 2017)

    Given eeg_feat (B, D) and clip_feat (B, D):
        gate = σ(W_eeg · eeg_feat + W_clip · clip_feat + b)
        fused = gate * eeg_feat + (1 - gate) * clip_feat

    At test time without CLIP features, gate → 1 (relies on EEG only).
    """
    def __init__(self, feat_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_eeg = nn.Linear(feat_dim, feat_dim)
        self.gate_clip = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, eeg_feat: torch.Tensor, clip_feat: torch.Tensor = None):
        if clip_feat is None:
            return eeg_feat
        gate = torch.sigmoid(self.gate_eeg(eeg_feat) + self.gate_clip(clip_feat))
        fused = gate * eeg_feat + (1.0 - gate) * clip_feat
        fused = self.dropout(fused)
        fused = self.norm(fused)
        return fused


# ── CLIP Projection ──────────────────────────────────────────────────────────

class CLIPProjector(nn.Module):
    """Projects CLIP embeddings (768-d) into EEG feature space (feat_dim).
    Two-layer MLP with LayerNorm."""
    def __init__(self, clip_dim: int = 768, feat_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Alignment Loss (optional contrastive component) ─────────────────────────

class EEGCLIPAlignmentLoss(nn.Module):
    """Soft contrastive loss that encourages EEG and CLIP features to align
    in the shared feature space. Uses label-aware weighting.

    For samples sharing labels, their EEG and CLIP features should be similar.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_feat: torch.Tensor, clip_feat: torch.Tensor,
                labels: torch.Tensor = None):
        """
        eeg_feat:  (B, D)
        clip_feat: (B, D)
        labels:    (B, C) multi-hot — used for label-aware weighting
        """
        eeg_n = F.normalize(eeg_feat, dim=-1)
        clip_n = F.normalize(clip_feat, dim=-1)

        # Simple MSE alignment in normalized space
        align_loss = F.mse_loss(eeg_n, clip_n)

        if labels is not None:
            # InfoNCE-style cross-modal alignment
            B = eeg_n.size(0)
            sim = eeg_n @ clip_n.T / self.temperature  # (B, B)
            # Diagonal elements are the positive pairs (same sample)
            targets = torch.arange(B, device=eeg_n.device)
            info_loss = (F.cross_entropy(sim, targets) + F.cross_entropy(sim.T, targets)) / 2
            return 0.5 * align_loss + 0.5 * info_loss

        return align_loss


# ── Main Experiment ──────────────────────────────────────────────────────────

class Exp_ClassificationClipFused(Exp_Basic):
    """Multi-label EEG classification with CLIP feature fusion + ERP averaging.

    Training pipeline:
    1. Load ERP-averaged EEG data + precomputed CLIP embeddings
    2. Encode EEG → eeg_feat (via ATM/CNN/etc.)
    3. Project CLIP → clip_feat (via CLIPProjector)
    4. Gated fusion → fused_feat
    5. Classify → 80-class multi-hot logits
    6. Loss = ASL + λ_align * AlignmentLoss
    """

    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super().__init__(args, logger)
        self.session = session
        self.best_val_acc = 0.0

        feat_dim = getattr(args, 'feat_dim', 256)
        clip_dim = getattr(args, 'clip_dim', 768)
        n_classes = args.num_classes
        dropout = getattr(args, 'dropout', 0.3)

        # CLIP projector: 768 → feat_dim
        self.clip_proj = CLIPProjector(clip_dim, feat_dim, dropout).to(self.device)
        # Gated fusion
        self.fusion = GatedFusion(feat_dim, dropout=0.1).to(self.device)
        # Classification head (from fused features)
        self.cls_head = nn.Linear(feat_dim, n_classes).to(self.device)
        # Alignment loss
        self.align_loss_fn = EEGCLIPAlignmentLoss(
            temperature=getattr(args, 'align_temperature', 0.1)
        ).to(self.device)
        self.lambda_align = getattr(args, 'lambda_align', 0.1)
        self.use_clip_feat = getattr(args, 'use_clip_feat', True)

    def _build_model(self):
        assert self.args.task == 'classification'
        model = self.model_dict[self.args.model](self.args)
        return model

    def _get_data(self):
        """Use the ERP-averaged data loader."""
        from utils.data_loader_erp_avg import Dataset_UDF_VIZ_ERP_Avg
        from torch.utils.data import DataLoader

        data_set_trn = Dataset_UDF_VIZ_ERP_Avg(self.args, seeds=self.args.seed)
        data_set_trn.get_flag('trn')
        data_set_val = Dataset_UDF_VIZ_ERP_Avg(self.args, seeds=self.args.seed)
        data_set_val.get_flag('val')

        bs = self.args.batch_size
        nw = self.args.num_workers
        pm = self.args.pin_memory

        loader_trn = DataLoader(data_set_trn, batch_size=bs, shuffle=True,
                                num_workers=nw, pin_memory=pm, drop_last=False)
        loader_val = DataLoader(data_set_val, batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=pm, drop_last=False)
        return data_set_trn, loader_trn, data_set_val, loader_val

    def _select_optimizer(self):
        params = (
            list(self.model.parameters())
            + list(self.clip_proj.parameters())
            + list(self.fusion.parameters())
            + list(self.cls_head.parameters())
        )
        lr = self.args.learning_rate
        wd = getattr(self.args, 'weight_decay', 1e-5)
        betas = getattr(self.args, 'betas', (0.9, 0.98))
        opt = getattr(self.args, 'optimizer', 'adamw')
        if opt == 'adam':
            return optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)

    def _select_criterion(self, pos_weight=None):
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.0)
        print(f"Using AsymmetricLoss (gamma_neg=2, gamma_pos=1, clip=0.0)")
        return criterion

    def _forward(self, batch, training=True):
        """Full forward pass: EEG encode → CLIP project → fusion → classify."""
        data = batch['data'].to(self.device, non_blocking=True)
        regs = batch['regs'].to(self.device, non_blocking=True)
        subj = batch['subjects'].to(self.device, non_blocking=True)

        # EEG encoding
        eeg_feat = self.model.forward_features(data)  # (B, feat_dim)

        # CLIP feature fusion
        clip_feat = None
        if self.use_clip_feat and 'clip_img' in batch:
            clip_img = batch['clip_img'].to(self.device, non_blocking=True)
            clip_cap = batch['clip_cap'].to(self.device, non_blocking=True)
            # Average image and caption CLIP embeddings
            clip_combined = (clip_img + clip_cap) / 2.0
            clip_feat = self.clip_proj(clip_combined)  # (B, feat_dim)

        # Gated fusion
        fused = self.fusion(eeg_feat, clip_feat)  # (B, feat_dim)

        # Classification
        logits = self.cls_head(fused)  # (B, num_classes)

        return logits, eeg_feat, clip_feat, regs

    def vali(self, valid_loader, criterion, epoch):
        total_loss = []
        all_preds = []
        all_truths = []
        self.model.eval()
        self.clip_proj.eval()
        self.fusion.eval()
        self.cls_head.eval()

        with torch.no_grad():
            for batch in valid_loader:
                logits, eeg_feat, clip_feat, regs = self._forward(batch, training=False)
                loss = criterion(logits, regs)
                total_loss.append(loss)
                all_preds.append(logits.detach())
                all_truths.append(regs.detach())

            all_preds = torch.cat(all_preds, dim=0).cpu()
            all_truths = torch.cat(all_truths, dim=0).cpu()
            total_loss = torch.mean(torch.stack(total_loss)).cpu()

        micro_f1, macro_f1 = compute_f1_micro_macro(all_preds, all_truths, threshold=0.3)
        _, mAP = compute_map(all_preds, all_truths)
        vali_acc = topk_multilabel_accuracy_torch(all_preds, all_truths, k=3)
        hamming = compute_hamming_loss(all_preds, all_truths, threshold=0.3)
        subset_acc = compute_subset_accuracy(all_preds, all_truths, threshold=0.3)

        if mAP > self.best_val_acc:
            self.best_val_acc = mAP
            self._vali_plot_per_class(all_preds, all_truths, epoch)
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "clip_proj_state_dict": self.clip_proj.state_dict(),
                "fusion_state_dict": self.fusion.state_dict(),
                "cls_head_state_dict": self.cls_head.state_dict(),
                "preds": all_preds,
                "truths": all_truths,
                "mAP": mAP,
            }, f"{self.logger.writer.log_dir}/checkpoint.pth")

        self.model.train()
        self.clip_proj.train()
        self.fusion.train()
        self.cls_head.train()
        return total_loss, vali_acc, micro_f1, macro_f1, mAP, hamming, subset_acc

    def _vali_plot_per_class(self, all_preds, all_truths, step, threshold=0.5):
        from utils.data_loader import ALL_CLA
        probs = torch.sigmoid(all_preds).numpy()
        y_true = all_truths.numpy().astype(int)
        y_pred = (probs >= threshold).astype(int)

        C = y_true.shape[1]
        per_p = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_r = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_f = f1_score(y_true, y_pred, average=None, zero_division=0)

        fig, ax = plt.subplots(figsize=(10, max(8, C * 0.22)))
        y_pos = np.arange(C)
        bar_h = 0.25
        ax.barh(y_pos - bar_h, per_p, height=bar_h, label='Precision', color='#4C72B0')
        ax.barh(y_pos, per_r, height=bar_h, label='Recall', color='#DD8452')
        ax.barh(y_pos + bar_h, per_f, height=bar_h, label='F1', color='#55A868')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ALL_CLA[:C], fontsize=6)
        ax.set_xlabel('Score')
        ax.set_title(f'Per-class P/R/F1 (epoch {step})')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        self.logger.save_fig(fig, step, name='per_class_prf')
        plt.close(fig)

        mcm = multilabel_confusion_matrix(y_true, y_pred)
        tp = mcm[:, 1, 1]
        fp = mcm[:, 0, 1]
        fn = mcm[:, 1, 0]
        tn = mcm[:, 0, 0]
        mat = np.stack([tp, fp, fn, tn], axis=1)
        fig2, ax2 = plt.subplots(figsize=(6, max(8, C * 0.22)))
        im = ax2.imshow(mat, aspect='auto', cmap='YlOrRd')
        ax2.set_yticks(np.arange(C))
        ax2.set_yticklabels(ALL_CLA[:C], fontsize=6)
        ax2.set_xticks([0, 1, 2, 3])
        ax2.set_xticklabels(['TP', 'FP', 'FN', 'TN'], fontsize=8)
        ax2.set_title(f'Multi-label Confusion (epoch {step})')
        for i in range(C):
            for j in range(4):
                ax2.text(j, i, f'{int(mat[i, j])}', ha='center', va='center', fontsize=5)
        fig2.colorbar(im, ax=ax2, shrink=0.6)
        fig2.tight_layout()
        self.logger.save_fig(fig2, step, name='multilabel_confusion')
        plt.close(fig2)

    def train(self, setting):
        train_data, train_loader, valid_data, valid_loader = self._get_data()
        path = os.path.join(self.args.loggerdir, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.epoch, eta_min=1e-6
        )
        criterion = self._select_criterion()
        early_stopping = EarlyStoppingCla(patience=self.args.patience, verbose=True)

        self.model.train()
        self.clip_proj.train()
        self.fusion.train()
        self.cls_head.train()

        # Print model info
        total_params = (
            sum(p.numel() for p in self.model.parameters())
            + sum(p.numel() for p in self.clip_proj.parameters())
            + sum(p.numel() for p in self.fusion.parameters())
            + sum(p.numel() for p in self.cls_head.parameters())
        )
        print(f"Total trainable parameters: {total_params:,}")
        print(f"ERP mode: {getattr(self.args, 'erp_mode', 'N/A')}")
        print(f"CLIP fusion: {self.use_clip_feat}")
        print(f"Lambda align: {self.lambda_align}")

        for epoch in range(self.args.epoch):
            epoch_time = time.time()
            train_losses = []
            train_losses_cls = []
            train_losses_align = []
            train_preds = []
            train_truths = []

            self.model.train()
            self.clip_proj.train()
            self.fusion.train()
            self.cls_head.train()

            for batch in train_loader:
                model_optim.zero_grad()

                logits, eeg_feat, clip_feat, regs = self._forward(batch, training=True)

                # Classification loss (ASL)
                loss_cls = criterion(logits, regs)

                # Alignment loss (EEG ↔ CLIP feature alignment)
                loss_align = torch.tensor(0.0, device=self.device)
                if clip_feat is not None and self.lambda_align > 0:
                    loss_align = self.align_loss_fn(eeg_feat, clip_feat, regs)

                loss = loss_cls + self.lambda_align * loss_align
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + list(self.clip_proj.parameters())
                    + list(self.fusion.parameters())
                    + list(self.cls_head.parameters()),
                    max_norm=10.0,
                )
                model_optim.step()

                train_losses.append(loss.item())
                train_losses_cls.append(loss_cls.item())
                train_losses_align.append(loss_align.item())
                train_preds.append(logits.detach())
                train_truths.append(regs.detach())

            scheduler.step()

            with torch.no_grad():
                train_preds_cat = torch.cat(train_preds, dim=0).cpu()
                train_truths_cat = torch.cat(train_truths, dim=0).cpu()
                train_acc = topk_multilabel_accuracy_torch(train_preds_cat, train_truths_cat, k=3)
                train_loss = np.mean(train_losses)
                train_cls = np.mean(train_losses_cls)
                train_align = np.mean(train_losses_align)

            # Validation
            vali_loss, vali_acc, micro_f1, macro_f1, mAP, hamming, subset_acc = \
                self.vali(valid_loader, criterion, epoch + 1)

            # Logging
            self.logger.log_scalar('train_loss', train_loss, epoch + 1)
            self.logger.log_scalar('train_loss_cls', train_cls, epoch + 1)
            self.logger.log_scalar('train_loss_align', train_align, epoch + 1)
            self.logger.log_scalar('valid_loss', vali_loss, epoch + 1)
            self.logger.log_scalar('train_acc', train_acc, epoch + 1)
            self.logger.log_scalar('valid_acc', vali_acc, epoch + 1)
            self.logger.log_scalar('micro_f1', micro_f1, epoch + 1)
            self.logger.log_scalar('macro_f1', macro_f1, epoch + 1)
            self.logger.log_scalar('mAP', mAP, epoch + 1)
            self.logger.log_scalar('hamming_loss', hamming, epoch + 1)
            self.logger.log_scalar('subset_acc', subset_acc, epoch + 1)
            self.logger.log_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)

            print(
                f"Epoch {epoch+1:3d} [{time.time()-epoch_time:.1f}s] | "
                f"TrnLoss={train_loss:.4f} (cls={train_cls:.4f} align={train_align:.4f}) | "
                f"TrnAcc={train_acc:.3f} | "
                f"ValLoss={vali_loss:.4f} ValAcc={vali_acc:.3f} | "
                f"mAP={mAP:.4f} microF1={micro_f1:.3f} macroF1={macro_f1:.3f} | "
                f"Hamming={hamming:.4f} SubsetAcc={subset_acc:.3f}"
            )

            early_stopping(mAP, self.model, path, epoch)

            if self.session is not None:
                self.session.report({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "vali_loss": vali_loss.item(),
                    "mAP": mAP,
                })

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model
