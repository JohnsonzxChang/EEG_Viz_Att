#!/usr/bin/env python3
"""
RSVP-COCO: Joint ATM Classification + CLIP Retrieval Training

Trains ATM encoder on 12-class RSVP EEG data with:
  1) Cross-entropy classification loss
  2) CLIP-style contrastive loss (EEG→Image alignment)
  3) Joint training (CE + CLIP)

Compares results with ML baseline (LDA: Top1=23.9%, Top3=43.6%).
Outputs figures to --save_dir/fig/.
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             top_k_accuracy_score, confusion_matrix, f1_score)

warnings.filterwarnings('ignore')

# ── Imports from project ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conf import BaseConfigRSVP
from encoder.atm_encoder import ATM_Encoder
from utils.data_loader_rsvp import Dataset_RSVP_COCO, RSVP_CATEGORIES, N_RSVP_CLASSES

# ── ML baseline from previous analysis ──────────────────────────────────────
ML_BASELINE = {
    'method': 'LDA + Feature Engineering',
    'top1': 0.2389,
    'top3': 0.4361,
    'bacc': 0.2389,
    'chance': 1.0 / N_RSVP_CLASSES,
}


# ═══ Modules ═════════════════════════════════════════════════════════════════

class EEGProjectionHead(nn.Module):
    """Maps EEG features → CLIP embedding space."""
    def __init__(self, in_dim, proj_dim=768, drop=0.3):
        super().__init__()
        hidden = proj_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        return F.normalize(self.norm(self.net(x)), dim=-1)


class CLIPContrastiveLoss(nn.Module):
    """Symmetric InfoNCE with learnable temperature."""
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, eeg_emb, clip_emb):
        B = eeg_emb.size(0)
        temp = self.log_temp.exp().clamp(0.01, 1.0)
        sim = eeg_emb @ clip_emb.T / temp
        labels = torch.arange(B, device=eeg_emb.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


# ═══ Data Loading ════════════════════════════════════════════════════════════

def build_dataloaders(config):
    """Create train/val datasets and loaders using the framework API."""
    ds_trn = Dataset_RSVP_COCO(config, seeds=config.seed)
    ds_trn.get_flag('trn')
    ds_val = Dataset_RSVP_COCO(config, seeds=config.seed)
    ds_val.get_flag('val')

    dl_trn = DataLoader(ds_trn, batch_size=config.batch_size,
                        shuffle=True, num_workers=0, pin_memory=True,
                        drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True,
                        drop_last=False)
    return ds_trn, dl_trn, ds_val, dl_val


# ═══ Evaluation ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_classification(model, loader, device, n_classes=N_RSVP_CLASSES):
    """Compute classification metrics on a dataloader."""
    model.eval()
    all_labels, all_preds, all_proba = [], [], []

    for batch in loader:
        x = batch['data'].to(device)
        y = batch['label'].to(device)
        logits = model(x)
        proba = F.softmax(logits, dim=-1)
        all_labels.append(y.cpu())
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_proba.append(proba.cpu())

    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    proba = torch.cat(all_proba).numpy()

    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    top3 = top_k_accuracy_score(labels, proba, k=3) if n_classes > 3 else acc
    f1_mac = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))

    return {
        'top1': float(acc), 'top3': float(top3), 'bacc': float(bacc),
        'f1_macro': float(f1_mac), 'cm': cm,
    }


@torch.no_grad()
def evaluate_retrieval(model, proj_head, loader, device, ks=[1, 3, 5, 10]):
    """Compute R@K retrieval metrics (EEG→Image)."""
    model.eval()
    proj_head.eval()
    all_eeg, all_img = [], []

    for batch in loader:
        x = batch['data'].to(device)
        img = batch['img_emb'].to(device)
        feat = model.forward_features(x)
        eeg_emb = proj_head(feat)
        all_eeg.append(eeg_emb.cpu())
        all_img.append(F.normalize(img, dim=-1).cpu())

    all_eeg = torch.cat(all_eeg)
    all_img = torch.cat(all_img)
    sim = all_eeg @ all_img.T
    N = all_eeg.size(0)
    labels = torch.arange(N)

    recalls = {}
    for k in ks:
        if k > N:
            continue
        topk = sim.topk(k, dim=1).indices
        hit = (topk == labels.unsqueeze(1)).any(dim=1)
        recalls[f'R@{k}'] = float(hit.float().mean().item())

    return recalls


# ═══ Training Modes ══════════════════════════════════════════════════════════

def train_classification_only(config, dl_trn, dl_val, device, save_dir):
    """Mode 1: Pure 12-class classification with CE loss."""
    print("\n" + "=" * 70)
    print("MODE 1: ATM Classification (CE Loss Only)")
    print("=" * 70)

    model = ATM_Encoder(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            betas=config.betas, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epoch, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = {'top1': 0, 'epoch': 0}
    best_state = None
    history = []

    for epoch in range(1, config.epoch + 1):
        model.train()
        losses = []
        t0 = time.time()

        for batch in dl_trn:
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        trn_loss = np.mean(losses)
        val = evaluate_classification(model, dl_val, device)

        if val['top1'] > best_val['top1']:
            best_val = {**val, 'epoch': epoch}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            'epoch': epoch, 'train_loss': trn_loss,
            'val_top1': val['top1'], 'val_top3': val['top3'], 'val_bacc': val['bacc'],
        })

        if epoch % 20 == 0 or epoch <= 5:
            print(f"  Ep {epoch:3d} | Loss={trn_loss:.4f} | "
                  f"Top1={val['top1']:.4f} Top3={val['top3']:.4f} "
                  f"BAcc={val['bacc']:.4f} | {time.time()-t0:.1f}s")

    # Final eval with best model
    model.load_state_dict(best_state)
    final = evaluate_classification(model, dl_val, device)
    print(f"\n  Best CE @ep{best_val['epoch']}: "
          f"Top1={final['top1']:.4f} Top3={final['top3']:.4f} "
          f"BAcc={final['bacc']:.4f}")

    return model, final, history


def train_clip_retrieval(config, dl_trn, dl_val, device, save_dir):
    """Mode 2: CLIP contrastive retrieval (EEG→Image)."""
    print("\n" + "=" * 70)
    print("MODE 2: ATM + CLIP Retrieval (InfoNCE Loss Only)")
    print("=" * 70)

    model = ATM_Encoder(config).to(device)
    proj = EEGProjectionHead(config.feat_dim, config.proj_dim,
                             drop=config.dropout).to(device)
    clip_loss = CLIPContrastiveLoss(config.temperature).to(device)
    alpha = config.alpha

    params = (list(model.parameters()) + list(proj.parameters())
              + [clip_loss.log_temp])
    optimizer = optim.AdamW(params, lr=config.learning_rate,
                            betas=config.betas, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epoch, eta_min=1e-6)

    best_r5 = 0
    history = []

    for epoch in range(1, config.epoch + 1):
        model.train()
        proj.train()
        losses = []
        t0 = time.time()

        for batch in dl_trn:
            x = batch['data'].to(device)
            img = batch['img_emb'].to(device)
            cap = batch['cap_emb'].to(device)

            optimizer.zero_grad()
            feat = model.forward_features(x)
            eeg_emb = proj(feat)
            img_n = F.normalize(img, dim=-1)
            cap_n = F.normalize(cap, dim=-1)

            loss = (alpha * clip_loss(eeg_emb, img_n)
                    + (1 - alpha) * clip_loss(eeg_emb, cap_n))
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        trn_loss = np.mean(losses)
        recalls = evaluate_retrieval(model, proj, dl_val, device, config.retrieval_ks)

        r5 = recalls.get('R@5', recalls.get('R@3', 0))
        if r5 > best_r5:
            best_r5 = r5
            best_recalls = recalls.copy()
            best_epoch = epoch

        history.append({'epoch': epoch, 'train_loss': trn_loss, **recalls})

        if epoch % 20 == 0 or epoch <= 5:
            rstr = ' '.join(f'{k}={v:.4f}' for k, v in recalls.items())
            temp = clip_loss.log_temp.exp().item()
            print(f"  Ep {epoch:3d} | Loss={trn_loss:.4f} | {rstr} | "
                  f"temp={temp:.4f} | {time.time()-t0:.1f}s")

    print(f"\n  Best Retrieval @ep{best_epoch}: "
          + ' '.join(f'{k}={v:.4f}' for k, v in best_recalls.items()))

    return model, proj, best_recalls, history


def train_joint(config, dl_trn, dl_val, device, save_dir):
    """Mode 3: Joint CE + CLIP loss (classification + retrieval)."""
    print("\n" + "=" * 70)
    print("MODE 3: Joint ATM (CE + CLIP) — Best of Both Worlds")
    print("=" * 70)

    model = ATM_Encoder(config).to(device)
    proj = EEGProjectionHead(config.feat_dim, config.proj_dim,
                             drop=config.dropout).to(device)
    clip_loss = CLIPContrastiveLoss(config.temperature).to(device)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    alpha = config.alpha
    ce_w = config.ce_weight
    clip_w = config.clip_weight

    params = (list(model.parameters()) + list(proj.parameters())
              + [clip_loss.log_temp])
    optimizer = optim.AdamW(params, lr=config.learning_rate,
                            betas=config.betas, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epoch, eta_min=1e-6)

    best_top1 = 0
    best_state = None
    best_proj_state = None
    history = []

    for epoch in range(1, config.epoch + 1):
        model.train()
        proj.train()
        losses_ce, losses_clip = [], []
        t0 = time.time()

        for batch in dl_trn:
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            img = batch['img_emb'].to(device)
            cap = batch['cap_emb'].to(device)

            optimizer.zero_grad()
            feat = model.forward_features(x)
            logits = model.cls_head(feat)
            eeg_emb = proj(feat)
            img_n = F.normalize(img, dim=-1)
            cap_n = F.normalize(cap, dim=-1)

            l_ce = ce_loss(logits, y)
            l_clip = (alpha * clip_loss(eeg_emb, img_n)
                      + (1 - alpha) * clip_loss(eeg_emb, cap_n))
            loss = ce_w * l_ce + clip_w * l_clip

            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            losses_ce.append(l_ce.item())
            losses_clip.append(l_clip.item())

        scheduler.step()
        trn_ce = np.mean(losses_ce)
        trn_clip = np.mean(losses_clip)

        val_cls = evaluate_classification(model, dl_val, device)
        val_ret = evaluate_retrieval(model, proj, dl_val, device, config.retrieval_ks)

        if val_cls['top1'] > best_top1:
            best_top1 = val_cls['top1']
            best_cls = {**val_cls}
            best_ret = val_ret.copy()
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_proj_state = {k: v.cpu().clone() for k, v in proj.state_dict().items()}

        history.append({
            'epoch': epoch, 'ce_loss': trn_ce, 'clip_loss': trn_clip,
            **{f'cls_{k}': v for k, v in val_cls.items() if k != 'cm'},
            **val_ret,
        })

        if epoch % 20 == 0 or epoch <= 5:
            rstr = ' '.join(f'{k}={v:.4f}' for k, v in val_ret.items())
            print(f"  Ep {epoch:3d} | CE={trn_ce:.4f} CLIP={trn_clip:.4f} | "
                  f"Top1={val_cls['top1']:.4f} Top3={val_cls['top3']:.4f} | "
                  f"{rstr} | {time.time()-t0:.1f}s")

    # Final eval
    model.load_state_dict(best_state)
    proj.load_state_dict(best_proj_state)
    final_cls = evaluate_classification(model, dl_val, device)
    final_ret = evaluate_retrieval(model, proj, dl_val, device, config.retrieval_ks)

    print(f"\n  Best Joint @ep{best_epoch}: "
          f"Top1={final_cls['top1']:.4f} Top3={final_cls['top3']:.4f} "
          f"BAcc={final_cls['bacc']:.4f}")
    print(f"  Retrieval: " + ' '.join(f'{k}={v:.4f}' for k, v in final_ret.items()))

    return model, proj, final_cls, final_ret, history


# ═══ Visualization ═══════════════════════════════════════════════════════════

def plot_comparison(results, save_dir):
    """Generate comparison figure: ML baseline vs deep learning models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('RSVP-COCO 12-Class: ML Baseline vs ATM Deep Learning\n'
                 'ERP-averaged (all trials), 360 images, 5-fold CV (ML) / 80-20 split (DL)',
                 fontsize=13, fontweight='bold')

    methods = ['ML Baseline\n(LDA)', 'ATM\n(CE Only)', 'ATM+CLIP\n(Joint)']
    colors = ['#9E9E9E', '#4CAF50', '#2196F3']

    # ── Panel 1: Top-1 / Top-3 / BAcc ──
    ax = axes[0]
    top1s = [ML_BASELINE['top1'], results['ce']['top1'], results['joint']['top1']]
    top3s = [ML_BASELINE['top3'], results['ce']['top3'], results['joint']['top3']]
    baccs = [ML_BASELINE['bacc'], results['ce']['bacc'], results['joint']['bacc']]
    x = np.arange(len(methods))
    w = 0.25
    ax.bar(x - w, top1s, w, color=colors, alpha=0.9, edgecolor='black', lw=0.5)
    ax.bar(x, top3s, w, color=colors, alpha=0.6, edgecolor='black', lw=0.5)
    ax.bar(x + w, baccs, w, color=colors, alpha=0.4, edgecolor='black', lw=0.5)
    for i in range(len(methods)):
        ax.text(i - w, top1s[i] + 0.01, f'{top1s[i]:.3f}', ha='center', fontsize=8)
        ax.text(i, top3s[i] + 0.01, f'{top3s[i]:.3f}', ha='center', fontsize=8)
        ax.text(i + w, baccs[i] + 0.01, f'{baccs[i]:.3f}', ha='center', fontsize=8)
    ax.axhline(ML_BASELINE['chance'], color='red', ls=':', lw=1.5,
               label=f'Chance ({ML_BASELINE["chance"]:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics')
    ax.legend(['Top-1', 'Top-3', 'BAcc', 'Chance'], loc='upper left', fontsize=8)
    ax.set_ylim(0, max(max(top3s) + 0.15, 0.6))
    ax.grid(True, alpha=0.2, axis='y')

    # ── Panel 2: Improvement over baseline ──
    ax = axes[1]
    metrics = ['Top-1', 'Top-3', 'BAcc']
    ce_improve = [(results['ce']['top1'] / ML_BASELINE['top1'] - 1) * 100,
                  (results['ce']['top3'] / ML_BASELINE['top3'] - 1) * 100,
                  (results['ce']['bacc'] / ML_BASELINE['bacc'] - 1) * 100]
    joint_improve = [(results['joint']['top1'] / ML_BASELINE['top1'] - 1) * 100,
                     (results['joint']['top3'] / ML_BASELINE['top3'] - 1) * 100,
                     (results['joint']['bacc'] / ML_BASELINE['bacc'] - 1) * 100]
    x = np.arange(len(metrics))
    ax.bar(x - 0.15, ce_improve, 0.3, color='#4CAF50', alpha=0.8, label='ATM (CE)')
    ax.bar(x + 0.15, joint_improve, 0.3, color='#2196F3', alpha=0.8, label='ATM+CLIP')
    for i in range(len(metrics)):
        ax.text(i - 0.15, ce_improve[i] + 1, f'{ce_improve[i]:+.1f}%', ha='center', fontsize=9)
        ax.text(i + 0.15, joint_improve[i] + 1, f'{joint_improve[i]:+.1f}%', ha='center', fontsize=9)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Improvement over ML Baseline (%)')
    ax.set_title('Relative Improvement')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # ── Panel 3: Training curves ──
    ax = axes[2]
    if 'joint_history' in results:
        hist = results['joint_history']
        epochs = [h['epoch'] for h in hist]
        t1 = [h.get('cls_top1', 0) for h in hist]
        t3 = [h.get('cls_top3', 0) for h in hist]
        ax.plot(epochs, t1, '-', color='#2196F3', lw=2, label='Joint Top-1')
        ax.plot(epochs, t3, '-', color='#2196F3', lw=1, alpha=0.5, label='Joint Top-3')
    if 'ce_history' in results:
        hist = results['ce_history']
        epochs = [h['epoch'] for h in hist]
        t1 = [h['val_top1'] for h in hist]
        ax.plot(epochs, t1, '--', color='#4CAF50', lw=2, label='CE Top-1')
    ax.axhline(ML_BASELINE['top1'], color='gray', ls=':', lw=2, label=f'ML Baseline ({ML_BASELINE["top1"]:.3f})')
    ax.axhline(ML_BASELINE['chance'], color='red', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Training Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_dl_vs_ml_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)

    # ── Confusion Matrix (Joint) ──
    cm = results['joint'].get('cm', None)
    if cm is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=max(0.3, cm_norm.max()))
        for i in range(N_RSVP_CLASSES):
            for j in range(N_RSVP_CLASSES):
                v = cm_norm[i, j]
                c = 'white' if v > 0.15 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=c)
        ax.set_xticks(range(N_RSVP_CLASSES))
        ax.set_yticks(range(N_RSVP_CLASSES))
        ax.set_xticklabels(RSVP_CATEGORIES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(RSVP_CATEGORIES, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'ATM+CLIP Joint: Confusion Matrix\n'
                     f'Top1={results["joint"]["top1"]:.3f} | '
                     f'Top3={results["joint"]["top3"]:.3f} | '
                     f'BAcc={results["joint"]["bacc"]:.3f}',
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Proportion', shrink=0.8)
        plt.tight_layout()
        p = os.path.join(save_dir, 'fig', 'rsvp_atm_clip_confusion.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close(fig)


# ═══ Main ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fif', type=str, default=None,
                        help='Override RSVP .fif path')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override save directory (default: same as fif)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip_weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Config
    config = BaseConfigRSVP()
    if args.fif:
        config.rsvp_fif_path = args.fif
    config.epoch = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.clip_weight = args.clip_weight
    config.seed = args.seed

    save_dir = args.save_dir or os.path.dirname(config.rsvp_fif_path)
    os.makedirs(os.path.join(save_dir, 'fig'), exist_ok=True)

    # Seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Data
    print("\n=== Loading RSVP-COCO data ===")
    ds_trn, dl_trn, ds_val, dl_val = build_dataloaders(config)

    # ── Mode 1: Classification only ──
    torch.manual_seed(config.seed)
    model_ce, res_ce, hist_ce = train_classification_only(
        config, dl_trn, dl_val, device, save_dir)

    # ── Mode 2: CLIP retrieval ──
    torch.manual_seed(config.seed)
    model_ret, proj_ret, res_ret, hist_ret = train_clip_retrieval(
        config, dl_trn, dl_val, device, save_dir)

    # ── Mode 3: Joint ──
    torch.manual_seed(config.seed)
    model_j, proj_j, res_joint, ret_joint, hist_joint = train_joint(
        config, dl_trn, dl_val, device, save_dir)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} {'Top-1':>8} {'Top-3':>8} {'BAcc':>8} {'Improve':>10}")
    print("-" * 70)
    print(f"{'Chance':<25} {ML_BASELINE['chance']:>8.4f} {3*ML_BASELINE['chance']:>8.4f} "
          f"{ML_BASELINE['chance']:>8.4f} {'—':>10}")
    print(f"{'ML Baseline (LDA)':<25} {ML_BASELINE['top1']:>8.4f} {ML_BASELINE['top3']:>8.4f} "
          f"{ML_BASELINE['bacc']:>8.4f} {'—':>10}")
    ce_imp = (res_ce['top1'] / ML_BASELINE['top1'] - 1) * 100
    print(f"{'ATM (CE only)':<25} {res_ce['top1']:>8.4f} {res_ce['top3']:>8.4f} "
          f"{res_ce['bacc']:>8.4f} {ce_imp:>+9.1f}%")
    j_imp = (res_joint['top1'] / ML_BASELINE['top1'] - 1) * 100
    print(f"{'ATM + CLIP (Joint)':<25} {res_joint['top1']:>8.4f} {res_joint['top3']:>8.4f} "
          f"{res_joint['bacc']:>8.4f} {j_imp:>+9.1f}%")
    print("-" * 70)
    print(f"\nRetrieval (Joint): " + ' '.join(f'{k}={v:.4f}' for k, v in ret_joint.items()))
    print(f"Retrieval (CLIP only): " + ' '.join(f'{k}={v:.4f}' for k, v in res_ret.items()))

    # ── Save results ──
    all_results = {
        'ml_baseline': ML_BASELINE,
        'ce': {k: v for k, v in res_ce.items() if k != 'cm'},
        'ce_history': hist_ce,
        'retrieval': res_ret,
        'retrieval_history': hist_ret,
        'joint': {k: v for k, v in res_joint.items() if k != 'cm'},
        'joint_retrieval': ret_joint,
        'joint_history': hist_joint,
        'config': {
            'model': config.model, 'enc_in': config.enc_in,
            'epochs': config.epoch, 'batch_size': config.batch_size,
            'lr': config.learning_rate, 'clip_weight': config.clip_weight,
            't_len': config.t_len, 'feat_dim': config.feat_dim,
            'proj_dim': config.proj_dim, 'e_layers': config.e_layers,
        }
    }
    jp = os.path.join(save_dir, 'fig', 'rsvp_dl_results.json')
    with open(jp, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON: {jp}")

    # ── Plot comparison ──
    plot_results = {
        'ce': res_ce,
        'joint': res_joint,
        'ce_history': hist_ce,
        'joint_history': hist_joint,
    }
    plot_comparison(plot_results, save_dir)

    print("\n=== ALL TRAINING COMPLETE ===")


if __name__ == '__main__':
    main()
