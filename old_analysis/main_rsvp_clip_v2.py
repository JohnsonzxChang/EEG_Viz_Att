#!/usr/bin/env python3
"""
RSVP-COCO V2: ATM + Category-Level CLIP Contrastive Learning

V1 Issue: Whole-image CLIP embeddings don't capture "attention target" info.
  Same COCO image → same CLIP emb, but different EEG (different target category).
  → Conflicting gradients → CLIP retrieval fails.

V2 Fix: Use CLIP *text* embeddings of category names as contrastive targets.
  "a photo of a dog" → 768-dim CLIP text embedding → unique per category.
  EEG should align with the attended category's text representation.

Training modes:
  1) CE only: ATM → 12-class cross-entropy
  2) Category-CLIP: ATM → projection → align with CLIP text of target category
  3) Joint: CE + Category-CLIP (supervised contrastive regularization)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conf import BaseConfigRSVP
from encoder.atm_encoder import ATM_Encoder
# from utils.data_loader_rsvp import Dataset_RSVP_COCO, RSVP_CATEGORIES, N_RSVP_CLASSES
from utils.data_loader_viz_aud import Dataset_ERP_VGG, N_ERP_CLASSES, ERP_CATEGORIES  # Reusing the ERP dataset class for RSVP

N_RSVP_CLASSES = N_ERP_CLASSES  # Reusing the ERP class count for RSVP
RSVP_CATEGORIES = ERP_CATEGORIES  # Reusing the ERP category names for RSVP

# ── ML baseline ─────────────────────────────────────────────────────────────
ML_BASELINE = {
    'method': 'LDA + Feature Engineering',
    'top1': 0.2389, 'top3': 0.4361, 'bacc': 0.2389,
    'chance': 1.0 / N_RSVP_CLASSES,
}


# ═══ Modules ═════════════════════════════════════════════════════════════════

class EEGProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=768, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        return F.normalize(self.norm(self.net(x)), dim=-1)


class CategoryCLIPLoss(nn.Module):
    """Supervised contrastive: pull EEG toward its category's CLIP text embedding.

    Unlike vanilla InfoNCE (1-to-1 pairing), here multiple samples share the
    same target embedding (all dogs → "a photo of a dog").
    We use a cross-entropy formulation: similarity with 12 class prototypes.
    """
    def __init__(self, cat_embeds: torch.Tensor, init_temp=0.1):
        """
        cat_embeds: (12, 768) L2-normalized CLIP text embeddings for each category.
        """
        super().__init__()
        self.register_buffer('prototypes', cat_embeds)  # (12, D)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, eeg_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        eeg_emb: (B, D) L2-normalized
        labels:  (B,) integer category labels
        """
        temp = self.log_temp.exp().clamp(0.01, 5.0)
        # similarity with all 12 category prototypes: (B, 12)
        logits = eeg_emb @ self.prototypes.T / temp
        return F.cross_entropy(logits, labels)

    def predict(self, eeg_emb: torch.Tensor):
        """Return (B, 12) logits for category prediction."""
        temp = self.log_temp.exp().clamp(0.01, 5.0)
        return eeg_emb @ self.prototypes.T / temp


# ═══ Category CLIP Embeddings ════════════════════════════════════════════════

def compute_category_clip_embeddings(config, device):
    """Compute CLIP text embeddings for each of the 12 RSVP categories."""
    cache_path = os.path.join(
        os.path.dirname(config.rsvp_fif_path), 'rsvp_category_clip.npz')

    if os.path.isfile(cache_path):
        npz = np.load(cache_path)
        cat_embs = torch.from_numpy(npz['cat_emb']).float().to(device)
        print(f"[V2] Loaded category CLIP embeddings from {cache_path}")
        return cat_embs

    print("[V2] Computing CLIP text embeddings for 12 categories...")
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_name = getattr(config, 'clip_model_name', 'openai/clip-vit-large-patch14')
        model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(clip_name)

        prompts = [f"a photo of a {cat}" for cat in RSVP_CATEGORIES]
        with torch.no_grad():
            inputs = processor(text=prompts, return_tensors='pt',
                               padding=True, truncation=True).to(device)
            text_emb = model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        cat_embs = text_emb.float()
        np.savez(cache_path, cat_emb=cat_embs.cpu().numpy(),
                 categories=RSVP_CATEGORIES)
        print(f"[V2] Saved category embeddings → {cache_path}")
        del model, processor
        torch.cuda.empty_cache()
    except ImportError:
        print("[V2] WARNING: transformers unavailable, using random category embeddings")
        cat_embs = torch.randn(N_RSVP_CLASSES, 768, device=device)
        cat_embs = F.normalize(cat_embs, dim=-1)

    return cat_embs


# ═══ Data ════════════════════════════════════════════════════════════════════

def build_dataloaders(config):
    # ds_trn = Dataset_RSVP_COCO(config, seeds=config.seed)
    # ds_trn.get_flag('trn')
    # ds_val = Dataset_RSVP_COCO(config, seeds=config.seed)
    # ds_val.get_flag('val')

    ds_trn = Dataset_ERP_VGG(config, seeds=config.seed)
    ds_trn.get_flag('trn')
    ds_val = Dataset_ERP_VGG(config, seeds=config.seed)
    ds_val.get_flag('val')

    dl_trn = DataLoader(ds_trn, batch_size=config.batch_size,
                        shuffle=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    return ds_trn, dl_trn, ds_val, dl_val


# ═══ Evaluation ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_cls(model, loader, device):
    model.eval()
    all_y, all_pred, all_proba = [], [], []
    for batch in loader:
        x = batch['data'].to(device)
        y = batch['label']
        logits = model(x)
        all_y.append(y)
        all_pred.append(logits.argmax(1).cpu())
        all_proba.append(F.softmax(logits, 1).cpu())
    y = torch.cat(all_y).numpy()
    pred = torch.cat(all_pred).numpy()
    proba = torch.cat(all_proba).numpy()
    return {
        'top1': float(accuracy_score(y, pred)),
        'top3': float(top_k_accuracy_score(y, proba, k=3)),
        'bacc': float(balanced_accuracy_score(y, pred)),
        'f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'cm': confusion_matrix(y, pred, labels=list(range(N_RSVP_CLASSES))),
    }


@torch.no_grad()
def evaluate_clip_cls(model, proj, clip_loss, loader, device):
    """Evaluate using CLIP category prototypes as classifier."""
    model.eval(); proj.eval()
    all_y, all_pred, all_proba = [], [], []
    for batch in loader:
        x = batch['data'].to(device)
        y = batch['label']
        feat = model.forward_features(x)
        eeg_emb = proj(feat)
        logits = clip_loss.predict(eeg_emb)
        all_y.append(y)
        all_pred.append(logits.argmax(1).cpu())
        all_proba.append(F.softmax(logits, 1).cpu())
    y = torch.cat(all_y).numpy()
    pred = torch.cat(all_pred).numpy()
    proba = torch.cat(all_proba).numpy()
    return {
        'top1': float(accuracy_score(y, pred)),
        'top3': float(top_k_accuracy_score(y, proba, k=3)),
        'bacc': float(balanced_accuracy_score(y, pred)),
        'f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'cm': confusion_matrix(y, pred, labels=list(range(N_RSVP_CLASSES))),
    }


# ═══ Training ════════════════════════════════════════════════════════════════

def train_ce(config, dl_trn, dl_val, device):
    """Mode 1: CE-only classification."""
    print("\n" + "=" * 70)
    print("MODE 1: ATM Classification (CE Only)")
    print("=" * 70)
    model = ATM_Encoder(config).to(device)
    opt = optim.AdamW(model.parameters(), lr=config.learning_rate,
                      betas=config.betas, weight_decay=config.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epoch, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = {'top1': 0}; best_state = None; hist = []
    for ep in range(1, config.epoch + 1):
        model.train(); losses = []; t0 = time.time()
        for b in dl_trn:
            x, y = b['data'].to(device), b['label'].to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        val = evaluate_cls(model, dl_val, device)
        if val['top1'] > best['top1']:
            best = {**val, 'epoch': ep}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        hist.append({'epoch': ep, 'loss': np.mean(losses), **{k: v for k, v in val.items() if k != 'cm'}})
        if ep % 20 == 0 or ep <= 5:
            print(f"  Ep {ep:3d} | L={np.mean(losses):.4f} | "
                  f"T1={val['top1']:.4f} T3={val['top3']:.4f} BA={val['bacc']:.4f} | {time.time()-t0:.1f}s")
    model.load_state_dict(best_state)
    final = evaluate_cls(model, dl_val, device)
    print(f"  Best @ep{best['epoch']}: T1={final['top1']:.4f} T3={final['top3']:.4f} BA={final['bacc']:.4f}")
    return model, final, hist


def train_cat_clip(config, dl_trn, dl_val, device, cat_embeds):
    """Mode 2: Category-CLIP contrastive only (zero-shot style)."""
    print("\n" + "=" * 70)
    print("MODE 2: ATM + Category-CLIP (Supervised Contrastive)")
    print("=" * 70)
    model = ATM_Encoder(config).to(device)
    proj = EEGProjectionHead(config.feat_dim, config.proj_dim, config.dropout).to(device)
    clip_loss = CategoryCLIPLoss(cat_embeds, init_temp=0.5).to(device)

    params = list(model.parameters()) + list(proj.parameters()) + [clip_loss.log_temp]
    opt = optim.AdamW(params, lr=config.learning_rate,
                      betas=config.betas, weight_decay=config.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epoch, eta_min=1e-6)
    best = {'top1': 0}; best_state = None; best_proj = None; hist = []
    for ep in range(1, config.epoch + 1):
        model.train(); proj.train(); losses = []; t0 = time.time()
        for b in dl_trn:
            x = b['data'].to(device)
            y = b['label'].to(device)
            opt.zero_grad()
            feat = model.forward_features(x)
            eeg_emb = proj(feat)
            loss = clip_loss(eeg_emb, y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        val = evaluate_clip_cls(model, proj, clip_loss, dl_val, device)
        temp = clip_loss.log_temp.exp().item()
        if val['top1'] > best['top1']:
            best = {**val, 'epoch': ep}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_proj = {k: v.cpu().clone() for k, v in proj.state_dict().items()}
        hist.append({'epoch': ep, 'loss': np.mean(losses), 'temp': temp,
                     **{k: v for k, v in val.items() if k != 'cm'}})
        if ep % 20 == 0 or ep <= 5:
            print(f"  Ep {ep:3d} | L={np.mean(losses):.4f} temp={temp:.3f} | "
                  f"T1={val['top1']:.4f} T3={val['top3']:.4f} BA={val['bacc']:.4f} | {time.time()-t0:.1f}s")
    model.load_state_dict(best_state); proj.load_state_dict(best_proj)
    final = evaluate_clip_cls(model, proj, clip_loss, dl_val, device)
    print(f"  Best @ep{best['epoch']}: T1={final['top1']:.4f} T3={final['top3']:.4f} BA={final['bacc']:.4f}")
    return model, proj, clip_loss, final, hist


def train_joint_v2(config, dl_trn, dl_val, device, cat_embeds):
    """Mode 3: Joint CE + Category-CLIP."""
    print("\n" + "=" * 70)
    print("MODE 3: Joint ATM (CE + Category-CLIP)")
    print("=" * 70)
    model = ATM_Encoder(config).to(device)
    proj = EEGProjectionHead(config.feat_dim, config.proj_dim, config.dropout).to(device)
    clip_loss = CategoryCLIPLoss(cat_embeds, init_temp=0.5).to(device)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_w = config.ce_weight
    clip_w = config.clip_weight

    params = list(model.parameters()) + list(proj.parameters()) + [clip_loss.log_temp]
    opt = optim.AdamW(params, lr=config.learning_rate,
                      betas=config.betas, weight_decay=config.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epoch, eta_min=1e-6)
    best = {'top1': 0}; best_state = None; best_proj = None; hist = []
    for ep in range(1, config.epoch + 1):
        model.train(); proj.train()
        losses_ce, losses_clip = [], []; t0 = time.time()
        for b in dl_trn:
            x = b['data'].to(device)
            y = b['label'].to(device)
            opt.zero_grad()
            feat = model.forward_features(x)
            logits = model.cls_head(feat)
            eeg_emb = proj(feat)
            l_ce = ce_loss(logits, y)
            l_clip = clip_loss(eeg_emb, y)
            loss = ce_w * l_ce + clip_w * l_clip
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses_ce.append(l_ce.item()); losses_clip.append(l_clip.item())
        sched.step()
        # Evaluate with BOTH heads
        val_ce = evaluate_cls(model, dl_val, device)
        val_clip = evaluate_clip_cls(model, proj, clip_loss, dl_val, device)
        # Ensemble: average logits from CE head and CLIP prototype head
        val_ens = evaluate_ensemble(model, proj, clip_loss, dl_val, device)
        temp = clip_loss.log_temp.exp().item()
        best_t1 = max(val_ce['top1'], val_clip['top1'], val_ens['top1'])
        if best_t1 > best['top1']:
            best = {'epoch': ep,
                    'ce_top1': val_ce['top1'], 'ce_top3': val_ce['top3'], 'ce_bacc': val_ce['bacc'],
                    'clip_top1': val_clip['top1'], 'clip_top3': val_clip['top3'],
                    'ens_top1': val_ens['top1'], 'ens_top3': val_ens['top3'], 'ens_bacc': val_ens['bacc'],
                    'top1': best_t1}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_proj = {k: v.cpu().clone() for k, v in proj.state_dict().items()}
        hist.append({
            'epoch': ep, 'ce_loss': np.mean(losses_ce), 'clip_loss': np.mean(losses_clip), 'temp': temp,
            'ce_top1': val_ce['top1'], 'ce_top3': val_ce['top3'],
            'clip_top1': val_clip['top1'], 'clip_top3': val_clip['top3'],
            'ens_top1': val_ens['top1'], 'ens_top3': val_ens['top3'],
        })
        if ep % 20 == 0 or ep <= 5:
            print(f"  Ep {ep:3d} | CE={np.mean(losses_ce):.4f} CLIP={np.mean(losses_clip):.4f} t={temp:.3f} | "
                  f"CE[T1={val_ce['top1']:.3f} T3={val_ce['top3']:.3f}] "
                  f"CLIP[T1={val_clip['top1']:.3f} T3={val_clip['top3']:.3f}] "
                  f"ENS[T1={val_ens['top1']:.3f} T3={val_ens['top3']:.3f}] | {time.time()-t0:.1f}s")
    model.load_state_dict(best_state); proj.load_state_dict(best_proj)
    final_ce = evaluate_cls(model, dl_val, device)
    final_clip = evaluate_clip_cls(model, proj, clip_loss, dl_val, device)
    final_ens = evaluate_ensemble(model, proj, clip_loss, dl_val, device)
    print(f"\n  Best @ep{best['epoch']}:")
    print(f"    CE head:   T1={final_ce['top1']:.4f} T3={final_ce['top3']:.4f} BA={final_ce['bacc']:.4f}")
    print(f"    CLIP head: T1={final_clip['top1']:.4f} T3={final_clip['top3']:.4f} BA={final_clip['bacc']:.4f}")
    print(f"    Ensemble:  T1={final_ens['top1']:.4f} T3={final_ens['top3']:.4f} BA={final_ens['bacc']:.4f}")
    return model, proj, clip_loss, final_ce, final_clip, final_ens, hist


@torch.no_grad()
def evaluate_ensemble(model, proj, clip_loss, loader, device, alpha=0.5):
    """Ensemble: weighted average of CE logits and CLIP prototype logits."""
    model.eval(); proj.eval()
    all_y, all_pred, all_proba = [], [], []
    for batch in loader:
        x = batch['data'].to(device)
        y = batch['label']
        feat = model.forward_features(x)
        logits_ce = model.cls_head(feat)
        eeg_emb = proj(feat)
        logits_clip = clip_loss.predict(eeg_emb)
        # Normalize both to same scale before combining
        logits = alpha * F.softmax(logits_ce, 1) + (1 - alpha) * F.softmax(logits_clip, 1)
        all_y.append(y)
        all_pred.append(logits.argmax(1).cpu())
        all_proba.append(logits.cpu())
    y = torch.cat(all_y).numpy()
    pred = torch.cat(all_pred).numpy()
    proba = torch.cat(all_proba).numpy()
    # Re-normalize proba for top_k_accuracy_score
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {
        'top1': float(accuracy_score(y, pred)),
        'top3': float(top_k_accuracy_score(y, proba, k=3)),
        'bacc': float(balanced_accuracy_score(y, pred)),
        'f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'cm': confusion_matrix(y, pred, labels=list(range(N_RSVP_CLASSES))),
    }


# ═══ Plotting ════════════════════════════════════════════════════════════════

def plot_results(results, save_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('RSVP-COCO 12-Class: ML Baseline vs ATM + Category-CLIP (V2)\n'
                 '360 ERP-averaged images, 32 EEG channels, [0, +500ms]',
                 fontsize=13, fontweight='bold')

    methods = ['ML Baseline\n(LDA)', 'ATM\n(CE Only)', 'ATM\n(Cat-CLIP)', 'ATM Joint\n(CE+CLIP)', 'Joint\nEnsemble']
    t1 = [ML_BASELINE['top1'], results['ce']['top1'], results['cat_clip']['top1'],
          results['joint_ce']['top1'], results['ensemble']['top1']]
    t3 = [ML_BASELINE['top3'], results['ce']['top3'], results['cat_clip']['top3'],
          results['joint_ce']['top3'], results['ensemble']['top3']]
    ba = [ML_BASELINE['bacc'], results['ce']['bacc'], results['cat_clip']['bacc'],
          results['joint_ce']['bacc'], results['ensemble']['bacc']]
    colors = ['#9E9E9E', '#4CAF50', '#FF9800', '#2196F3', '#9C27B0']

    # Panel 1: Absolute metrics
    ax = axes[0]
    x = np.arange(len(methods))
    w = 0.22
    bars1 = ax.bar(x - w, t1, w, color=colors, alpha=0.9, edgecolor='black', lw=0.5)
    bars2 = ax.bar(x, t3, w, color=colors, alpha=0.55, edgecolor='black', lw=0.5)
    bars3 = ax.bar(x + w, ba, w, color=colors, alpha=0.35, edgecolor='black', lw=0.5)
    for i in range(len(methods)):
        ax.text(i - w, t1[i] + 0.008, f'{t1[i]:.3f}', ha='center', fontsize=7, fontweight='bold')
        ax.text(i, t3[i] + 0.008, f'{t3[i]:.3f}', ha='center', fontsize=7)
        ax.text(i + w, ba[i] + 0.008, f'{ba[i]:.3f}', ha='center', fontsize=7)
    ax.axhline(ML_BASELINE['chance'], color='red', ls=':', lw=1.5, label=f'Chance ({ML_BASELINE["chance"]:.3f})')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Score'); ax.set_title('Classification Metrics')
    ax.legend(['Top-1', 'Top-3', 'BAcc', 'Chance'], loc='upper left', fontsize=8)
    ax.set_ylim(0, max(max(t3) + 0.12, 0.65)); ax.grid(True, alpha=0.2, axis='y')

    # Panel 2: Improvement
    ax = axes[1]
    labels = ['ATM CE', 'Cat-CLIP', 'Joint CE', 'Ensemble']
    imp_t1 = [(t1[i+1]/ML_BASELINE['top1'] - 1)*100 for i in range(4)]
    imp_t3 = [(t3[i+1]/ML_BASELINE['top3'] - 1)*100 for i in range(4)]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, imp_t1, 0.3, color=[colors[i+1] for i in range(4)], alpha=0.9, label='Top-1 Improve')
    ax.bar(x + 0.15, imp_t3, 0.3, color=[colors[i+1] for i in range(4)], alpha=0.5, label='Top-3 Improve')
    for i in range(4):
        ax.text(i - 0.15, imp_t1[i] + 1, f'{imp_t1[i]:+.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + 0.15, imp_t3[i] + 1, f'{imp_t3[i]:+.1f}%', ha='center', fontsize=8)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Improvement over ML (%)')
    ax.set_title('Relative Improvement vs LDA Baseline')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')

    # Panel 3: Training curves
    ax = axes[2]
    if 'joint_history' in results:
        h = results['joint_history']
        ep = [e['epoch'] for e in h]
        ax.plot(ep, [e['ce_top1'] for e in h], '-', color='#2196F3', lw=1.5, label='Joint CE-head T1')
        ax.plot(ep, [e['clip_top1'] for e in h], '-', color='#FF9800', lw=1.5, label='Joint CLIP-head T1')
        ax.plot(ep, [e['ens_top1'] for e in h], '-', color='#9C27B0', lw=2, label='Joint Ensemble T1')
    if 'ce_history' in results:
        h = results['ce_history']
        ep = [e['epoch'] for e in h]
        ax.plot(ep, [e['top1'] for e in h], '--', color='#4CAF50', lw=1.5, label='CE-only T1')
    ax.axhline(ML_BASELINE['top1'], color='gray', ls=':', lw=2, label=f'ML Baseline ({ML_BASELINE["top1"]:.3f})')
    ax.axhline(ML_BASELINE['chance'], color='red', ls=':', lw=1, alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Top-1')
    ax.set_title('Training Curves'); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = os.path.join(save_dir, 'fig', 'rsvp_v2_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}"); plt.close(fig)

    # Confusion matrix (best model)
    best_key = max(['ce', 'cat_clip', 'joint_ce', 'ensemble'],
                   key=lambda k: results[k]['top1'])
    cm = results[best_key].get('cm', None)
    if cm is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=max(0.3, cm_n.max()))
        for i in range(N_RSVP_CLASSES):
            for j in range(N_RSVP_CLASSES):
                v = cm_n[i, j]; c = 'white' if v > 0.15 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=c)
        ax.set_xticks(range(N_RSVP_CLASSES)); ax.set_yticks(range(N_RSVP_CLASSES))
        ax.set_xticklabels(RSVP_CATEGORIES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(RSVP_CATEGORIES, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'Best Model ({best_key}): T1={results[best_key]["top1"]:.3f} '
                     f'T3={results[best_key]["top3"]:.3f} BA={results[best_key]["bacc"]:.3f}',
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Proportion', shrink=0.8); plt.tight_layout()
        p = os.path.join(save_dir, 'fig', 'rsvp_v2_confusion.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved: {p}"); plt.close(fig)


# ═══ Main ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fif', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip_weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = BaseConfigRSVP()
    if args.fif: config.rsvp_fif_path = args.fif
    config.epoch = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.clip_weight = args.clip_weight
    config.seed = args.seed

    save_dir = args.save_dir or os.path.dirname(config.rsvp_fif_path)
    os.makedirs(os.path.join(save_dir, 'fig'), exist_ok=True)

    torch.manual_seed(config.seed); np.random.seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name()}")

    # Category CLIP embeddings
    cat_embeds = compute_category_clip_embeddings(config, device)
    print(f"Category CLIP embeddings shape: {cat_embeds.shape}")

    # Data
    print("\n=== Loading RSVP-COCO data ===")
    ds_trn, dl_trn, ds_val, dl_val = build_dataloaders(config)

    # Mode 1: CE only
    torch.manual_seed(config.seed)
    m1, r1, h1 = train_ce(config, dl_trn, dl_val, device)

    # Mode 2: Category-CLIP only
    torch.manual_seed(config.seed)
    m2, p2, cl2, r2, h2 = train_cat_clip(config, dl_trn, dl_val, device, cat_embeds)

    # Mode 3: Joint
    torch.manual_seed(config.seed)
    m3, p3, cl3, r3_ce, r3_clip, r3_ens, h3 = train_joint_v2(config, dl_trn, dl_val, device, cat_embeds)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (V2 — Category-Level CLIP)")
    print("=" * 70)
    print(f"{'Method':<28} {'Top-1':>8} {'Top-3':>8} {'BAcc':>8} {'Improve':>10}")
    print("-" * 70)
    print(f"{'Chance':<28} {ML_BASELINE['chance']:>8.4f} {3*ML_BASELINE['chance']:>8.4f} "
          f"{ML_BASELINE['chance']:>8.4f} {'—':>10}")
    print(f"{'ML Baseline (LDA)':<28} {ML_BASELINE['top1']:>8.4f} {ML_BASELINE['top3']:>8.4f} "
          f"{ML_BASELINE['bacc']:>8.4f} {'—':>10}")

    for name, r in [('ATM (CE only)', r1), ('ATM (Cat-CLIP)', r2),
                     ('Joint: CE head', r3_ce), ('Joint: CLIP head', r3_clip),
                     ('Joint: Ensemble', r3_ens)]:
        imp = (r['top1'] / ML_BASELINE['top1'] - 1) * 100
        print(f"{name:<28} {r['top1']:>8.4f} {r['top3']:>8.4f} {r['bacc']:>8.4f} {imp:>+9.1f}%")
    print("-" * 70)

    # Save
    all_res = {
        'ml_baseline': ML_BASELINE,
        'ce': {k: v for k, v in r1.items() if k != 'cm'}, 'ce_history': h1,
        'cat_clip': {k: v for k, v in r2.items() if k != 'cm'}, 'cat_clip_history': h2,
        'joint_ce': {k: v for k, v in r3_ce.items() if k != 'cm'},
        'joint_clip': {k: v for k, v in r3_clip.items() if k != 'cm'},
        'ensemble': {k: v for k, v in r3_ens.items() if k != 'cm'},
        'joint_history': h3,
        'config': {'model': config.model, 'enc_in': config.enc_in,
                   'epochs': config.epoch, 'batch_size': config.batch_size,
                   'lr': config.learning_rate, 'clip_weight': config.clip_weight},
    }
    jp = os.path.join(save_dir, 'fig', 'rsvp_v2_results.json')
    with open(jp, 'w') as f:
        json.dump(all_res, f, indent=2, default=str)
    print(f"\nJSON: {jp}")

    plot_results({**all_res,
                  'ce': r1, 'cat_clip': r2, 'joint_ce': r3_ce, 'ensemble': r3_ens}, save_dir)
    print("\n=== V2 TRAINING COMPLETE ===")


if __name__ == '__main__':
    main()
