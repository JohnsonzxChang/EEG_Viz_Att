"""
Prototype-Based Multi-Label Evaluation
=======================================
Diagnostic tool for metric learning: builds per-class centroid prototypes
from training embeddings, evaluates validation set via cosine similarity.

This measures embedding space geometry quality independently of the cls_head.
The gap between logit-based mAP and proto_mAP reveals whether Circle Loss /
Proxy-Anchor are effectively organizing the embedding space.

Usage:
    result = prototype_eval(model, train_loader, val_loader, device, num_classes=79)
    # result = {'proto_mAP': ..., 'proto_micro_f1': ..., 'proto_macro_f1': ...}
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract L2-normalized embeddings and labels from a DataLoader.

    Uses model.forward_all() to get the projection z (not backbone h).
    """
    all_z, all_y = [], []
    model.eval()
    for batch_x, batch_y, batch_id in loader:
        batch_x  = batch_x.to(device, non_blocking=True)
        batch_id = batch_id.to(device, non_blocking=True)

        if hasattr(model, 'forward_all'):
            z, _ = model.forward_all(batch_x, batch_id)
        else:
            z = model.forward_features(batch_x, batch_id)

        z = F.normalize(z, dim=-1)
        all_z.append(z.cpu())
        all_y.append(batch_y.cpu())

    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0)


@torch.no_grad()
def build_prototypes(train_z, train_y, num_classes):
    """Compute per-class mean embedding prototype on the unit sphere.

    Returns:
        prototypes: (C, D) L2-normalized.
        valid_mask: (C,) bool, True if class has >= 1 training sample.
    """
    D = train_z.size(1)
    prototypes = torch.zeros(num_classes, D)
    valid_mask = torch.zeros(num_classes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (train_y[:, c] > 0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            prototypes[c] = train_z[idx].mean(dim=0)
            valid_mask[c] = True

    prototypes = F.normalize(prototypes, dim=-1)
    return prototypes, valid_mask


def prototype_eval(model, train_loader, val_loader, device,
                   num_classes=79, tau=10.0, threshold=0.5):
    """Full prototype evaluation pipeline.

    Args:
        tau:       temperature for sigmoid(tau * cosine_sim). Default 10.0.
        threshold: decision threshold on pseudo-probabilities. Default 0.5.

    Returns:
        dict with proto_mAP, proto_micro_f1, proto_macro_f1.
    """
    train_z, train_y = extract_embeddings(model, train_loader, device)
    val_z,   val_y   = extract_embeddings(model, val_loader,   device)

    prototypes, valid_mask = build_prototypes(train_z, train_y, num_classes)

    # Cosine similarity: (N_val, C)
    sim = val_z @ prototypes.T
    scores = torch.sigmoid(tau * sim).numpy()

    y_true = val_y.numpy().astype(int)
    y_pred = (scores >= threshold).astype(int)

    # Only evaluate on classes with training prototypes
    valid = valid_mask.numpy()
    scores_v = scores[:, valid]
    y_true_v = y_true[:, valid]
    y_pred_v = y_pred[:, valid]

    proto_mAP = float(average_precision_score(
        y_true_v, scores_v, average='macro'))
    proto_micro_f1 = float(f1_score(
        y_true_v, y_pred_v, average='micro', zero_division=0))
    proto_macro_f1 = float(f1_score(
        y_true_v, y_pred_v, average='macro', zero_division=0))

    return {
        'proto_mAP':      proto_mAP,
        'proto_micro_f1': proto_micro_f1,
        'proto_macro_f1': proto_macro_f1,
    }
