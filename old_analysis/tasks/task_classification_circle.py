"""
Exp_ClassificationCircle
========================
Multi-label classification with   L = L_ASL  +  λ · L_Circle

Key difference from Exp_ClassificationM
----------------------------------------
* Each training step calls  model.forward_all(x)  to obtain both
  the intermediate *feature* vector (needed for Circle Loss) and the
  final classification *logits* (needed for ASL).
* Validation is inherited unchanged — it only evaluates logit-based
  metrics (mAP, F1, Hamming …).
* No existing files are modified.
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
from utils.tools import EarlyStoppingCla
from losses import MultiLabelCircleLoss
from .task_classification_multilabel import (
    Exp_ClassificationM,
    topk_multilabel_accuracy_torch,
)

warnings.filterwarnings('ignore')


class Exp_ClassificationCircle(Exp_ClassificationM):
    """ASL + Circle Loss hybrid trainer.

    Circle Loss is applied on L2-normalised intermediate features so that
    the encoder is explicitly trained to produce a semantically structured
    embedding space: EEG epochs sharing COCO labels are pulled together,
    while unrelated epochs are pushed apart.

    Requires the encoder to expose  forward_all(x)  →  (feat, logits).
    All standard encoders in this repo (CNN, TFEncoder, ATM …) satisfy
    this interface.
    """

    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super().__init__(args, logger, session)

        self.circle_loss_fn = MultiLabelCircleLoss(
            gamma=getattr(args, 'circle_gamma', 64),
            margin=getattr(args, 'circle_margin', 0.25),
            use_jaccard=getattr(args, 'circle_jaccard', True),
        ).to(self.device)

        self.circle_lambda = getattr(args, 'circle_lambda', 0.1)

    # ── helpers ───────────────────────────────────────────────────────

    def _get_feat_and_logits(
        self, batch_x: torch.Tensor, batch_id: torch.Tensor
    ):
        """Return (feat, logits) with a single forward pass when possible."""
        if hasattr(self.model, 'forward_all'):
            return self.model.forward_all(batch_x, batch_id)
        # Fallback: two passes (slightly wasteful, covers any new encoder)
        feat   = self.model.forward_features(batch_x, batch_id)
        logits = self.model(batch_x, batch_id)
        return feat, logits

    # ── training loop (overrides parent) ──────────────────────────────

    def train(self, setting: str):
        train_data, train_loader, valid_data, valid_loader = self._get_data()
        path = os.path.join(self.args.loggerdir, setting)
        os.makedirs(path, exist_ok=True)
        time_now = time.time()

        train_steps  = len(train_loader)
        early_stopping = EarlyStoppingCla(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler   = optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.epoch, eta_min=1e-6
        )
        pos_weight = getattr(train_data, 'pos_weight', None)
        criterion  = self._select_criterion(pos_weight=pos_weight)

        self.model = self.model.to(self.device, non_blocking=True)
        self._print_model()

        for epoch in range(self.args.epoch):
            iter_count  = 0
            train_loss  = []
            train_loss_asl    = []
            train_loss_circle = []
            train_preds  = []
            train_truths = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_id) in enumerate(train_loader):
                batch_x  = batch_x.to(self.device, non_blocking=True)
                batch_y  = batch_y.to(self.device, non_blocking=True)
                batch_id = batch_id.to(self.device, non_blocking=True)
                iter_count += 1

                model_optim.zero_grad()

                # ── forward ───────────────────────────────────────────
                feat, logits = self._get_feat_and_logits(batch_x, batch_id)

                # ── ASL (classification) loss ─────────────────────────
                l_asl = self.do_loss(logits, batch_y)

                # ── Circle Loss (metric learning on features) ─────────
                z      = F.normalize(feat, dim=-1)   # unit hypersphere
                l_circ = self.circle_loss_fn(z, batch_y)

                loss = l_asl + self.circle_lambda * l_circ

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                model_optim.step()

                train_loss.append(loss)
                train_loss_asl.append(l_asl.detach())
                train_loss_circle.append(l_circ.detach())
                train_preds.append(logits.detach())
                train_truths.append(batch_y.detach())

            scheduler.step()

            with torch.no_grad():
                train_loss        = torch.mean(torch.stack(train_loss)).cpu()
                train_loss_asl    = torch.mean(torch.stack(train_loss_asl)).cpu()
                train_loss_circle = torch.mean(torch.stack(train_loss_circle)).cpu()
                train_preds_cat   = torch.cat(train_preds,  dim=0).cpu()
                train_truths_cat  = torch.cat(train_truths, dim=0).cpu()
                train_acc = topk_multilabel_accuracy_torch(
                    train_preds_cat, train_truths_cat, k=3
                )

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.1f}s")
            vali_loss, vali_acc, micro_f1, macro_f1, mAP, hamming, subset_acc = \
                self.vali(train_data, train_loader, valid_data, valid_loader,
                          criterion, epoch + 1)

            # ── TensorBoard ───────────────────────────────────────────
            self.logger.log_scalar('train_loss',        train_loss,        epoch + 1)
            self.logger.log_scalar('train_loss_asl',    train_loss_asl,    epoch + 1)
            self.logger.log_scalar('train_loss_circle', train_loss_circle, epoch + 1)
            self.logger.log_scalar('valid_loss',        vali_loss,         epoch + 1)
            self.logger.log_scalar('train_acc',         train_acc,         epoch + 1)
            self.logger.log_scalar('valid_acc',         vali_acc,          epoch + 1)
            self.logger.log_scalar('micro_f1',          micro_f1,          epoch + 1)
            self.logger.log_scalar('macro_f1',          macro_f1,          epoch + 1)
            self.logger.log_scalar('mAP',               mAP,               epoch + 1)
            self.logger.log_scalar('hamming_loss',      hamming,           epoch + 1)
            self.logger.log_scalar('subset_acc',        subset_acc,        epoch + 1)
            self.logger.log_scalar('lr', scheduler.get_last_lr()[0],       epoch + 1)

            print(
                f"Epoch: {epoch+1}, Steps: {train_steps} | "
                f"Loss: {train_loss:.4f}  ASL: {train_loss_asl:.4f}  "
                f"Circle: {train_loss_circle:.4f} | "
                f"Train Acc: {train_acc:.3f}  Vali Acc: {vali_acc:.3f}  "
                f"mAP: {mAP:.3f}  Hamming: {hamming:.4f}"
            )
            early_stopping(vali_acc, self.model, path, epoch)

            if self.session is not None:
                self.session.report({
                    "epoch":      epoch,
                    "train_loss": train_loss.item(),
                    "vali_loss":  vali_loss.item(),
                })

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model
