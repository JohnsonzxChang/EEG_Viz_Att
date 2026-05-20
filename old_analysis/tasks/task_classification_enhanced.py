"""
Exp_ClassificationEnhanced
==========================
Enhanced multi-label contrastive learning trainer composing:
  1. Multi-Label Stratified Batch Sampler  (rare-class pair coverage)
  2. Proxy-Anchor Loss                     (learnable class proxies)
  3. Memory Queue                          (expanded effective batch for Circle Loss)
  4. Prototype-Based Evaluation            (embedding space diagnostic)
  5. MLP Contrastive Projector             (decoupled from cls_head)
  6. EEG Data Augmentation                 (noise, channel dropout, smoothing)

Inherits from Exp_ClassificationCircle (encoder-agnostic).

Combined loss:
  L = L_ASL + lambda_circle * L_Circle(z_aug, y_aug) + lambda_proxy * L_Proxy(z, y)
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from conf import BaseConfig
from utils import LoggerFile
from utils.tools import EarlyStoppingCla
from encoder.contrastive_wrapper import EncoderWithProjector
from utils.eeg_augment import AugmentedEEGDataset
from utils.multilabel_sampler import MultiLabelStratifiedBatchSampler
from utils.memory_queue import EmbeddingQueue
from utils.knn_eval import prototype_eval
from losses.proxy_anchor_loss import ProxyAnchorLoss
from .task_classification_circle import Exp_ClassificationCircle, topk_multilabel_accuracy_torch

warnings.filterwarnings('ignore')


class Exp_ClassificationEnhanced(Exp_ClassificationCircle):
    """Enhanced trainer: Sampler + Proxy-Anchor + Queue + Prototype eval."""

    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super().__init__(args, logger, session)

        proj_dim = getattr(args, 'proj_dim', 128)

        # Proxy-Anchor Loss (learnable proxies — must be in optimizer)
        self.proxy_loss_fn = ProxyAnchorLoss(
            num_classes=args.num_classes,
            embedding_dim=proj_dim,
            scale=getattr(args, 'proxy_scale', 32.0),
            delta=getattr(args, 'proxy_delta', 0.1),
        ).to(self.device)
        self.proxy_lambda = getattr(args, 'proxy_lambda', 0.3)

        # ── Constant Loss Weighting ─────────────────────────
        # Using lambda values from config to prevent homoscedastic overfitting
        self.circle_lambda = getattr(args, 'circle_lambda', 0.5)
        self.proxy_lambda = getattr(args, 'proxy_lambda', 0.3)
        # Memory Queue
        self.queue = EmbeddingQueue(
            queue_size=getattr(args, 'queue_size', 512),
            embed_dim=proj_dim,
            num_classes=args.num_classes,
            device=self.device,
        )

        self.proto_eval_freq = getattr(args, 'proto_eval_freq', 5)

    # ── model: wrap any encoder with MLP projector ───────────────────

    def _build_model(self):
        assert self.args.task == 'classification'
        # Use model_dict to select encoder (encoder-agnostic)
        backbone = self.model_dict[self.args.model](self.args).to(self.device)

        feat_dim = getattr(self.args, 'feat_dim', 256)
        proj_dim = getattr(self.args, 'proj_dim', 128)

        model = EncoderWithProjector(backbone, feat_dim, proj_dim).to(self.device)
        return model

    # ── data: augmentation + multi-label sampler ─────────────────────

    def _get_data(self):
        # Get base data from grandparent (Exp_ClassificationM._get_data)
        from utils.data_factory import get_data_loader_cutt
        train_data, _, valid_data, valid_loader = get_data_loader_cutt(self.args)

        # Wrap training data with augmentation
        aug_train = AugmentedEEGDataset(
            train_data,
            is_train=True,
            noise_std=getattr(self.args, 'aug_noise_std', 0.05),
            chan_drop=getattr(self.args, 'aug_chan_drop', 3),
            smooth_k=getattr(self.args, 'aug_smooth_k', 3),
        )

        # Multi-label stratified batch sampler
        base_labels = train_data.all_regs  # (N_trn, 79) numpy after get_flag('trn')
        sampler = MultiLabelStratifiedBatchSampler(
            labels=base_labels,
            batch_size=self.args.batch_size,
            K_anchor=getattr(self.args, 'sampler_k_anchor', 2),
            num_batches=getattr(self.args, 'sampler_num_batches', None),
        )

        # batch_sampler replaces batch_size + shuffle + drop_last
        train_loader = DataLoader(
            aug_train,
            batch_sampler=sampler,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

        return aug_train, train_loader, valid_data, valid_loader

    # ── optimizer: include proxy parameters with severe damping ───

    def _select_optimizer(self):
        # We heavily damp the learning rate of the proxies (100x smaller) and
        # increase their weight decay. This prevents the learning proxies from
        # overfitting to the training batch geometry and forces them to act
        # as stable global anchors.
        params = [
            {
                'params': self.model.parameters(),
            },
            {
                'params': self.proxy_loss_fn.parameters(),
                'lr': self.args.learning_rate * 0.01,
                'weight_decay': 0.1
            }
        ]
        return optim.AdamW(
            params,
            lr=self.args.learning_rate,
            betas=self.args.betas,
            weight_decay=self.args.weight_decay,
        )

    # ── training loop ────────────────────────────────────────────────

    def train(self, setting: str):
        train_data, train_loader, valid_data, valid_loader = self._get_data()
        path = os.path.join(self.args.loggerdir, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingCla(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.epoch, eta_min=1e-6
        )
        pos_weight = getattr(train_data, 'pos_weight', None)
        criterion = self._select_criterion(pos_weight=pos_weight)

        self.model = self.model.to(self.device, non_blocking=True)
        self._print_model()
        
        print(f"\n[Decoupled Training] Continuous Blending Mode enabled.")
        print(f"ASL weight will ramp up continuously from 0.0 to 1.0 quadratically over {self.args.epoch} epochs.\n")

        asl_on = np.zeros(self.args.epoch)
        for i in range((self.args.epoch//2)):
            if i % 20 < 4:
                asl_on[i+self.args.epoch//2] = (i%20) / (4*100) 
            else:
                asl_on[i+self.args.epoch//2] = 0

        for epoch in range(self.args.epoch):
            # ── Continuous Decoupled Training Switch ──
            # Calculate a continuous tracking weight that starts at 0 and grows to 1
            # Using a quadratic curve so the early epochs are heavily dominated by Contrastive Learning
            asl_weight = asl_on[epoch]
            contrastive_weight = 1.0 - asl_weight
            
            iter_count = 0
            train_loss = []
            train_loss_asl = []
            train_loss_circle = []
            train_loss_proxy = []
            train_preds = []
            train_truths = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_id) in enumerate(train_loader):
                batch_x  = batch_x.to(self.device, non_blocking=True)
                batch_y  = batch_y.to(self.device, non_blocking=True)
                batch_id = batch_id.to(self.device, non_blocking=True)
                iter_count += 1

                model_optim.zero_grad()

                # ── forward ──────────────────────────────────────────
                feat, logits = self._get_feat_and_logits(batch_x, batch_id)

                # ── ASL loss ─────────────────────────────────────────
                l_asl = self.do_loss(logits, batch_y)

                # ── Circle Loss with queue-augmented batch ───────────
                z = F.normalize(feat, dim=-1)
                z_aug, y_aug = self.queue.augmented_batch(z, batch_y)
                l_circ = self.circle_loss_fn(z_aug, y_aug)
                self.queue.enqueue(z, batch_y)

                # ── Proxy-Anchor Loss ────────────────────────────────
                l_proxy = self.proxy_loss_fn(z, batch_y)

                # ── Loss Calculation per Stage ──
                # Smooth continuous transitioning between Contrastive embedding space definition and ASL space optimization
                loss = asl_weight * l_asl + contrastive_weight * (self.circle_lambda * l_circ + self.proxy_lambda * l_proxy)

                loss.backward()
                
                # Clip gradients
                active_params = list(self.model.parameters()) + list(self.proxy_loss_fn.parameters())
                if len(active_params) > 0:
                    nn.utils.clip_grad_norm_(active_params, max_norm=10.0)
                
                model_optim.step()

                train_loss.append(loss.detach())
                train_loss_asl.append(l_asl.detach())
                train_loss_circle.append(l_circ.detach())
                train_loss_proxy.append(l_proxy.detach())
                train_preds.append(logits.detach())
                train_truths.append(batch_y.detach())

            # scheduler.step()

            with torch.no_grad():
                train_loss        = torch.mean(torch.stack(train_loss)).cpu()
                train_loss_asl    = torch.mean(torch.stack(train_loss_asl)).cpu()
                train_loss_circle = torch.mean(torch.stack(train_loss_circle)).cpu()
                train_loss_proxy  = torch.mean(torch.stack(train_loss_proxy)).cpu()
                train_preds_cat   = torch.cat(train_preds, dim=0).cpu()
                train_truths_cat  = torch.cat(train_truths, dim=0).cpu()
                
                from tasks.task_classification_multilabel import compute_subset_accuracy
                train_acc = compute_subset_accuracy(
                    train_preds_cat, train_truths_cat, threshold=0.3
                )

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.1f}s")
            vali_loss, vali_loss_asl, vali_loss_circle, vali_loss_proxy, vali_acc, micro_f1, macro_f1, mAP, hamming, subset_acc = \
                self.vali(train_data, train_loader, valid_data, valid_loader,
                          criterion, epoch + 1)

            # ── Prototype evaluation (every N epochs) ────────────────
            if (epoch + 1) % self.proto_eval_freq == 0:
                proto_result = prototype_eval(
                    self.model, train_loader, valid_loader,
                    device=self.device,
                    num_classes=self.args.num_classes,
                    tau=getattr(self.args, 'proto_tau', 10.0),
                )
                self.logger.log_scalar('proto_mAP',      proto_result['proto_mAP'],      epoch + 1)
                self.logger.log_scalar('proto_micro_f1', proto_result['proto_micro_f1'], epoch + 1)
                self.logger.log_scalar('proto_macro_f1', proto_result['proto_macro_f1'], epoch + 1)
                print(f"  [proto] mAP={proto_result['proto_mAP']:.3f}  "
                      f"micro_f1={proto_result['proto_micro_f1']:.3f}  "
                      f"macro_f1={proto_result['proto_macro_f1']:.3f}")

            # ── TensorBoard ──────────────────────────────────────────
            self.logger.log_scalar('train_loss',         train_loss,         epoch + 1)
            self.logger.log_scalar('train_loss_asl',     train_loss_asl,     epoch + 1)
            self.logger.log_scalar('train_loss_circle',  train_loss_circle,  epoch + 1)
            self.logger.log_scalar('train_loss_proxy',   train_loss_proxy,   epoch + 1)
            self.logger.log_scalar('valid_loss',         vali_loss,          epoch + 1)
            self.logger.log_scalar('valid_loss_asl',     vali_loss_asl,      epoch + 1)
            self.logger.log_scalar('valid_loss_circle',  vali_loss_circle,   epoch + 1)
            self.logger.log_scalar('valid_loss_proxy',   vali_loss_proxy,    epoch + 1)
            self.logger.log_scalar('train_acc',          train_acc,          epoch + 1)
            self.logger.log_scalar('valid_acc',          vali_acc,           epoch + 1)
            self.logger.log_scalar('micro_f1',           micro_f1,           epoch + 1)
            self.logger.log_scalar('macro_f1',           macro_f1,           epoch + 1)
            self.logger.log_scalar('mAP',                mAP,                epoch + 1)
            self.logger.log_scalar('hamming_loss',       hamming,            epoch + 1)
            self.logger.log_scalar('subset_acc',         subset_acc,         epoch + 1)
            self.logger.log_scalar('lr', scheduler.get_last_lr()[0],         epoch + 1)

            print(
                f"Epoch: {epoch+1}, Steps: {train_steps} | "
                f"Loss: {train_loss:.4f}  ASL: {train_loss_asl:.4f} (w={asl_weight:.4f}) "
                f"Circle: {train_loss_circle:.4f}  Proxy: {train_loss_proxy:.4f} (w={contrastive_weight:.2f}) | "
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

    # ── validation override: track multi-component loss ──────────────

    def vali(self, train_data, train_loader, valid_data, valid_loader, criterion, epoch):
        total_loss = []
        total_loss_asl = []
        total_loss_circle = []
        total_loss_proxy = []
        all_preds = []
        all_truths = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_id) in enumerate(valid_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_id = batch_id.to(self.device, non_blocking=True)

                feat, outputs = self._get_feat_and_logits(batch_x, batch_id)
                all_preds.append(outputs)
                all_truths.append(batch_y)
                
                # Compute individual metrics
                l_asl = self.do_loss(outputs, batch_y)
                
                z = F.normalize(feat, dim=-1)
                l_circ = self.circle_loss_fn(z, batch_y)
                l_proxy = self.proxy_loss_fn(z, batch_y)
                
                loss_val = l_asl + self.circle_lambda * l_circ + self.proxy_lambda * l_proxy

                total_loss.append(loss_val)
                total_loss_asl.append(l_asl)
                total_loss_circle.append(l_circ)
                total_loss_proxy.append(l_proxy)
            
            all_preds = torch.cat(all_preds, dim=0).cpu()
            all_truths = torch.cat(all_truths, dim=0).cpu()
            
            total_loss = torch.mean(torch.stack(total_loss)).cpu()
            total_loss_asl = torch.mean(torch.stack(total_loss_asl)).cpu()
            total_loss_circle = torch.mean(torch.stack(total_loss_circle)).cpu()
            total_loss_proxy = torch.mean(torch.stack(total_loss_proxy)).cpu()

        from tasks.task_classification_multilabel import compute_f1_micro_macro, compute_map, compute_hamming_loss, compute_subset_accuracy
        
        micro_f1, macro_f1 = compute_f1_micro_macro(all_preds, all_truths, threshold=0.3)
        _, mAP = compute_map(all_preds, all_truths)
        vali_acc = compute_subset_accuracy(all_preds, all_truths, threshold=0.3)
        hamming = compute_hamming_loss(all_preds, all_truths, threshold=0.3)
        subset_acc = vali_acc
        
        if mAP > self.best_val_acc:
            self.best_val_acc = mAP
            self.vali_plot_per_class(all_preds, all_truths, epoch)
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "preds": all_preds,
                "truths": all_truths,
                "acc": vali_acc,
            }, f"{self.logger.writer.log_dir}/checkpoint.pth")
            
        self.model.train()
        return total_loss, total_loss_asl, total_loss_circle, total_loss_proxy, vali_acc, micro_f1, macro_f1, mAP, hamming, subset_acc
