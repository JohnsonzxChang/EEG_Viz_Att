from conf import BaseConfig
from utils import LoggerFile
from utils.data_factory_openimages import get_data_loader_cutt
from .task_base import Exp_Basic
from utils.tools import EarlyStoppingCla, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch
import torch.profiler
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

from encoder.tracking_wrapper import EncoderWithTemporalCrossAttn

class Exp_AttentionTracking(Exp_Basic):
    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super(Exp_AttentionTracking, self).__init__(args, logger)
        self.best_val_acc = 0.0
        self.session = session

    def _build_model(self):
        assert 'tracking' in self.args.task, f"Task must be tracking, got {self.args.task}"
        base_model = self.model_dict[self.args.model](self.args)
        model = EncoderWithTemporalCrossAttn(
            base_model, 
            feat_dim=self.args.feat_dim, 
            proj_dim=self.args.proj_dim, 
            num_classes=1,
            use_eeg=getattr(self.args, 'use_eeg', True)
        ).to(self.device)
        return model

    def _get_data(self):
        train_data, train_loader, valid_data, valid_loader = get_data_loader_cutt(self.args)
        return train_data, train_loader, valid_data, valid_loader

    def _print_model(self):
        from torchinfo import summary
        with torch.no_grad():
            print(summary(self.model, (2, len(self.args.chn_sel), self.args.t_len)))

    def _select_optimizer(self):
        params = self.model.parameters()
        assert self.args.optimizer in ['sgd', 'adam', 'adamw', 'nadam', 'radam', 'adamax']
        if self.args.optimizer == 'sgd':
            model_optim = optim.SGD(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            model_optim = optim.Adam(params, lr=self.args.learning_rate, betas=self.args.betas, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(params, lr=self.args.learning_rate, betas=self.args.betas, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'nadam':
            model_optim = optim.NAdam(params, lr=self.args.learning_rate, betas=self.args.betas, weight_decay=self.args.weight_decay, momentum_decay=self.args.momentum)
        elif self.args.optimizer == 'radam':
            model_optim = optim.RAdam(params, lr=self.args.learning_rate, betas=self.args.betas, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            model_optim = optim.Adamax(params, lr=self.args.learning_rate, betas=self.args.betas, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, pos_weight=None):
        # 1. Classification Error for Object Preference (e.g., matching SAM query to target image)
        self.criterion_cls = nn.BCEWithLogitsLoss()
        
        # 2. Bounding Box loss for simultaneous spatial positioning
        self.criterion_bbox = nn.SmoothL1Loss()
        
        print(f"Using Temporal Cross-Attention Preference Losses (BCE + SmoothL1 BBox)")
        return self.criterion_cls
    
    def compute_iou(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        return iou

    def compute_ciou_loss(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + 1e-6
        rho2 = ((b1_x1 + b1_x2) - (b2_x1 + b2_x2)) ** 2 / 4 + ((b1_y1 + b1_y2) - (b2_y1 + b2_y2)) ** 2 / 4

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / torch.clamp(h2, min=1e-6)) - torch.atan(w1 / torch.clamp(h1, min=1e-6)), 2)
        
        with torch.no_grad():
            alpha = v / torch.clamp((1 - iou + v), min=1e-6)
            
        ciou = iou - (rho2 / c2 + v * alpha)
        return 1 - ciou

    def do_loss(self, pref_logits, target_cls, bbox_pred, bbox_true):
        # Calculate matching preference
        loss_cls = self.criterion_cls(pref_logits, target_cls)
        
        # BBox spatial regression is only solved for TRUE authentic matches!!!
        # Mask out distractor pairings (target_cls == 0.0) from wrecking spatial weights
        # bbox_mask = (target_cls.squeeze(1) == 1.0)
        
        # if bbox_mask.sum() > 0:
        pred_boxes = bbox_pred#[bbox_mask]
        true_boxes = bbox_true#[bbox_mask]
        
        l1_loss = torch.nn.functional.l1_loss(pred_boxes, true_boxes, reduction='mean')
        ciou_loss = self.compute_ciou_loss(pred_boxes, true_boxes).mean()
        
        lambda_1 = 0  # CIoU weight
        lambda_2 = 1  # L1 weight
        
        loss_bbox = lambda_1 * ciou_loss + lambda_2 * l1_loss
        # else:
        #     loss_bbox = torch.tensor(0.0).to(target_cls.device)
            
        return loss_cls, loss_bbox

    def vali(self, train_data, train_loader, valid_data, valid_loader, criterion, epoch):
        total_loss_cls = []
        total_loss_bbox = []
        total_iou = []
        all_preds = []
        all_truths = []
        all_bbox_preds = []
        all_bbox_truths = []
        all_img_paths = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, img_ids, bbox, target_cls_val, batch_id, img_path) in enumerate(valid_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                
                # RAM direct retrieval (Bypasses Windows PyTorch IPC Bottleneck!)
                img_emb = torch.stack([valid_data.image_data[i_id]['image_emb'] for i_id in img_ids], dim=0)
                img_emb = img_emb.to(self.device, non_blocking=True)
                
                bbox = bbox.to(self.device, non_blocking=True)
                
                # Assign dynamic distractors or matches (0.0 or 1.0)
                target_cls = target_cls_val.unsqueeze(1).to(self.device)

                with autocast():
                    pref_logits, bbox_pred, attn_weights = self.model.forward_all(batch_x, batch_id=batch_id, img_emb=img_emb)
                    l_cls, l_box = self.do_loss(pref_logits, target_cls, bbox_pred, bbox)
                    iou = self.compute_iou(bbox_pred, bbox)
                
                total_loss_cls.append(l_cls.float())
                total_loss_bbox.append(l_box.float())
                total_iou.append(iou.float().mean())
                
                probs = torch.sigmoid(pref_logits)
                all_preds.append(probs.detach().cpu())
                all_truths.append(target_cls.cpu())
                all_bbox_preds.append(bbox_pred.detach().cpu())
                all_bbox_truths.append(bbox.cpu())
                all_img_paths.extend(img_path)
                
            total_loss_cls = torch.mean(torch.stack(total_loss_cls)).cpu()
            total_loss_bbox = torch.mean(torch.stack(total_loss_bbox)).cpu()
            total_iou = torch.mean(torch.stack(total_iou)).cpu()

            all_bbox_preds = torch.cat(all_bbox_preds, dim=0)
            all_bbox_truths = torch.cat(all_bbox_truths, dim=0)

        all_preds = torch.cat(all_preds, dim=0)
        all_truths = torch.cat(all_truths, dim=0)
        acc = ((all_preds > 0.5) == all_truths).float().mean().item()
        
        vali_acc = acc
        total_loss = total_loss_cls + total_loss_bbox * 5.0
        
        if epoch % 20 == 1:
            self.best_val_acc = vali_acc
            
            # Hook the image plots
            self.vali_plot_bboxes(all_bbox_preds.numpy(), all_bbox_truths.numpy(), epoch, all_img_paths)
            
        self.model.train()
        return total_loss_cls, total_loss_bbox, vali_acc, total_iou

    def vali_plot_bboxes(self, preds: np.ndarray, truths: np.ndarray, step: int, img_paths: list, num_samples: int = 16):
        import matplotlib.patches as patches
        from PIL import Image
        num_samples = min(num_samples, len(preds))
        cols = 4
        rows = int(np.ceil(num_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.25))
        if num_samples == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        for i in range(num_samples):
            ax = axes[i]
            
            # Draw real background image!
            if i < len(img_paths) and os.path.exists(img_paths[i]):
                img = Image.open(img_paths[i]).convert("RGB")
                
                # We need to map the aspect ratio back to exactly what SAM expects (16:9 canvas with black bars!)
                # Re-using the padding logic to ensure perfect bbox overlay representation
                target_size = (1920, 1080)
                img_ratio = img.width / img.height
                target_ratio = target_size[0] / target_size[1]
                if img_ratio > target_ratio:
                    new_w = target_size[0]
                    new_h = int(target_size[0] / img_ratio)
                else:
                    new_h = target_size[1]
                    new_w = int(target_size[1] * img_ratio)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                new_img = Image.new("RGB", target_size, (0, 0, 0))
                paste_x, paste_y = (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                ax.imshow(new_img, extent=[0, 1, 1, 0])
            else:
                ax.set_facecolor('#1a1a1a')
            
            tx1, ty1, tx2, ty2 = truths[i]
            rect_true = patches.Rectangle((tx1, ty1), tx2 - tx1, ty2 - ty1, linewidth=3, edgecolor='#00FF00', facecolor='none', label='Truth')
            ax.add_patch(rect_true)
            
            px1, py1, px2, py2 = preds[i]
            rect_pred = patches.Rectangle((px1, py1), px2 - px1, py2 - py1, linewidth=2, edgecolor='#FF3333', facecolor='none', linestyle='--', label='Pred')
            ax.add_patch(rect_pred)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(1080/1920)
            if i == 0: ax.legend(loc='lower left', prop={'size': 6})
                
        for i in range(num_samples, len(axes)): axes[i].axis('off')
        fig.tight_layout()
        self.logger.save_fig(fig, step, name='spatial_attention_bboxes')
        plt.close(fig)

    def train(self, setting):
        train_data, train_loader, valid_data, valid_loader = self._get_data()
        # print trn and val data num 
        print(f"Train data num: {len(train_data)}")
        print(f"Valid data num: {len(valid_data)}") 

        path = os.path.join(self.args.loggerdir, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingCla(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.epoch, eta_min=1e-6)
        pos_weight = getattr(train_data, 'pos_weight', None)
        criterion = self._select_criterion(pos_weight=pos_weight)

        self.model = self.model.to(self.device, non_blocking=True)
        self._print_model()

        print(f"\n[Temporal Object Grounding] Cross-Attention Object Preference Classification Enabled.\n")

        start_epoch = 0
        if hasattr(self.args, 'resume') and self.args.resume:
            if os.path.isfile(self.args.resume):
                print(f"==> Resuming from checkpoint '{self.args.resume}'")
                checkpoint = torch.load(self.args.resume, map_location=self.device)
                start_epoch = checkpoint.get('epoch', 0)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"==> Loaded checkpoint '{self.args.resume}' (epoch {start_epoch})")
            else:
                print(f"==> No checkpoint found at '{self.args.resume}'")

        for epoch in range(start_epoch, self.args.epoch):
            iter_count = 0
            train_loss_cls = []
            train_loss_bbox = []
            train_iou = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, img_ids, bbox, target_cls_val, batch_id, img_path) in enumerate(train_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                
                # RAM direct retrieval (Bypasses Windows PyTorch IPC Bottleneck!)
                img_emb = torch.stack([train_data.image_data[i_id]['image_emb'] for i_id in img_ids], dim=0)
                img_emb = img_emb.to(self.device, non_blocking=True)
                
                bbox = bbox.to(self.device, non_blocking=True)
                
                target_cls = target_cls_val.unsqueeze(1).to(self.device)
                
                iter_count += 1
                model_optim.zero_grad()
                
                with autocast():
                    pref_logits, bbox_pred, attn_weights = self.model.forward_all(batch_x, batch_id=batch_id, img_emb=img_emb)
                    l_cls, l_box = self.do_loss(pref_logits, target_cls, bbox_pred, bbox)
                    
                    # Weight spatial learning a bit higher
                    loss = l_box# * 5.0  l_cls + 
                
                scaler.scale(loss).backward()
                scaler.unscale_(model_optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                scaler.step(model_optim)
                scaler.update()
                
                iou = self.compute_iou(bbox_pred.detach(), bbox)
                
                train_loss_cls.append(l_cls)
                train_loss_bbox.append(l_box)
                train_iou.append(iou.mean())

            with torch.no_grad():
                train_loss_cls = torch.mean(torch.stack(train_loss_cls)).cpu()
                train_loss_bbox = torch.mean(torch.stack(train_loss_bbox)).cpu()
                train_iou = torch.mean(torch.stack(train_iou)).cpu()

            print("Epoch: {} cost time: {:.1f}s".format(epoch + 1, time.time() - epoch_time))
            vali_loss_cls, vali_loss_bbox, vali_acc, vali_iou = self.vali(
                train_data, train_loader, valid_data, valid_loader, criterion, epoch + 1
            )
            
            self.logger.log_scalar('loss/train_cls', train_loss_cls, epoch + 1)
            self.logger.log_scalar('loss/valid_cls', vali_loss_cls, epoch + 1)
            self.logger.log_scalar('loss/train_bbox', train_loss_bbox, epoch + 1)
            self.logger.log_scalar('loss/valid_bbox', vali_loss_bbox, epoch + 1)
            
            self.logger.log_scalar('metrics/valid_acc', vali_acc, epoch + 1)
            self.logger.log_scalar('metrics/train_iou', train_iou, epoch + 1)
            self.logger.log_scalar('metrics/valid_iou', vali_iou, epoch + 1)
            
            print(
                "Epoch: {0} | TrCls: {1:.4f} TrBox: {2:.4f} TrIoU: {3:.3f} | VaCls: {4:.4f} VaBox: {5:.4f} VaAcc: {6:.3f} VaIoU: {7:.3f}".format(
                    epoch + 1, train_loss_cls, train_loss_bbox, train_iou, vali_loss_cls, vali_loss_bbox, vali_acc, vali_iou
                )
            )
            
            early_stopping(vali_acc, self.model, path, epoch)
            
            # Save latest checkpoint state for resuming
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": model_optim.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, os.path.join(path, "checkpoint_latest.pth"))

            if self.session is not None:
                self.session.report(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss_cls.item(),
                        "vali_loss": vali_loss_cls.item()
                    }
                )

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model