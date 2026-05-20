from conf import BaseConfig
from utils import LoggerFile
from utils.data_factory_openimages import get_data_loader_cutt
from .task_base import Exp_Basic
from utils.tools import EarlyStoppingCla
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.ops import generalized_box_iou_loss

warnings.filterwarnings('ignore')

from encoder.pref_predictor import PreferenceBboxPredictor

class TotalLoss(nn.Module):
    def __init__(self, lambda1=0.5, lambda2=0.3, lambda3=0.1, temperature=0.07):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.tau = temperature
        # Linear projection to align EEG with visual tokens space for contrastive loss
        self.eeg_proj = nn.Linear(256, 768)

    def forward(self, preference_logits, attn_maps, eeg_cls,
                object_features, object_mask, target_idx,
                pred_bbox, target_bbox):
        B, K, N = attn_maps.shape

        # Loss 1: Preference Classification Loss
        L_pref = F.cross_entropy(preference_logits, target_idx)

        # Loss 2: EEG-Visual Contrastive Loss
        proj_eeg = self.eeg_proj(eeg_cls)  # (B, 768)
        pos_feat = object_features[torch.arange(B), target_idx]  # (B, 768)
        
        all_obj = object_features.reshape(B * K, -1)  # (B*K, 768)
        sim_all = torch.mm(proj_eeg, all_obj.T) / self.tau  # (B, B*K)
        
        # Mask out padded visual features across the entire batch to prevent false contrastive matching
        pad_mask = ~object_mask.view(-1) # (B*K)
        sim_all = sim_all.masked_fill(pad_mask.unsqueeze(0), -1e4)
        
        labels = target_idx + torch.arange(B, device=target_idx.device) * K
        L_contra = F.cross_entropy(sim_all, labels)

        # Loss 3: Temporal Consistency Loss - Removed Fixation
        L_temporal = torch.tensor(0.0).to(preference_logits.device)

        # Loss 4: Bbox Regression Loss
        # Ensure coordinates are in x1,y1,x2,y2 format and properly normalized
        L_bbox = generalized_box_iou_loss(pred_bbox, target_bbox, reduction='mean')

        L_total = L_pref + self.lambda1 * L_contra + self.lambda2 * L_temporal + self.lambda3 * L_bbox

        return L_total, {
            'L_pref': L_pref.item(),
            'L_contra': L_contra.item(),
            'L_temporal': L_temporal.item(),
            'L_bbox': L_bbox.item(),
        }

class Exp_NewAttentionTracking(Exp_Basic):
    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super(Exp_NewAttentionTracking, self).__init__(args, logger)
        self.best_val_acc = 0.0
        self.session = session

    def _build_model(self):
        assert 'new_attention_tracking' in self.args.task, f"Task must be new_attention_tracking, got {self.args.task}"
        model = PreferenceBboxPredictor(self.args).to(self.device)
        return model

    def _get_data(self):
        train_data, train_loader, valid_data, valid_loader = get_data_loader_cutt(self.args)
        return train_data, train_loader, valid_data, valid_loader

    def _select_optimizer(self):
        params = self.model.parameters()
        return optim.AdamW(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        self.criterion = TotalLoss(
            lambda1=self.args.lambda1,
            lambda2=self.args.lambda2,
            lambda3=self.args.lambda3,
            temperature=self.args.temperature
        ).to(self.device)
        return self.criterion
    
    def vali(self, train_data, train_loader, valid_data, valid_loader, criterion, epoch):
        total_loss = []
        all_preds = []
        all_truths = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (eeg_data, obj_feat, obj_bbox, object_mask, target_idx, target_bbox, img_id, subject_id, img_path) in enumerate(valid_loader):
                eeg_data = eeg_data.to(self.device, non_blocking=True)
                obj_feat = obj_feat.to(self.device, non_blocking=True)
                obj_bbox = obj_bbox.to(self.device, non_blocking=True)
                object_mask = object_mask.to(self.device, non_blocking=True)
                target_idx = target_idx.to(self.device, non_blocking=True)
                target_bbox = target_bbox.to(self.device, non_blocking=True)
                
                with autocast():
                    pred_bbox, logits, attn_maps, eeg_cls = self.model(eeg_data, obj_feat, obj_bbox, object_mask)
                    l_total, loss_dict = criterion(
                        logits, attn_maps, eeg_cls, obj_feat, object_mask, target_idx, pred_bbox, target_bbox
                    )
                
                total_loss.append(l_total.item())
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_truths.append(target_idx.cpu())
                
        all_preds = torch.cat(all_preds, dim=0)
        all_truths = torch.cat(all_truths, dim=0)
        acc = (all_preds == all_truths).float().mean().item()
        total_loss_mean = sum(total_loss) / len(total_loss)
        
        if epoch % 1 == 1:
            self.best_val_acc = max(self.best_val_acc, acc)
            
        self.model.train()
        return total_loss_mean, acc

    def train(self, setting):
        train_data, train_loader, valid_data, valid_loader = self._get_data()
        path = os.path.join(self.args.loggerdir, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        early_stopping = EarlyStoppingCla(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.epoch, eta_min=1e-6)
        criterion = self._select_criterion()

        print(f"\n[Temporal Object Grounding] Beginning End-to-End Fine-tuning for Architecture 2.\n")

        for epoch in range(self.args.epoch):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()

            for i, (eeg_data, obj_feat, obj_bbox, object_mask, target_idx, target_bbox, img_id, subject_id, img_path) in enumerate(train_loader):
                eeg_data = eeg_data.to(self.device, non_blocking=True)
                obj_feat = obj_feat.to(self.device, non_blocking=True)
                obj_bbox = obj_bbox.to(self.device, non_blocking=True)
                object_mask = object_mask.to(self.device, non_blocking=True)
                target_idx = target_idx.to(self.device, non_blocking=True)
                target_bbox = target_bbox.to(self.device, non_blocking=True)
                
                model_optim.zero_grad()
                
                with autocast():
                    # For training: using soft_bbox selection which is output natively
                    pred_bbox, logits, attn_maps, eeg_cls = self.model(eeg_data, obj_feat, obj_bbox, object_mask)
                    
                    l_total, loss_dict = criterion(
                        logits, attn_maps, eeg_cls, obj_feat, object_mask, target_idx, pred_bbox, target_bbox
                    )

                scaler.scale(l_total).backward()
                scaler.unscale_(model_optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(model_optim)
                scaler.update()

                train_loss.append(l_total.item())
                iter_count += 1
                
                # Debug tracking logic
                if i % 10 == 0:
                    print(f"\tIt: {i} | T_Loss: {l_total.item():.4f} | " + 
                          f"Pref: {loss_dict['L_pref']:.3f} | Contra: {loss_dict['L_contra']:.3f} | " + 
                          f"Temp: {loss_dict['L_temporal']:.3f} | Bbox: {loss_dict['L_bbox']:.3f}")

            train_loss_mean = sum(train_loss) / len(train_loss)
            scheduler.step()
            
            print("Epoch: {} cost time: {:.1f}s".format(epoch + 1, time.time() - epoch_time))
            vali_loss, vali_acc = self.vali(train_data, train_loader, valid_data, valid_loader, criterion, epoch + 1)
            
            self.logger.log_scalar('loss/train_total', train_loss_mean, epoch + 1)
            self.logger.log_scalar('loss/valid_total', vali_loss, epoch + 1)
            self.logger.log_scalar('metrics/valid_acc', vali_acc, epoch + 1)
            
            print("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} Vali Acc: {3:.3f}".format(
                epoch + 1, train_loss_mean, vali_loss, vali_acc)
            )
            
            # early_stopping(vali_acc, self.model, path, epoch)
            
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

        return self.model
