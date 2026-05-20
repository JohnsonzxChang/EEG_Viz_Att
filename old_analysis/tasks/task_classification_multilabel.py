from conf import BaseConfig
from utils import LoggerFile
from utils.data_factory import get_data_loader_cutt
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
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')

def topk_multilabel_accuracy_torch(logits, targets, k=3):
    """
    logits:  (B, C)
    targets: (B, C), 0/1
    """
    # 取每个样本 top-k 预测类别
    # values: (B, k), indices: (B, k)
    _, topk_idx = torch.topk(logits, k=k, dim=1)
    # 构造一个 (B, k) 的 hit 矩阵：对应位置的 target 是否为1
    B = targets.size(0)
    row_idx = torch.arange(B, device=logits.device).unsqueeze(1)  # (B,1)
    hits = targets[row_idx, topk_idx] > 0                        # (B,k)
    # 每个样本 top-k 中是否命中任意一个真实类
    sample_hit = hits.any(dim=1).float()                         # (B,)
    return sample_hit.mean().item()

def compute_map(logits, targets):
    """
    logits:  (B, C)  未过 sigmoid 的输出
    targets: (B, C)  0/1 标签
    返回: per-class AP (C,) 和 mAP (标量)
    """
    # 转概率
    probs = torch.sigmoid(logits).detach().numpy()   # (B, C)
    y_true = targets.detach().numpy().astype(int)    # (B, C)

    # average=None -> 返回每个类别的 AP
    ap_per_class = average_precision_score(
        y_true,
        probs,
        average=None
    )  # shape: (C,)
    
    # 动态排除没有正样本的类再求均值，防止被人工拖累成0
    valid_classes = (np.sum(y_true, axis=0) > 0)
    
    if np.sum(valid_classes) == 0:
        mAP = 0.0
    else:
        mAP = np.mean(ap_per_class[valid_classes])
        
    return ap_per_class, mAP

def compute_f1_micro_macro(logits, targets, threshold=0.3):
    probs = torch.sigmoid(logits).detach().numpy()   # (B, C)
    y_true = targets.detach().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)              # (B, C)
    # 微平均：全都混成一团真伪正负计算，不怕某些类别没数据
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    # 宏平均：先逐类计算，然后只提取验证集有GroundTruth正样本的那些类做平均
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    valid_classes = (np.sum(y_true, axis=0) > 0)
    
    if np.sum(valid_classes) == 0:
        macro_f1 = 0.0
    else:
        macro_f1 = np.mean(f1_per_class[valid_classes])
    return micro_f1, macro_f1

def compute_hamming_loss(logits, targets, threshold=0.3):
    """所有 (sample, class) 对中预测错误的比例"""
    probs = torch.sigmoid(logits).detach().numpy()
    y_true = targets.detach().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)
    return np.mean(y_true != y_pred)

def compute_subset_accuracy(logits, targets, threshold=0.3):
    """预测标签集合与真实完全一致的比例 (exact match ratio)"""
    probs = torch.sigmoid(logits).detach().numpy()
    y_true = targets.detach().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)
    return np.mean(np.all(y_pred == y_true, axis=1))

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification (ICCV 2021).

    gamma_neg: focusing parameter for negatives (higher = suppress easy-neg more, reduce FP)
    gamma_pos: focusing parameter for positives
    clip:      probability margin — hard-clamp negative prob, kills trivial-neg gradient
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric probability clipping
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Binary cross-entropy components
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets                  # p  for positives
            pt1 = xs_neg * (1.0 - targets)          # 1-p for negatives
            pt = pt0 + pt1
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss *= torch.pow(1.0 - pt, gamma)

        return -loss.sum() / logits.size(0)


class Exp_ClassificationM(Exp_Basic):
    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super(Exp_ClassificationM, self).__init__(args, logger)
        self.best_val_acc = 0.0
        self.session = session

    def _build_model(self):
        assert self.args.task == 'classification', f"Task must be 'classification', got {self.args.task}"
        model = self.model_dict[self.args.model](self.args).to(self.device)
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
        # Relaxed ASL parameters: Less extreme penalization of False Positives 
        # Since contrastive learning groups dense clusters, strict ASL pushes true positives below threshold
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.0)
        print(f"Using AsymmetricLoss (gamma_neg=2, gamma_pos=1, clip=0.0) - Relaxed settings for Contrastive Harmony")
        self.criterion = [[criterion, 1.0]]
        return criterion
    
    def do_loss(self, pred, true):
        loss = 0
        for cc in self.criterion:
            loss += (cc[0](pred, true) * cc[1])
        return loss 

    def vali(self, train_data, train_loader, valid_data, valid_loader, criterion, epoch):
        total_loss = []
        train_acc = []
        all_preds = []
        all_truths = []
        trn_preds = []
        trn_truths = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_id) in enumerate(valid_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_id = batch_id.to(self.device, non_blocking=True)

                outputs = self.model(batch_x, batch_id).detach()
                all_preds.append(outputs)
                all_truths.append(batch_y)
                y = batch_y
                loss = self.do_loss(outputs, y)
                total_loss.append(loss)
                # train_acc.append((y == outputs.argmax(-1)).sum())
            
            all_preds = torch.cat(all_preds, dim=0).cpu()
            all_truths = torch.cat(all_truths, dim=0).cpu()
            total_loss = torch.mean(torch.stack(total_loss)).cpu()

        # train_acc = torch.sum(torch.stack(train_acc)).cpu() / len(valid_data)
        # train_acc = (all_truths == all_preds.argmax(-1)).sum() / all_truths.shape[0]
        micro_f1, macro_f1 = compute_f1_micro_macro(all_preds, all_truths, threshold=0.3)
        _, mAP = compute_map(all_preds, all_truths)
        vali_acc = topk_multilabel_accuracy_torch(all_preds, all_truths, k=3)
        hamming = compute_hamming_loss(all_preds, all_truths, threshold=0.3)
        subset_acc = compute_subset_accuracy(all_preds, all_truths, threshold=0.3)
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
        return total_loss, vali_acc, micro_f1, macro_f1, mAP, hamming, subset_acc
    
    def vali_plot_per_class(self, all_preds: torch.Tensor, all_truths: torch.Tensor, step: int, threshold=0.5):
        """Per-class Precision / Recall / F1 柱状图 — 多标签版的"混淆矩阵" """
        from utils.data_loader import ALL_CLA
        probs = torch.sigmoid(all_preds).numpy()
        y_true = all_truths.numpy().astype(int)
        y_pred = (probs >= threshold).astype(int)

        C = y_true.shape[1]
        per_p = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_r = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_f = f1_score(y_true, y_pred, average=None, zero_division=0)

        # --- 1) Per-class P/R/F1 水平柱状图 ---
        fig, ax = plt.subplots(figsize=(10, max(8, C * 0.22)))
        y_pos = np.arange(C)
        bar_h = 0.25
        ax.barh(y_pos - bar_h, per_p, height=bar_h, label='Precision', color='#4C72B0')
        ax.barh(y_pos,          per_r, height=bar_h, label='Recall',    color='#DD8452')
        ax.barh(y_pos + bar_h, per_f, height=bar_h, label='F1',        color='#55A868')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ALL_CLA[:C], fontsize=6)
        ax.set_xlabel('Score')
        ax.set_title(f'Per-class P / R / F1  (epoch {step})')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        self.logger.save_fig(fig, step, name='per_class_prf')
        plt.close(fig)

        # --- 2) 多标签混淆热力图 (每类 TP/FP/FN/TN) ---
        mcm = multilabel_confusion_matrix(y_true, y_pred)  # (C, 2, 2)
        tp = mcm[:, 1, 1]
        fp = mcm[:, 0, 1]
        fn = mcm[:, 1, 0]
        tn = mcm[:, 0, 0]
        mat = np.stack([tp, fp, fn, tn], axis=1)  # (C, 4)

        fig2, ax2 = plt.subplots(figsize=(6, max(8, C * 0.22)))
        im = ax2.imshow(mat, aspect='auto', cmap='YlOrRd')
        ax2.set_yticks(np.arange(C))
        ax2.set_yticklabels(ALL_CLA[:C], fontsize=6)
        ax2.set_xticks([0, 1, 2, 3])
        ax2.set_xticklabels(['TP', 'FP', 'FN', 'TN'], fontsize=8)
        ax2.set_title(f'Multi-label Confusion  (epoch {step})')
        # 在格子内标注数值
        for i in range(C):
            for j in range(4):
                ax2.text(j, i, f'{int(mat[i, j])}', ha='center', va='center', fontsize=5, color='black')
        fig2.colorbar(im, ax=ax2, shrink=0.6)
        fig2.tight_layout()
        self.logger.save_fig(fig2, step, name='multilabel_confusion')
        plt.close(fig2)

    def tsne_plot(self, features:np.ndarray, labels:np.ndarray, step:int):
        from sklearn.manifold import TSNE
        from scipy.special import softmax
        tsne = TSNE(n_components=2, init='pca', random_state=self.args.seed)
        features = softmax(features, axis=-1)
        reduced_features = tsne.fit_transform(features)
        fig, ax = plt.subplots(1,1)
        sca = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        fig.colorbar(sca, ax=ax)
        ax.set_title('t-SNE Visualization of Features')
        self.logger.save_fig(fig, step, name='tsneFig')


    def train(self, setting):
        train_data, train_loader, valid_data, valid_loader = self._get_data()
        path = os.path.join(self.args.loggerdir, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingCla(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.epoch, eta_min=1e-6)
        pos_weight = getattr(train_data, 'pos_weight', None)
        criterion = self._select_criterion(pos_weight=pos_weight)

        self.model = self.model.to(self.device, non_blocking=True)

        self._print_model()

        for epoch in range(self.args.epoch):
            iter_count = 0
            train_loss = []
            train_preds = []
            train_truths = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_id) in enumerate(train_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_id = batch_id.to(self.device, non_blocking=True)
                iter_count += 1
                model_optim.zero_grad()

                outputs = self.model(batch_x, batch_id)
                loss = self.do_loss(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                model_optim.step()
                train_loss.append(loss)
                train_preds.append(outputs.detach())
                train_truths.append(batch_y.detach())

            scheduler.step()

            with torch.no_grad():
                train_loss = torch.mean(torch.stack(train_loss)).cpu()
                train_preds_cat = torch.cat(train_preds, dim=0).cpu()
                train_truths_cat = torch.cat(train_truths, dim=0).cpu()
                train_acc = topk_multilabel_accuracy_torch(train_preds_cat, train_truths_cat, k=3)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            vali_loss, vali_acc, micro_f1, macro_f1, mAP, hamming, subset_acc = self.vali(train_data, train_loader, valid_data, valid_loader, criterion, epoch + 1)
            self.logger.log_scalar('train_loss', train_loss, epoch + 1)
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
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Vali Loss: {4:.3f} Vali Acc: {5:.3f} mAP: {6:.3f} Hamming: {7:.4f} SubsetAcc: {8:.3f}"
                .format(epoch + 1, train_steps, train_loss, train_acc, vali_loss, vali_acc, mAP, hamming, subset_acc))
            early_stopping(vali_acc, self.model, path, epoch)

            if self.session is not None:
                self.session.report(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss.item(),
                        "vali_loss": vali_loss.item()
                    }
                )

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model