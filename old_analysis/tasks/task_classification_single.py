from conf import BaseConfig
from utils import LoggerFile
from utils.data_factory_single import get_data_loader_cutt
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

class Exp_ClassificationSingle(Exp_Basic):
    def __init__(self, args: BaseConfig, logger: LoggerFile = None, session=None):
        super(Exp_ClassificationSingle, self).__init__(args, logger)
        self.best_val_acc = 0.0
        self.session = session

    def _build_model(self):
        assert 'classification' in self.args.task, f"Task must be classification, got {self.args.task}"
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
        # Single-label classification uses CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        print(f"Using standard CrossEntropyLoss for strictly Single-Label Supervision")
        self.criterion = [[criterion, 1.0]]
        return criterion
    
    def do_loss(self, pred, true):
        loss = 0
        for cc in self.criterion:
            loss += (cc[0](pred, true) * cc[1])
        return loss 

    def vali(self, train_data, train_loader, valid_data, valid_loader, criterion, epoch):
        total_loss = []
        all_preds = []
        all_truths = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_id) in enumerate(valid_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_id = batch_id.to(self.device, non_blocking=True)

                outputs = self.model(batch_x, batch_id).detach()
                all_preds.append(outputs)
                all_truths.append(batch_y)
                loss = self.do_loss(outputs, batch_y)
                total_loss.append(loss)
            
            all_preds = torch.cat(all_preds, dim=0).cpu()
            all_truths = torch.cat(all_truths, dim=0).cpu()
            total_loss = torch.mean(torch.stack(total_loss)).cpu()

        # metrics for single label
        preds_cls = all_preds.argmax(dim=-1).numpy()
        truths_cls = all_truths.numpy()
        
        vali_acc = (preds_cls == truths_cls).mean()
        macro_f1 = f1_score(truths_cls, preds_cls, average="macro", zero_division=0)

        if vali_acc > self.best_val_acc:
            self.best_val_acc = vali_acc
            self.vali_plot_confusion(preds_cls, truths_cls, epoch, num_classes=all_preds.size(-1))
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "preds": all_preds,
                "truths": all_truths,
                "acc": vali_acc,
            }, f"{self.logger.writer.log_dir}/checkpoint.pth")
            
        self.model.train()
        return total_loss, vali_acc, macro_f1
    
    def vali_plot_confusion(self, preds_cls: np.ndarray, truths_cls: np.ndarray, step: int, num_classes: int):
        """Standard multi-class confusion matrix plot"""
        from utils.data_loader_single import ALL_CLA_OPENIMAGE as ALL_CLA
        
        cm = confusion_matrix(truths_cls, preds_cls, labels=np.arange(num_classes))
        
        fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.22), max(8, num_classes * 0.22)))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=ALL_CLA[:num_classes], 
               yticklabels=ALL_CLA[:num_classes],
               title=f'Confusion Matrix (epoch {step})',
               ylabel='True label',
               xlabel='Predicted label')
               
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
        plt.setp(ax.get_yticklabels(), fontsize=6)
        
        fig.tight_layout()
        self.logger.save_fig(fig, step, name='confusion_matrix')
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.epoch, eta_min=1e-6)
        pos_weight = getattr(train_data, 'pos_weight', None)
        criterion = self._select_criterion(pos_weight=pos_weight)

        self.model = self.model.to(self.device, non_blocking=True)
        self._print_model()

        print(f"\n[Supervised Learning] Single Label Mode Enabled.\n")

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

            # scheduler.step()

            with torch.no_grad():
                train_loss = torch.mean(torch.stack(train_loss)).cpu()
                train_preds_cat = torch.cat(train_preds, dim=0).cpu()
                train_truths_cat = torch.cat(train_truths, dim=0).cpu()
                
                preds_cls = train_preds_cat.argmax(dim=-1).numpy()
                truths_cls = train_truths_cat.numpy()
                train_acc = (preds_cls == truths_cls).mean()

            print("Epoch: {} cost time: {:.1f}s".format(epoch + 1, time.time() - epoch_time))
            vali_loss, vali_acc, macro_f1 = self.vali(train_data, train_loader, valid_data, valid_loader, criterion, epoch + 1)
            
            self.logger.log_scalar('train_loss', train_loss, epoch + 1)
            self.logger.log_scalar('valid_loss', vali_loss, epoch + 1)
            self.logger.log_scalar('train_acc', train_acc, epoch + 1)
            self.logger.log_scalar('valid_acc', vali_acc, epoch + 1)
            self.logger.log_scalar('macro_f1', macro_f1, epoch + 1)
            self.logger.log_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)
            
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Train Acc: {3:.3f} Vali Loss: {4:.4f} Vali Acc: {5:.3f} Macro-F1: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, train_acc, vali_loss, vali_acc, macro_f1))
            
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

        return self.model