"""
Config for CLIP-Fused EEG Classification with ERP Averaging.

Inherits from BaseConfigViz and adds:
  - ERP averaging settings (erp_mode, erp_k, erp_n_aug)
  - CLIP fusion settings (clip_dim, feat_dim, lambda_align)
  - Adjusted training hyperparams for the fused model

References:
  - ATM (Li et al., NeurIPS 2024): ERP averaging + temporal-spatial encoding
  - NICE (Song et al., 2023): K-trial sub-averaging for SNR improvement
  - EEG-CLIP: Cross-modal alignment between EEG and CLIP
  - BraVL: Brain-Visual-Language trimodal learning
"""

import os
from datetime import datetime
from .base_cla_viz import BaseConfigViz


class BaseConfigClipFused(BaseConfigViz):
    def __init__(self):
        super().__init__()
        self.task = 'classification'
        self.data = 'udf_viz_erp_avg'  # uses the new ERP-averaged data loader

        # ── ERP Averaging Settings ──
        # 'image_avg':  average ALL trials per image (max SNR, fewer samples)
        # 'ktrial_avg': randomly sub-average K trials (moderate SNR, data augmentation)
        # 'both':       K-trial for train, image-avg for val (recommended)
        self.erp_mode = 'both'
        self.erp_k = 4          # number of trials to average in ktrial mode
        self.erp_n_aug = 5      # number of augmented samples per image in ktrial mode

        # ── CLIP Fusion Settings ──
        self.use_clip_feat = True
        self.clip_dim = 768         # CLIP ViT-L/14 output dim
        self.feat_dim = 256         # EEG encoder feature dim (also fusion dim)
        self.lambda_align = 0.1     # weight for EEG↔CLIP alignment loss
        self.align_temperature = 0.1  # InfoNCE temperature for alignment

        # CLIP embedding cache
        self.clip_cache_path = './data/coco/processed_train2017/zx/clip_embeds_openai-clip-vit-large-patch14-bfb313c2.npz'
        self.clip_model_name = 'openai/clip-vit-large-patch14'

        # ── Model Settings ──
        self.model = 'ATM'          # ATM encoder (best for EEG visual decoding)
        self.enc_in = 33
        self.t_len = 1000
        self.t0 = 2000
        self.e_layers = 3
        self.n_heads = 4
        self.d_model = 128
        self.d_ff = 256
        self.dropout = 0.5          # moderate regularization

        # ── Training Settings ──
        self.epoch = 150
        self.batch_size = 128       # smaller than raw (512) since ERP-avg reduces samples
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.betas = (0.9, 0.98)
        self.optimizer = 'adamw'
        self.patience = 30

        # ── Logging ──
        self.loggerdir = './logs_ClipFused'
        self.comment = f'{datetime.now()}-clipfused-{self.model}-{self.erp_mode}'
