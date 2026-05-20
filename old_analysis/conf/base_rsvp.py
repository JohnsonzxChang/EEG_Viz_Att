"""Config for RSVP-COCO experiments (12-class classification + CLIP retrieval)."""

import os
import sys
from .base_cla_viz import BaseConfigViz
from datetime import datetime


class BaseConfigRSVP(BaseConfigViz):
    """RSVP-COCO: joint classification + CLIP-contrastive training."""

    def __init__(self):
        super().__init__()

        # ── Task & Data ──────────────────────────────────────────────────
        self.task = 'rsvp_clip'
        self.data = 'rsvp_coco'

        # RSVP .fif path
        self.rsvp_fif_path = None  # set to your .fif path, e.g. r'D:\data\epochs_big-epo.fif'

        # ERP averaging
        self.erp_k = None            # None = use ALL trials per image
        self.use_post_stim = True     # use [0, +500ms] window

        # ── EEG Encoder ──────────────────────────────────────────────────
        self.model = 'ATM'
        self.enc_in = 32              # 32 pure EEG channels (no eye-tracking)
        self.t_len = 500              # post-stimulus 500ms @ 1000Hz
        self.num_classes = 12         # 12 target RSVP categories
        self.chn_sel = list(range(32))

        # ATM hyperparams
        self.e_layers = 2             # channel transformer layers
        self.n_heads = 4
        self.d_model = 128
        self.d_ff = 256
        self.feat_dim = 256           # encoder output dim

        # ── CLIP Projection ──────────────────────────────────────────────
        self.proj_dim = 768           # CLIP ViT-L/14 dimension
        self.temperature = 0.07      # learnable init
        self.alpha = 0.5             # weight for image vs caption loss
        self.retrieval_ks = [1, 3, 5, 10]

        # CLIP model & cache
        self.clip_model_name = 'openai/clip-vit-large-patch14'
        self.clip_cache_path = None   # auto-detect or compute on first run

        # ── Training ─────────────────────────────────────────────────────
        self.epoch = 200
        self.batch_size = 32          # 360 images → ~9 batches/epoch
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.betas = (0.9, 0.98)
        self.optimizer = 'adamw'
        self.dropout = 0.5
        self.patience = 30

        # Joint loss weighting: total = ce_weight * CE + clip_weight * CLIP
        self.ce_weight = 1.0
        self.clip_weight = 0.5

        # Logging
        self.loggerdir = './logs_RSVP'

        self.comment = f'{datetime.now()}-rsvp-clip-{self.model}'
