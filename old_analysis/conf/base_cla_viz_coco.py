import os
import sys
from .base_cla_viz import BaseConfigViz
from datetime import datetime


class BaseConfigVizCoCo(BaseConfigViz):
    """Config for EEG-to-CLIP contrastive retrieval task."""
    def __init__(self):
        super().__init__()
        # Task & data
        self.task = 'retrieval'
        self.data = 'udf_viz_pair'

        # Encoder output dim → projection head input
        self.feat_dim = 256
        # CLIP embedding dim (ViT-L/14 = 768)
        self.proj_dim = 768

        # Contrastive loss
        self.temperature = 0.07   # init temperature (learnable)
        self.alpha = 0.5          # img-loss weight; (1-alpha) for caption loss

        # Retrieval evaluation
        self.retrieval_ks = [1, 5, 10, 50]

        # CLIP offline cache path (NPZ with img_ids / img_emb / cap_emb)
        self.clip_cache_path = './data/coco/processed_train2017/zx/clip_embeds_openai-clip-vit-large-patch14-bfb313c2.npz'

        # Logging
        self.loggerdir = './logs_Retrieval'

        # Model & training
        self.model = 'ATM'
        self.e_layers = 1           # keep encoder small for tiny dataset
        self.dropout = 0.5          # stronger regularization
        self.epoch = 150
        self.patience = 25
        self.batch_size = 64        # must be << N_train (~336)
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.betas = (0.9, 0.98)
        self.optimizer = 'adamw'

        self.comment = f'{datetime.now()}-retrieval-{self.model}'
