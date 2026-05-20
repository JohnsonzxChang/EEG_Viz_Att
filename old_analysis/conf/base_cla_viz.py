import torch
import os
import sys
from .base import BaseConfig
from datetime import datetime
from torch import optim
import time
import json

class BaseConfigViz(BaseConfig):
    def __init__(self):
        super(BaseConfigViz, self).__init__()
        self.task = 'classification'  # 'classification', 'regression', 'anomaly_detection', 'forecasting', 'reinforcement_learning' 'forecasting'
        self.epoch = 100
        self.early_stop = 5000
        self.patience = 5000
        self.device = 'cuda'
        self.loggerdir = './logs_Visual'
        self.plot_trn = True

        # DataLoader settings
        self.seed = 10086
        self.num_workers = min(8, os.cpu_count() or 4)
        self.pin_memory = True
        self.batch_size = 512
        self.flag = 'val'
        self.data = 'udf_viz_m' # 'ss_bench' or 'udf_ssmr' or 'udf_mivr' 'udf_mivr_pred' 'udf_mivr_cla' 'udf_emg_vr'
        self.num_subjects = 35
        self.subjects_val = None
        self.subjects_trn = 1
        self.num_classes = 80 # 80 - person removed
        self.classes_val = None
        self.classes_trn = None
        self.num_trials = 6
        self.trials_val = None
        self.trials_trn = None
        self.chn_sel = list(range(33)) # list(range(32)) #[53, 54, 55, 56, 57, 58, 59, 61, 62, 63] list(range(64))  # select all channels by default
        self.data_type = 'x0|x'
        self.mask_type = 'extrapolation'
        self.mask_ratio = 0.2
        self.t0 = 2000
        self.t_len = 1000
        self.mux = 3

        # COCO image path (used by data loaders for image retrieval tasks)
        # Falls back to DATA_ROOT/data/coco/train2017 if not set
        _dr = os.environ.get('DATA_ROOT', os.getcwd())
        self.coco_img_path = os.path.join(_dr, 'data', 'coco', 'train2017')

        # Model settings
        # 'CNN': CNN_Encoder,
        # 'Transformer': TFEncoder,
        # 'TransForcast' : TransForcast,
        # 'EEGNet':EEGNet,
        # 'FAPEM': fapem,
        # 'LSTM' : lstm_encoder
        # HHNeuron
        self.model = 'CNN' #  
        self.enc_in = 33
        # self.seq_len = 400 # 1000Hz * 0.4s = 400
        self.e_layers = 3
        self.n_heads = 4
        self.d_ff = 512
        self.d_model = 64
        self.patch_len = 9
        self.stride = 3
        self.activation = 'relu'
        self.use_tpatch = True
        self.use_ssm = True
        self.ssm_kernel_size = 3
        self.d_layers = 3
        self.pred_len = 50


        self.dropout = 0.8
        self.comment = f'{datetime.now()}-vizVIcla-{self.model}'

        # "batch_size": 32,
        # "beta1": 0.7639596238409772,
        # "beta2": 0.657751589914447,
        # "epochs": 100,
        # "learning_rate": 0.0008680873025075397,
        # "momentum": 0.6913490092653194,
        # "optimizer": "adam",
        # "weight_decay": 1.2499456713029488e-06
        # Optimizer settings
        # 'sgd', 'adam', 'adamw', 'nadam', 'radam', 'adamax'
        self.optimizer = 'adamw'
        self.learning_rate = 8e-4
        self.gamma = 0
        self.momentum = 0.69
        self.weight_decay = 1.25e-6
        self.betas = (0.7639596238409772, 0.657751589914447) # 0.96

