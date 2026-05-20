import torch
from .model_config import ModelConfig
from datetime import datetime
from torch import optim
import time
import json 

class BaseConfig(ModelConfig):
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.task = 'regression'  # 'classification', 'regression', 'anomaly_detection', 'forecasting', 'reinforcement_learning' 'forecasting'
        self.epoch = 1000
        self.early_stop = 500
        self.patience = 500
        self.device = 'cuda'
        self.loggerdir = './logs/'
        self.plot_trn = True

        # DataLoader settings
        self.seed = 10086
        self.num_workers = 24
        self.pin_memory = True
        self.batch_size = 32
        self.flag = 'val'
        self.data = 'udf_ssmr' # 'ss_bench' or 'udf_ssmr' or 'udf_mivr' 'udf_mivr_pred' 'udf_mivr_cla' 'udf_emg_vr'
        self.num_subjects = 35
        self.subjects_val = None
        self.subjects_trn = 1
        self.num_classes = 2
        self.classes_val = None
        self.classes_trn = None
        self.num_trials = 6
        self.trials_val = None
        self.trials_trn = None
        self.chn_sel = list(range(32)) # list(range(32)) #[53, 54, 55, 56, 57, 58, 59, 61, 62, 63] list(range(64))  # select all channels by default
        self.data_type = 'x0|x'
        self.mask_type = 'extrapolation'
        self.mask_ratio = 0.2
        self.t0 = 125
        self.t_len = 50
        self.mux = 3

        # Model settings
        # 'CNN': CNN_Encoder,
        # 'Transformer': TFEncoder,
        # 'TransForcast' : TransForcast,
        # 'EEGNet':EEGNet,
        # 'FAPEM': fapem,
        # 'LSTM' : lstm_encoder
        # 'Regformer' : RegressionTransformer
        self.model = 'Regformer' #  
        self.enc_in = 32
        # self.seq_len = 400 # 1000Hz * 0.4s = 400
        self.e_layers = 3
        self.n_heads = 4
        self.d_ff = 1024
        self.d_model = 64
        self.patch_len = 9
        self.stride = 3
        self.activation = 'relu'
        self.use_tpatch = True
        self.use_ssm = True
        self.ssm_kernel_size = 3
        self.d_layers = 3
        self.pred_len = 50
        self.factor = 5


        self.dropout = 0.3
        self.comment = f'{datetime.now()}-ssmr-{self.model}'

#           "batch_size": 32,
#   "beta1": 0.7639596238409772,
#   "beta2": 0.657751589914447,
#   "epochs": 100,
#   "learning_rate": 0.0008680873025075397,
#   "momentum": 0.6913490092653194,
#   "optimizer": "adam",
#   "weight_decay": 1.2499456713029488e-06
        # Optimizer settings
        # 'sgd', 'adam', 'adamw', 'nadam', 'radam', 'adamax'
        self.optimizer = 'adamw'
        # lr: float | Tensor = 0.001, betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01, amsgrad: bool = False, 
        self.learning_rate = 1e-3 # 8e-4
        self.gamma = 0
        self.momentum = 0.69
        self.weight_decay = 0.01 # 1.25e-6
        self.betas = (0.9, 0.999) # (0.7639596238409772, 0.657751589914447) # 0.96

        # Loss config
        # 'mae', 'mse', 'huber', 'quantile', 'cos'
        self.loss_config = {
            # 'mae' : 1.0, 
            'mse' : 1.0, 
            # 'huber' : [1.0, 0.12], 
            # 'quantile' : [1.0, 0.9], 
            # 'cos' : 1.0,
        }

        # RL settings
        self.rl_model = 'A2C'  # 'PPO' or 'A2C'
        self.train_steps = 200000
        self.max_steps = 300  # max steps per episode
    
    def json(self):
        """Convert configuration to a JSON-serializable dictionary.
        
        Returns:
            dict: Configuration parameters as a dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            # Handle special cases like torch devices and numpy arrays
            if isinstance(value, torch.device):
                value = str(value)
            elif hasattr(value, 'tolist'):  # For numpy arrays
                value = value.tolist()
            config_dict[key] = value
        
        return config_dict

