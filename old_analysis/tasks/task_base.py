import torch as th
import os
import sys

from conf import BaseConfig
from utils import LoggerFile# , DataInterface
from encoder import CNN_Encoder, TFEncoder, EEGNet, fapem_encoder, lstm_encoder, TransForcast, HHMambaEncoder, RegressionTransformer, ATM_Encoder
from encoder.tvit_encoder import TViT_Encoder


class Exp_Basic(object):
    def __init__(self, args: BaseConfig, logger: LoggerFile = None):
        self.args = args
        mp = os.path.join(self.args.loggerdir, self.args.task)
        os.path.exists(mp) or os.makedirs(mp)
        self.logger = logger if logger is not None else LoggerFile(mp, args.comment, self.args)
        self.model_dict = {
            'CNN': CNN_Encoder,
            'Transformer': TFEncoder,
            'EEGNet':EEGNet,
            'FAPEM': fapem_encoder,
            'LSTM' : lstm_encoder,
            'TransForcast' : TransForcast,
            'HHNeuron' : HHMambaEncoder,
            'Regformer': RegressionTransformer,
            'ATM': ATM_Encoder,
            'TViT': TViT_Encoder,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.device == 'cuda':
            device = th.device('cuda')
            print('Use GPU: cuda')
        elif self.args.device == 'mps':
            device = th.device('mps')
            print('Use GPU: mps')
        else:
            device = th.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
