import torch as th
from torch import nn

from conf.base import BaseConfig

class lstm_encoder(nn.Module):
    def __init__(self, conf:BaseConfig):
        super().__init__()
        self.conf = conf

    def forward(self, x, idid=None):
        pass 