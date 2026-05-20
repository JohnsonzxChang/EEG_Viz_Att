from conf.base import BaseConfig
from .fapem.model_cnn import da_encoder, da_classifier, da_encoder_refine
from .fapem.net import (
    _get_norm, _get_non_linear, _get_non_linear_fcn,
    Freq_Attention_Multi_Head
)
import torch as th
from torch import nn

PARA_BASE = {
        'N': [64, 64, 64, 64, 64, 64],
        't': [15, 13, 7, 5], #[3, 7],
        's': [2, 2, 2, 2],
        'F': 1000,
        'H': 3,
        'non_linear': 'leakrelu',
        'norm': 'none',
        'drop': [0.15, 0.15, 0.7], #[0.1, 0.1, 0.92],
        'compress': 8,
        'head': 6,
        'emb_dim': 60,
        'bias':[True, False],
        'ica': 10,
        "mid_dim": 80,
        "num_person": 35,
        'banks': 3,
    }

class fapem_encoder(nn.Module):
    def __init__(self, conf:BaseConfig):
        super().__init__()
        self.conf = conf 
        self.encoder = da_encoder(params=PARA_BASE, Mlist=conf.chn_sel, T0=conf.t_len, device=conf.device)
        # self.encoder2 = da_encoder(params=PARA_BASE, Mlist=conf.chn_sel, T0=conf.t_len, device=conf.device)
        self.grid = nn.Sequential(
            nn.Dropout(PARA_BASE['drop'][-1]),
            _get_non_linear(PARA_BASE['non_linear']),
            nn.Linear(self.encoder.flat, PARA_BASE["mid_dim"], bias=PARA_BASE['bias'][-1])
        )
        # self.regeresser2 = nn.Sequential(
        #     nn.Dropout(PARA_BASE['drop'][-1]),
        #     _get_non_linear(PARA_BASE['non_linear']),
        #     nn.Linear(self.encoder.flat, PARA_BASE["mid_dim"], bias=PARA_BASE['bias'][-1]),
        #     _get_non_linear(PARA_BASE['non_linear']),
        #     nn.Dropout(PARA_BASE['drop'][1]),
        #     nn.Linear(PARA_BASE["mid_dim"], 1, bias=PARA_BASE['bias'][-1]),
        # )
        # self.to_grid = nn.Linear(PARA_BASE["mid_dim"], 10*10, bias=False)
        if self.conf.task == 'classification':
            self.regeresser1 = nn.Sequential(
             _get_non_linear(PARA_BASE['non_linear']),
            nn.Dropout(PARA_BASE['drop'][1]),
            nn.Linear(PARA_BASE["mid_dim"], self.conf.num_classes, bias=PARA_BASE['bias'][-1])
        )
        else:
            self.regeresser1 = nn.Sequential(
                _get_non_linear(PARA_BASE['non_linear']),
                nn.Dropout(PARA_BASE['drop'][1]),
                nn.Linear(PARA_BASE["mid_dim"], 8+2, bias=PARA_BASE['bias'][-1])
            )

    def forward(self, x, sub=None):
        # x = x.unsqueeze(1)
        x, _ = self.encoder(x)
        # x2, _ = self.encoder2(x)
        # x = self.grid(th.cat([x1, x2], dim=-1))
        x = self.grid(x)
        # grid_pred = self.to_grid(x).view(-1, 100)
        x = self.regeresser1(x)
        if self.conf.task == 'classification':
            return x
        else:
            return x[:, :8], x[:, 8:] #, th.sigmoid(grid_pred)