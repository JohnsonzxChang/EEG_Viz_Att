import torch as th
import torch.nn as nn
import einops.layers.torch as eth
import math

from .net import _get_norm, _get_non_linear, _get_non_linear_fcn, Freq_Multi_Head


class id_net(nn.Module):
    def __init__(self, para, T0 = 50, device='cpu', M=10, emb_dim=200):
        super().__init__()
        # N = para['N_id'] # [40, 120, emb_dim]
        # assert N[-1] == emb_dim
        self.feature = nn.Parameter(th.rand((35, emb_dim)))

    def forward(self, id):
        id = self.feature[id,:]
        id = id / th.sqrt(th.sum(id ** 2) + 1e-6)
        return id

class sia_net(nn.Module):
    def __init__(self, params=None, M=10, T0=50, device='cpu', emb_dim=200, no_pre=False):
        super().__init__()
        N = params['N_id']  # [40, 120, emb_dim]
        assert N[-1] == emb_dim
        Fs = params['F']  # 250
        self.H = params['H']  # 3
        non_linear = params['non_linear']  # 'relu'
        norm = params['norm']
        drop = params['drop_id']  # [0.2, 0.2]
        self.drop = drop
        self.device = device

        self.preprocess = nn.Identity() if no_pre else nn.Sequential(
            nn.Conv2d(3, N[0], (1, 1), bias=False, padding='same'),
            _get_norm(norm, N[0]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )
        self.feature = nn.Sequential(
            nn.Conv2d(N[0], N[1], (1, 7), padding='same'),
            _get_norm(norm, N[0]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
            eth.Rearrange('b c m t -> (b m) c t'),
            # nn.Dropout(drop[0]),
            Freq_Multi_Head(N[1], T0, Fs, self.H, drop[0], 'none', non_linear, device, compress=8, bias=False, Head=6)
        )
        self.linear = nn.Sequential(
            eth.Rearrange('(b m) c -> b m c', m=M),
            # nn.Conv1d(M, 4*M, 5, padding='same'),
            # nn.Dropout(drop[1]),
            eth.Rearrange('b m c -> b (m c)'),
            nn.Linear(N[-2] * M, N[-1]),
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.feature(x)
        x = self.linear(x)
        x = x / (th.sqrt(th.sum(x ** 2, dim=-1, keepdim=True)) + 1e-6)
        return x

class sia_net_simple(nn.Module):
    def __init__(self, params=None, M=10, T0=50, device='cpu', emb_dim=200, no_pre=False):
        super().__init__()
        N = params['N_id']  # [40, 120, emb_dim]
        assert N[-1] == emb_dim
        Fs = params['F']  # 250
        self.H = 1 # params['H']  # 3
        non_linear = params['non_linear']  # 'relu'
        norm = params['norm']
        drop = params['drop_id']  # [0.2, 0.2]
        self.drop = drop
        self.device = device

        freq_list = th.linspace(8,15.8, 40)
        assert freq_list.shape[0] == 40
        position = th.arange(0, T0, requires_grad=False).unsqueeze(1) / Fs
        pe = []
        for i in range(self.H):
            pe.append(th.sin(math.pi * 2 * position * freq_list * (i + 1)))  # T, F*H
            # pe.append(th.cos(math.pi * 2 * position * freq_list * (i + 1)))  # T, F*H
        self.ref = th.cat(pe, dim=-1).to(device).T.requires_grad_(False)
        assert N[1] == self.ref.shape[0]

        self.preprocess = nn.Identity() if no_pre else nn.Sequential(
            nn.Conv2d(3, N[0], (1, 1), bias=False, padding='same'),
            # _get_norm(norm, N[0]),
            # _get_non_linear(non_linear),
            # nn.Dropout(drop[0]),
        )
        self.feature = nn.Sequential(
            nn.Conv2d(N[0], N[1], (1, 7), padding='same'),
            _get_norm(norm, N[0]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )
        self.linear = nn.Sequential(
            eth.Rearrange('b c d m -> b (c d m)'),
            nn.Dropout(drop[1]),
            nn.Linear(N[1]**2 * M, N[2]),
        )

    def get_coeff(self, x):
        x = (x - x.mean(dim=-1, keepdim=True)) / (1e-8 + x.std(dim=-1, keepdim=True))
        cor = th.einsum('bcmt, ct -> bcm', x, self.ref)
        x = cor.unsqueeze(-1).expand_as(x) * x
        cor = th.einsum('bcmt, bdmt -> bcdm', x, x)
        return cor

    def forward(self, x):
        x = self.preprocess(x)
        x = self.feature(x)
        x = self.get_coeff(x)
        x = self.linear(x)
        x = x.renorm(p=2, dim=-1, maxnorm=1)
        return x