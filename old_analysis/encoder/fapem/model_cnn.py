import torch as th
import torch.nn as nn
import torch.nn.functional as FF
from einops import rearrange
import einops.layers.torch as eth

from .net import (
    _get_norm, _get_non_linear, _get_non_linear_fcn,
    Freq_Attention_Multi_Head
)

class da_encoder(nn.Module):
    def __init__(self, params=None, Mlist=range(10), T0=50, device='cpu',
                 DEBUG=False, return_x_prime=False, val_id=0, mid=120):
        super().__init__()
        N = params['N']
        t = params['t']
        s = params['s']
        emb = params['emb_dim']
        norm = params['norm']
        non_linear = params['non_linear']
        drop = params['drop']  # [0.6, 0.6, 0.95]
        bias = [True, False]
        TS = T0
        FS = params['F']
        self.num_person = params['num_person']
        self.H = params['H']
        self.drop = drop
        self.device = device
        ica = params['ica']
        M = len(Mlist)
        NN = params['banks']
        # TN = [T0 // 2, T0 // 2]  # params['TN']
        # FN = params['FN']
        tmpT = T0 
        tmpF = FS
        TN, FN = [], []
        for i in range(len(params['s'])):
            tmpT = tmpT // params['s'][i]
            tmpF = tmpF / params['s'][i]
            TN.append(tmpT)
            FN.append(tmpF)

        self.compress = params['compress']  # 8
        self.head = params['head']  # 6

        self.id_embedding = nn.Embedding(self.num_person, emb, max_norm=1)

        self.id_operation = nn.Sequential(
            nn.Linear(emb, M * ica, bias=False),
            # nn.Dropout(drop[0]),
            eth.Rearrange('b (m n) -> b m n', m=M),
        )

        self.preprocess = nn.Sequential(
            nn.Conv2d(NN, N[0], (1, 1), bias=bias[0], padding='same'),
            # _get_norm(norm, N[0]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )

        self.chn_combination = nn.Sequential(
            nn.Conv2d(N[0], N[1], (ica, 1), bias=False),
            _get_norm(norm, N[1]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )
        self.chn_attention = nn.Sequential(
            Freq_Attention_Multi_Head(N[1], TS, FS, self.H, self.drop[1], norm, non_linear, self.device,
                                      compress=self.compress,
                                      bias=bias[1], norm_dim=1, Head=self.head),
        )

        self.tempo_fea, self.tempo_down, self.tempo_att = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        hh = [1, 1, 1, 1]  # [1,2,3]# [3,2,1] # [1,1,1]
        for i in range(len(t)):
            tmp = self._create_tempo_layers(t[i], s[i], N[i + 1], N[i + 2], TN[i], FN[i], bias, norm, non_linear,
                                            har=hh[i])
            self.tempo_fea.append(tmp[0])
            self.tempo_down.append(tmp[1])
            self.tempo_att.append(tmp[2])

        self.flat = N[-1] * TN[-1]
        self.function_resnet = _get_non_linear_fcn(non_linear)
        print(f'final flatten number is {self.flat}...')

    def _create_tempo_layers(self, K, S, Cin, Cout, TN, Fs, bias, norm, non_linear, har):
        feature = nn.Sequential(
            nn.Conv2d(Cin, Cout, (1, K), (1, S), padding=(0, K // 2), bias=bias[0]),
            _get_norm(norm, Cout),
            _get_non_linear(non_linear),
            nn.Dropout(self.drop[0])
        )
        downsample = nn.Sequential(
            nn.Conv2d(Cin, Cout, 1, (1, S), bias=False),
            _get_norm(norm, Cout),
        ) if S != 1 else nn.Identity()
        attention = nn.Sequential(
            Freq_Attention_Multi_Head(Cout, TN, Fs, self.H, self.drop[1], norm, non_linear, self.device,
                                      compress=self.compress,
                                      bias=bias[1], Head=self.head),
        ) if Fs > 70 else nn.Identity()
        return feature, downsample, attention

    def _get_chn_self_corr(self, x):
        # x: b c m t -> b c m m
        x = (x - x.mean(dim=-1, keepdim=True)) / (1e-5 + x.std(dim=-1, keepdim=True))
        corr = th.einsum('bcmt,bcnt->bcmn', x, x)
        return th.log(th.mean(th.norm(corr, dim=(2, 3), p=2), dim=1))

    def forward(self, x, id=None):
        if id is None:
            id = th.zeros((x.shape[0]), dtype=th.long, device=x.device)
        id = self.id_embedding(id)
        x = self.preprocess(x)
        # addition = self._get_chn_self_corr(x)
        x_p = th.einsum('bcmt,bmn->bcnt', x, self.id_operation(id))
        # addition = self._get_chn_self_corr(x) - addition
        x = self.chn_combination(x_p)
        x = self.function_resnet(x + self.chn_attention(x), inplace=True)
        x = FF.dropout(x, self.drop[1], training=self.training)
        for i in range(len(self.tempo_fea)):
            xtmp = self.tempo_fea[i](x)
            xtmp = self.tempo_att[i](xtmp)
            x = self.function_resnet(xtmp + self.tempo_down[i](x), inplace=True)
            if not i == len(self.tempo_fea) - 1:
                x = FF.dropout(x, self.drop[0], training=self.training)
        x = rearrange(x, 'b c 1 t -> b (c t)')
        return x, x_p

class da_encoder_refine(nn.Module):
    def __init__(self, params=None, Mlist=range(10), T0=50, device='cpu',
                 DEBUG=False, return_x_prime=False, val_id=0, mid=120):
        super().__init__()
        N = params['N']
        t = params['t']
        s = params['s']
        emb = params['emb_dim']
        norm = params['norm']
        non_linear = params['non_linear']
        drop = params['drop']  # [0.6, 0.6, 0.95]
        bias = [True, False]
        TS = T0
        FS = params['F']
        self.L = 40
        self.num_person = 35
        self.H = params['H']
        self.drop = drop
        self.device = device
        ica = 10
        M = len(Mlist)
        TN = [T0 // 2, T0 // 2]  # params['TN']
        FN = params['FN']
        self.compress = params['compress']  # 8
        self.head = params['head']  # 6

        self.id_embedding = nn.Embedding(self.num_person, emb, max_norm=1)

        self.id_operation = nn.Sequential(
            nn.Linear(emb, M * ica, bias=False),
            # nn.Dropout(drop[0]),
            eth.Rearrange('b (m n) -> b m n', m=M),
        )

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, N[0], (1, 1), bias=bias[0], padding='same'),
            # _get_norm(norm, N[0]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )

        self.chn_combination = nn.Sequential(
            nn.Conv2d(N[0], N[1], (ica, 1), bias=False),
            _get_norm(norm, N[1]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )
        self.chn_attention = nn.Sequential(
            Freq_Attention_Multi_Head(N[1], TS, FS, self.H, self.drop[1], norm, non_linear, self.device,
                                      compress=self.compress,
                                      bias=bias[1], norm_dim=1, Head=self.head),
        )

        self.tempo_fea, self.tempo_down, self.tempo_att = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        hh = [1, 1, 1]  # [1,2,3]# [3,2,1] # [1,1,1]
        for i in range(len(t)):
            tmp = self._create_tempo_layers(t[i], s[i], N[i + 1], N[i + 2], TN[i], FN[i], bias, norm, non_linear,
                                            har=hh[i])
            self.tempo_fea.append(tmp[0])
            self.tempo_down.append(tmp[1])
            self.tempo_att.append(tmp[2])

        self.flat = N[-1] * TN[-1]
        self.function_resnet = _get_non_linear_fcn(non_linear)
        print(f'final flatten number is {self.flat}...')

    def _create_tempo_layers(self, K, S, Cin, Cout, TN, Fs, bias, norm, non_linear, har):
        feature = nn.Sequential(
            nn.Conv2d(Cin, Cout, (1, K), (1, S), padding=(0, K // 2), bias=bias[0]),
            _get_norm(norm, Cout),
            _get_non_linear(non_linear),
            nn.Dropout(self.drop[0])
        )
        downsample = nn.Sequential(
            nn.Conv2d(Cin, Cout, 1, (1, S), bias=False),
            _get_norm(norm, Cout),
        ) if S != 1 else nn.Identity()
        attention = nn.Sequential(
            Freq_Attention_Multi_Head(Cout, TN, Fs, self.H, self.drop[1], norm, non_linear, self.device,
                                      compress=self.compress,
                                      bias=bias[1], Head=self.head),
        ) if Fs > 70 else nn.Identity()
        return feature, downsample, attention

    def forward(self, x, id=None):
        id = self.id_embedding(id)
        x_pre = self.preprocess(x)
        # addition = self._get_chn_self_corr(x)
        x = th.einsum('bcmt,bmn->bcnt', x_pre, self.id_operation(id))
        # addition = self._get_chn_self_corr(x) - addition
        x = self.chn_combination(x)
        x = self.function_resnet(x + self.chn_attention(x), inplace=True)
        x = FF.dropout(x, self.drop[1], training=self.training)
        for i in range(len(self.tempo_fea)):
            xtmp = self.tempo_fea[i](x)
            xtmp = self.tempo_att[i](xtmp)
            x = self.function_resnet(xtmp + self.tempo_down[i](x), inplace=True)
            if not i == len(self.tempo_fea) - 1:
                x = FF.dropout(x, self.drop[0], training=self.training)
        # x = rearrange(x, 'b c 1 t -> b (c t)')
        return x, x_pre

class da_classifier(nn.Module):
    def __init__(self, params=None, Mlist=range(10), T0=50, device='cpu', bias=False, same_emb=True,
                 DEBUG=False, return_x_prime=False, val_id=0, mid=120):
        super().__init__()
        self.L = 40
        drop = params['drop']  # [0.6, 0.6, 0.95]
        TN = params['TN']
        N = params['N']
        non_linear = params['non_linear']
        emb_dim = params['emb_cla_dim'] if same_emb else params['emb_dim']
        self.emb_dim = nn.Embedding(35, emb_dim, max_norm=1) if same_emb else None
        self.flat = N[-1] * T0 // 2  + emb_dim# TN[-1]
        self.fc = nn.Sequential(
            nn.Dropout(drop[-1]),
            _get_non_linear(non_linear),
            nn.Linear(self.flat, mid, bias=bias),
            _get_non_linear(non_linear),
            nn.Dropout(drop[1]),
            nn.Linear(mid, self.L, bias=bias),
        )

    def forward(self, z, id, idp):
        if self.emb_dim is not None:
            idp = self.emb_dim(id)
        x = self.fc(th.cat((z, idp), dim=-1))
        return x
    
    def forward2(self, z, idp):
        x = self.fc(th.cat((z, idp), dim=-1))
        return x