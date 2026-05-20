import torch as th
import torch.nn as nn
import torch.nn.functional as FF
from einops import rearrange
import einops.layers.torch as eth
import math
import numpy as np
from torch.autograd import Function
import matplotlib.pyplot as plt

new_freq_list = th.tensor(
    [22.22, 23.81, 25.64, 27.78,
    30.30, 33.33, 37.04, 41.67])
old_freq_list = th.linspace(8, 15.8, 40)

class corr_layer(nn.Module):
    def __init__(self, params=None, M=10, T0=50, device='cpu',
                    DEBUG=False, emb_dim=200, id_norm=True):
        super().__init__()
        self.DEBUG = DEBUG
        self.id_norm = id_norm
        N = params['N']  # [40, 80, 120, 120, 120]
        # assert N[1] == emb_dim
        FS = params['F']  # 250
        self.H = params['H']  # 3
        non_linear = params['non_linear']  # 'relu'
        norm = params['norm']
        drop = params['drop']  # [0.25, 0.25, 0.95] # [0.6, 0.6, 0.95]
        t = params['t']  # [3, 7, 7]
        s = params['s']  # [2, 1, 1]
        TN = params['TN']  # [50, 50, 50]
        FN = params['FN']  # [125, 125, 125]
        self.compress = params['compress'] # 8
        self.head = params['head'] # 6
        bias = [True, False]
        TS = T0
        self.L = 40
        self.drop = drop
        self.device = device
        ica = 10

        self.id_operation = nn.Sequential(
            nn.Linear(emb_dim, M * ica, bias=False),
            nn.Dropout(drop[0]),
            eth.Rearrange('b (m n) -> b m n', m=M),
        )

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, N[0], (1, 1), bias=False, padding='same'),
            _get_norm(norm, N[0]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )

        self.chn_combination = nn.Sequential(
            nn.Conv2d(N[0], N[1], (ica,1), bias=bias[0]),
            _get_norm(norm, N[1]),
            _get_non_linear(non_linear),
            nn.Dropout(drop[0]),
        )
        self.chn_attention = nn.Sequential(
            Freq_Attention_Multi_Head(N[1], TS, FS, self.H, self.drop[1], norm, non_linear, self.device,
            compress=self.compress, bias=bias[1], norm_dim=1, Head=self.head),
        )

        self.tempo_fea, self.tempo_down, self.tempo_att = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for i in range(len(t)):
            tmp = self._create_tempo_layers(t[i], s[i], N[i+1], N[i+2], TN[i], FN[i], bias, norm, non_linear)
            self.tempo_fea.append(tmp[0])
            self.tempo_down.append(tmp[1])
            self.tempo_att.append(tmp[2])

        self.fc = nn.Linear(N[-1]*TN[-1], self.L)
        self.flat = N[-1]*TN[-1]

    def _create_tempo_layers(self, K, S, Cin, Cout, TN, Fs, bias, norm, non_linear):
        feature = nn.Sequential(
            nn.Conv2d(Cin, Cout, (1,K), (1,S), padding=(0,K//2), bias=bias[0]),
            _get_norm(norm, Cout),
            _get_non_linear(non_linear),
            nn.Dropout(self.drop[0])
        )
        downsample = nn.Sequential(
            nn.Conv2d(Cin, Cout, 1, (1,S), bias=False),
            _get_norm(norm, Cout),
        ) if S != 1 else nn.Identity()
        attention = nn.Sequential(
            Freq_Attention_Multi_Head(Cout, TN, Fs, self.H, self.drop[1], norm, non_linear,
            self.device, compress=self.compress, bias=bias[1], Head=self.head),
        ) if Fs > 70 else nn.Identity()
        return feature, downsample, attention
    def forward_pre(self, x):
        return self.preprocess(x.squeeze())

    def forward(self, x, id=None):
        # if self.DEBUG:
        #     assert  id is not None
        #     id = id.squeeze()
        # if self.DEBUG:
        #     print(x.shape)
        # if self.id_norm:
        #     id = id / (th.sqrt(th.sum(id ** 2, dim=-1, keepdim=True)) + 1e-6)

        x = th.einsum('bcmt,bmn->bcnt', x, self.id_operation(id))
        x = self.chn_combination(x)
        x = FF.relu(x + self.chn_attention(x), inplace=True)
        x = FF.dropout(x, self.drop[1], training=self.training)
        for i in range(len(self.tempo_fea)):
            xtmp = self.tempo_fea[i](x)
            # if self.DEBUG:
            #     print(xtmp.shape)
            xtmp = self.tempo_att[i](xtmp)
            x = FF.relu(xtmp + self.tempo_down[i](x),inplace=True)
            if i == len(self.tempo_fea)-1:
                x = FF.dropout(x, self.drop[2], training=self.training)
            else:
                x = FF.dropout(x, self.drop[0], training=self.training)
        x = rearrange(x, 'b c 1 t -> b (c t)')
        x = self.fc(x)
        return x

class Freq_Attention(nn.Module):
    def __init__(self, C, TN, Fs, H, drop, conv_norm='batch', non_linear='gelu', device='cpu', compress=4, norm_dim=2, har=1,
                 bias=False, eps=1e-5, FreqList=th.linspace(8, 15.8, 40)):
        super().__init__()
        self.eps = eps
        self.harmonic = har
        self.H = H
        self.TN = TN
        self.Fs = Fs
        self.C = C
        self.Fea = FreqList.shape[0] * H
        self.upsample = int(FreqList.max() * H * 25 / Fs)
        self.up_mat = th.zeros((TN, self.upsample * TN)).to(device).requires_grad_(False)
        for i in range(TN // 2):
            for j in range(self.upsample * 2):
                self.up_mat[i * 2, i * 2 * self.upsample + j] = (j / self.upsample - 1) * (j / self.upsample - 2) / ((-1) * (-2))
                self.up_mat[min(i * 2 + 1, TN - 1), i * 2 * self.upsample + j] = (j / self.upsample) * (j / self.upsample - 2) / ((1) * (-1))
                self.up_mat[min(i * 2 + 2, TN - 1), i * 2 * self.upsample + j] = (j / self.upsample) * (j / self.upsample - 1) / ((1) * (2))

        print(f'sample freq is {Fs} with time length {TN}, upsample is {self.upsample}')
        position = th.arange(0, self.TN * self.upsample, requires_grad=False).unsqueeze(1) / Fs
        pe_sin = []
        pe_cos = []
        for i in range(self.H):
            pe_sin.append(th.sin(math.pi * 2 * position * FreqList * self.harmonic))  # T, F
            pe_cos.append(th.cos(math.pi * 2 * position * FreqList * self.harmonic))  # T, F
        self.pe_sin = th.cat(pe_sin, dim=-1).to(device).T
        self.pe_cos = th.cat(pe_cos, dim=-1).to(device).T

        self.embed = nn.Sequential(
            nn.Conv1d(C, self.Fea, kernel_size=1, bias=bias),
        )
        self.decode = nn.Sequential(
            nn.Linear(self.Fea, C//compress, bias=False),
            # _get_norm(conv_norm, C//compress, dim=1),
            _get_non_linear(non_linear),
            nn.Linear(C//compress, C, bias=False),
            nn.Dropout(drop)
        )


    def corr(self, x, eps=1e-5):
        # x: {B, C, T}
        # return: {B, C}
        assert x.shape[1:] == self.pe_sin.shape == (self.Fea, self.TN*self.upsample), f'input shape is {x.shape},,{self.Fea},{self.TN},,{self.pe_sin.shape}'
        xy = th.einsum('bct,ct->bc', x, self.pe_sin) ** 2 + th.einsum('bct,ct->bc', x, self.pe_cos) ** 2
        return xy

    def upsample_x(self, x):
        return th.matmul(x, self.up_mat)#[:,:, :-2 * self.upsample]

    def forward(self, x):
        # x: {B, C, 1, T}, y: {B, C}
        x = x.squeeze(2)
        # print(x.shape)
        y = self.embed(x)
        # print(y.shape)
        y = self.corr(self.upsample_x(y))
        y = FF.sigmoid(self.decode(y).unsqueeze(2))
        return (x * y).unsqueeze(2)

class Freq_Attention_Multi_Head(nn.Module):
    def __init__(self, C, TN, Fs, H, drop, conv_norm='batch', non_linear='gelu', device='cpu', compress=4, norm_dim=2, Head=5, 
                 use_upsample=True,
                 bias=False, eps=1e-5, FreqList=new_freq_list):
        super().__init__()
        self.eps = eps
        self.head = Head
        self.H = H
        self.TN = TN
        self.Fs = Fs
        self.C = C
        self.F = FreqList.shape[0]
        self.Fea = FreqList.shape[0] * H
        self.upsample = int(FreqList.max() * H * 25 / Fs)
        self.up_mat = th.zeros((TN, self.upsample * TN)).to(device).requires_grad_(False)
        for i in range(TN // 2):
            for j in range(self.upsample * 2):
                self.up_mat[i * 2, i * 2 * self.upsample + j] = (j / self.upsample - 1) * (j / self.upsample - 2) / ((-1) * (-2))
                self.up_mat[min(i * 2 + 1, TN - 1), i * 2 * self.upsample + j] = (j / self.upsample) * (j / self.upsample - 2) / ((1) * (-1))
                self.up_mat[min(i * 2 + 2, TN - 1), i * 2 * self.upsample + j] = (j / self.upsample) * (j / self.upsample - 1) / ((1) * (2))

        print(f'sample freq is {Fs} with time length {TN}, upsample is {self.upsample}')
        self.use_upsample = use_upsample
        if use_upsample:
            position = th.arange(0, self.TN * self.upsample, requires_grad=False).unsqueeze(1) / (Fs*self.upsample) # Fs or (Fs*self.upsample) ?????? TO-DO
        else:
            position = th.arange(0, self.TN, requires_grad=False).unsqueeze(1) / Fs
        pe_sin = []
        pe_cos = []
        for i in range(self.H):
            pe_sin.append(th.sin(math.pi * 2 * position * FreqList * (i+1)))  # T, F*H
            pe_cos.append(th.cos(math.pi * 2 * position * FreqList * (i+1)))  # T, F*H
        self.pe_sin = th.cat(pe_sin, dim=-1).to(device).T.requires_grad_(False)
        self.pe_cos = th.cat(pe_cos, dim=-1).to(device).T.requires_grad_(False)

        self.embed = nn.Sequential(
            nn.Conv1d(C, self.Fea * Head, kernel_size=1, bias=bias),
        )
        self.harmonic = nn.Parameter(th.rand((H)))
        self.harmonic_non_linear = nn.Sequential(
            _get_non_linear(non_linear),
        )
        self.decode = nn.Sequential(
            nn.Linear(FreqList.shape[0] * Head, C//compress, bias=False),
            # _get_norm(conv_norm, C//compress, dim=1),
            _get_non_linear(non_linear),
            nn.Linear(C//compress, C, bias=False),
        )
        self.drop = drop


    def corr(self, x, eps=1e-5):
        # x: {B, C, T}
        # return: {B, C}
        # assert x.shape[1:] == self.pe_sin.shape == (self.Fea, self.TN*self.upsample), f'input shape is {x.shape},,{self.Fea},{self.TN},,{self.pe_sin.shape}'
        xy = th.einsum('bct,ct->bc', x, self.pe_sin) ** 2 + th.einsum('bct,ct->bc', x, self.pe_cos) ** 2
        assert th.sum(th.isnan(xy)) == 0, print(xy.shape)
        return xy


    def upsample_x(self, x):
        return th.matmul(x, self.up_mat)#[:,:, :-2 * self.upsample]

    def forward(self, x):
        # x: {B, C, 1, T}, y: {B, C}
        x = x.squeeze(2)
        # print(x.shape)
        y = self.embed(x)
        # print(y.shape) 
        # TO-DO: add harmonic and check again
        if self.use_upsample:
            y = th.cat([self.corr(
                self.upsample_x(y[:,i * self.Fea :(i+1) * self.Fea ,:])
                ) for i in range(self.head)], dim=-1).reshape((-1, self.head, self.H, self.F)) # ((40) 3) 5
        else:
            y = th.cat([self.corr(
                y[:,i * self.Fea :(i+1) * self.Fea ,:]
                ) for i in range(self.head)], dim=-1).reshape((-1, self.head, self.H, self.F))
        y = th.einsum('bahf,h->baf', y, self.harmonic)
        y = self.decode(rearrange(y, 'b a f -> b (a f)')).unsqueeze(2)
        y = FF.sigmoid(y)
        y = FF.dropout(y, self.drop, training=self.training)
        return (x * y).unsqueeze(2)

class Freq_Multi_Head(nn.Module):
    def __init__(self, C, TN, Fs, H, drop, conv_norm='batch', non_linear='gelu', device='cpu', compress=4, norm_dim=2, Head=5,
                 bias=False, eps=1e-5, FreqList=th.linspace(8, 15.8, 40)):
        super().__init__()
        self.eps = eps
        self.head = Head
        self.H = H
        self.TN = TN
        self.Fs = Fs
        self.C = C
        self.F = FreqList.shape[0]
        self.Fea = FreqList.shape[0] * H
        self.upsample = int(FreqList.max() * H * 25 / Fs)
        self.up_mat = th.zeros((TN, self.upsample * TN)).to(device).requires_grad_(False)
        for i in range(TN // 2):
            for j in range(self.upsample * 2):
                self.up_mat[i * 2, i * 2 * self.upsample + j] = (j / self.upsample - 1) * (j / self.upsample - 2) / ((-1) * (-2))
                self.up_mat[min(i * 2 + 1, TN - 1), i * 2 * self.upsample + j] = (j / self.upsample) * (j / self.upsample - 2) / ((1) * (-1))
                self.up_mat[min(i * 2 + 2, TN - 1), i * 2 * self.upsample + j] = (j / self.upsample) * (j / self.upsample - 1) / ((1) * (2))

        print(f'sample freq is {Fs} with time length {TN}, upsample is {self.upsample}')
        position = th.arange(0, self.TN * self.upsample, requires_grad=False).unsqueeze(1) / Fs
        pe_sin = []
        pe_cos = []
        for i in range(self.H):
            pe_sin.append(th.sin(math.pi * 2 * position * FreqList * (i+1)))  # T, F*H
            pe_cos.append(th.cos(math.pi * 2 * position * FreqList * (i+1)))  # T, F*H
        self.pe_sin = th.cat(pe_sin, dim=-1).to(device).T.requires_grad_(False)
        self.pe_cos = th.cat(pe_cos, dim=-1).to(device).T.requires_grad_(False)

        self.embed = nn.Sequential(
            nn.Conv1d(C, self.Fea * Head, kernel_size=1, bias=bias),
        )
        self.harmonic = nn.Parameter(th.rand((H)))
        self.harmonic_non_linear = nn.Sequential(
            _get_non_linear(non_linear),
        )
        self.decode = nn.Sequential(
            nn.Linear(FreqList.shape[0] * Head, C//compress, bias=False),
            # _get_norm(conv_norm, C//compress, dim=1),
            _get_non_linear(non_linear),
            nn.Linear(C//compress, C, bias=False),
        )
        self.drop = drop

    def corr(self, x, eps=1e-5):
        # x: {B, C, T}
        # return: {B, C}
        assert x.shape[1:] == self.pe_sin.shape == (self.Fea, self.TN*self.upsample), f'input shape is {x.shape},,{self.Fea},{self.TN},,{self.pe_sin.shape}'
        xy = th.einsum('bct,ct->bc', x, self.pe_sin) ** 2 + th.einsum('bct,ct->bc', x, self.pe_cos) ** 2
        assert th.sum(th.isnan(xy)) == 0, print(xy.shape)
        return xy

    def upsample_x(self, x):
        return th.matmul(x, self.up_mat)#[:,:, :-2 * self.upsample]

    def forward(self, x):
        # x: {B, C, 1, T}, y: {B, C}
        x = x.squeeze(2)
        # print(x.shape)
        y = self.embed(x)
        # print(y.shape)
        y = th.cat([self.corr(self.upsample_x(y[:,i * self.Fea :(i+1) * self.Fea ,:])) for i in range(self.head)], dim=-1).reshape((-1, self.head, self.H, self.F)) # ((40) 3) 5
        y = th.einsum('bahf,h->baf', y, self.harmonic)
        y = self.decode(rearrange(y, 'b a f -> b (a f)'))
        y = FF.sigmoid(y)
        y = FF.dropout(y, self.drop, training=self.training)
        return y

def _get_norm(name, planes, dim=2):
    if name == 'batch':
        if dim == 2:
            return nn.BatchNorm2d(planes)
        else:
            return nn.BatchNorm1d(planes)
    elif name == 'instance':
        if dim == 2:
            return nn.InstanceNorm2d(planes)
        else:
            return nn.InstanceNorm1d(planes)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError('Invalid normalization method: {}'.format(name))

def _get_non_linear(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leakrelu':
        return nn.LeakyReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError('Invalid non-linear method: {}'.format(name))

def _get_non_linear_fcn(name):
    if name == 'relu':
        return FF.relu
    elif name == 'leakrelu':
        return FF.leaky_relu
    elif name == 'tanh':
        return FF.tanh
    elif name == 'sigmoid':
        return FF.sigmoid
    elif name == 'gelu':
        return FF.gelu
    elif name == 'none':
        return lambda x: x
    else:
        raise ValueError('Invalid non-linear method: {}'.format(name))

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    mnist_classes = [f'{i}' for i in range(35)]
    figure = plt.figure(figsize=(10,10))
    for i in range(35):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    return figure

def extract_embeddings(dataloader, model, device):
    with th.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.to(device)
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

