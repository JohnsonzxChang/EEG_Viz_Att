import numpy as np
import torch
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
from conf.base import BaseConfig
from torch.nn.modules.module import _addindent


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            # DepthwiseConv2D =======================
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate))
        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate))
        return nn.Sequential(block1, block2)


    def ClassifierBlock(self, inputSize, n_classes, use_cla=True):
        if use_cla:
            return nn.Sequential(
                nn.Linear(inputSize, n_classes, bias=False),
                nn.Softmax(dim=1))
        else:
            return nn.Sequential(
                nn.Linear(inputSize, n_classes, bias=False))

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, conf:BaseConfig):
        super(EEGNet, self).__init__()
        n_classes = conf.num_classes
        channels = len(conf.chn_sel)
        samples = conf.t_len
        dropoutRate = conf.dropout
        kernelLength=64
        kernelLength2=16
        F1=8
        D=2
        F2=16
        use_cla=False
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate

        self.blocks = self.InitialBlocks(dropoutRate)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, samples)
        if use_cla is None:
            self.classifierBlock = nn.Identity()
        else:
            assert isinstance(use_cla, bool), "use_cla must be a boolean value"
            self.classifierBlock = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], n_classes) if use_cla else self.ClassifierBlock(conf.feat_dim, n_classes, False)
        self.fea_block = nn.Linear(self.F2 * self.blockOutputSize[1], conf.feat_dim)

    def forward_all(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        x = x.unsqueeze(1)
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        # print(x.shape)
        x = self.fea_block(x)
        logits = self.classifierBlock(x)
        return x, logits

    def forward(self, x, padding_mask=None, enc_self_mask=None, dec_self_mask=None):
        x, _ = self.forward_all(x, padding_mask=padding_mask, enc_self_mask=enc_self_mask, dec_self_mask=dec_self_mask)
        return x

def categorical_cross_entropy(y_pred, y_true):
    # y_pred = y_pred.cuda()
    # y_true = y_true.cuda()
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr