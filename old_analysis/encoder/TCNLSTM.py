import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from collections import deque
import os
from scipy import signal

from conf.base import BaseConfig

# 导入原始代码中的模型定义类
class ChannelAttention(torch.nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.fc = torch.nn.Linear(channels, channels)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # x: b t m
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # print(avg_pool.shape)
        channel_weights = self.sigmoid(self.fc(avg_pool))
        return x * channel_weights

class TCNBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=15, dilation=1):
        super(TCNBlock, self).__init__()
        self.padding = (kernel_size-1) * dilation
        self.conv = torch.nn.Conv1d(
            input_channels, 
            output_channels, 
            kernel_size=kernel_size, 
            padding=self.padding,
            dilation=dilation
        )
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(output_channels)
        self.residual = torch.nn.Conv1d(input_channels, output_channels, kernel_size=1) if input_channels != output_channels else torch.nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x_conv = self.conv(x)
        
        if residual.size(2) != x_conv.size(2):
            if residual.size(2) > x_conv.size(2):
                residual = residual[:, :, :x_conv.size(2)]
            else:
                x_conv = x_conv[:, :, :residual.size(2)]
        
        x_conv = self.batch_norm(x_conv)
        x_conv = self.relu(x_conv)
        
        return x_conv + residual

class lstm_encoder(torch.nn.Module):
    def __init__(self, conf:BaseConfig):
        super(lstm_encoder, self).__init__()
        
        self.input_channels = len(conf.chn_sel)
        self.output_size = conf.num_classes
        
        self.channel_attention = ChannelAttention(self.input_channels)
        
        self.tcn1 = TCNBlock(self.input_channels, 64, dilation=1)
        self.tcn2 = TCNBlock(64, 64, dilation=2)
        self.tcn3 = TCNBlock(64, 64, dilation=4)
        
        self.lstm1 = torch.nn.LSTM(self.input_channels + 64, 50, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(50, 50, batch_first=True, dropout=0.3)
        self.lstm3 = torch.nn.LSTM(50, 50, batch_first=True, dropout=0.3)
        self.lstm4 = torch.nn.LSTM(50, 50, batch_first=True, dropout=0.3)
        self.lstm5 = torch.nn.LSTM(50, 50, batch_first=True, dropout=0.3)
        self.lstm6 = torch.nn.LSTM(50, 50, batch_first=True, dropout=0.3)
        
        self.fc = torch.nn.Linear(50, self.output_size)
    
    def forward(self, x, t=None):
        batch_size = x.size(0)
        # print(x.shape) # b m t
        x = self.channel_attention(x.transpose(1, 2))
        
        x_tcn = x.transpose(1, 2) # b m t
        
        tcn_out = self.tcn1(x_tcn)
        tcn_out = self.tcn2(tcn_out)
        tcn_out = self.tcn3(tcn_out)
        
        tcn_out = tcn_out.transpose(1, 2)# b t m
        
        if tcn_out.size(1) != x.size(1):
            min_time_steps = min(tcn_out.size(1), x.size(1))
            tcn_out = tcn_out[:, :min_time_steps, :]
            x = x[:, :min_time_steps, :]
        
        x_combined = torch.cat([tcn_out, x], dim=2)
        
        x_lstm, _ = self.lstm1(x_combined)
        x_lstm, _ = self.lstm2(x_lstm)
        x_lstm, _ = self.lstm3(x_lstm)
        x_lstm, _ = self.lstm4(x_lstm)
        x_lstm, _ = self.lstm5(x_lstm)
        x_lstm, (h_n, _) = self.lstm6(x_lstm)
        
        last_output = h_n[-1, :, :]
        
        output = self.fc(last_output)
        
        return output

