import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAverage(nn.Module):
    """
    Moving average block to extract the trend component of a time series.
    """
    def __init__(self, kernel_size: int, stride: int):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        # Calculate padding to maintain the same output length
        # For stride=1, output_length = (input_length + 2*padding - kernel_size) / 1 + 1
        # If output_length = input_length, then 2*padding - kernel_size + 1 = 0
        # So, padding = (kernel_size - 1) // 2
        self.padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_seq_len, features]
        # Permute to [batch_size, features, input_seq_len] for AvgPool1d
        x_permuted = x.permute(0, 2, 1)

        # Apply AvgPool1d
        moving_avg = self.avg(x_permuted)

        # If kernel_size is even, AvgPool1d with padding=(kernel_size-1)//2 might result in
        # output length input_seq_len - 1. We need to pad the end to match input_seq_len.
        # This ensures the output sequence length is exactly the same as the input.
        if self.kernel_size % 2 == 0:
            moving_avg = F.pad(moving_avg, (0, 1), mode='replicate') # Pad last dimension by 1 at the end

        # Permute back to [batch_size, input_seq_len, features]
        moving_avg = moving_avg.permute(0, 2, 1)
        return moving_avg

class DLinear(nn.Module):
    """
    DLinear model for time series forecasting.

    Args:
        input_seq_len (int): The length of the input time series sequence.
        pred_seq_len (int): The length of the prediction horizon.
        features (int): The number of features (variates) in the time series.
        kernel_size (int, optional): The kernel size for the moving average. Defaults to 25.
        individual (bool, optional): If True, uses individual linear layers for each feature.
                                     If False, uses shared linear layers across all features.
                                     Defaults to False.
    """
    def __init__(self, input_seq_len: int, pred_seq_len: int, features: int, kernel_size: int = 25, individual: bool = False):
        super(DLinear, self).__init__()
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.features = features
        self.individual = individual

        # Moving average layer for trend decomposition
        self.moving_average = MovingAverage(kernel_size, stride=1)

        if self.individual:
            # Individual linear layers for each feature
            self.linear_trend = nn.ModuleList()
            self.linear_season = nn.ModuleList()
            for i in range(self.features):
                self.linear_trend.append(nn.Linear(self.input_seq_len, self.pred_seq_len))
                self.linear_season.append(nn.Linear(self.input_seq_len, self.pred_seq_len))
        else:
            # Shared linear layers across all features
            self.linear_trend = nn.Linear(self.input_seq_len, self.pred_seq_len)
            self.linear_season = nn.Linear(self.input_seq_len, self.pred_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_seq_len, features]

        # Decomposition
        moving_mean = self.moving_average(x)
        trend = moving_mean
        season = x - moving_mean

        # Apply linear layers
        if self.individual:
            # For individual linear layers, process each feature separately
            # Input to linear layer for each feature will be [batch_size, input_seq_len]
            # Output will be [batch_size, pred_seq_len]
            # We stack these outputs to get [batch_size, pred_seq_len, features]
            trend_output = []
            season_output = []
            for i in range(self.features):
                trend_output.append(self.linear_trend[i](trend[:, :, i]))
                season_output.append(self.linear_season[i](season[:, :, i]))
            trend_output = torch.stack(trend_output, dim=-1)
            season_output = torch.stack(season_output, dim=-1)
        else:
            # For shared linear layers, reshape and apply
            # Input to linear layer: [batch_size, features, input_seq_len]
            # Output from linear layer: [batch_size, features, pred_seq_len]
            # Permute back to [batch_size, pred_seq_len, features]
            trend_output = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            season_output = self.linear_season(season.permute(0, 2, 1)).permute(0, 2, 1)

        # Sum the outputs
        output = trend_output + season_output
        return output
