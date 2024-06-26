from typing import Tuple
import torch.nn as nn
import torch
import numpy as np
from .resnet import ResBlockWrapper


class ConvRNNCell(nn.Module):
    def __init__(self, channel_multiplier, in_channels, hidden_channels, kernel_size, bias=True):
        super(ConvRNNCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bias = bias

        if isinstance(kernel_size, Tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.x2h = nn.Conv2d(in_channels=in_channels,
                             out_channels=hidden_channels * channel_multiplier,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)

        self.h2h = nn.Conv2d(in_channels=in_channels,
                             out_channels=hidden_channels * channel_multiplier,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        self.Wc = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_channels)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx):
        # Inputs:
        #       input: of shape (batch_size, in_channels, height_size, width_size)
        #       hx: of shape (batch_size, hidden_channels, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_channels, height_size, width_size)]
        raise NotImplementedError


class Conv2dGRUCell(ConvRNNCell):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super(Conv2dGRUCell, self).__init__(3, in_channels, hidden_channels, kernel_size, bias)

    def forward(self, input, hx):
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy


class Resnet2dGRUCell(Conv2dGRUCell):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super(Resnet2dGRUCell, self).__init__(in_channels, hidden_channels, kernel_size=kernel_size, bias=bias)

    def _build_model(self):
        self.x2h = ResBlockWrapper(
            nn.Sequential(
                nn.Conv2d(self.in_channels, 8, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.hidden_channels * self.channel_multiplier,
                                   3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
            ), self.in_channels, self.hidden_channels * self.channel_multiplier)

        self.h2h = ResBlockWrapper(
            nn.Sequential(
                nn.Conv2d(self.hidden_channels, 8, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.hidden_channels * self.channel_multiplier,
                                   3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
            ), self.hidden_channels, self.hidden_channels * self.channel_multiplier)

        self.Wc = None
