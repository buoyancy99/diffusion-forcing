from typing import Type

from torch import nn as nn


class ResBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation: Type[nn.Module] = nn.ReLU):
        super(ResBlock2d, self).__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResBlock1d(nn.Module):
    def __init__(self, in_planes, planes, activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, planes),
            self.activation,
            nn.Linear(planes, planes),
        )

        self.shortcut = nn.Identity() if in_planes == planes else nn.Linear(in_planes, planes)

    def forward(self, x):
        out = self.activation(self.shortcut(x) + self.mlp(x))
        return out


class ResBlockWrapper(nn.Module):
    def __init__(self, model: nn.Module, in_planes=None, out_planes=None):
        super().__init__()
        self.model = model
        self.shortcut = (
            nn.Identity()
            if in_planes == out_planes
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return self.model(x) + self.shortcut(x)
