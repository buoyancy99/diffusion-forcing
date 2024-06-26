from typing import Type, Optional

import torch
from torch import nn as nn


class SimpleMlp(nn.Module):
    """
    A class for very simple multi layer perceptron
    """
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, n_layers=2,
                 activation: Type[nn.Module] = nn.ReLU, output_activation: Optional[Type[nn.Module]] = None):
        super(SimpleMlp, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()] * (n_layers - 2))
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_activation:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
