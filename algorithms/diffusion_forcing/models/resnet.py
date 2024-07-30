from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        emb_dim: Optional[int] = None,
        groups: int = 8,
        eps=1e-6,
    ):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps),
            nn.SiLU(),
            nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim_out, eps=eps),
            nn.SiLU(),
            nn.Conv3d(dim_out, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        self.emb_layers = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim_out * 2),
            )
            if emb_dim is not None
            else None
        )

        self.skip_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        h = self.in_layers(x)

        if self.emb_layers is not None:
            assert (
                emb is not None
            ), "Noise level embedding is required for this ResnetBlock"
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            emb = self.emb_layers(emb)
            emb = rearrange(emb, "b f c -> b c f 1 1")
            scale, shift = emb.chunk(2, dim=1)

            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h)

        return self.skip_conv(x) + h


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(
            dim, dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        return self.conv(x)
