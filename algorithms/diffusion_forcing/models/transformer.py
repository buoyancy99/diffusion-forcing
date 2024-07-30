import torch
import torch.nn as nn
import math
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Transformer(nn.Module):
    def __init__(
        self,
        x_dim,
        external_cond_dim=0,
        size=128,
        num_layers=4,
        nhead=4,
        dim_feedforward=512,
        dropout=0.0,
    ):
        super(Transformer, self).__init__()
        self.external_cond_dim = external_cond_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        k_embed_dim = size // 2
        self.t_embed = SinusoidalPosEmb(dim=size)
        self.k_embed = SinusoidalPosEmb(dim=k_embed_dim)
        self.init_mlp = nn.Sequential(
            nn.Linear(x_dim + k_embed_dim + external_cond_dim, size),
            nn.ReLU(),
            nn.Linear(size, size),
        )
        self.out = nn.Linear(size, x_dim)

    def forward(self, x, k, external_cond=None, is_causal=False):
        # x.shape (T, B, C)
        # k.shape (T, B)
        # optional external_cond.shape (T, B, C)

        seq_len, batch_size, _ = x.shape
        k_embed = rearrange(self.k_embed(k.flatten()), "(t b) d -> t b d", t=seq_len)
        x = torch.cat((x, k_embed), dim=-1)
        if external_cond is not None:
            x = torch.cat((x, external_cond), dim=-1)
        x = self.init_mlp(x)
        x = x + self.t_embed(torch.arange(seq_len, device=x.device)[:, None])

        mask = nn.Transformer.generate_square_subsequent_mask(len(x), x.device) if is_causal else None
        x = self.transformer(x, mask=mask, is_causal=is_causal)
        x = self.out(x)

        return x


if __name__ == "__main__":
    model = Transformer(x_dim=10)
    x = torch.randn(100, 32, 10)
    k = torch.randint(0, 10, (100, 32))
    out = model(x, k)
    print(out.shape)
