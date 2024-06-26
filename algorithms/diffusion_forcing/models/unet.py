from functools import partial
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from einops import rearrange, repeat


from .utils import default, exists, cast_tuple, divisible_by
from .gru import Conv2dGRUCell
from .resnet import ResBlock1d
from .sin_emb import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb
from .attend import Attend


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        """
        :param dim: input channel
        :param dim_out:  output channel
        :param emb_dim: extra embedding to fuse, such as time or control
        :param groups: group for conv2d
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2)) if exists(emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(emb):
            emb = self.mlp(emb)
            emb = rearrange(emb, "b c -> b c 1 1")
            scale_shift = emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out.contiguous())


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b).contiguous(), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,  # base number of channels that controls network size
        init_dim=None,  # this is not in_channels but the one after in_channels
        out_dim=None,
        z_cond_dim=None,
        external_cond_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_cond=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,  # defaults to full attention only for inner most layer
        flash_attn=True,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        self.z_cond_dim = z_cond_dim
        self.external_cond_dim = external_cond_dim
        input_channels = channels * (2 if self_condition else 1)
        input_channels += z_cond_dim if z_cond_dim else 0

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_emb_dim = dim * 4
        external_cond_emb_dim = dim * 2

        self.emb_dim = time_emb_dim + external_cond_emb_dim if external_cond_dim else time_emb_dim
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_cond

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_cond)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.external_cond_mlp = (
            nn.Sequential(
                nn.Linear(external_cond_dim, external_cond_emb_dim),
                nn.GELU(),
                nn.Linear(external_cond_emb_dim, external_cond_emb_dim),
            )
            if self.external_cond_dim
            else None
        )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, emb_dim=self.emb_dim),
                        block_klass(dim_in, dim_in, emb_dim=self.emb_dim),
                        attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, emb_dim=self.emb_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, emb_dim=self.emb_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, emb_dim=self.emb_dim),
                        block_klass(dim_out + dim_in, dim_out, emb_dim=self.emb_dim),
                        attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, emb_dim=self.emb_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, z_cond, external_cond=None, x_self_cond=None):
        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        if self.z_cond_dim:
            x = torch.cat((z_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        emb = self.time_mlp(time)
        if self.external_cond_dim:
            if external_cond is None:
                external_cond_emb = torch.zeros((emb.shape[0], self.external_cond_dim)).to(emb)
            else:
                external_cond_emb = self.external_cond_mlp(external_cond)
            emb = torch.cat([emb, external_cond_emb], -1)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            h.append(x)

            x = block2(x, emb)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, emb)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, emb)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, emb)
        return self.final_conv(x)


class TransitionUnet(Unet):
    def __init__(
        self,
        z_channel,
        x_channel,
        external_cond_dim=None,
        network_size=32,
        num_gru_layers=1,
        self_condition=False,
    ):
        super().__init__(
            network_size,
            channels=x_channel,
            out_dim=z_channel,
            external_cond_dim=external_cond_dim,
            z_cond_dim=z_channel,
            self_condition=self_condition,
        )
        self.z_channel = z_channel
        self.x_channel = x_channel
        self.num_gru_layers = num_gru_layers
        self.self_condition = self_condition
        self.gru = Conv2dGRUCell(z_channel, z_channel) if num_gru_layers else None

        if num_gru_layers > 1:
            raise NotImplementedError("num_gru_layers > 1 is not implemented yet for TransitionUnet.")

    def forward(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        z_next = super().forward(x, t, z_cond, external_cond, x_self_cond)
        if self.num_gru_layers:
            z_next = self.gru(z_next, z_cond)

        return z_next


class TransitionMlp(nn.Module):
    def __init__(
        self,
        z_dim,
        x_dim,
        external_cond_dim=None,
        network_size=16,
        num_gru_layers=1,
        num_mlp_layers=4,
        self_condition=False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.external_cond_dim = external_cond_dim
        self.network_size = network_size * 32
        self.num_gru_layers = num_gru_layers
        self.num_mlp_layers = num_mlp_layers
        self.z_cond_dim = z_dim
        self.self_condition = self_condition

        fourier_dim = network_size
        time_emb_dim = network_size
        external_cond_emb_dim = network_size
        input_dim = x_dim * (2 if self_condition else 1) + z_dim + time_emb_dim
        input_dim += external_cond_emb_dim if external_cond_dim else 0

        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.external_cond_mlp = (
            nn.Sequential(
                nn.Linear(external_cond_dim, external_cond_emb_dim),
                nn.GELU(),
                nn.Linear(external_cond_emb_dim, external_cond_emb_dim),
            )
            if self.external_cond_dim
            else None
        )

        if self.num_mlp_layers <= 0:
            self.mlp_before_gru = None
            self.mlp_after_gru = None
        elif self.num_mlp_layers == 1:
            self.mlp_before_gru = ResBlock1d(input_dim, z_dim, activation=nn.GELU)
            self.mlp_after_gru = nn.Linear(z_dim, z_dim)
        else:
            self.mlp_before_gru = []
            self.mlp_before_gru.append(ResBlock1d(input_dim, self.network_size, activation=nn.GELU))
            for _ in range(1, self.num_mlp_layers - 1):
                self.mlp_before_gru.append(ResBlock1d(self.network_size, self.network_size, activation=nn.GELU))
            self.mlp_before_gru.append(nn.Linear(self.network_size, z_dim))
            self.mlp_before_gru = nn.Sequential(*self.mlp_before_gru)
            self.mlp_after_gru = []
            self.mlp_after_gru.append(ResBlock1d(z_dim, self.network_size, activation=nn.GELU))
            for _ in range(1, self.num_mlp_layers - 1):
                self.mlp_after_gru.append(ResBlock1d(self.network_size, self.network_size, activation=nn.GELU))
            self.mlp_after_gru.append(nn.Linear(self.network_size, z_dim))
            self.mlp_after_gru = nn.Sequential(*self.mlp_after_gru)

        if self.num_gru_layers:
            assert z_dim % self.num_gru_layers == 0
            self.gru_stack = nn.ModuleList()
            stack_input_dim = z_dim if self.num_mlp_layers else input_dim
            self.gru_stack.append(nn.GRUCell(stack_input_dim, z_dim // self.num_gru_layers))
            for _ in range(1, self.num_gru_layers):
                self.gru_stack.append(nn.GRUCell(z_dim // self.num_gru_layers, z_dim // self.num_gru_layers))
        else:
            self.gru_stack = None

    def forward(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        if not self.num_mlp_layers and not self.num_gru_layers:
            # only for sweeps
            return z_cond

        if self.z_cond_dim:
            x = torch.cat((z_cond, x), dim=-1)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=-1)
        emb = self.time_mlp(t)
        if self.external_cond_dim:
            if external_cond is None:
                external_cond_emb = torch.zeros((emb.shape[0], self.external_cond_dim)).to(emb)
            else:
                external_cond_emb = self.external_cond_mlp(external_cond)
            emb = torch.cat([emb, external_cond_emb], -1)

        x = torch.cat([x, emb], -1)

        if self.num_mlp_layers:
            x = self.mlp_before_gru(x)

        if self.num_gru_layers:
            z_cond = z_cond.chunk(self.num_gru_layers, dim=-1)
            outputs = []
            for i in range(self.num_gru_layers):
                x = self.gru_stack[i](x, z_cond[i])
                outputs.append(x)
            x = torch.cat(outputs, dim=-1)

        if self.num_mlp_layers:
            x = self.mlp_after_gru(x)

        return x
