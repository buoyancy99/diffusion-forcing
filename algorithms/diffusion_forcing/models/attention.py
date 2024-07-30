from typing import Optional
from functools import wraps
from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from .embeddings import TimestepEmbedding, Timesteps
from .utils import get_einops_wrapped_module


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        bias: bool = False,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(query_dim, self.inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(self.inner_dim, query_dim)

        # determine efficient attention configs for cuda and cpu

        self.cpu_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.MATH,
            SDPBackend.EFFICIENT_ATTENTION,
        ]

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major >= 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_backends = [SDPBackend.FLASH_ATTENTION]
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_backends = [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def forward(self, hidden_states: torch.Tensor, is_causal: bool = False):
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # Flash / Memory efficient attention leads to cuda errors for large batch sizes
        backends = (
            ([SDPBackend.MATH] if q.shape[0] >= 65536 else self.cuda_backends)
            if q.is_cuda
            else self.cpu_backends
        )
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        with sdpa_kernel(backends=backends):
            # pylint: disable=E1102
            hidden_states = F.scaled_dot_product_attention(
                query=q, key=k, value=v, is_causal=is_causal
            )

        hidden_states = rearrange(hidden_states, "b h n d -> b n (h d)")
        hidden_states = hidden_states.to(q.dtype)

        # linear proj
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        use_linear: bool = False,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        attn_klass = LinearAttention if use_linear else Attention
        self.attn = attn_klass(
            query_dim=dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb
        )

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        return x + self.attn(self.norm(x), is_causal=is_causal)


class LinearAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(query_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, query_dim)

        if rotary_emb is not None:
            raise NotImplementedError(
                "Rotary embeddings not implemented for linear attention"
            )

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        if is_causal:
            raise NotImplementedError(
                "Causal masking not implemented for linear attention"
            )
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h d n", h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h d n -> b n (h d)")
        return self.to_out(out)


class _TemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        is_causal: bool = True,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.attn_block = AttentionBlock(dim, heads, dim_head, rotary_emb=rotary_emb)
        self.time_pos_embedding = (
            nn.Sequential(
                Timesteps(dim),
                TimestepEmbedding(in_channels=dim, time_embed_dim=dim * 4, out_dim=dim),
            )
            if rotary_emb is None
            else None
        )
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor):
        if self.time_pos_embedding is not None:
            num_frames = x.shape[1]
            time_emb = self.time_pos_embedding(
                torch.arange(num_frames, device=x.device)
            )
            x = x + time_emb
        x = self.attn_block(x, is_causal=self.is_causal)
        return x


SpatialAttentionBlock = get_einops_wrapped_module(
    AttentionBlock, "b c t h w", "(b t) (h w) c"
)

TemporalAttentionBlock = get_einops_wrapped_module(
    _TemporalAttentionBlock, "b c t h w", "(b h w) t c"
)
