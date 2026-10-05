"""The from-scratch Decisions Model: a plain Transformer encoder (pre-LN, RoPE, GELU MLP) with two heads.

  masked-word head  tied to the input embedding, used for pretraining
  decision head     one logit per [A] marker; softmax over them gives the choice and its probability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SCRATCH
from .text import PAD, VOCAB

SIZES = {"xxs": (64, 2), "xs": (128, 4), "s": (256, 6), "m": (384, 8), "l": (512, 8)}  # (width, layers); heads = width / 64
assert tuple(SIZES) == SCRATCH
MAX_LEN = 512
HEAD_DIM = 64


def rope(T: int, head_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / 10_000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    t = torch.arange(T, device=device).float()[:, None] * inv[None]
    return t.cos()[None, :, None, :], t.sin()[None, :, None, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:  # x: (batch, T, heads, head_dim)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.h = max(1, d // HEAD_DIM)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, cos, sin, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Decider(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.mlm_bias = nn.Parameter(torch.zeros(VOCAB))
        self.score = nn.Linear(d, 1)
        self.pad_id = PAD
        nn.init.normal_(self.emb.weight, std=0.02)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        cos, sin = rope(T, self.emb.weight.size(1) // self.blocks[0].h, ids.device)
        mask = (ids != PAD)[:, None, None, :]
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return self.ln(x)

    def mlm_loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        h = self.encode(ids)
        sel = targets != -100
        return F.cross_entropy(h[sel] @ self.emb.weight.T + self.mlm_bias, targets[sel])

    def decide(self, ids: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        """(batch, T) logits; only [A] positions are finite."""
        return self.score(self.encode(ids)).squeeze(-1).float().masked_fill(~markers, -1e4)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build(size: str) -> Decider:
    return Decider(*SIZES[size])
