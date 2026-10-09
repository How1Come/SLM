"""A tiny decoder-only transformer (RoPE, RMSNorm, tied embeddings) sized for laptop CPUs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    max_len: int = 256
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(max_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        causal = cache is None or T > 1
        if cache is not None and T > 1 and k.shape[2] != T:
            raise ValueError("multi-token decode with a non-empty cache is not supported")
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, cache)
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        cos, sin = rope_cache(cfg.max_len * 2, cfg.d_model // cfg.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, caches: list[dict] | None = None, pos: int = 0) -> torch.Tensor:
        T = idx.shape[1]
        cos, sin = self.rope_cos[pos : pos + T], self.rope_sin[pos : pos + T]
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 64,
        temperature: float = 0.9,
        top_k: int = 40,
        stop_ids: tuple[int, ...] = (),
        pad_id: int = 0,
    ) -> tuple[list[list[int]], int]:
        """Batched sampling with a KV cache. Prompts are left-truncated to a common length.

        Returns (continuations, forward_tokens_processed) so callers can account compute.
        """
        was_training = self.training
        self.eval()
        plen = min(len(p) for p in prompts)
        plen = max(1, min(plen, self.cfg.max_len - max_new_tokens))
        idx = torch.tensor([p[-plen:] for p in prompts], dtype=torch.long)
        B = idx.shape[0]
        caches = [dict() for _ in self.blocks]
        logits = self(idx, caches=caches, pos=0)[:, -1]
        forward_tokens = B * plen
        out = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool)
        for step in range(max_new_tokens):
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            for b in range(B):
                if not done[b]:
                    t = int(nxt[b])
                    out[b].append(t)
                    if t in stop_ids:
                        done[b] = True
            if bool(done.all()):
                break
            logits = self(nxt, caches=caches, pos=plen + step)[:, -1]
            forward_tokens += B
        if was_training:
            self.train()
        return out, forward_tokens
