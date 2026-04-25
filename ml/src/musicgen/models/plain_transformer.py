from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PlainTransformerConfig:
    vocab_size: int = 195
    block_size: int = 1024
    n_layers: int = 6
    n_embed: int = 512
    n_head: int = 8
    d_ff: int = 2048
    bars_per_sample: int = 8
    dropout: float = 0.1

    # Attributes: 4 dims (polyphony, intensity, velocity, density), each binned 0..7
    num_attributes: int = 4
    attribute_bins: int = 8
    attribute_embed_dim: int = 32  # 4*32 = 128


class AttributeConditioner(nn.Module):
    """
    attributes: LongTensor [B, bars, 4] with values in 0..7
    returns:    FloatTensor [B, bars, n_embed]
    """

    def __init__(self, cfg: PlainTransformerConfig):
        super().__init__()
        assert cfg.num_attributes == 4
        assert cfg.attribute_embed_dim * cfg.num_attributes == 128

        self.embeds = nn.ModuleList(
            [nn.Embedding(cfg.attribute_bins, cfg.attribute_embed_dim) for _ in range(cfg.num_attributes)]
        )
        self.proj = nn.Linear(cfg.attribute_embed_dim * cfg.num_attributes, cfg.n_embed)

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        # attributes: [B, bars, 4]
        a0 = self.embeds[0](attributes[..., 0])
        a1 = self.embeds[1](attributes[..., 1])
        a2 = self.embeds[2](attributes[..., 2])
        a3 = self.embeds[3](attributes[..., 3])
        A = torch.cat([a0, a1, a2, a3], dim=-1)  # [B, bars, 128]
        return self.proj(A)  # [B, bars, n_embed]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: PlainTransformerConfig):
        super().__init__()
        assert cfg.n_embed % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embed // cfg.n_head

        self.qkv = nn.Linear(cfg.n_embed, 3 * cfg.n_embed, bias=True)
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=True)
        self.dropout = float(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)

        # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # PyTorch uses optimized SDPA under the hood when available.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.proj(y)


class FiLMBlock(nn.Module):
    def __init__(self, cfg: PlainTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.attn = CausalSelfAttention(cfg)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Conditioned FiLM: gamma/beta per token timestep, derived from cond_tok [B,T,C]
        self.film = nn.Linear(cfg.n_embed, 2 * cfg.n_embed, bias=True)

        self.ln2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embed, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.n_embed),
        )
        self.mlp_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cond_tok: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        h = self.ln1(x)
        gamma_beta = self.film(cond_tok)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        x = x + self.resid_drop(self.attn(h))

        x = x + self.mlp_drop(self.mlp(self.ln2(x)))
        return x


class PlainTransformerDecoder(nn.Module):
    """
    Decoder-only baseline conditioned ONLY on per-bar attributes via FiLM.

    Inputs:
      - X:           LongTensor [B, T]
      - bar_indices: LongTensor [B, T] with values in [0..bars-1]
      - attributes:  LongTensor [B, bars, 4] with values in [0..7]
      - targets:     optional LongTensor [B, T] for teacher forcing loss

    Outputs:
      dict with logits [B,T,vocab] and optional loss.
    """

    def __init__(self, cfg: PlainTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embed)

        self.attr = AttributeConditioner(cfg)
        self.blocks = nn.ModuleList([FiLMBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=True)

    def forward(
        self,
        X: torch.Tensor,
        bar_indices: torch.Tensor,
        attributes: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        pad_id: Optional[int] = None,
    ) -> dict:
        B, T = X.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length T={T} exceeds block_size={self.cfg.block_size}")

        # Token + position embeddings
        pos = torch.arange(T, device=X.device)
        x = self.tok_emb(X) + self.pos_emb(pos)[None, :, :]

        # Build per-bar conditions and broadcast to tokens using bar_indices
        cond_bars = self.attr(attributes)  # [B, bars, C]
        idx = bar_indices.to(dtype=torch.long).clamp_(0, self.cfg.bars_per_sample - 1)
        cond_tok = cond_bars.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.cfg.n_embed))  # [B, T, C]

        for blk in self.blocks:
            x = blk(x, cond_tok)

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab]

        out = {"logits": logits}
        if targets is not None:
            if pad_id is None:
                raise ValueError("pad_id is required when computing loss with targets.")

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=int(pad_id),
                reduction="mean",
            )
            n_tokens = int((targets != int(pad_id)).sum().item())
            out.update({"loss": loss, "n_tokens": n_tokens})

        return out

