from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VAEConfig:
    vocab_size: int = 195
    block_size: int = 1024

    # Encoder (bidirectional)
    enc_n_layers: int = 6
    enc_n_embed: int = 512
    enc_n_head: int = 8
    enc_d_ff: int = 2048
    z_dim: int = 128
    enc_dropout: float = 0.1

    # Attributes
    bars_per_sample: int = 8
    num_attributes: int = 4
    attribute_bins: int = 8
    attribute_embed_dim: int = 32  # 4*32 = 128

    # Conductor
    cond_hidden: int = 384
    cond_layers: int = 2

    # Decoder (causal)
    dec_n_layers: int = 6
    dec_n_embed: int = 512
    dec_n_head: int = 8
    dec_d_ff: int = 2048
    dec_dropout: float = 0.1


class AttributeEmbedder(nn.Module):
    """
    attributes: LongTensor [B, bars, 4] with values 0..7
    returns:    FloatTensor [B, bars, 128]
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        assert cfg.num_attributes == 4
        self.embeds = nn.ModuleList(
            [nn.Embedding(cfg.attribute_bins, cfg.attribute_embed_dim) for _ in range(cfg.num_attributes)]
        )

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        a0 = self.embeds[0](attributes[..., 0])
        a1 = self.embeds[1](attributes[..., 1])
        a2 = self.embeds[2](attributes[..., 2])
        a3 = self.embeds[3](attributes[..., 3])
        return torch.cat([a0, a1, a2, a3], dim=-1)  # [B, bars, 128]


def broadcast_by_bar_indices(bars_x_c: torch.Tensor, bar_indices: torch.Tensor) -> torch.Tensor:
    """
    bars_x_c:  [B, bars, C]
    bar_indices: [B, T] values in 0..bars-1
    returns:   [B, T, C]
    """
    B, bars, C = bars_x_c.shape
    idx = bar_indices.to(dtype=torch.long).clamp_(0, bars - 1)
    return bars_x_c.gather(1, idx.unsqueeze(-1).expand(B, idx.size(1), C))


class BiTransformerEncoder(nn.Module):
    """
    Bidirectional encoder with learned [CLS] token, outputs mu/logvar for z_p.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.enc_n_embed)
        self.pos_emb = nn.Embedding(cfg.block_size + 1, cfg.enc_n_embed)  # +1 for CLS position
        self.cls = nn.Parameter(torch.zeros(cfg.enc_n_embed))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.enc_n_embed,
            nhead=cfg.enc_n_head,
            dim_feedforward=cfg.enc_d_ff,
            dropout=float(cfg.enc_dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.enc_n_layers)
        self.ln_f = nn.LayerNorm(cfg.enc_n_embed)

        self.mu = nn.Linear(cfg.enc_n_embed, cfg.z_dim)
        self.logvar = nn.Linear(cfg.enc_n_embed, cfg.z_dim)

        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = X.shape
        if T != self.cfg.block_size:
            raise ValueError(f"Encoder expects block_size={self.cfg.block_size}, got T={T}")

        tok = self.tok_emb(X)  # [B,T,C]
        cls = self.cls.view(1, 1, -1).expand(B, 1, -1)  # [B,1,C]
        h = torch.cat([cls, tok], dim=1)  # [B,T+1,C]

        pos = torch.arange(T + 1, device=X.device)
        h = h + self.pos_emb(pos)[None, :, :]

        h = self.encoder(h)
        h = self.ln_f(h)
        h_cls = h[:, 0, :]  # [B,C]
        return self.mu(h_cls), self.logvar(h_cls)


class Conductor(nn.Module):
    """
    Deterministic GRU that maps z_p -> z_k sequence of length bars_per_sample.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.h0_proj = nn.Linear(cfg.z_dim, cfg.cond_layers * cfg.cond_hidden)
        self.bar_emb = nn.Embedding(cfg.bars_per_sample, cfg.cond_hidden)
        self.gru = nn.GRU(
            input_size=cfg.cond_hidden,
            hidden_size=cfg.cond_hidden,
            num_layers=cfg.cond_layers,
            batch_first=True,
        )

    def forward(self, z_p: torch.Tensor) -> torch.Tensor:
        # z_p: [B,128]
        B = z_p.size(0)
        h0 = self.h0_proj(z_p).view(self.cfg.cond_layers, B, self.cfg.cond_hidden).contiguous()

        bar_ids = torch.arange(self.cfg.bars_per_sample, device=z_p.device)
        x = self.bar_emb(bar_ids)[None, :, :].expand(B, -1, -1)  # [B,bars,384]

        z_seq, _hn = self.gru(x, h0)  # [B,bars,384]
        return z_seq


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=True)
        self.proj = nn.Linear(n_embed, n_embed, bias=True)
        self.dropout = 0.0

    def set_dropout(self, p: float) -> None:
        self.dropout = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FiLMBlock(nn.Module):
    def __init__(self, n_embed: int, n_head: int, d_ff: int, *, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed=n_embed, n_head=n_head)
        self.attn.set_dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.film = nn.Linear(n_embed, 2 * n_embed, bias=True)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, n_embed),
        )
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond_tok: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        gamma, beta = self.film(cond_tok).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        x = x + self.resid_drop(self.attn(h))
        x = x + self.mlp_drop(self.mlp(self.ln2(x)))
        return x


class FiLMDecoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dec_n_embed)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dec_n_embed)

        self.blocks = nn.ModuleList(
            [
                FiLMBlock(cfg.dec_n_embed, cfg.dec_n_head, cfg.dec_d_ff, dropout=float(cfg.dec_dropout))
                for _ in range(cfg.dec_n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.dec_n_embed)
        self.head = nn.Linear(cfg.dec_n_embed, cfg.vocab_size, bias=True)

    def forward(self, X: torch.Tensor, cond_tok: torch.Tensor) -> torch.Tensor:
        B, T = X.shape
        if T != self.cfg.block_size:
            raise ValueError(f"Decoder expects block_size={self.cfg.block_size}, got T={T}")

        pos = torch.arange(T, device=X.device)
        x = self.tok_emb(X) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x, cond_tok)
        x = self.ln_f(x)
        return self.head(x)  # [B,T,vocab]


class MusicVAE(nn.Module):
    """
    Conditional VAE: Encoder -> z_p -> Conductor -> z_k; Decoder conditioned via FiLM on C_k=[z_k;A_k].
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.attr = AttributeEmbedder(cfg)  # [B,8,128]
        self.encoder = BiTransformerEncoder(cfg)
        self.conductor = Conductor(cfg)  # [B,8,384]
        self.decoder = FiLMDecoder(cfg)

        # Project A_k to 128 (already) then concat with z_k(384) -> 512 matches dec_n_embed
        if cfg.cond_hidden + (cfg.attribute_embed_dim * cfg.num_attributes) != cfg.dec_n_embed:
            raise ValueError(
                f"Expected cond_hidden(={cfg.cond_hidden}) + attr_dim(={cfg.attribute_embed_dim*cfg.num_attributes}) "
                f"== dec_n_embed(={cfg.dec_n_embed})"
            )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence_free_bits(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        free_bits_lambda_bits: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Free-bits (hinge) KL to fight posterior collapse (spec).

        Returns:
          - kl_loss: scalar (mean over batch) in nats
          - kl_per_dim_mean: [z_dim] mean KL per dimension in nats
          - active_units: scalar count of dims where KL_dim_mean > lambda_nats
        """
        # KL per dim per sample (nats): [B, z_dim]
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

        # Mean over batch, per dim: [z_dim]
        kl_per_dim_mean = kl_per_dim.mean(dim=0)

        # Spec lambda is in bits; convert to nats.
        lambda_nats = float(free_bits_lambda_bits) * math.log(2.0)

        # Apply free-bits hinge *after* averaging across batch per dimension.
        # This is typically the more stable variant (vs. hinging per-sample).
        kl_per_dim_hinged_mean = torch.clamp(kl_per_dim_mean, min=lambda_nats)  # [z_dim]
        kl_loss = kl_per_dim_hinged_mean.sum()

        active_units = (kl_per_dim_mean > lambda_nats).sum()
        return kl_loss, kl_per_dim_mean, active_units

    def forward(
        self,
        X: torch.Tensor,
        bar_indices: torch.Tensor,
        attributes: torch.Tensor,
        *,
        z_p: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        pad_id: Optional[int] = None,
        beta: float = 1.0,
    ) -> dict:
        mu: Optional[torch.Tensor]
        logvar: Optional[torch.Tensor]
        if z_p is None:
            mu, logvar = self.encoder(X)
            z_p = self.reparameterize(mu, logvar)
        else:
            mu, logvar = None, None
            if z_p.ndim != 2 or z_p.shape[0] != X.shape[0]:
                raise ValueError(f"z_p must have shape [B,z_dim], got {tuple(z_p.shape)}")
            if z_p.shape[1] != self.cfg.z_dim:
                raise ValueError(f"z_p dim mismatch: expected {self.cfg.z_dim}, got {z_p.shape[1]}")

        z_k = self.conductor(z_p)  # [B,8,384]

        A_k = self.attr(attributes)  # [B,8,128]
        C_k = torch.cat([z_k, A_k], dim=-1)  # [B,8,512]
        cond_tok = broadcast_by_bar_indices(C_k, bar_indices)  # [B,1024,512]

        logits = self.decoder(X, cond_tok)

        out = {"logits": logits, "mu": mu, "logvar": logvar, "z_p": z_p, "z_k": z_k}

        if mu is not None and logvar is not None:
            loss_kl, kl_per_dim_mean, active_units = self.kl_divergence_free_bits(
                mu, logvar, free_bits_lambda_bits=1.0
            )
            out["loss_kl"] = loss_kl
            out["kl_per_dim_mean"] = kl_per_dim_mean
            out["active_units"] = active_units

        if targets is not None:
            if pad_id is None:
                raise ValueError("pad_id is required when computing reconstruction loss.")
            if mu is None or logvar is None:
                raise ValueError("targets provided but z_p was passed explicitly; KL stats are undefined in this mode.")
            loss_recon = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=int(pad_id),
                reduction="mean",
            )
            out["loss_recon"] = loss_recon
            out["loss_total"] = loss_recon + float(beta) * loss_kl

        return out

