from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from musicgen.inference.grammar import (
    GrammarVocabIndex,
    allowed_next_ids,
    build_vocab_index,
    mask_logits_inplace,
)
from musicgen.inference.sample import sample_next_token


ModelType = Literal["plain", "vae", "simple_vae"]


@dataclass(frozen=True)
class GenerationConfig:
    block_size: int = 1024
    bars_per_sample: int = 8
    temperature: float = 1.0
    top_p: float = 0.9


def compute_bar_indices_prefix(tokens: list[int], *, bar_id: int, bars_per_sample: int) -> list[int]:
    """
    Mirror preprocessing's `compute_bar_indices` but for a prefix (can end early).

    Rule:
      - Increment bar counter when seeing Bar token.
      - Assign each token to current bar (Bar token belongs to the new bar).
      - Before first Bar, assign bar 0.
    """
    cur = -1
    max_bar = bars_per_sample - 1
    out: list[int] = []
    for tid in tokens:
        if int(tid) == int(bar_id):
            cur += 1
        if cur < 0:
            cur = 0
        if cur > max_bar:
            cur = max_bar
        out.append(int(cur))
    return out


@torch.no_grad()
def generate_remi_tokens(
    *,
    model_type: ModelType,
    model,
    attributes: torch.Tensor,
    pad_id: int,
    bos_id: int,
    bar_id: int,
    id_to_token: dict[int, str],
    cfg: GenerationConfig,
    device: torch.device,
    z_p: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> list[int]:
    """
    Autoregressive generation with grammar masking, switching bar-level conditioning based on generated Bar tokens.

    `attributes` is expected to be binned ints shaped [1, bars_per_sample, 4].
    """
    if attributes.ndim != 3 or attributes.shape[0] != 1 or attributes.shape[1] != cfg.bars_per_sample or attributes.shape[2] != 4:
        raise ValueError(f"attributes must have shape [1,{cfg.bars_per_sample},4], got {tuple(attributes.shape)}")

    model.eval()
    model.to(device)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    vocab_index: GrammarVocabIndex = build_vocab_index(id_to_token)

    # Start with BOS then force first token via grammar (BOS -> Bar).
    tokens: list[int] = [int(bos_id)]
    bar_count = 0

    # Preallocate full block buffer for models that require fixed length.
    X_full = torch.full((1, cfg.block_size), int(pad_id), dtype=torch.long, device=device)

    # If model uses z_p (VAEs), set it up once.
    z_p_in = None
    if model_type in ("vae", "simple_vae"):
        if z_p is None:
            # Standard Normal sample
            z_p_in = torch.randn((1, getattr(model.cfg, "z_dim", 128)), generator=gen, device=device)
        else:
            z_p_in = z_p.to(device=device, dtype=torch.float32)
            if z_p_in.ndim != 2 or z_p_in.shape[0] != 1:
                raise ValueError(f"z_p must have shape [1,z_dim], got {tuple(z_p_in.shape)}")
        # Some models expect z_p as argument, others store inside forward; we unify below.

    for t in range(cfg.block_size - 1):
        # Fill prefix into fixed buffer
        X_full.fill_(int(pad_id))
        X_full[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)

        # Build bar_indices for full buffer: prefix computed, suffix set to current bar.
        bar_idx_prefix = compute_bar_indices_prefix(tokens, bar_id=bar_id, bars_per_sample=cfg.bars_per_sample)
        cur_bar = int(bar_idx_prefix[-1]) if bar_idx_prefix else 0
        bar_indices = torch.full((1, cfg.block_size), cur_bar, dtype=torch.long, device=device)
        bar_indices[0, : len(bar_idx_prefix)] = torch.tensor(bar_idx_prefix, dtype=torch.long, device=device)

        # Forward
        if model_type == "plain":
            out = model(X_full, bar_indices=bar_indices, attributes=attributes)
        elif model_type == "simple_vae":
            out = model(X_full, bar_indices=bar_indices, attributes=attributes, z_p=z_p_in)
        elif model_type == "vae":
            out = model(X_full, bar_indices=bar_indices, attributes=attributes, z_p=z_p_in)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        logits = out["logits"] if isinstance(out, dict) else getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError("Model forward must return dict with key 'logits'.")

        # Next-token logits at current position
        step_logits = logits[0, len(tokens) - 1].float().clone()

        # Grammar mask
        allow_bar = True
        prev_id = int(tokens[-1])
        allowed = allowed_next_ids(
            prev_id,
            id_to_token=id_to_token,
            vocab_index=vocab_index,
            bar_id=bar_id,
            pad_id=pad_id,
            bos_id=bos_id,
            allow_bar=allow_bar,
        )
        mask_logits_inplace(step_logits, allowed_ids=allowed)

        next_id = sample_next_token(step_logits, top_p=cfg.top_p, temperature=cfg.temperature, generator=gen)

        # Stop when the model wants to start a 9th bar (i.e., we've already generated 8 Bar tokens).
        if int(next_id) == int(bar_id) and bar_count >= cfg.bars_per_sample:
            break

        tokens.append(int(next_id))
        if int(next_id) == int(bar_id):
            bar_count += 1

    # Pad to full block_size for downstream compatibility.
    if len(tokens) < cfg.block_size:
        tokens = tokens + [int(pad_id)] * (cfg.block_size - len(tokens))
    else:
        tokens = tokens[: cfg.block_size]

    return tokens

