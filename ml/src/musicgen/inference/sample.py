from __future__ import annotations

import math
from typing import Optional

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    top_p: float = 0.9,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """
    Sample one token index from unnormalized logits using temperature + nucleus (top-p) sampling.

    Args:
        logits: 1D tensor [vocab]
        top_p: nucleus probability mass (0<top_p<=1)
        temperature: softmax temperature (>0)
        generator: optional torch RNG generator for reproducibility
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected logits shape [vocab], got {tuple(logits.shape)}")
    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError("top_p must be in (0, 1].")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0.")

    # Temperature
    if float(temperature) != 1.0:
        logits = logits / float(temperature)

    # Stabilize + softmax
    probs = torch.softmax(logits - logits.max(), dim=-1)

    # Top-p: sort by prob desc, keep smallest set whose cumulative mass >= top_p
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= float(top_p)
    # Always keep at least the best token
    keep[0] = True
    # Include the first token that crosses top_p
    if keep.sum().item() < keep.numel():
        first_excluded = int(keep.sum().item())
        keep[first_excluded] = True

    filt_probs = sorted_probs[keep]
    filt_idx = sorted_idx[keep]
    filt_probs = filt_probs / filt_probs.sum()

    choice = torch.multinomial(filt_probs, num_samples=1, replacement=False, generator=generator)
    next_id = int(filt_idx[int(choice.item())].item())
    return next_id

