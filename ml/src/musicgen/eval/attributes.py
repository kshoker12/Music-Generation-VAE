from __future__ import annotations

"""
Thin wrappers around the canonical preprocessing-compatible attribute logic.

We reuse the logic from `musicgen.analysis.control_eval` because it explicitly
documents and implements:
- raw per-bar stat extraction from REMI tokens
- binning via saved global quantile thresholds
"""

from pathlib import Path
from typing import Any

import numpy as np

from musicgen.analysis.control_eval import (  # canonical implementations
    ATTRIBUTE_NAMES,
    bin_attributes as _bin_attributes,
    compute_raw_attributes_per_bar_from_tokens as _compute_raw_attrs_from_tokens,
    load_thresholds as _load_thresholds,
)


def load_quantile_thresholds(quantiles_path: str | Path | None = None) -> dict[str, list[float]]:
    """Load global quantile thresholds (7 edges per attribute)."""
    return _load_thresholds(quantiles_path)


def compute_raw_attributes_per_bar_from_tokens(
    x_tokens_1024: list[int],
    *,
    id_to_token: dict[int, str],
    bar_token_id: int,
    bars_per_sample: int,
) -> list[list[float]]:
    """Return raw per-bar stats shaped [bars_per_sample,4]."""
    return _compute_raw_attrs_from_tokens(
        x_tokens_1024,
        id_to_token=id_to_token,
        bar_token_id=bar_token_id,
        bars_per_sample=bars_per_sample,
    )


def bin_attributes(
    raw_attrs_barsx4: list[list[float]] | np.ndarray,
    *,
    thresholds: dict[str, list[float]],
    bars_per_sample: int,
) -> list[list[int]]:
    """Quantize raw attributes to bins 0..7, shape [bars_per_sample,4]."""
    if isinstance(raw_attrs_barsx4, np.ndarray):
        raw = raw_attrs_barsx4.astype(float).tolist()
    else:
        raw = raw_attrs_barsx4
    return _bin_attributes(raw, thresholds, bars_per_sample=bars_per_sample)


def reduce_barwise_bins_to_piece_bins(attrs_bins_barsx4: list[list[int]] | np.ndarray) -> list[int]:
    """
    Reduce [bars,4] integer bins to a single [4] bin vector by averaging over bars
    then rounding (matches latent viz convention).
    """
    arr = np.asarray(attrs_bins_barsx4, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected shape [bars,4], got {arr.shape}")
    mean_bins = arr.mean(axis=0)
    return np.rint(mean_bins).astype(int).tolist()


def pretty_attr_vector(v4: list[int] | tuple[int, int, int, int]) -> str:
    """Small helper for filenames/logging."""
    return "[" + ",".join(str(int(x)) for x in v4) + "]"

