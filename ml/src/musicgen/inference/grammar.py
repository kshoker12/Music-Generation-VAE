from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

import torch


TokenType = Literal[
    "PAD",
    "BOS",
    "Bar",
    "Position",
    "Tempo",
    "Pitch",
    "Velocity",
    "Duration",
    "Other",
]


def token_type(token_str: str) -> TokenType:
    # miditok uses strings like "Bar_None", "Position_0", "Pitch_60", "PAD_None"
    head = token_str.split("_", 1)[0]
    if head == "PAD":
        return "PAD"
    if head == "BOS":
        return "BOS"
    if head == "Bar":
        return "Bar"
    if head == "Position":
        return "Position"
    if head == "Tempo":
        return "Tempo"
    if head == "Pitch":
        return "Pitch"
    if head == "Velocity":
        return "Velocity"
    if head == "Duration":
        return "Duration"
    return "Other"


@dataclass(frozen=True)
class GrammarVocabIndex:
    ids_by_type: dict[TokenType, list[int]]


def build_vocab_index(id_to_token: dict[int, str]) -> GrammarVocabIndex:
    ids_by_type: dict[TokenType, list[int]] = {
        "PAD": [],
        "BOS": [],
        "Bar": [],
        "Position": [],
        "Tempo": [],
        "Pitch": [],
        "Velocity": [],
        "Duration": [],
        "Other": [],
    }
    for tid, s in id_to_token.items():
        ids_by_type[token_type(s)].append(int(tid))
    for k in ids_by_type:
        ids_by_type[k] = sorted(ids_by_type[k])
    return GrammarVocabIndex(ids_by_type=ids_by_type)


def allowed_next_ids(
    prev_token_id: int,
    *,
    id_to_token: dict[int, str],
    vocab_index: GrammarVocabIndex,
    bar_id: int,
    pad_id: int,
    bos_id: int,
    allow_bar: bool = True,
) -> list[int]:
    """
    Compute allowed next token IDs based on previous token (local grammar).

    Minimum constraints per spec:
      - Bar -> Position
      - Pitch -> Velocity
    Plus safe constraints to keep REMI ordering sane:
      - BOS -> Bar
      - Velocity -> Duration
    """
    prev_str = id_to_token[int(prev_token_id)]
    prev_type = token_type(prev_str)

    if prev_token_id == int(bos_id) or prev_type == "BOS":
        return [int(bar_id)]

    if prev_token_id == int(bar_id) or prev_type == "Bar":
        # Must start bar with a position.
        return vocab_index.ids_by_type["Position"]

    if prev_type == "Pitch":
        return vocab_index.ids_by_type["Velocity"]

    if prev_type == "Velocity":
        return vocab_index.ids_by_type["Duration"]

    if prev_type == "Tempo":
        # Tempo typically appears at a bar start before the first position.
        return vocab_index.ids_by_type["Position"]

    if prev_type == "Position":
        # After choosing a position we can emit a note (Pitch) or close the bar.
        allowed: list[int] = []
        allowed += vocab_index.ids_by_type["Pitch"]
        allowed += vocab_index.ids_by_type["Tempo"]
        if allow_bar:
            allowed.append(int(bar_id))
        return sorted(set(allowed))

    if prev_type == "Duration":
        # After a note event completes, we can emit another Pitch in same position,
        # move to another Position, change Tempo, or end the bar.
        allowed = []
        allowed += vocab_index.ids_by_type["Pitch"]
        allowed += vocab_index.ids_by_type["Position"]
        allowed += vocab_index.ids_by_type["Tempo"]
        if allow_bar:
            allowed.append(int(bar_id))
        return sorted(set(allowed))

    # Fallback: allow most musical tokens, disallow PAD/BOS.
    allowed = []
    allowed += vocab_index.ids_by_type["Bar"] if allow_bar else []
    allowed += vocab_index.ids_by_type["Position"]
    allowed += vocab_index.ids_by_type["Tempo"]
    allowed += vocab_index.ids_by_type["Pitch"]
    # Do not allow BOS/PAD mid-generation.
    allowed = [i for i in allowed if i not in (int(pad_id), int(bos_id))]
    return sorted(set(allowed))


def mask_logits_inplace(
    logits: torch.Tensor,
    *,
    allowed_ids: Iterable[int],
) -> torch.Tensor:
    """
    In-place mask: set disallowed logits to -inf.
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected logits [vocab], got {tuple(logits.shape)}")
    allowed = torch.tensor(list(allowed_ids), device=logits.device, dtype=torch.long)
    mask = torch.ones((logits.size(0),), device=logits.device, dtype=torch.bool)
    mask[allowed] = False
    logits[mask] = float("-inf")
    return logits

