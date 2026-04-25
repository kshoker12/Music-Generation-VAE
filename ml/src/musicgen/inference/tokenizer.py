from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any


@dataclass(frozen=True)
class InferenceTokenizer:
    tok: Any
    pad_id: int
    bos_id: int
    bar_id: int
    id_to_token: dict[int, str]
    token_to_id: dict[str, int]


def load_remi_tokenizer() -> InferenceTokenizer:
    """
    Build the miditok REMI tokenizer with the same configuration as preprocessing.

    This mirrors `make_tokenizer()` in `ml/scripts/preprocess_all.py` to ensure ID compatibility.
    """
    import miditok
    from miditok import REMI, TokenizerConfig

    # Keep in sync with `ml/scripts/preprocess_all.py`.
    PITCH_RANGE = (21, 108)
    NUM_VELOCITIES = 32

    sig = inspect.signature(TokenizerConfig)
    params = set(sig.parameters.keys())

    base_kwargs = dict(
        pitch_range=PITCH_RANGE,
        beat_res={(0, 4): 4, (4, 8): 2},
        num_velocities=NUM_VELOCITIES,
        use_time_signatures=False,
        use_tempos=True,
        use_sustain_pedals=False,
    )
    cfg_kwargs = dict(base_kwargs)

    if "special_tokens" in params:
        cfg_kwargs["special_tokens"] = ["PAD", "BOS"]
    if "use_pitchdrum_tokens" in params:
        cfg_kwargs["use_pitchdrum_tokens"] = False
    if "sustain_pedal_duration" in params:
        cfg_kwargs["sustain_pedal_duration"] = True
    for k, v in {
        "use_chords": False,
        "use_rests": False,
        "use_programs": False,
        "use_pitch_bends": False,
        "use_pitch_intervals": False,
    }.items():
        if k in params:
            cfg_kwargs[k] = v

    tok = REMI(TokenizerConfig(**cfg_kwargs))
    if len(tok.vocab) != 195:
        raise RuntimeError(f"Tokenizer vocab does not match spec (expected 195). Got {len(tok.vocab)}.")

    vocab: dict[str, int] = tok.vocab

    def find_id(pred) -> int:
        matches = [(s, i) for (s, i) in vocab.items() if pred(s)]
        if len(matches) != 1:
            preview = sorted(matches)[:10]
            raise RuntimeError(f"Expected 1 match, got {len(matches)}. Preview: {preview}")
        return int(matches[0][1])

    pad_id = vocab.get("PAD_None")
    if pad_id is None:
        pad_id = find_id(lambda s: s.startswith("PAD"))

    bos_id = vocab.get("BOS_None")
    if bos_id is None:
        bos_id = find_id(lambda s: s.startswith("BOS"))

    bar_id = vocab.get("Bar_None")
    if bar_id is None:
        bar_id = find_id(lambda s: s.split("_")[0] == "Bar")

    id_to_token = {int(i): str(s) for (s, i) in vocab.items()}

    return InferenceTokenizer(
        tok=tok,
        pad_id=int(pad_id),
        bos_id=int(bos_id),
        bar_id=int(bar_id),
        id_to_token=id_to_token,
        token_to_id={str(k): int(v) for k, v in vocab.items()},
    )

