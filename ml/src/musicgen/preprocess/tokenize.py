from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerConfigStub:
    name: str = "miditok.REMI"
    vocab_size: int = 163


def tokenize_midi_file(_: str, __: TokenizerConfigStub) -> list[int]:
    """
    Stub: later implement MIDI -> REMI token ids using miditok.
    """
    raise NotImplementedError("tokenize_midi_file is a stub (Kaggle/local impl later).")

