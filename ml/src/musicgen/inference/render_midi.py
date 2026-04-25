from __future__ import annotations

from pathlib import Path
from typing import Any


def tokens_to_midi(tokens: list[int], tok: Any):
    """
    Convert token ids -> a MidiFile-like object.

    miditok has slightly different APIs across versions, so we try a few.
    """
    ids = [int(x) for x in tokens]
    # miditok expects sequences shaped like ('I','T') even for single-stream tokenizers.
    # Provide a 2D list [[...]] to avoid ambiguous auto-unsqueeze behavior.
    ids_2d = [ids]

    if hasattr(tok, "tokens_to_midi"):
        return tok.tokens_to_midi(ids_2d)
    if hasattr(tok, "decode"):
        # Some miditok versions expose decode(ids) -> MidiFile
        return tok.decode(ids_2d)
    if hasattr(tok, "tokens_to_track"):
        # Last resort: may require wrapping; keep explicit error for now.
        raise RuntimeError("Tokenizer has tokens_to_track but no tokens_to_midi/decode; unsupported miditok version.")
    raise RuntimeError("Tokenizer does not support detokenization (missing tokens_to_midi/decode).")


def write_midi(midi_obj: Any, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # symusic ScoreTick (miditok backend) supports dump_midi(path)
    if hasattr(midi_obj, "dump_midi"):
        midi_obj.dump_midi(str(out_path))
        return out_path

    # miditoolkit MidiFile supports .dump(path)
    if hasattr(midi_obj, "dump"):
        midi_obj.dump(str(out_path))
        return out_path

    # pretty_midi PrettyMIDI supports .write(path)
    if hasattr(midi_obj, "write"):
        midi_obj.write(str(out_path))
        return out_path

    raise RuntimeError(f"Unsupported MIDI object type for writing: {type(midi_obj)}")

