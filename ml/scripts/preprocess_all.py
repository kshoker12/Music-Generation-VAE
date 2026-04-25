from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm.auto import tqdm


# -----------------------------
# Constants / paths
# -----------------------------

BARS_PER_SAMPLE = 8
BLOCK_SIZE = 1024
PITCH_RANGE = (21, 108)
NUM_VELOCITIES = 32
TRANSPOSITIONS = list(range(12))  # 0..11

ATTRIBUTE_NAMES = [
    "polyphony_rate",
    "rhythmic_intensity",
    "velocity_dynamics",
    "note_density",
]


# -----------------------------
# Chunk-quality filtering (strict, Option 1)
# -----------------------------

MIN_ACTIVE_BARS = 6
MIN_TOTAL_ONSETS = 40
MAX_LEADING_EMPTY_BARS = 1  # drop if >= 2
MIN_MEAN_RHYTHMIC_INTENSITY = 0.10


def chunk_passes_filter(raw_attrs_barsx4: list[list[float]], *, bars_per_sample: int) -> bool:
    """Return True if this chunk is kept for training.

    Expects raw attrs per bar: [polyphony_rate, rhythmic_intensity, vel_std, note_density].
    """
    if len(raw_attrs_barsx4) != bars_per_sample:
        return False

    note_density = [float(r[3]) for r in raw_attrs_barsx4]
    rhythmic_intensity = [float(r[1]) for r in raw_attrs_barsx4]

    active_bars = sum(1 for v in note_density if v > 0.0)
    total_onsets = float(sum(note_density))

    leading_empty = 0
    for v in note_density:
        if v == 0.0:
            leading_empty += 1
        else:
            break

    mean_ri = float(sum(rhythmic_intensity) / bars_per_sample) if bars_per_sample > 0 else 0.0

    if active_bars < MIN_ACTIVE_BARS:
        return False
    if total_onsets < float(MIN_TOTAL_ONSETS):
        return False
    if leading_empty > MAX_LEADING_EMPTY_BARS:
        return False
    if mean_ri < float(MIN_MEAN_RHYTHMIC_INTENSITY):
        return False
    return True


def repo_root() -> Path:
    # ml/scripts/preprocess_all.py -> repo root is 2 parents up from ml/
    return Path(__file__).resolve().parents[2]


def paths():
    root = repo_root()
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"

    return {
        "root": root,
        "data": data_dir,
        "artifacts": artifacts_dir,
        "splits": artifacts_dir / "splits",
        "attributes": artifacts_dir / "attributes",
        "preprocessed": artifacts_dir / "preprocessed",
    }


# -----------------------------
# Memmap writer (Kaggle-friendly)
# -----------------------------


class MemmapAppendWriter:
    """Append rows to fixed-width memmaps, growing capacity as needed.

    Stores:
    - X: uint16 (or uint8) shape [N, BLOCK_SIZE]
    - bar_indices: uint8 shape [N, BLOCK_SIZE]
    - attributes: uint8 shape [N, BARS_PER_SAMPLE, 4]
    """

    def __init__(self, out_dir: Path, *, x_dtype: np.dtype, block_size: int, bars_per_sample: int):
        self.out_dir = out_dir
        self.block_size = int(block_size)
        self.bars_per_sample = int(bars_per_sample)
        self.x_dtype = np.dtype(x_dtype)
        self.bi_dtype = np.dtype(np.uint8)
        self.a_dtype = np.dtype(np.uint8)

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.x_path = self.out_dir / ("X.u16.memmap" if self.x_dtype == np.uint16 else "X.u8.memmap")
        self.bi_path = self.out_dir / "bar_indices.u8.memmap"
        self.a_path = self.out_dir / "attributes.u8.memmap"

        self._capacity = 0
        self._n = 0
        self._X = None
        self._BI = None
        self._A = None

    @property
    def n(self) -> int:
        return int(self._n)

    def _alloc(self, capacity: int) -> None:
        capacity = int(capacity)
        if capacity <= self._capacity:
            return

        # flush and drop old views before resizing
        self.flush()
        self._X = self._BI = self._A = None

        # compute byte sizes and grow files
        x_bytes = capacity * self.block_size * self.x_dtype.itemsize
        bi_bytes = capacity * self.block_size * self.bi_dtype.itemsize
        a_bytes = capacity * self.bars_per_sample * 4 * self.a_dtype.itemsize

        for path, nbytes in [(self.x_path, x_bytes), (self.bi_path, bi_bytes), (self.a_path, a_bytes)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab"):
                pass
            os.truncate(path, nbytes)

        self._X = np.memmap(self.x_path, dtype=self.x_dtype, mode="r+", shape=(capacity, self.block_size))
        self._BI = np.memmap(self.bi_path, dtype=self.bi_dtype, mode="r+", shape=(capacity, self.block_size))
        self._A = np.memmap(self.a_path, dtype=self.a_dtype, mode="r+", shape=(capacity, self.bars_per_sample, 4))
        self._capacity = capacity

    def append(self, X_row: list[int], BI_row: list[int], A_row: list[list[int]]) -> None:
        if self._n >= self._capacity:
            new_cap = 1 if self._capacity == 0 else self._capacity * 2
            self._alloc(new_cap)
        assert self._X is not None and self._BI is not None and self._A is not None
        i = self._n
        self._X[i, :] = np.asarray(X_row, dtype=self.x_dtype)
        self._BI[i, :] = np.asarray(BI_row, dtype=self.bi_dtype)
        self._A[i, :, :] = np.asarray(A_row, dtype=self.a_dtype)
        self._n += 1

    def finalize(self) -> None:
        """Truncate files to final row count."""
        self.flush()
        n = int(self._n)
        os.truncate(self.x_path, n * self.block_size * self.x_dtype.itemsize)
        os.truncate(self.bi_path, n * self.block_size * self.bi_dtype.itemsize)
        os.truncate(self.a_path, n * self.bars_per_sample * 4 * self.a_dtype.itemsize)

    def flush(self) -> None:
        if self._X is not None:
            self._X.flush()
        if self._BI is not None:
            self._BI.flush()
        if self._A is not None:
            self._A.flush()


# -----------------------------
# Small IO helpers
# -----------------------------


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_split(path: Path) -> list[tuple[str, str]]:
    rows = read_jsonl(path)
    return [(r["zip"], r["member"]) for r in rows]


def rm_if_exists(p: Path) -> None:
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


# -----------------------------
# MIDI transforms (bytes in/out)
# -----------------------------


def _midibytes_to_midifile(midi_bytes: bytes):
    from miditoolkit import MidiFile

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        return MidiFile(tmp.name)


def _midifile_to_midibytes(midi) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        midi.dump(tmp.name)
        tmp.flush()
        tmp.seek(0)
        return tmp.read()


def load_midi_from_zip(zip_path: Path, member: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as z:
        return z.read(member)


def strip_leading_silence(midi_bytes: bytes) -> bytes:
    midi = _midibytes_to_midifile(midi_bytes)

    min_start = None
    for inst in midi.instruments:
        for n in inst.notes:
            if min_start is None or n.start < min_start:
                min_start = n.start

    if min_start is None or min_start <= 0:
        return midi_bytes

    shift = int(min_start)
    for inst in midi.instruments:
        for n in inst.notes:
            n.start = max(0, n.start - shift)
            n.end = max(0, n.end - shift)
        for cc in getattr(inst, "control_changes", []) or []:
            cc.time = max(0, cc.time - shift)
        for pb in getattr(inst, "pitch_bends", []) or []:
            pb.time = max(0, pb.time - shift)

    for tempo in getattr(midi, "tempo_changes", []) or []:
        tempo.time = max(0, tempo.time - shift)
    for ts in getattr(midi, "time_signature_changes", []) or []:
        ts.time = max(0, ts.time - shift)
    for ks in getattr(midi, "key_signature_changes", []) or []:
        ks.time = max(0, ks.time - shift)

    return _midifile_to_midibytes(midi)


def transpose_midi(midi_bytes: bytes, semitones: int, pitch_range: tuple[int, int]) -> Optional[bytes]:
    if semitones == 0:
        return midi_bytes

    midi = _midibytes_to_midifile(midi_bytes)
    lo, hi = pitch_range

    for inst in midi.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            p = n.pitch + semitones
            if p < lo or p > hi:
                return None

    for inst in midi.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            n.pitch += semitones

    return _midifile_to_midibytes(midi)


def midi_duration_seconds(midi_bytes: bytes) -> float:
    midi = _midibytes_to_midifile(midi_bytes)

    end_tick = 0
    for inst in midi.instruments:
        for n in inst.notes:
            end_tick = max(end_tick, int(n.end))

    if end_tick <= 0:
        return 0.0

    tpq = float(getattr(midi, "ticks_per_beat", 480) or 480)
    tempo_changes = list(getattr(midi, "tempo_changes", []) or [])
    tempo_changes.sort(key=lambda t: int(getattr(t, "time", 0)))

    # Ensure a tempo at tick 0.
    if not tempo_changes or int(getattr(tempo_changes[0], "time", 0)) != 0:
        class _T:
            time = 0
            tempo = 120.0

        tempo_changes = [_T()] + tempo_changes

    secs = 0.0
    for i, tc in enumerate(tempo_changes):
        t0 = int(getattr(tc, "time", 0))
        bpm = float(getattr(tc, "tempo", 120.0) or 120.0)
        t1 = end_tick
        if i + 1 < len(tempo_changes):
            t1 = min(end_tick, int(getattr(tempo_changes[i + 1], "time", end_tick)))
        if t1 <= t0:
            continue
        beats = (t1 - t0) / tpq
        secs += beats * (60.0 / bpm)
        if t1 >= end_tick:
            break
    return float(secs)


# -----------------------------
# Tokenizer (miditok REMI)
# -----------------------------


def make_tokenizer():
    import inspect
    import miditok
    from miditok import REMI, TokenizerConfig

    sig = inspect.signature(TokenizerConfig)
    params = set(sig.parameters.keys())

    base_kwargs = dict(
        pitch_range=PITCH_RANGE,
        beat_res={(0, 4): 4, (4, 8): 2},
        num_velocities=NUM_VELOCITIES,
        use_time_signatures=False,
        use_tempos=True,
        # sustain is used but folded into Duration tokens
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

    vocab = tok.vocab

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

    return tok, int(pad_id), int(bos_id), int(bar_id)


def tokenize_midi_bytes(tokenizer, midi_bytes: bytes) -> list[int]:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        if hasattr(tokenizer, "encode"):
            seq = tokenizer.encode(tmp.name)
        else:
            seq = tokenizer.midi_to_tokens(tmp.name)

    if isinstance(seq, list):
        if len(seq) != 1:
            raise RuntimeError(f"Expected 1 token stream, got {len(seq)}")
        seq = seq[0]

    ids = getattr(seq, "ids", None)
    if ids is None:
        raise RuntimeError("miditok returned an object without `.ids`")
    return [int(x) for x in ids]


# -----------------------------
# Chunking + tensors
# -----------------------------


def split_into_bar_chunks(tokens: list[int], bar_token_id: int, bars_per_sample: int) -> list[list[int]]:
    if bars_per_sample <= 0:
        raise ValueError("bars_per_sample must be > 0")
    bar_starts = [i for i, t in enumerate(tokens) if t == bar_token_id]
    if not bar_starts:
        return []

    chunks: list[list[int]] = []
    for k in range(0, len(bar_starts) - bars_per_sample + 1, bars_per_sample):
        start = bar_starts[k]
        end = bar_starts[k + bars_per_sample] if (k + bars_per_sample) < len(bar_starts) else len(tokens)
        ch = tokens[start:end]
        if ch and ch[0] == bar_token_id and sum(1 for t in ch if t == bar_token_id) == bars_per_sample:
            chunks.append(ch)
    return chunks


def enforce_block_size(tokens: list[int], pad_id: int, block_size: int) -> Optional[list[int]]:
    if len(tokens) > block_size:
        return None
    if len(tokens) == block_size:
        return tokens
    return tokens + [pad_id] * (block_size - len(tokens))


def build_xy(x: list[int], pad_id: int) -> tuple[list[int], list[int]]:
    return x, (x[1:] + [pad_id])


def compute_bar_indices(x_tokens: list[int], bar_token_id: int, bars_per_sample: int) -> list[int]:
    bar_indices: list[int] = []
    cur = -1
    max_bar = bars_per_sample - 1
    for tok in x_tokens:
        if tok == bar_token_id:
            cur += 1
        if cur < 0:
            cur = 0
        if cur > max_bar:
            raise ValueError(f"Found >{bars_per_sample} bars inside a {len(x_tokens)}-token chunk")
        bar_indices.append(cur)
    if cur != max_bar:
        raise ValueError(f"Expected end bar {max_bar}, got {cur}")
    return bar_indices


def _parse_duration_beats(token: str) -> float:
    raw = token.split("_", 1)[1]
    try:
        return float(raw)
    except Exception:
        parts = raw.split(".")
        if len(parts) == 3:
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
            if c == 0:
                raise ValueError(f"Cannot parse duration token: {token}")
            return float(a + (b / c))
        raise ValueError(f"Cannot parse duration token: {token}")


def compute_raw_attributes_per_bar_from_tokens(
    x_tokens_1024: list[int],
    *,
    id_to_token: dict[int, str],
    bar_token_id: int,
    bars_per_sample: int,
) -> list[list[float]]:
    if len(x_tokens_1024) != BLOCK_SIZE:
        raise ValueError(f"Expected {BLOCK_SIZE} tokens, got {len(x_tokens_1024)}")

    onsets = [0 for _ in range(bars_per_sample)]
    pos_has_onset = [set() for _ in range(bars_per_sample)]
    velocities: list[list[int]] = [[] for _ in range(bars_per_sample)]
    global_intervals: list[tuple[float, float]] = []

    cur_bar = -1
    cur_pos = 0

    def parse_pos(tok: str) -> int:
        return int(tok.split("_", 1)[1])

    def parse_vel(tok: str) -> int:
        return int(tok.split("_", 1)[1])

    i = 0
    while i < len(x_tokens_1024):
        tid = x_tokens_1024[i]
        tok = id_to_token.get(int(tid))
        if tok is None:
            i += 1
            continue
        typ = tok.split("_", 1)[0]

        if tid == bar_token_id or typ == "Bar":
            cur_bar += 1
            cur_pos = 0
            i += 1
            continue

        if cur_bar < 0:
            i += 1
            continue
        if cur_bar >= bars_per_sample:
            break

        if typ == "Position":
            cur_pos = parse_pos(tok)
            i += 1
            continue

        if typ == "Pitch" and i + 2 < len(x_tokens_1024):
            vtok = id_to_token.get(int(x_tokens_1024[i + 1]), "")
            dtok = id_to_token.get(int(x_tokens_1024[i + 2]), "")
            if vtok.startswith("Velocity_") and dtok.startswith("Duration_"):
                vel = parse_vel(vtok)
                dur = _parse_duration_beats(dtok)

                onsets[cur_bar] += 1
                pos_has_onset[cur_bar].add(cur_pos)
                velocities[cur_bar].append(vel)

                s = cur_bar * 4.0 + (cur_pos * 0.25)
                e = s + dur
                global_intervals.append((s, e))

                i += 3
                continue

        i += 1

    attrs: list[list[float]] = []
    for b in range(bars_per_sample):
        note_density = float(onsets[b])
        rhythmic_intensity = float(len(pos_has_onset[b]) / 16.0)
        vel_std = float(np.std(np.asarray(velocities[b], dtype=float), ddof=0)) if velocities[b] else 0.0

        poly_samples = []
        for p in range(16):
            t = b * 4.0 + (p * 0.25)
            active = 0
            for s, e in global_intervals:
                if s <= t < e:
                    active += 1
            poly_samples.append(active)
        polyphony_rate = float(sum(poly_samples) / 16.0)

        attrs.append([polyphony_rate, rhythmic_intensity, vel_std, note_density])
    return attrs


def fit_quantile_thresholds(train_raw_attrs: list[list[float]], num_bins: int = 8) -> dict[str, list[float]]:
    if num_bins != 8:
        raise ValueError("Spec requires 8 bins (0..7)")
    arr = np.asarray(train_raw_attrs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected shape [N,4], got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("No training rows provided for quantile fitting")

    quantiles = [i / num_bins for i in range(1, num_bins)]
    thresholds: dict[str, list[float]] = {}
    for j, name in enumerate(ATTRIBUTE_NAMES):
        th = np.quantile(arr[:, j], quantiles, method="linear")
        thresholds[name] = [float(x) for x in th.tolist()]
    return thresholds


def bin_attributes(raw_attrs_barsx4: list[list[float]], thresholds: dict[str, list[float]], bars_per_sample: int) -> list[list[int]]:
    arr = np.asarray(raw_attrs_barsx4, dtype=float)
    if arr.ndim != 2 or arr.shape != (bars_per_sample, 4):
        raise ValueError(f"Expected shape [{bars_per_sample},4], got {arr.shape}")

    out = np.zeros_like(arr, dtype=int)
    for j, name in enumerate(ATTRIBUTE_NAMES):
        th = np.asarray(thresholds[name], dtype=float)
        if th.shape != (7,):
            raise ValueError(f"Expected 7 thresholds for {name}, got shape {th.shape}")
        out[:, j] = np.digitize(arr[:, j], th, right=False)
    return out.astype(int).tolist()


# -----------------------------
# Worker globals + init
# -----------------------------

W_TOKENIZER = None
W_ID_TO_TOKEN = None
W_THRESHOLDS = None
W_PAD_ID = None
W_BAR_ID = None
W_PITCH_RANGE = None
W_BARS = None
W_TRANSPOSITIONS = None


def _worker_init(pitch_range, num_velocities, bar_id, pad_id, quantiles_json, bars_per_sample, transpositions):
    global W_TOKENIZER, W_ID_TO_TOKEN, W_THRESHOLDS, W_PAD_ID, W_BAR_ID, W_PITCH_RANGE, W_BARS, W_TRANSPOSITIONS

    import inspect
    from miditok import REMI as _REMI, TokenizerConfig as _TokenizerConfig

    W_PAD_ID = int(pad_id)
    W_BAR_ID = int(bar_id)
    W_PITCH_RANGE = tuple(pitch_range)
    W_BARS = int(bars_per_sample)
    W_TRANSPOSITIONS = [int(x) for x in transpositions]

    sig = inspect.signature(_TokenizerConfig)
    params = set(sig.parameters.keys())

    cfg_kwargs = dict(
        pitch_range=tuple(pitch_range),
        beat_res={(0, 4): 4, (4, 8): 2},
        num_velocities=int(num_velocities),
        use_time_signatures=False,
        use_tempos=True,
        use_sustain_pedals=False,
    )
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

    W_TOKENIZER = _REMI(_TokenizerConfig(**cfg_kwargs))
    if len(W_TOKENIZER.vocab) != 195:
        raise RuntimeError(f"Tokenizer vocab mismatch in worker: {len(W_TOKENIZER.vocab)}")
    W_ID_TO_TOKEN = {i: s for (s, i) in W_TOKENIZER.vocab.items()}

    payload = json.loads(Path(quantiles_json).read_text(encoding="utf-8"))
    W_THRESHOLDS = payload["thresholds"]


def _worker_tokenize(midi_bytes: bytes) -> list[int]:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        seq = W_TOKENIZER.encode(tmp.name)
    if isinstance(seq, list):
        if len(seq) != 1:
            raise RuntimeError(f"Expected 1 token stream, got {len(seq)}")
        seq = seq[0]
    ids = getattr(seq, "ids", None)
    if ids is None:
        raise RuntimeError("miditok returned an object without `.ids`")
    return [int(x) for x in ids]


def _worker_strip_silence(midi_bytes: bytes) -> bytes:
    from miditoolkit import MidiFile

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        midi = MidiFile(tmp.name)

    min_start = None
    for inst in midi.instruments:
        for n in inst.notes:
            if min_start is None or n.start < min_start:
                min_start = n.start
    if min_start is None or min_start <= 0:
        return midi_bytes

    shift = int(min_start)
    for inst in midi.instruments:
        for n in inst.notes:
            n.start = max(0, n.start - shift)
            n.end = max(0, n.end - shift)
        for cc in getattr(inst, "control_changes", []) or []:
            cc.time = max(0, cc.time - shift)
        for pb in getattr(inst, "pitch_bends", []) or []:
            pb.time = max(0, pb.time - shift)

    for tempo in getattr(midi, "tempo_changes", []) or []:
        tempo.time = max(0, tempo.time - shift)
    for ts in getattr(midi, "time_signature_changes", []) or []:
        ts.time = max(0, ts.time - shift)
    for ks in getattr(midi, "key_signature_changes", []) or []:
        ks.time = max(0, ks.time - shift)

    return _midifile_to_midibytes(midi)


def _worker_transpose(midi_bytes: bytes, semitones: int) -> Optional[bytes]:
    if semitones == 0:
        return midi_bytes
    from miditoolkit import MidiFile

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        midi = MidiFile(tmp.name)

    lo, hi = W_PITCH_RANGE
    for inst in midi.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            p = n.pitch + semitones
            if p < lo or p > hi:
                return None

    for inst in midi.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            n.pitch += semitones

    return _midifile_to_midibytes(midi)


def _worker_split_bars(ids: list[int]) -> list[list[int]]:
    return split_into_bar_chunks(ids, W_BAR_ID, W_BARS)


def _worker_enforce_block(ch: list[int]) -> Optional[list[int]]:
    return enforce_block_size(ch, W_PAD_ID, BLOCK_SIZE)


def _worker_build_xy(x: list[int]) -> tuple[list[int], list[int]]:
    return build_xy(x, W_PAD_ID)


def _worker_bar_indices(x: list[int]) -> list[int]:
    return compute_bar_indices(x, W_BAR_ID, W_BARS)


def _worker_raw_attrs(x_1024: list[int]) -> list[list[float]]:
    return compute_raw_attributes_per_bar_from_tokens(
        x_1024,
        id_to_token=W_ID_TO_TOKEN,
        bar_token_id=W_BAR_ID,
        bars_per_sample=W_BARS,
    )


def _worker_bin_attrs(raw_barsx4: list[list[float]]) -> list[list[int]]:
    return bin_attributes(raw_barsx4, W_THRESHOLDS, W_BARS)


def process_piece(zip_path: str, member: str) -> dict:
    with zipfile.ZipFile(zip_path, "r") as z:
        b = z.read(member)

    b = _worker_strip_silence(b)

    ids_base = _worker_tokenize(b)
    bar_count = sum(1 for t in ids_base if t == W_BAR_ID)
    dur_sec_piece = midi_duration_seconds(b)
    sec_per_bar = (dur_sec_piece / bar_count) if bar_count > 0 else 0.0

    examples = []
    chunks_total = 0
    chunks_kept = 0
    chunks_filtered_out = 0
    transpositions_kept = 0

    for st in (W_TRANSPOSITIONS or TRANSPOSITIONS):
        bt = _worker_transpose(b, st)
        if bt is None:
            continue
        transpositions_kept += 1

        ids = _worker_tokenize(bt)
        chunks = _worker_split_bars(ids)
        for ch in chunks:
            chunks_total += 1
            xb = _worker_enforce_block(ch)
            if xb is None:
                continue
            chunks_kept += 1

            X, Y = _worker_build_xy(xb)
            BI = _worker_bar_indices(X)
            raw = _worker_raw_attrs(X)
            if not chunk_passes_filter(raw, bars_per_sample=W_BARS):
                chunks_filtered_out += 1
                continue
            attrs = _worker_bin_attrs(raw)
            examples.append({"X": X, "Y": Y, "bar_indices": BI, "attributes": attrs})

    return {
        "member": member,
        "bar_count": bar_count,
        "sec_per_bar": float(sec_per_bar),
        "examples": examples,
        "chunks_total": chunks_total,
        "chunks_kept": chunks_kept,
        "chunks_filtered_out": chunks_filtered_out,
        "transpositions_kept": transpositions_kept,
    }


# -----------------------------
# Quantiles (fit if missing)
# -----------------------------


def fit_and_write_quantiles(
    *,
    out_json: Path,
    split_keys: list[tuple[str, str]],
    tokenizer,
    pad_id: int,
    bar_id: int,
    max_pieces: Optional[int],
):
    p = paths()
    data_dir = p["data"]

    id_to_token = {i: s for (s, i) in tokenizer.vocab.items()}

    rows: list[list[float]] = []
    keys = split_keys if max_pieces is None else split_keys[:max_pieces]
    for zip_name, member in tqdm(keys, desc="fit quantiles (train base)", total=len(keys)):
        b = load_midi_from_zip(data_dir / zip_name, member)
        b = strip_leading_silence(b)
        ids = tokenize_midi_bytes(tokenizer, b)
        chunks = split_into_bar_chunks(ids, bar_id, BARS_PER_SAMPLE)
        for ch in chunks:
            xb = enforce_block_size(ch, pad_id, BLOCK_SIZE)
            if xb is None:
                continue
            raw = compute_raw_attributes_per_bar_from_tokens(
                xb, id_to_token=id_to_token, bar_token_id=bar_id, bars_per_sample=BARS_PER_SAMPLE
            )
            if not chunk_passes_filter(raw, bars_per_sample=BARS_PER_SAMPLE):
                continue
            rows.extend(raw)

    thresholds = fit_quantile_thresholds(rows, num_bins=8)
    payload = {"num_bins": 8, "attribute_names": ATTRIBUTE_NAMES, "thresholds": thresholds}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return thresholds


# -----------------------------
# Split runner
# -----------------------------


def run_split_parallel(
    *,
    split_name: str,
    keys: list[tuple[str, str]],
    thresholds_json: Path,
    out_pt: Path,
    tmp_dir: Path,
    shard_size: int,
    workers: int,
    transpositions: list[int],
    write_test_txt: bool,
    test_txt_dir: Path,
    id_to_token: dict[int, str],
    tokenizer_pad_id: int,
    tokenizer_bar_id: int,
) -> dict:
    p = paths()
    data_dir = p["data"]

    rm_if_exists(out_pt)
    rm_if_exists(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    # test-only dumps
    fx = fy = fbi = fa = None
    if write_test_txt:
        test_txt_dir.mkdir(parents=True, exist_ok=True)
        fx = (test_txt_dir / "test.X.txt").open("w", encoding="utf-8")
        fy = (test_txt_dir / "test.Y.txt").open("w", encoding="utf-8")
        fbi = (test_txt_dir / "test.bar_indices.txt").open("w", encoding="utf-8")
        fa = (test_txt_dir / "test.attributes.txt").open("w", encoding="utf-8")

    shard_idx = 0
    buf_X: list[list[int]] = []
    buf_Y: list[list[int]] = []
    buf_BI: list[list[int]] = []
    buf_A: list[list[list[int]]] = []

    pieces_done = 0
    chunks_total = 0
    chunks_kept = 0
    chunks_filtered_out = 0
    total_seconds = 0.0

    futures: dict = {}
    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(
            PITCH_RANGE,
            NUM_VELOCITIES,
            tokenizer_bar_id,
            tokenizer_pad_id,
            str(thresholds_json),
            BARS_PER_SAMPLE,
            transpositions,
        ),
    ) as ex:
        # Bounded submission to avoid RAM spikes from thousands of in-flight futures.
        max_in_flight = max(32, int(workers) * 4)
        submit_idx = 0

        def submit_one(i: int) -> None:
            zip_name, member = keys[i]
            zip_path = str(data_dir / zip_name)
            futures[ex.submit(process_piece, zip_path, member)] = i

        while submit_idx < len(keys) and len(futures) < max_in_flight:
            submit_one(submit_idx)
            submit_idx += 1

        pending: dict[int, dict] = {}
        next_idx = 0

        pbar = tqdm(total=len(keys), desc=f"{split_name} pieces")
        try:
            while futures:
                done, _not_done = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = futures.pop(fut)
                    pending[idx] = fut.result()

                while submit_idx < len(keys) and len(futures) < max_in_flight:
                    submit_one(submit_idx)
                    submit_idx += 1

                while next_idx in pending:
                    r = pending.pop(next_idx)
                    next_idx += 1
                    pieces_done += 1

                    chunks_total += int(r["chunks_total"])
                    chunks_kept += int(r["chunks_kept"])
                    chunks_filtered_out += int(r.get("chunks_filtered_out", 0))

                    # duration estimate from worker-reported seconds per bar (avoids re-loading MIDI here)
                    sec_per_bar = float(r.get("sec_per_bar", 0.0))

                    for exm in r["examples"]:
                        buf_X.append(exm["X"])
                        buf_Y.append(exm["Y"])
                        buf_BI.append(exm["bar_indices"])
                        buf_A.append(exm["attributes"])
                        total_seconds += BARS_PER_SAMPLE * sec_per_bar

                        if write_test_txt:
                            assert fx and fy and fbi and fa
                            fx.write(" ".join(id_to_token.get(int(i), "UNK") for i in exm["X"]) + "\n")
                            fy.write(" ".join(id_to_token.get(int(i), "UNK") for i in exm["Y"]) + "\n")
                            fbi.write(" ".join(str(int(i)) for i in exm["bar_indices"]) + "\n")
                            flat_attrs = [str(int(v)) for row in exm["attributes"] for v in row]
                            fa.write(" ".join(flat_attrs) + "\n")

                        if len(buf_X) >= shard_size:
                            shard_path = tmp_dir / f"shard_{shard_idx:05d}.pt"
                            torch.save(
                                {
                                    "X": torch.tensor(buf_X, dtype=torch.long),
                                    "Y": torch.tensor(buf_Y, dtype=torch.long),
                                    "bar_indices": torch.tensor(buf_BI, dtype=torch.long),
                                    "attributes": torch.tensor(buf_A, dtype=torch.long),
                                },
                                shard_path,
                            )
                            shard_idx += 1
                            buf_X, buf_Y, buf_BI, buf_A = [], [], [], []

                    pbar.update(1)
                    pbar.set_postfix(
                        examples=(chunks_kept),
                        kept_ratio=(chunks_kept / chunks_total) if chunks_total else 0.0,
                    )
        finally:
            pbar.close()

    if buf_X:
        shard_path = tmp_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(
            {
                "X": torch.tensor(buf_X, dtype=torch.long),
                "Y": torch.tensor(buf_Y, dtype=torch.long),
                "bar_indices": torch.tensor(buf_BI, dtype=torch.long),
                "attributes": torch.tensor(buf_A, dtype=torch.long),
            },
            shard_path,
        )
        shard_idx += 1

    shard_files = sorted(tmp_dir.glob("shard_*.pt"))

    # Low-RAM merge: 2-pass (count then fill) to avoid holding all shards in memory.
    total_n = 0
    shard_ns: list[int] = []
    for sp in shard_files:
        sh = torch.load(sp, map_location="cpu")
        n = int(sh["X"].shape[0])
        shard_ns.append(n)
        total_n += n
        del sh

    X = torch.empty((total_n, BLOCK_SIZE), dtype=torch.long)
    Y = torch.empty((total_n, BLOCK_SIZE), dtype=torch.long)
    BI = torch.empty((total_n, BLOCK_SIZE), dtype=torch.long)
    A = torch.empty((total_n, BARS_PER_SAMPLE, 4), dtype=torch.long)

    offset = 0
    for sp, n in tqdm(list(zip(shard_files, shard_ns)), desc=f"merge {split_name}", total=len(shard_files)):
        sh = torch.load(sp, map_location="cpu")
        X[offset : offset + n].copy_(sh["X"])
        Y[offset : offset + n].copy_(sh["Y"])
        BI[offset : offset + n].copy_(sh["bar_indices"])
        A[offset : offset + n].copy_(sh["attributes"])
        offset += n
        del sh

    meta = {
        "split": split_name,
        "num_examples": int(X.shape[0]),
        "tokens_per_example": BLOCK_SIZE,
        "total_tokens": int(X.shape[0]) * BLOCK_SIZE,
        "estimated_seconds": float(total_seconds),
        "estimated_hours": float(total_seconds / 3600.0),
        "pieces_seen": int(len(keys)),
        "transpositions_attempted": int(len(transpositions)),
        "chunks_total": int(chunks_total),
        "chunks_kept": int(chunks_kept),
        "kept_ratio": float(chunks_kept / chunks_total) if chunks_total else 0.0,
        "chunks_filtered_out": int(chunks_filtered_out),
        "filter_ratio": float(chunks_kept / (chunks_kept + chunks_filtered_out))
        if (chunks_kept + chunks_filtered_out)
        else 0.0,
        "bars_per_sample": int(BARS_PER_SAMPLE),
    }

    torch.save({"X": X, "Y": Y, "bar_indices": BI, "attributes": A, "meta": meta}, out_pt)

    if write_test_txt:
        assert fx and fy and fbi and fa
        fx.close()
        fy.close()
        fbi.close()
        fa.close()

    return meta


def run_split_memmap(
    *,
    split_name: str,
    keys: list[tuple[str, str]],
    thresholds_json: Path,
    out_dir: Path,
    workers: int,
    shard_size: int,
    transpositions: list[int],
    tokenizer_pad_id: int,
    tokenizer_bar_id: int,
    pad_id: int,
    bar_id: int,
) -> dict:
    """Run a split and stream outputs directly to memmaps.

    Writes:
      out_dir/<split_name>/{X.u16.memmap, bar_indices.u8.memmap, attributes.u8.memmap, meta.json}
    """
    p = paths()
    data_dir = p["data"]

    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    writer = MemmapAppendWriter(split_dir, x_dtype=np.uint16, block_size=BLOCK_SIZE, bars_per_sample=BARS_PER_SAMPLE)

    chunks_total = 0
    chunks_kept = 0
    chunks_filtered_out = 0
    total_seconds = 0.0

    futures: dict = {}
    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(
            PITCH_RANGE,
            NUM_VELOCITIES,
            tokenizer_bar_id,
            tokenizer_pad_id,
            str(thresholds_json),
            BARS_PER_SAMPLE,
            transpositions,
        ),
    ) as ex:
        max_in_flight = max(32, int(workers) * 4)
        submit_idx = 0

        def submit_one(i: int) -> None:
            zip_name, member = keys[i]
            zip_path = str(data_dir / zip_name)
            futures[ex.submit(process_piece, zip_path, member)] = i

        while submit_idx < len(keys) and len(futures) < max_in_flight:
            submit_one(submit_idx)
            submit_idx += 1

        pending: dict[int, dict] = {}
        next_idx = 0
        pbar = tqdm(total=len(keys), desc=f"{split_name} pieces (memmap)")
        try:
            while futures:
                done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = futures.pop(fut)
                    pending[idx] = fut.result()

                while submit_idx < len(keys) and len(futures) < max_in_flight:
                    submit_one(submit_idx)
                    submit_idx += 1

                while next_idx in pending:
                    r = pending.pop(next_idx)
                    next_idx += 1

                    chunks_total += int(r["chunks_total"])
                    chunks_kept += int(r["chunks_kept"])
                    chunks_filtered_out += int(r.get("chunks_filtered_out", 0))
                    sec_per_bar = float(r.get("sec_per_bar", 0.0))

                    for exm in r["examples"]:
                        writer.append(exm["X"], exm["bar_indices"], exm["attributes"])
                        total_seconds += BARS_PER_SAMPLE * sec_per_bar

                    pbar.update(1)
                    pbar.set_postfix(examples=writer.n, kept_ratio=(chunks_kept / chunks_total) if chunks_total else 0.0)
        finally:
            pbar.close()

    writer.finalize()

    meta = {
        "split": split_name,
        "num_examples": int(writer.n),
        "tokens_per_example": int(BLOCK_SIZE),
        "total_tokens": int(writer.n) * int(BLOCK_SIZE),
        "estimated_seconds": float(total_seconds),
        "estimated_hours": float(total_seconds / 3600.0),
        "pieces_seen": int(len(keys)),
        "transpositions_attempted": int(len(transpositions)),
        "chunks_total": int(chunks_total),
        "chunks_kept": int(chunks_kept),
        "kept_ratio": float(chunks_kept / chunks_total) if chunks_total else 0.0,
        "chunks_filtered_out": int(chunks_filtered_out),
        "filter_ratio": float(chunks_kept / (chunks_kept + chunks_filtered_out))
        if (chunks_kept + chunks_filtered_out)
        else 0.0,
        "bars_per_sample": int(BARS_PER_SAMPLE),
        "memmap": {
            "X": str(writer.x_path),
            "bar_indices": str(writer.bi_path),
            "attributes": str(writer.a_path),
            "X_dtype": "uint16",
            "bar_indices_dtype": "uint8",
            "attributes_dtype": "uint8",
        },
        "token_ids": {"pad_id": int(pad_id), "bar_id": int(bar_id)},
        "notes": "Y is not stored; derive as shift(X)+pad at training time.",
    }
    (split_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pieces", type=int, default=None, help="Limit pieces per split (smoke test).")
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4))
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument(
        "--out-subdir",
        type=str,
        default="preprocessed",
        help="Artifacts subdir for outputs (default: artifacts/preprocessed). Use e.g. preprocessed_smoke2 for tests.",
    )
    ap.add_argument(
        "--output-format",
        type=str,
        default="pt",
        choices=("pt", "memmap"),
        help="Output format: 'pt' (default) or 'memmap' (Kaggle-friendly streaming arrays).",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated subset of splits to run, e.g. 'val,test'. Default runs all.",
    )
    args = ap.parse_args()

    p = paths()
    splits_dir: Path = p["splits"]
    attr_dir: Path = p["attributes"]
    preproc_dir: Path = p["artifacts"] / args.out_subdir

    tok, pad_id, _bos_id, bar_id = make_tokenizer()

    train_keys = load_split(splits_dir / "train.jsonl")
    val_keys = load_split(splits_dir / "val.jsonl")
    test_keys = load_split(splits_dir / "test.jsonl")

    if args.max_pieces is not None:
        train_keys = train_keys[: args.max_pieces]
        val_keys = val_keys[: args.max_pieces]
        test_keys = test_keys[: args.max_pieces]

    q_path = attr_dir / "quantiles.json"
    if q_path.exists():
        payload = json.loads(q_path.read_text(encoding="utf-8"))
    else:
        thresholds = fit_and_write_quantiles(
            out_json=q_path,
            split_keys=train_keys,
            tokenizer=tok,
            pad_id=pad_id,
            bar_id=bar_id,
            max_pieces=None if args.max_pieces is None else args.max_pieces,
        )
        payload = {"num_bins": 8, "attribute_names": ATTRIBUTE_NAMES, "thresholds": thresholds}

    # ensure quantiles file exists for workers
    if not q_path.exists():
        q_path.parent.mkdir(parents=True, exist_ok=True)
        q_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    splits_wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
    id_to_token = {i: s for (s, i) in tok.vocab.items()}
    transpositions_train = list(TRANSPOSITIONS)
    transpositions_eval = [0]

    if args.output_format == "pt":
        if "train" in splits_wanted:
            meta_train = run_split_parallel(
                split_name="train",
                keys=train_keys,
                thresholds_json=q_path,
                out_pt=preproc_dir / "train.pt",
                tmp_dir=preproc_dir / "_tmp_train",
                shard_size=args.shard_size,
                workers=args.workers,
                transpositions=transpositions_train,
                write_test_txt=False,
                test_txt_dir=preproc_dir,
                id_to_token=id_to_token,
                tokenizer_pad_id=pad_id,
                tokenizer_bar_id=bar_id,
            )
            print("train meta:", meta_train)

        if "val" in splits_wanted:
            meta_val = run_split_parallel(
                split_name="val",
                keys=val_keys,
                thresholds_json=q_path,
                out_pt=preproc_dir / "val.pt",
                tmp_dir=preproc_dir / "_tmp_val",
                shard_size=args.shard_size,
                workers=args.workers,
                transpositions=transpositions_eval,
                write_test_txt=False,
                test_txt_dir=preproc_dir,
                id_to_token=id_to_token,
                tokenizer_pad_id=pad_id,
                tokenizer_bar_id=bar_id,
            )
            print("val meta:", meta_val)

        if "test" in splits_wanted:
            for name in ["test.X.txt", "test.Y.txt", "test.bar_indices.txt", "test.attributes.txt"]:
                rm_if_exists(preproc_dir / name)
            meta_test = run_split_parallel(
                split_name="test",
                keys=test_keys,
                thresholds_json=q_path,
                out_pt=preproc_dir / "test.pt",
                tmp_dir=preproc_dir / "_tmp_test",
                shard_size=args.shard_size,
                workers=args.workers,
                transpositions=transpositions_eval,
                write_test_txt=True,
                test_txt_dir=preproc_dir,
                id_to_token=id_to_token,
                tokenizer_pad_id=pad_id,
                tokenizer_bar_id=bar_id,
            )
            print("test meta:", meta_test)
    else:
        # memmap output
        if "train" in splits_wanted:
            meta_train = run_split_memmap(
                split_name="train",
                keys=train_keys,
                thresholds_json=q_path,
                out_dir=preproc_dir,
                workers=args.workers,
                shard_size=args.shard_size,
                transpositions=transpositions_train,
                tokenizer_pad_id=pad_id,
                tokenizer_bar_id=bar_id,
                pad_id=pad_id,
                bar_id=bar_id,
            )
            print("train meta:", meta_train)
        if "val" in splits_wanted:
            meta_val = run_split_memmap(
                split_name="val",
                keys=val_keys,
                thresholds_json=q_path,
                out_dir=preproc_dir,
                workers=args.workers,
                shard_size=args.shard_size,
                transpositions=transpositions_eval,
                tokenizer_pad_id=pad_id,
                tokenizer_bar_id=bar_id,
                pad_id=pad_id,
                bar_id=bar_id,
            )
            print("val meta:", meta_val)
        if "test" in splits_wanted:
            meta_test = run_split_memmap(
                split_name="test",
                keys=test_keys,
                thresholds_json=q_path,
                out_dir=preproc_dir,
                workers=args.workers,
                shard_size=args.shard_size,
                transpositions=transpositions_eval,
                tokenizer_pad_id=pad_id,
                tokenizer_bar_id=bar_id,
                pad_id=pad_id,
                bar_id=bar_id,
            )
            print("test meta:", meta_test)


if __name__ == "__main__":
    main()

