"""A/B compare: baseline MIDI chunks vs tokenize→decode chunks for one random split piece.

Baseline: slice the *preprocessed* MIDI (strip silence + transpose) into bar windows (tick-based, time sig aware).
Decoded: same preprocessed MIDI → tokenize → bar-chunk → block pad/drop → decode to MIDI.

Why baseline uses preprocessed MIDI:
- ``preprocess_all.py`` strips leading silence before tokenization, so slicing the raw original will not align.
- Some token bar-windows are dropped if > BLOCK_SIZE; we align using the kept bar-window indices.

Uses the same helpers as ``preprocess_all.py`` (loaded dynamically so ``python ml/scripts/...`` works).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from miditoolkit import Instrument, MidiFile, Note, TempoChange, TimeSignature


def _load_preprocess_all():
    here = Path(__file__).resolve()
    path = here.parent / "preprocess_all.py"
    spec = importlib.util.spec_from_file_location("preprocess_all", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


P = _load_preprocess_all()


@dataclass(frozen=True)
class PieceRef:
    zip: str
    member: str


@dataclass(frozen=True)
class KeptChunk:
    bar_window_index: int
    x_1024: list[int]
    unpadded_len: int


def _end_tick(midi: MidiFile) -> int:
    end_tick = 0
    for inst in midi.instruments:
        for n in inst.notes:
            end_tick = max(end_tick, int(n.end))
        for cc in getattr(inst, "control_changes", []) or []:
            end_tick = max(end_tick, int(cc.time))
        for pb in getattr(inst, "pitch_bends", []) or []:
            end_tick = max(end_tick, int(pb.time))
    for t in getattr(midi, "tempo_changes", []) or []:
        end_tick = max(end_tick, int(t.time))
    for ts in getattr(midi, "time_signature_changes", []) or []:
        end_tick = max(end_tick, int(ts.time))
    return int(end_tick)


def _sorted_time_sigs(midi: MidiFile) -> list[TimeSignature]:
    tss = list(getattr(midi, "time_signature_changes", []) or [])
    tss.sort(key=lambda x: int(x.time))
    if not tss or int(tss[0].time) != 0:
        tss = [TimeSignature(4, 4, 0)] + tss
    return tss


def _bar_start_ticks(midi: MidiFile) -> list[int]:
    tpq = int(getattr(midi, "ticks_per_beat", 480) or 480)
    tss = _sorted_time_sigs(midi)
    end_tick = _end_tick(midi)
    starts: list[int] = []
    cur_tick = 0
    ts_idx = 0
    while cur_tick <= end_tick:
        while ts_idx + 1 < len(tss) and int(tss[ts_idx + 1].time) <= cur_tick:
            ts_idx += 1
        ts = tss[ts_idx]
        beats_per_bar = float(ts.numerator) * (4.0 / float(ts.denominator))
        bar_len = int(round(beats_per_bar * tpq))
        if bar_len <= 0:
            break
        starts.append(cur_tick)
        cur_tick += bar_len
    return starts


def _last_change_before(changes: list, tick: int, default):
    best = None
    for c in changes:
        if int(c.time) <= tick:
            best = c
        else:
            break
    return best if best is not None else default


def slice_midifile_to_chunk(midi: MidiFile, start_tick: int, end_tick: int) -> MidiFile:
    out = MidiFile()
    out.ticks_per_beat = midi.ticks_per_beat
    out.instruments = []

    for inst in midi.instruments:
        new_inst = Instrument(
            program=int(getattr(inst, "program", 0)),
            is_drum=bool(getattr(inst, "is_drum", False)),
            name=str(getattr(inst, "name", "")),
        )
        for n in inst.notes:
            s, e = int(n.start), int(n.end)
            if e <= start_tick or s >= end_tick:
                continue
            ns = max(s, start_tick) - start_tick
            ne = min(e, end_tick) - start_tick
            if ne <= ns:
                continue
            new_inst.notes.append(Note(velocity=int(n.velocity), pitch=int(n.pitch), start=int(ns), end=int(ne)))

        new_inst.control_changes = []
        for cc in getattr(inst, "control_changes", []) or []:
            t = int(cc.time)
            if start_tick <= t < end_tick:
                cc2 = type(cc)(number=int(getattr(cc, "number", 0)), value=int(getattr(cc, "value", 0)), time=t - start_tick)
                new_inst.control_changes.append(cc2)
        new_inst.pitch_bends = []
        for pb in getattr(inst, "pitch_bends", []) or []:
            t = int(pb.time)
            if start_tick <= t < end_tick:
                pb2 = type(pb)(pitch=int(getattr(pb, "pitch", 0)), time=t - start_tick)
                new_inst.pitch_bends.append(pb2)

        if new_inst.notes or new_inst.control_changes or new_inst.pitch_bends:
            out.instruments.append(new_inst)

    tempos = list(getattr(midi, "tempo_changes", []) or [])
    tempos.sort(key=lambda x: int(x.time))
    base_tempo = _last_change_before(tempos, start_tick, default=TempoChange(tempo=120.0, time=0))
    out.tempo_changes = [TempoChange(tempo=float(base_tempo.tempo), time=0)]
    for t in tempos:
        tt = int(t.time)
        if start_tick <= tt < end_tick:
            out.tempo_changes.append(TempoChange(tempo=float(t.tempo), time=tt - start_tick))

    tss = _sorted_time_sigs(midi)
    base_ts = _last_change_before(tss, start_tick, default=TimeSignature(4, 4, 0))
    out.time_signature_changes = [TimeSignature(int(base_ts.numerator), int(base_ts.denominator), 0)]
    for ts in tss:
        tt = int(ts.time)
        if start_tick <= tt < end_tick:
            out.time_signature_changes.append(TimeSignature(int(ts.numerator), int(ts.denominator), tt - start_tick))

    return out


def load_split_keys(repo: Path, split: str) -> list[PieceRef]:
    rows = P.read_jsonl(repo / "artifacts" / "splits" / f"{split}.jsonl")
    return [PieceRef(zip=r["zip"], member=r["member"]) for r in rows]


def extract_member(repo: Path, ref: PieceRef) -> bytes:
    with zipfile.ZipFile(repo / "data" / ref.zip, "r") as z:
        return z.read(ref.member)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline vs decoded MIDI chunks for one random piece.")
    ap.add_argument("--split", choices=("train", "val", "test"), default="val")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-chunks", type=int, default=5)
    ap.add_argument("--semitones", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=200, help="Resample until enough baseline bars + kept decoded chunks exist.")
    args = ap.parse_args()

    repo = P.repo_root()
    rng = random.Random(args.seed)
    seed_used = args.seed if args.seed is not None else rng.randrange(2**32)
    rng = random.Random(seed_used)

    keys = load_split_keys(repo, args.split)
    tok, pad_id, _bos_id, bar_id = P.make_tokenizer()

    bars = int(P.BARS_PER_SAMPLE)
    ref: PieceRef | None = None
    orig_bytes: bytes | None = None
    proc_bytes: bytes | None = None
    midi_proc: MidiFile | None = None
    bar_starts_proc: list[int] | None = None
    kept_chunks: list[KeptChunk] | None = None

    for _ in range(args.max_tries):
        cand = keys[rng.randrange(len(keys))]
        b0 = extract_member(repo, cand)
        b1 = P.strip_leading_silence(b0)
        bt = P.transpose_midi(b1, int(args.semitones), P.PITCH_RANGE)
        if bt is None:
            continue

        m_proc = P._midibytes_to_midifile(bt)
        bs_proc = _bar_start_ticks(m_proc)
        if len(bs_proc) < (args.num_chunks * bars + 1):
            continue

        ids = P.tokenize_midi_bytes(tok, bt)
        token_chunks = P.split_into_bar_chunks(ids, bar_id, bars)
        kept: list[KeptChunk] = []
        for bar_window_index, ch in enumerate(token_chunks):
            unpadded_len = len(ch)
            xb = P.enforce_block_size(ch, pad_id, P.BLOCK_SIZE)
            if xb is not None:
                kept.append(KeptChunk(bar_window_index=bar_window_index, x_1024=xb, unpadded_len=unpadded_len))
            if len(kept) >= args.num_chunks:
                break
        if len(kept) < args.num_chunks:
            continue

        ref = cand
        orig_bytes = b0
        proc_bytes = bt
        midi_proc = m_proc
        bar_starts_proc = bs_proc
        kept_chunks = kept
        break

    if (
        ref is None
        or orig_bytes is None
        or proc_bytes is None
        or midi_proc is None
        or bar_starts_proc is None
        or kept_chunks is None
    ):
        raise SystemExit(
            f"Could not find a piece with {args.num_chunks} baseline+decoded chunks in {args.max_tries} tries."
        )

    piece_id = f"{Path(ref.zip).stem}__{Path(ref.member).stem}"
    out_dir = repo / "artifacts" / "ab_compare" / piece_id
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_chunks = []
    for i, kept in enumerate(kept_chunks[: args.num_chunks]):
        b0i = kept.bar_window_index * bars
        b1i = (kept.bar_window_index + 1) * bars
        start_tick = bar_starts_proc[b0i]
        end_tick = bar_starts_proc[b1i]
        ch = slice_midifile_to_chunk(midi_proc, start_tick, end_tick)
        out_mid = out_dir / f"baseline_chunk_{i:02d}.mid"
        ch.dump(str(out_mid))
        baseline_chunks.append(
            {
                "i": i,
                "bar_window_index": kept.bar_window_index,
                "start_tick": start_tick,
                "end_tick": end_tick,
                "path": str(out_mid),
            }
        )

    decoded_chunks = []
    for i, kept in enumerate(kept_chunks[: args.num_chunks]):
        ids_1d = list(kept.x_1024)
        while ids_1d and ids_1d[-1] == pad_id:
            ids_1d.pop()
        score = tok.decode([ids_1d])
        out_mid = out_dir / f"decoded_chunk_{i:02d}.mid"
        score.dump_midi(str(out_mid))
        decoded_chunks.append(
            {
                "i": i,
                "bar_window_index": kept.bar_window_index,
                "unpadded_len": kept.unpadded_len,
                "path": str(out_mid),
            }
        )

    manifest = {
        "seed": seed_used,
        "split": args.split,
        "piece": asdict(ref),
        "out_dir": str(out_dir),
        "bars_per_sample": bars,
        "block_size": int(P.BLOCK_SIZE),
        "semitones": int(args.semitones),
        "vocab_size": len(tok.vocab),
        "baseline_chunks": baseline_chunks,
        "decoded_chunks": decoded_chunks,
        "notes": "Baseline = preprocessed MIDI (strip+transpose) sliced by bar ticks, aligned by kept bar-window indices. Decoded = tokenize→chunk→pad/drop→decode.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("wrote:", out_dir)
    print("seed:", seed_used)
    print("piece:", ref.zip, ref.member)
    print("baseline_chunks:", len(baseline_chunks))
    print("decoded_chunks:", len(decoded_chunks))


if __name__ == "__main__":
    main()
