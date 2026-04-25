"""Write human-readable TXT dumps from memmap datasets for quick verification.

Given a memmap split directory containing `meta.json` + memmap files, this script writes:
- dump.X.txt            (space-separated token ids)
- dump.bar_indices.txt  (space-separated bar indices 0..bars-1 per token)
- dump.attributes.txt   (per-example flattened [bars,4] integers; bars*4 numbers)

All values are derived directly from the memmaps (no recomputation from MIDI).
"""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path

import numpy as np


def _load_meta(split_dir: Path) -> dict:
    meta_path = split_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"Missing {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_preprocess_all():
    here = Path(__file__).resolve()
    path = here.parent / "preprocess_all.py"
    spec = importlib.util.spec_from_file_location("preprocess_all", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _id_to_token_map() -> dict[int, str]:
    P = _load_preprocess_all()
    tok, _pad_id, _bos_id, _bar_id = P.make_tokenizer()
    return {int(i): str(s) for (s, i) in tok.vocab.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="artifacts/preprocessed_memmap_toy",
        help="Root directory containing train/val/test split folders.",
    )
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--max-examples", type=int, default=200, help="How many examples to dump per split.")
    ap.add_argument(
        "--x-as-tokens",
        action="store_true",
        help="If set, writes dump.X.txt as vocab token strings instead of integer IDs.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    id_to_tok = _id_to_token_map() if args.x_as_tokens else None

    for split in splits:
        split_dir = root / split
        meta = _load_meta(split_dir)

        N = int(meta.get("num_examples") or meta.get("num_examples".upper(), 0))
        L = int(meta.get("block_size") or meta.get("tokens_per_example") or meta.get("tokens_per_example".upper(), 0) or 0)
        bars = int(meta.get("bars_per_sample") or 0)

        if N <= 0 or L <= 0 or bars <= 0:
            raise RuntimeError(f"Bad meta in {split_dir}/meta.json (N={N}, L={L}, bars={bars})")

        # support both meta formats:
        # - pt_parts_to_memmap.py: meta["paths"] + meta["dtypes"]
        # - preprocess_all.py memmap: meta["memmap"] (paths + dtype strings)
        if "paths" in meta:
            x_path = Path(meta["paths"]["X"])
            bi_path = Path(meta["paths"]["bar_indices"])
            a_path = Path(meta["paths"]["attributes"])
            x_dtype = np.uint16 if str(meta.get("dtypes", {}).get("X")) == "uint16" else np.uint8
        else:
            x_path = Path(meta["memmap"]["X"])
            bi_path = Path(meta["memmap"]["bar_indices"])
            a_path = Path(meta["memmap"]["attributes"])
            x_dtype = np.uint16 if str(meta["memmap"].get("X_dtype", "uint16")) == "uint16" else np.uint8

        X = np.memmap(x_path, dtype=x_dtype, mode="r", shape=(N, L))
        BI = np.memmap(bi_path, dtype=np.uint8, mode="r", shape=(N, L))
        A = np.memmap(a_path, dtype=np.uint8, mode="r", shape=(N, bars, 4))

        k = min(int(args.max_examples), N)

        fx = (split_dir / "dump.X.txt").open("w", encoding="utf-8")
        fbi = (split_dir / "dump.bar_indices.txt").open("w", encoding="utf-8")
        fa = (split_dir / "dump.attributes.txt").open("w", encoding="utf-8")

        try:
            for i in range(k):
                if id_to_tok is None:
                    fx.write(" ".join(str(int(v)) for v in X[i].tolist()) + "\n")
                else:
                    fx.write(" ".join(id_to_tok.get(int(v), "UNK") for v in X[i].tolist()) + "\n")
                fbi.write(" ".join(str(int(v)) for v in BI[i].tolist()) + "\n")
                flat_attrs = [str(int(v)) for row in A[i].tolist() for v in row]
                fa.write(" ".join(flat_attrs) + "\n")
        finally:
            fx.close()
            fbi.close()
            fa.close()

        print(f"wrote {k} examples to {split_dir}/dump.*.txt")


if __name__ == "__main__":
    main()

