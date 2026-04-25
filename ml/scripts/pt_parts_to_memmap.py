"""Convert preprocessed `.pt` part files into Kaggle-friendly memory-mapped arrays.

Expected input (existing artifacts):
- artifacts/preprocessed/train.pt : manifest with meta.parts pointing to train_parts/train.part_*.pt
  OR you can pass --parts-dir directly.

Output layout (default):
artifacts/preprocessed_memmap/<split>/
  - X.u16.memmap           shape [N, 1024] dtype uint16
  - bar_indices.u8.memmap  shape [N, 1024] dtype uint8
  - attributes.u8.memmap   shape [N, 8, 4] dtype uint8
  - meta.json

Notes:
- Does NOT store Y; compute Y on the fly in the DataLoader: Y = shift(X) + pad_id at end.
- Streaming conversion: never loads full dataset into RAM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


def _load_preprocess_all():
    here = Path(__file__).resolve()
    path = here.parent / "preprocess_all.py"
    spec = importlib.util.spec_from_file_location("preprocess_all", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _parts_from_manifest(manifest_pt: Path) -> list[Path]:
    obj = torch.load(manifest_pt, map_location="cpu")
    meta = obj.get("meta") if isinstance(obj, dict) else None
    if not isinstance(meta, dict) or "parts" not in meta:
        raise RuntimeError(f"{manifest_pt} does not look like a parts manifest (missing meta.parts)")
    parts = meta["parts"]
    if not isinstance(parts, list) or not parts:
        raise RuntimeError(f"{manifest_pt} meta.parts is empty")
    return [Path(p) for p in parts]


def _parts_from_dir(parts_dir: Path) -> list[Path]:
    parts = sorted(parts_dir.glob("*.pt"))
    if not parts:
        raise RuntimeError(f"No .pt parts found in {parts_dir}")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    ap.add_argument("--manifest-pt", type=str, default=None, help="Path to split manifest .pt with meta.parts")
    ap.add_argument("--parts-dir", type=str, default=None, help="Directory containing split part .pt files")
    ap.add_argument("--out-root", type=str, default="artifacts/preprocessed_memmap", help="Output root directory")
    ap.add_argument("--dtype-x", type=str, default="uint16", choices=("uint16", "uint8"))
    args = ap.parse_args()

    P = _load_preprocess_all()
    repo = P.repo_root()

    if args.manifest_pt is None and args.parts_dir is None:
        # default to train manifest produced by the resume merge
        args.manifest_pt = str(repo / "artifacts" / "preprocessed" / f"{args.split}.pt")

    if args.manifest_pt is not None:
        parts = _parts_from_manifest(Path(args.manifest_pt))
    else:
        parts = _parts_from_dir(Path(args.parts_dir))

    out_dir = (repo / args.out_root / args.split).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer ids (for meta + downstream)
    tok, pad_id, _bos_id, bar_id = P.make_tokenizer()
    pad_id = int(pad_id)
    bar_id = int(bar_id)
    vocab_size = int(len(tok.vocab))

    # pass 1: count N + confirm shapes
    total_n = 0
    block_size = None
    bars = None
    for sp in tqdm(parts, desc="count parts"):
        sh = torch.load(sp, map_location="cpu")
        X = sh["X"]
        A = sh["attributes"]
        n = int(X.shape[0])
        total_n += n
        if block_size is None:
            block_size = int(X.shape[1])
            bars = int(A.shape[1])
        del sh

    if block_size is None or bars is None:
        raise RuntimeError("Could not infer shapes from parts")

    if int(block_size) != int(P.BLOCK_SIZE):
        raise RuntimeError(f"BLOCK_SIZE mismatch: parts have {block_size}, preprocess_all has {P.BLOCK_SIZE}")
    if int(bars) != int(P.BARS_PER_SAMPLE):
        raise RuntimeError(f"BARS_PER_SAMPLE mismatch: parts have {bars}, preprocess_all has {P.BARS_PER_SAMPLE}")

    x_dtype = np.uint16 if args.dtype_x == "uint16" else np.uint8
    bi_dtype = np.uint8
    a_dtype = np.uint8

    x_path = out_dir / ("X.u16.memmap" if x_dtype == np.uint16 else "X.u8.memmap")
    bi_path = out_dir / "bar_indices.u8.memmap"
    a_path = out_dir / "attributes.u8.memmap"

    # allocate memmaps
    _ensure_parent(x_path)
    _ensure_parent(bi_path)
    _ensure_parent(a_path)

    X_mm = np.memmap(x_path, dtype=x_dtype, mode="w+", shape=(total_n, block_size))
    BI_mm = np.memmap(bi_path, dtype=bi_dtype, mode="w+", shape=(total_n, block_size))
    A_mm = np.memmap(a_path, dtype=a_dtype, mode="w+", shape=(total_n, bars, 4))

    # pass 2: stream copy
    offset = 0
    for sp in tqdm(parts, desc="write memmaps"):
        sh = torch.load(sp, map_location="cpu")
        X = sh["X"].to(torch.int64).cpu().numpy()
        BI = sh["bar_indices"].to(torch.int64).cpu().numpy()
        A = sh["attributes"].to(torch.int64).cpu().numpy()
        n = int(X.shape[0])

        # cast and write
        X_mm[offset : offset + n, :] = X.astype(x_dtype, copy=False)
        BI_mm[offset : offset + n, :] = BI.astype(bi_dtype, copy=False)
        A_mm[offset : offset + n, :, :] = A.astype(a_dtype, copy=False)
        offset += n
        del sh

    X_mm.flush()
    BI_mm.flush()
    A_mm.flush()

    meta = {
        "split": args.split,
        "num_examples": int(total_n),
        "block_size": int(block_size),
        "bars_per_sample": int(bars),
        "vocab_size": int(vocab_size),
        "token_ids": {"pad_id": int(pad_id), "bar_id": int(bar_id)},
        "paths": {
            "X": str(x_path),
            "bar_indices": str(bi_path),
            "attributes": str(a_path),
        },
        "dtypes": {
            "X": str(np.dtype(x_dtype)),
            "bar_indices": str(np.dtype(bi_dtype)),
            "attributes": str(np.dtype(a_dtype)),
        },
        "notes": [
            "Y is not stored; derive as shift(X) + pad_id at end.",
            "Values are stored as compact integer memmaps for Kaggle streaming.",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote memmaps to", out_dir)
    print("meta:", json.dumps({k: meta[k] for k in ['split','num_examples','block_size','bars_per_sample']}, indent=2))


if __name__ == "__main__":
    main()

