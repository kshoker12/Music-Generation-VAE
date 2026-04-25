from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _load_tokenizer_vocab_size() -> int:
    """
    Reuse preprocessing tokenizer config to get vocab size.
    We avoid importing the whole package layout assumptions by loading the script module directly.
    """
    import importlib.util

    p = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_all.py"
    spec = importlib.util.spec_from_file_location("preprocess_all", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    tok, _pad_id, _bos_id, _bar_id = mod.make_tokenizer()
    return int(len(tok.vocab))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-dir",
        type=str,
        default="artifacts/preprocessed_memmap_toy/train",
        help="Path to a memmap split dir containing meta.json and *.memmap files.",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()

    # Ensure we can import musicgen from ml/src when run from repo root.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml" / "src"))

    from musicgen.data.memmap_dataset import MemmapMusicDataset, make_dataloader
    from musicgen.models.plain_transformer import PlainTransformerConfig, PlainTransformerDecoder

    ds = MemmapMusicDataset(args.split_dir)
    dl = make_dataloader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(dl))

    vocab_size = _load_tokenizer_vocab_size()
    cfg = PlainTransformerConfig(vocab_size=vocab_size)
    model = PlainTransformerDecoder(cfg).to(args.device)
    model.eval()

    X = batch["X"].to(args.device)
    Y = batch["Y"].to(args.device)
    bar_indices = batch["bar_indices"].to(args.device)
    attributes = batch["attributes"].to(args.device)

    with torch.no_grad():
        out = model(X, bar_indices, attributes, targets=Y, pad_id=ds.meta.pad_id)

    logits = out["logits"]
    loss = out["loss"]
    n_tokens = out["n_tokens"]
    print(f"OK: logits={tuple(logits.shape)} loss={float(loss):.6f} n_tokens={n_tokens} vocab={vocab_size}")


if __name__ == "__main__":
    main()

