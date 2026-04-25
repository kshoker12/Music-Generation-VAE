from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch


def _load_vocab_size_and_token_ids() -> tuple[int, int]:
    """
    Load tokenizer vocab size + pad_id via preprocess_all tokenizer config.
    Returns: (vocab_size, pad_id)
    """
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "ml" / "scripts" / "preprocess_all.py"
    spec = importlib.util.spec_from_file_location("preprocess_all", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    tok, pad_id, _bos_id, _bar_id = mod.make_tokenizer()
    return int(len(tok.vocab)), int(pad_id)


@torch.no_grad()
def _eval_loss(model, dl, *, device: torch.device, pad_id: int, max_batches: int) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        X = batch["X"].to(device, non_blocking=True)
        Y = batch["Y"].to(device, non_blocking=True)
        bar_indices = batch["bar_indices"].to(device, non_blocking=True)
        attributes = batch["attributes"].to(device, non_blocking=True)
        out = model(X, bar_indices, attributes, targets=Y, pad_id=pad_id)
        losses.append(float(out["loss"].item()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", type=str, default="artifacts/preprocessed_memmap_toy/train")
    ap.add_argument("--val-split", type=str, default="artifacts/preprocessed_memmap_toy/val")
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=25)
    ap.add_argument("--amp", action="store_true", help="Use AMP (only meaningful on CUDA).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml" / "src"))

    from musicgen.data.memmap_dataset import MemmapMusicDataset, make_dataloader
    from musicgen.models.plain_transformer import PlainTransformerConfig, PlainTransformerDecoder

    vocab_size, pad_id_tok = _load_vocab_size_and_token_ids()

    train_ds = MemmapMusicDataset(args.train_split)
    val_ds = MemmapMusicDataset(args.val_split) if Path(args.val_split).exists() else None

    # Prefer pad_id from split meta (should match tokenizer); keep tokenizer for sanity.
    pad_id = int(train_ds.meta.pad_id)
    if pad_id != pad_id_tok:
        print(f"WARNING: pad_id mismatch: split meta {pad_id} vs tokenizer {pad_id_tok}")

    train_dl = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_dl = (
        make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        if val_ds is not None
        else None
    )

    device = torch.device(args.device)
    cfg = PlainTransformerConfig(vocab_size=vocab_size)
    model = PlainTransformerDecoder(cfg).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # Simple infinite iterator over train_dl
    it = iter(train_dl)

    t0 = time.time()
    ema = None
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)

        X = batch["X"].to(device, non_blocking=True)
        Y = batch["Y"].to(device, non_blocking=True)
        bar_indices = batch["bar_indices"].to(device, non_blocking=True)
        attributes = batch["attributes"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            out = model(X, bar_indices, attributes, targets=Y, pad_id=pad_id)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loss_f = float(loss.item())
        ema = loss_f if ema is None else (0.95 * ema + 0.05 * loss_f)

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            toks = out.get("n_tokens", 0)
            toks_per_s = (toks * args.log_every / dt) if dt > 0 and step % args.log_every == 0 else None
            msg = f"step {step:5d} | loss {loss_f:.4f} | ema {ema:.4f}"
            if toks_per_s is not None:
                msg += f" | tok/s ~{toks_per_s:,.0f}"
            print(msg)
            t0 = time.time()

        if val_dl is not None and (step % args.eval_every == 0 or step == args.steps):
            val_loss = _eval_loss(model, val_dl, device=device, pad_id=pad_id, max_batches=args.eval_batches)
            print(f"eval @ step {step:5d} | val_loss {val_loss:.4f}")

    # Quick overfit sanity suggestion
    print("Done. If loss doesn’t decrease, try smaller batch/overfit 1 batch, or verify FiLM conditioning varies with attributes.")


if __name__ == "__main__":
    main()

