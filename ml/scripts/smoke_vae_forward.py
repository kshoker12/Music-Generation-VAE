from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-dir",
        type=str,
        default="artifacts/preprocessed_memmap_toy/train",
        help="Memmap split dir (train/val/test) with meta.json and memmap files.",
    )
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml" / "src"))

    from musicgen.data.memmap_dataset import MemmapMusicDataset, make_dataloader
    from musicgen.models.vae import MusicVAE, VAEConfig

    ds = MemmapMusicDataset(args.split_dir)
    dl = make_dataloader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(dl))

    device = torch.device(args.device)
    cfg = VAEConfig(vocab_size=195, block_size=ds.meta.tokens_per_example, bars_per_sample=ds.meta.bars_per_sample)
    model = MusicVAE(cfg).to(device)
    model.eval()

    X = batch["X"].to(device)
    Y = batch["Y"].to(device)
    bar_indices = batch["bar_indices"].to(device)
    attributes = batch["attributes"].to(device)

    with torch.no_grad():
        out = model(X, bar_indices, attributes, targets=Y, pad_id=ds.meta.pad_id, beta=args.beta)

    logits = out["logits"]
    assert logits.shape == (args.batch_size, ds.meta.tokens_per_example, cfg.vocab_size), logits.shape
    assert out["mu"].shape == (args.batch_size, cfg.z_dim)
    assert out["logvar"].shape == (args.batch_size, cfg.z_dim)
    assert out["z_k"].shape == (args.batch_size, cfg.bars_per_sample, cfg.cond_hidden)

    loss_recon = float(out["loss_recon"].item())
    loss_kl = float(out["loss_kl"].item())
    loss_total = float(out["loss_total"].item())
    print(
        f"OK: logits={tuple(logits.shape)} recon={loss_recon:.4f} kl={loss_kl:.4f} "
        f"beta={args.beta} total={loss_total:.4f}"
    )


if __name__ == "__main__":
    main()

