#!/usr/bin/env python3
"""Batch-generate MIDI files for MOS evaluation across all three models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import torch

BARS_PER_SAMPLE = 8
BLOCK_SIZE = 1024

ModelType = Literal["plain", "vae", "simple_vae"]
Category = Literal["lullaby", "virtuoso", "crescendo"]

MODELS: list[tuple[str, ModelType, str]] = [
    ("plain_transformer", "plain", "ml/src/musicgen/runs/plain_v3/ckpt.pt"),
    ("simple_vae", "simple_vae", "ml/src/musicgen/runs/simple_v2/ckpt.pt"),
    ("hierarchical_vae", "vae", "ml/src/musicgen/runs/vae_v3/ckpt.pt"),
]

ALL_CATEGORIES: list[Category] = ["lullaby", "virtuoso", "crescendo"]

log = logging.getLogger(__name__)


def _crescendo_bin(bar_idx: int) -> int:
    return int(round(1 + 6 * bar_idx / 7))


def build_lullaby_attributes(sample_idx: int) -> list[list[int]]:
    """Odd samples use bin 1; even samples use bin 2 (all four attributes uniform)."""
    bin_val = 1 if sample_idx % 2 == 1 else 2
    row = [bin_val, bin_val, bin_val, bin_val]
    return [row[:] for _ in range(BARS_PER_SAMPLE)]


def build_virtuoso_attributes(sample_idx: int) -> list[list[int]]:
    """Odd samples use bin 6; even samples use bin 7 (all four attributes uniform)."""
    bin_val = 6 if sample_idx % 2 == 1 else 7
    row = [bin_val, bin_val, bin_val, bin_val]
    return [row[:] for _ in range(BARS_PER_SAMPLE)]


def build_crescendo_attributes(_sample_idx: int) -> list[list[int]]:
    """Polyphony + rhythm fixed at bin 4; velocity + density ramp 1→7 across bars."""
    rows: list[list[int]] = []
    for bar_idx in range(BARS_PER_SAMPLE):
        ramp = _crescendo_bin(bar_idx)
        rows.append([4, 4, ramp, ramp])
    return rows


ATTRIBUTE_BUILDERS = {
    "lullaby": build_lullaby_attributes,
    "virtuoso": build_virtuoso_attributes,
    "crescendo": build_crescendo_attributes,
}


def load_and_build_model(
    *,
    model_type: ModelType,
    ckpt_path: str,
    block_size: int,
    bars_per_sample: int,
) -> torch.nn.Module:
    from musicgen.inference.service import _build_model, _load_checkpoint

    ckpt = _load_checkpoint(ckpt_path)
    model = _build_model(model_type, ckpt, block_size=block_size, bars=bars_per_sample)
    model.eval()
    return model


def generate_sample(
    *,
    model: torch.nn.Module,
    model_type: ModelType,
    attributes: list[list[int]],
    tok_bundle: Any,
    gen_cfg: Any,
    device: torch.device,
    seed: int,
) -> Any:
    from musicgen.inference.generate import generate_remi_tokens
    from musicgen.inference.render_midi import tokens_to_midi

    attrs_t = torch.tensor(attributes, dtype=torch.long, device=device).unsqueeze(0)
    common = dict(
        model_type=model_type,
        model=model,
        attributes=attrs_t,
        pad_id=tok_bundle.pad_id,
        bos_id=tok_bundle.bos_id,
        bar_id=tok_bundle.bar_id,
        id_to_token=tok_bundle.id_to_token,
        cfg=gen_cfg,
        device=device,
        seed=seed,
    )
    if model_type == "plain":
        tokens = generate_remi_tokens(**common)
    else:
        tokens = generate_remi_tokens(**common, z_p=None)
    return tokens_to_midi(tokens, tok_bundle.tok)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-generate MIDI files for MOS evaluation.")
    ap.add_argument("--out-dir", type=str, default="artifacts/mos_evaluation")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated subset: lullaby,virtuoso,crescendo",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml" / "src"))

    from musicgen.inference.generate import GenerationConfig
    from musicgen.inference.render_midi import write_midi
    from musicgen.inference.service import _resolve_ckpt_path
    from musicgen.inference.tokenizer import load_remi_tokenizer

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.categories:
        categories: list[Category] = [c.strip() for c in args.categories.split(",")]  # type: ignore[assignment]
        for c in categories:
            if c not in ALL_CATEGORIES:
                raise ValueError(f"Unknown category: {c!r}")
    else:
        categories = list(ALL_CATEGORIES)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_bundle = load_remi_tokenizer()
    gen_cfg = GenerationConfig(
        block_size=BLOCK_SIZE,
        bars_per_sample=BARS_PER_SAMPLE,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    num_samples = int(args.num_samples)

    for category in categories:
        category_dir = out_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        build_attrs = ATTRIBUTE_BUILDERS[category]

        for file_prefix, model_type, ckpt_rel in MODELS:
            ckpt_path = _resolve_ckpt_path(ckpt_rel)
            log.info("Loading %s from %s", file_prefix, ckpt_path)
            model = load_and_build_model(
                model_type=model_type,
                ckpt_path=ckpt_path,
                block_size=BLOCK_SIZE,
                bars_per_sample=BARS_PER_SAMPLE,
            )
            model.to(device)

            try:
                for sample_idx in range(1, num_samples + 1):
                    seed = int(args.base_seed) + sample_idx
                    attributes = build_attrs(sample_idx)
                    out_path = category_dir / f"{file_prefix}_sample_{sample_idx:02d}.mid"
                    entry: dict[str, Any] = {
                        "category": category,
                        "model": file_prefix,
                        "model_type": model_type,
                        "sample_idx": sample_idx,
                        "seed": seed,
                        "attributes": attributes,
                        "output_path": str(out_path.relative_to(repo_root)),
                        "status": "ok",
                        "error": None,
                    }
                    try:
                        midi_obj = generate_sample(
                            model=model,
                            model_type=model_type,
                            attributes=attributes,
                            tok_bundle=tok_bundle,
                            gen_cfg=gen_cfg,
                            device=device,
                            seed=seed,
                        )
                        write_midi(midi_obj, out_path)
                        log.info("Wrote %s", out_path)
                    except Exception as exc:
                        log.exception(
                            "Failed category=%s model=%s sample=%02d",
                            category,
                            file_prefix,
                            sample_idx,
                        )
                        entry["status"] = "error"
                        entry["error"] = str(exc)
                        failures.append(entry)
                    manifest.append(entry)
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    manifest_path = out_dir / "manifest.json"
    summary = {
        "total": len(manifest),
        "ok": sum(1 for e in manifest if e["status"] == "ok"),
        "failed": len(failures),
        "categories": categories,
        "num_samples": num_samples,
        "base_seed": int(args.base_seed),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "device": str(device),
        "entries": manifest,
    }
    manifest_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote manifest to %s", manifest_path)
    log.info("Done: %d ok, %d failed", summary["ok"], summary["failed"])

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
