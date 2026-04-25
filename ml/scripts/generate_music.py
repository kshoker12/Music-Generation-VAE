from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch


def _parse_phase(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Phase must have 4 comma-separated ints, e.g. '3,3,3,3'")
    vals = [int(x) for x in parts]
    for v in vals:
        if not (0 <= v <= 7):
            raise ValueError("Attribute bins must be in [0..7].")
    return vals


def _build_attributes(
    *,
    bars_per_sample: int,
    attrs_json: Optional[str],
    phase1: Optional[str],
    phase2: Optional[str],
    phase_split: int,
) -> torch.Tensor:
    if attrs_json is not None:
        arr = json.loads(attrs_json)
        if not (isinstance(arr, list) and len(arr) == bars_per_sample):
            raise ValueError(f"--attributes-json must be a JSON list of length {bars_per_sample}.")
        out = []
        for row in arr:
            if not (isinstance(row, list) and len(row) == 4):
                raise ValueError("--attributes-json rows must have length 4.")
            out.append([int(x) for x in row])
        return torch.tensor(out, dtype=torch.long).unsqueeze(0)

    p1 = _parse_phase(phase1 or "3,3,3,3")
    p2 = _parse_phase(phase2 or "3,3,3,3")
    if not (0 <= phase_split <= bars_per_sample):
        raise ValueError("--phase-split must be within [0..bars_per-sample].")
    rows = [p1 for _ in range(phase_split)] + [p2 for _ in range(bars_per_sample - phase_split)]
    return torch.tensor(rows, dtype=torch.long).unsqueeze(0)


def _load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


def _resolve_default_ckpt(*, repo_root: Path, model_type: str) -> Path:
    """
    Resolve a default v1 checkpoint path under `runs/`.
    """
    runs_root = repo_root / "ml" / "src" / "musicgen" / "runs"
    cand_dirs = [
        runs_root / f"{model_type}_v1",
        runs_root / model_type / "v1",
        runs_root / model_type,
    ]
    cand_files = [
        "ckpt.pt",
        "checkpoint.pt",
        "model.pt",
        "last.pt",
        "best.pt",
    ]
    for d in cand_dirs:
        for fn in cand_files:
            p = d / fn
            if p.exists():
                return p
        # fallback: any .pt in that dir
        if d.exists() and d.is_dir():
            pts = sorted(d.glob("*.pt"))
            if pts:
                return pts[-1]

    tried = [str((d / fn).relative_to(repo_root)) for d in cand_dirs for fn in cand_files]
    raise FileNotFoundError(
        "No default checkpoint found. Tried:\n  - " + "\n  - ".join(tried) + "\n\nPass --ckpt explicitly."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Grammar-masked REMI generation to MIDI.")
    ap.add_argument("--model-type", type=str, required=True, choices=("plain", "vae", "simple_vae"))
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to saved checkpoint (.pt). If omitted, uses v1 under ./runs for plain/vae.",
    )
    ap.add_argument(
        "--out-mid",
        type=str,
        default=None,
        help="Output MIDI path. If relative (or omitted), will be written under artifacts/attributes/dumps/.",
    )
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--bars-per-sample", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)

    # Attribute controls
    ap.add_argument(
        "--attributes-json",
        type=str,
        default=None,
        help="JSON string of shape [bars_per_sample,4] with bins 0..7. Overrides phase args.",
    )
    ap.add_argument("--phase1", type=str, default=None, help="Bars [0..phase_split-1] bins, e.g. '2,5,3,4'")
    ap.add_argument("--phase2", type=str, default=None, help="Bars [phase_split..] bins, e.g. '6,2,6,7'")
    ap.add_argument("--phase-split", type=int, default=4)

    # Optional explicit z_p (JSON list of floats length z_dim)
    ap.add_argument("--z-p-json", type=str, default=None, help="Optional JSON list (len=z_dim) to fix z_p.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml" / "src"))

    from musicgen.inference.generate import GenerationConfig, generate_remi_tokens
    from musicgen.inference.render_midi import tokens_to_midi, write_midi
    from musicgen.inference.tokenizer import load_remi_tokenizer

    device = torch.device(args.device)
    tok_bundle = load_remi_tokenizer()

    dumps_root = repo_root / "artifacts" / "dumps"
    if args.out_mid is None:
        out_mid_path = dumps_root / f"out_{args.model_type}.mid"
    else:
        out_mid_path = Path(args.out_mid)
        if not out_mid_path.is_absolute():
            out_mid_path = dumps_root / out_mid_path

    if args.ckpt is None:
        if args.model_type == "simple_vae":
            raise FileNotFoundError("Simple VAE checkpoint not available by default yet. Pass --ckpt explicitly.")
        ckpt_path = _resolve_default_ckpt(repo_root=repo_root, model_type=args.model_type)
    else:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_checkpoint(str(ckpt_path))
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict")))
    if state is None:
        # Sometimes saved as raw state dict.
        state = ckpt

    # Instantiate model + cfg
    if args.model_type == "plain":
        from musicgen.models.plain_transformer import PlainTransformerConfig, PlainTransformerDecoder

        cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        cfg = PlainTransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else PlainTransformerConfig()
        cfg = PlainTransformerConfig(
            **{**cfg.__dict__, "vocab_size": 195, "block_size": int(args.block_size), "bars_per_sample": int(args.bars_per_sample)}
        )
        model = PlainTransformerDecoder(cfg)
    elif args.model_type == "simple_vae":
        from musicgen.models.simple_vae import SimpleMusicVAE, SimpleVAEConfig

        cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        cfg = SimpleVAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else SimpleVAEConfig()
        cfg = SimpleVAEConfig(
            **{**cfg.__dict__, "vocab_size": 195, "block_size": int(args.block_size), "bars_per_sample": int(args.bars_per_sample)}
        )
        model = SimpleMusicVAE(cfg)
    elif args.model_type == "vae":
        from musicgen.models.vae import MusicVAE, VAEConfig

        cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        cfg = VAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else VAEConfig()
        cfg = VAEConfig(
            **{**cfg.__dict__, "vocab_size": 195, "block_size": int(args.block_size), "bars_per_sample": int(args.bars_per_sample)}
        )
        model = MusicVAE(cfg)
    else:
        raise ValueError(args.model_type)

    model.load_state_dict(state, strict=True)

    attributes = _build_attributes(
        bars_per_sample=int(args.bars_per_sample),
        attrs_json=args.attributes_json,
        phase1=args.phase1,
        phase2=args.phase2,
        phase_split=int(args.phase_split),
    ).to(device)

    z_p = None
    if args.z_p_json is not None:
        z_list = json.loads(args.z_p_json)
        if not isinstance(z_list, list):
            raise ValueError("--z-p-json must be a JSON list of floats.")
        z_p = torch.tensor([float(x) for x in z_list], dtype=torch.float32).unsqueeze(0)

    gen_cfg = GenerationConfig(
        block_size=int(args.block_size),
        bars_per_sample=int(args.bars_per_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    tokens = generate_remi_tokens(
        model_type=args.model_type,
        model=model,
        attributes=attributes,
        pad_id=tok_bundle.pad_id,
        bos_id=tok_bundle.bos_id,
        bar_id=tok_bundle.bar_id,
        id_to_token=tok_bundle.id_to_token,
        cfg=gen_cfg,
        device=device,
        z_p=z_p,
        seed=int(args.seed) if args.seed is not None else None,
    )

    midi_obj = tokens_to_midi(tokens, tok_bundle.tok)
    out = write_midi(midi_obj, out_mid_path)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

