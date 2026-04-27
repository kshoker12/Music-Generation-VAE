"""
CLI driver for the controllability + z-effect evaluation of vae_v3.

Usage:
    python ml/scripts/run_control_eval.py            # full sweep (~32 + 12 MIDIs)
    python ml/scripts/run_control_eval.py --device cpu --run-id smoke

The heavy lifting lives in ml/src/musicgen/analysis/control_eval.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_ml_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    ml_src = repo_root / "ml" / "src"
    s = str(ml_src)
    if s not in sys.path:
        sys.path.insert(0, s)


def _parse_attrs_json(s: str) -> list[list[int]]:
    arr = json.loads(s)
    if not (isinstance(arr, list) and len(arr) == 8):
        raise argparse.ArgumentTypeError("Each --z-config-* must be a JSON list of length 8.")
    rows = []
    for row in arr:
        if not (isinstance(row, list) and len(row) == 4):
            raise argparse.ArgumentTypeError("Rows must have length 4.")
        rows.append([int(x) for x in row])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Controllability + z-effect evaluation for VAE v3.")
    ap.add_argument("--out-dir", type=str, default="artifacts/control_eval")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument(
        "--ckpt",
        type=str,
        default="ml/src/musicgen/runs/vae_v3/ckpt.pt",
        help="Path to VAE v3 checkpoint (.pt).",
    )
    ap.add_argument(
        "--quantiles",
        type=str,
        default="artifacts/attributes/quantiles.json",
        help="Path to global quantile thresholds JSON.",
    )
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--bars-per-sample", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.9)

    # Attribute sweep
    ap.add_argument("--attr-seeds", type=int, nargs="+", default=[2026])
    ap.add_argument("--attr-z-seed", type=int, default=2026)
    ap.add_argument("--neutral-bin", type=int, default=4)

    # z sweep
    ap.add_argument("--z-seeds", type=int, nargs="+", default=[11, 42, 123, 256, 1024, 4096])
    ap.add_argument(
        "--z-config-neutral",
        type=_parse_attrs_json,
        default=None,
        help="Override the 'neutral' z-sweep config as JSON [8][4] of bins.",
    )
    ap.add_argument(
        "--z-config-phased",
        type=_parse_attrs_json,
        default=None,
        help="Override the 'phased_low_to_high' z-sweep config as JSON [8][4] of bins.",
    )
    args = ap.parse_args()

    _ensure_ml_on_path()
    from musicgen.analysis.control_eval import ControlEvalConfig, run_control_eval

    z_configs: dict[str, list[list[int]]] = {
        "neutral": args.z_config_neutral or [[4, 4, 4, 4]] * int(args.bars_per_sample),
        "phased_low_to_high": args.z_config_phased
        or ([[1, 1, 2, 1]] * 4 + [[6, 6, 6, 7]] * 4),
    }

    cfg = ControlEvalConfig(
        out_dir=str(args.out_dir),
        run_id=args.run_id,
        ckpt_path=str(args.ckpt),
        quantiles_path=str(args.quantiles),
        device=str(args.device),
        block_size=int(args.block_size),
        bars_per_sample=int(args.bars_per_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        attr_seeds=[int(s) for s in args.attr_seeds],
        attr_z_seed=int(args.attr_z_seed),
        neutral_bin=int(args.neutral_bin),
        z_seeds=[int(s) for s in args.z_seeds],
        z_attr_configs=z_configs,
    )
    out = run_control_eval(cfg)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
