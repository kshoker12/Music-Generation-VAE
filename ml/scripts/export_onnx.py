from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


class PlainWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, bar_indices, attributes):
        out = self.model(X, bar_indices, attributes)
        return out["logits"]


class VAEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, bar_indices, attributes, z_p):
        out = self.model(X, bar_indices, attributes, z_p=z_p)
        return out["logits"]


def _dump_vocab_json(out_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml" / "src"))
    from musicgen.inference.tokenizer import load_remi_tokenizer

    tb = load_remi_tokenizer()
    payload = {
        "pad_id": tb.pad_id,
        "bos_id": tb.bos_id,
        "bar_id": tb.bar_id,
        "id_to_token": {str(k): v for k, v in tb.id_to_token.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, choices=("plain", "vae", "simple_vae"))
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out-onnx", required=True, type=str)
    ap.add_argument("--out-vocab-json", default="web/public/models/vocab.json")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--bars", type=int, default=8)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ml" / "src"))

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict")))
    if state is None:
        state = ckpt

    vocab_size = 195
    T = int(args.block_size)
    bars = int(args.bars)

    if args.model_type == "plain":
        from musicgen.models.plain_transformer import PlainTransformerConfig, PlainTransformerDecoder

        cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        cfg = PlainTransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else PlainTransformerConfig()
        cfg = PlainTransformerConfig(**{**cfg.__dict__, "vocab_size": vocab_size, "block_size": T, "bars_per_sample": bars})
        model = PlainTransformerDecoder(cfg).eval()
        model.load_state_dict(state, strict=True)
        wrapped = PlainWrapper(model)
        example_inputs = (
            torch.zeros((1, T), dtype=torch.long),
            torch.zeros((1, T), dtype=torch.long),
            torch.zeros((1, bars, 4), dtype=torch.long),
        )
        input_names = ["X", "bar_indices", "attributes"]
    else:
        if args.model_type == "vae":
            from musicgen.models.vae import MusicVAE, VAEConfig

            cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
            cfg = VAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else VAEConfig()
            cfg = VAEConfig(**{**cfg.__dict__, "vocab_size": vocab_size, "block_size": T, "bars_per_sample": bars})
            model = MusicVAE(cfg).eval()
        else:
            from musicgen.models.simple_vae import SimpleMusicVAE, SimpleVAEConfig

            cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
            cfg = SimpleVAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else SimpleVAEConfig()
            cfg = SimpleVAEConfig(**{**cfg.__dict__, "vocab_size": vocab_size, "block_size": T, "bars_per_sample": bars})
            model = SimpleMusicVAE(cfg).eval()

        model.load_state_dict(state, strict=True)
        wrapped = VAEWrapper(model)
        z_dim = int(getattr(model.cfg, "z_dim", 128))
        example_inputs = (
            torch.zeros((1, T), dtype=torch.long),
            torch.zeros((1, T), dtype=torch.long),
            torch.zeros((1, bars, 4), dtype=torch.long),
            torch.zeros((1, z_dim), dtype=torch.float32),
        )
        input_names = ["X", "bar_indices", "attributes", "z_p"]

    out_onnx = Path(args.out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        example_inputs,
        str(out_onnx),
        input_names=input_names,
        output_names=["logits"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamo=False,
    )

    _dump_vocab_json(repo_root / args.out_vocab_json)
    print(f"Wrote ONNX: {out_onnx}")
    print(f"Wrote vocab: {repo_root / args.out_vocab_json}")


if __name__ == "__main__":
    main()

