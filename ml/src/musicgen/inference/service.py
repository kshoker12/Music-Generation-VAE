from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Optional

import torch

from musicgen.inference.generate import GenerationConfig, generate_remi_tokens
from musicgen.inference.render_midi import midi_to_bytes, tokens_to_midi
from musicgen.inference.tokenizer import load_remi_tokenizer

ModelType = Literal["plain", "vae", "simple_vae"]


@dataclass(frozen=True)
class GenerateRequest:
    model_type: ModelType
    attributes: list[list[int]]  # [8,4]
    seed: int
    temperature: float
    top_p: float
    block_size: int = 1024
    bars_per_sample: int = 8
    ckpt_path: Optional[str] = None


def _default_ckpt(model_type: ModelType) -> str:
    # Defaults match local repo layout; override on RunPod via env vars.
    if model_type == "plain":
        return os.environ.get("DEFAULT_CKPT_PLAIN", "ml/src/musicgen/runs/plain_v1/ckpt.pt")
    if model_type == "vae":
        return os.environ.get("DEFAULT_CKPT_VAE", "ml/src/musicgen/runs/vae_v1/ckpt.pt")
    return os.environ.get("DEFAULT_CKPT_SIMPLE_VAE", "ml/src/musicgen/runs/simple_vae_v1/ckpt.pt")


def _repo_root() -> Path:
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Best-effort search upwards from this file for a repo root that contains ml/src/musicgen.
    here = Path(__file__).resolve()
    for p in [here.parents[i] for i in range(0, 10)]:
        if (p / "ml" / "src" / "musicgen").is_dir():
            return p
    # Fall back to CWD; callers can still pass absolute ckpt paths.
    return Path.cwd()


def _resolve_ckpt_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((_repo_root() / p).resolve())


def _load_checkpoint(path: str) -> dict:
    # PyTorch 2.x supports weights_only; training checkpoints are full pickles.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


class _ModelCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self._models: dict[str, torch.nn.Module] = {}

    def get(self, key: str, factory) -> torch.nn.Module:
        with self._lock:
            if key in self._models:
                return self._models[key]
            m = factory()
            m.eval()
            self._models[key] = m
            return m


_MODEL_CACHE = _ModelCache()


def _build_model(model_type: ModelType, ckpt: dict, *, block_size: int, bars: int) -> torch.nn.Module:
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict")))
    if state is None:
        state = ckpt

    vocab_size = 195

    if model_type == "plain":
        from musicgen.models.plain_transformer import PlainTransformerConfig, PlainTransformerDecoder

        cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        cfg = PlainTransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else PlainTransformerConfig()
        cfg = PlainTransformerConfig(
            **{**cfg.__dict__, "vocab_size": vocab_size, "block_size": int(block_size), "bars_per_sample": int(bars)}
        )
        model = PlainTransformerDecoder(cfg)
    elif model_type == "vae":
        from musicgen.models.vae import MusicVAE, VAEConfig

        cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        cfg = VAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else VAEConfig()
        cfg = VAEConfig(**{**cfg.__dict__, "vocab_size": vocab_size, "block_size": int(block_size), "bars_per_sample": int(bars)})
        model = MusicVAE(cfg)
    elif model_type == "simple_vae":
        from musicgen.models.simple_vae import SimpleMusicVAE, SimpleVAEConfig

        cfg_dict = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        cfg = SimpleVAEConfig(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else SimpleVAEConfig()
        cfg = SimpleVAEConfig(
            **{**cfg.__dict__, "vocab_size": vocab_size, "block_size": int(block_size), "bars_per_sample": int(bars)}
        )
        model = SimpleMusicVAE(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state, strict=True)
    return model


def generate_midi_bytes(req: GenerateRequest) -> bytes:
    if len(req.attributes) != int(req.bars_per_sample):
        raise ValueError(f"attributes must be [{req.bars_per_sample},4]")
    for row in req.attributes:
        if len(row) != 4:
            raise ValueError("Each attribute row must have length 4.")
        for v in row:
            if not (0 <= int(v) <= 7):
                raise ValueError("Attribute bins must be in [0..7].")

    ckpt_path_raw = req.ckpt_path or _default_ckpt(req.model_type)
    ckpt_path = _resolve_ckpt_path(ckpt_path_raw)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_checkpoint(ckpt_path)
    cache_key = f"{req.model_type}:{ckpt_path}:{req.block_size}:{req.bars_per_sample}"

    def factory() -> torch.nn.Module:
        return _build_model(req.model_type, ckpt, block_size=req.block_size, bars=req.bars_per_sample)

    model = _MODEL_CACHE.get(cache_key, factory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tok_bundle = load_remi_tokenizer()
    attrs = torch.tensor(req.attributes, dtype=torch.long, device=device).unsqueeze(0)

    torch.manual_seed(int(req.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(req.seed))

    gen_cfg = GenerationConfig(
        block_size=int(req.block_size),
        bars_per_sample=int(req.bars_per_sample),
        temperature=float(req.temperature),
        top_p=float(req.top_p),
    )

    with torch.no_grad():
        tokens = generate_remi_tokens(
            model_type=req.model_type,
            model=model,
            attributes=attrs,
            pad_id=tok_bundle.pad_id,
            bos_id=tok_bundle.bos_id,
            bar_id=tok_bundle.bar_id,
            id_to_token=tok_bundle.id_to_token,
            cfg=gen_cfg,
            device=device,
            z_p=None,
            seed=int(req.seed),
        )

    midi_obj = tokens_to_midi(tokens, tok_bundle.tok)
    return midi_to_bytes(midi_obj)
