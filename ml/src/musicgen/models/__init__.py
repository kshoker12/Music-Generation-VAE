"""Model modules (baseline + VAE)."""

from .plain_transformer import PlainTransformerConfig, PlainTransformerDecoder
from .simple_vae import SimpleMusicVAE, SimpleVAEConfig
from .vae import MusicVAE, VAEConfig

__all__ = [
    "PlainTransformerConfig",
    "PlainTransformerDecoder",
    "SimpleVAEConfig",
    "SimpleMusicVAE",
    "VAEConfig",
    "MusicVAE",
]

