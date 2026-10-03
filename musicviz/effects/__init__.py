"""Efectos de post-proceso aplicados al frame completo tras componer las capas."""
from __future__ import annotations

from ..config import (
    BloomEffect,
    ChromaticEffect,
    ColorEffect,
    EffectConfig,
    FilmGrainEffect,
    GlitchEffect,
    KaleidoEffect,
    PixelateEffect,
    RadialBlurEffect,
    ScanlinesEffect,
    ShakeEffect,
    StrobeEffect,
    VignetteEffect,
)
from .base import Effect
from .basic import (
    Bloom,
    Chromatic,
    ColorGrade,
    FilmGrain,
    Kaleido,
    Pixelate,
    RadialBlur,
    Scanlines,
    Shake,
    Strobe,
    Vignette,
)
from .glitch import Glitch

_REGISTRY = {
    GlitchEffect: Glitch,
    BloomEffect: Bloom,
    ChromaticEffect: Chromatic,
    ShakeEffect: Shake,
    VignetteEffect: Vignette,
    ColorEffect: ColorGrade,
    PixelateEffect: Pixelate,
    StrobeEffect: Strobe,
    KaleidoEffect: Kaleido,
    RadialBlurEffect: RadialBlur,
    ScanlinesEffect: Scanlines,
    FilmGrainEffect: FilmGrain,
}


def build_effect(cfg: EffectConfig) -> Effect:
    for cfg_type, cls in _REGISTRY.items():
        if isinstance(cfg, cfg_type):
            return cls(cfg)
    raise ValueError(f"Tipo de efecto no soportado: {type(cfg).__name__}")


__all__ = ["Effect", "build_effect"]
