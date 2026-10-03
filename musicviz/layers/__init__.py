"""Capas visuales: cada una dibuja sobre el lienzo según las características del frame."""
from __future__ import annotations

from ..config import (
    BarsLayer,
    CircleLayer,
    ImageLayer,
    LayerConfig,
    ParticlesLayer,
    ProgressLayer,
    TextLayer,
    WaveformLayer,
)
from .background import Background
from .bars import Bars
from .base import Layer
from .circle import CircleSpectrum
from .image import ImageOverlay
from .particles import Particles
from .progress import ProgressBar
from .text import TextOverlay
from .waveform import Waveform

_REGISTRY = {
    BarsLayer: Bars,
    CircleLayer: CircleSpectrum,
    WaveformLayer: Waveform,
    ParticlesLayer: Particles,
    ImageLayer: ImageOverlay,
    TextLayer: TextOverlay,
    ProgressLayer: ProgressBar,
}


def build_layer(cfg: LayerConfig) -> Layer:
    for cfg_type, cls in _REGISTRY.items():
        if isinstance(cfg, cfg_type):
            return cls(cfg)
    raise ValueError(f"Tipo de capa no soportado: {type(cfg).__name__}")


__all__ = ["Layer", "Background", "build_layer"]
