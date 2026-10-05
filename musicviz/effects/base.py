from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import EffectBase
from ..render.canvas import RenderContext

C = TypeVar("C", bound=EffectBase)


class Effect(Generic[C]):
    """Efecto de post-proceso sobre la imagen float32 RGB [0,1] del frame.

    `drive(frame)` devuelve la intensidad efectiva (0..∞) según el disparador configurado
    (always / beat / kick / bass / energy / drop / treble) y la ventana temporal start/end.
    """

    def __init__(self, cfg: C):
        self.cfg = cfg
        self.ctx: RenderContext | None = None

    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        self.ctx = ctx

    def drive(self, frame: FrameFeatures) -> float:
        cfg = self.cfg
        if cfg.start is not None and frame.time < cfg.start:
            return 0.0
        if cfg.end is not None and frame.time > cfg.end:
            return 0.0
        return frame.drive(cfg.trigger, cfg.threshold) * cfg.intensity * frame.intensity

    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:  # pragma: no cover - interfaz
        raise NotImplementedError
