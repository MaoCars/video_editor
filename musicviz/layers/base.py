from __future__ import annotations

from typing import Generic, TypeVar

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import LayerBase
from ..render.canvas import Canvas, RenderContext

C = TypeVar("C", bound=LayerBase)


class Layer(Generic[C]):
    """Base de capa. `prepare` se llama una vez con todo el análisis (permite precalcular);
    `render` se llama por frame y debe ser determinista (los frames se renderizan en paralelo)."""

    def __init__(self, cfg: C):
        self.cfg = cfg
        self.ctx: RenderContext | None = None

    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        self.ctx = ctx

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:  # pragma: no cover - interfaz
        raise NotImplementedError

    # Ayudas comunes -------------------------------------------------------
    def composite(self, canvas: Canvas, layer, glow: float | None = None) -> None:
        cfg = self.cfg
        ctx = self.ctx
        assert ctx is not None
        canvas.composite(
            layer,
            opacity=cfg.opacity,
            blend=cfg.blend,
            glow=cfg.glow if glow is None else glow,
            glow_radius=ctx.px(cfg.glow_radius),
        )
