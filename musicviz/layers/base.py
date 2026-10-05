from __future__ import annotations

from typing import Generic, TypeVar

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import LayerBase
from ..render.canvas import Canvas, RenderContext
from .animation import AnimState, anim_state

C = TypeVar("C", bound=LayerBase)


class Layer(Generic[C]):
    """Base de capa. `prepare` se llama una vez con todo el análisis (permite precalcular);
    `render` se llama por frame y debe ser determinista (los frames se renderizan en paralelo)."""

    def __init__(self, cfg: C):
        self.cfg = cfg
        self.ctx: RenderContext | None = None

    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        self.ctx = ctx
        self.duration = features.duration
        self._anim = AnimState()

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:  # pragma: no cover - interfaz
        raise NotImplementedError

    # Animación / ventana temporal ----------------------------------------
    def begin_frame(self, frame: FrameFeatures) -> bool:
        """Calcula el estado de animación del frame. Devuelve False si la capa no se dibuja."""
        assert self.ctx is not None
        self._anim = anim_state(self.cfg, frame.time, self.duration, self.ctx.min_dim)
        return self._anim.visible

    @property
    def anim(self) -> AnimState:
        return self._anim

    # Ayudas comunes -------------------------------------------------------
    def composite(self, canvas: Canvas, layer, glow: float | None = None) -> None:
        cfg = self.cfg
        ctx = self.ctx
        assert ctx is not None
        canvas.composite(
            layer,
            opacity=cfg.opacity * self._anim.alpha,
            blend=cfg.blend,
            glow=cfg.glow if glow is None else glow,
            glow_radius=ctx.px(cfg.glow_radius),
        )
