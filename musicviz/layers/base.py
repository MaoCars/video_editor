from __future__ import annotations

from typing import Generic, TypeVar

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import LayerBase
from ..config import PALETTE
from ..render.canvas import Canvas, RenderContext
from ..utils.color import gradient, gradient_from_stops, lut_to_int
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
        self._intensity = 1.0
        self._lut_key: bytes | None = None
        self._lut_dyn: tuple | None = None
        colors = getattr(self.cfg, "colors", None)
        static = gradient(["#ffffff"] if colors is None or colors == [PALETTE] else colors, 256)
        self._lut_static = (static, lut_to_int(static))

    # Colores -------------------------------------------------------------
    @property
    def uses_palette(self) -> bool:
        return getattr(self.cfg, "colors", None) == [PALETTE]

    def luts(self, frame: FrameFeatures) -> tuple:
        """(LUT float (256,4), LUT de tuplas enteras) según los colores fijos o la paleta de la sección."""
        if self.uses_palette and frame.palette is not None:
            key = frame.palette.tobytes()
            if self._lut_dyn is None or key != self._lut_key:
                lut = gradient_from_stops(frame.palette, 256)
                self._lut_dyn = (lut, lut_to_int(lut))
                self._lut_key = key
            return self._lut_dyn
        return self._lut_static

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:  # pragma: no cover - interfaz
        raise NotImplementedError

    # Animación / ventana temporal ----------------------------------------
    def begin_frame(self, frame: FrameFeatures) -> bool:
        """Calcula el estado de animación del frame. Devuelve False si la capa no se dibuja."""
        assert self.ctx is not None
        self._anim = anim_state(self.cfg, frame.time, self.duration, self.ctx.min_dim)
        self._intensity = float(frame.intensity)
        return self._anim.visible

    @property
    def anim(self) -> AnimState:
        return self._anim

    @property
    def intensity(self) -> float:
        """Intensidad de la sección activa (1 = normal)."""
        return self._intensity

    # Ayudas comunes -------------------------------------------------------
    def composite(self, canvas: Canvas, layer, glow: float | None = None) -> None:
        cfg = self.cfg
        ctx = self.ctx
        assert ctx is not None
        canvas.composite(
            layer,
            opacity=cfg.opacity * self._anim.alpha,
            blend=cfg.blend,
            glow=(cfg.glow if glow is None else glow) * self._intensity,
            glow_radius=ctx.px(cfg.glow_radius),
        )
