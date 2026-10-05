"""Animaciones de entrada/salida y ventana temporal de las capas (deterministas por frame)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import LayerBase


def ease(kind: str, t: float) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    if kind == "linear":
        return t
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if kind == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    if kind == "back":  # rebasa un poco y vuelve (easeOutBack)
        c1, c3 = 1.70158, 2.70158
        u = t - 1.0
        return 1.0 + c3 * u * u * u + c1 * u * u
    if kind == "bounce":  # easeOutBounce
        n1, d1 = 7.5625, 2.75
        if t < 1 / d1:
            return n1 * t * t
        if t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        if t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        t -= 2.625 / d1
        return n1 * t * t + 0.984375
    return t


@dataclass
class AnimState:
    visible: bool = True
    alpha: float = 1.0   # multiplicador de opacidad
    dx: float = 0.0      # desplazamiento en px
    dy: float = 0.0
    scale: float = 1.0   # multiplicador de tamaño
    blur: float = 0.0    # 0..1 (sólo texto/imagen)

    @property
    def is_identity(self) -> bool:
        return self.alpha >= 0.999 and abs(self.dx) < 0.5 and abs(self.dy) < 0.5 and abs(self.scale - 1.0) < 1e-3 and self.blur <= 1e-3


def _apply(kind: str, q: float, state: AnimState, distance: float, easing: str) -> None:
    """q: progreso de visibilidad 0 (invisible) .. 1 (totalmente visible)."""
    if kind == "none":
        return
    e = ease(easing, q)
    state.alpha *= min(max(e if easing not in ("back", "bounce") else q, 0.0), 1.0)
    if kind == "slide_left":
        state.dx += (1.0 - e) * distance
    elif kind == "slide_right":
        state.dx -= (1.0 - e) * distance
    elif kind == "slide_up":
        state.dy += (1.0 - e) * distance
    elif kind == "slide_down":
        state.dy -= (1.0 - e) * distance
    elif kind == "zoom_in":
        state.scale *= 0.4 + 0.6 * e
    elif kind == "zoom_out":
        state.scale *= 1.6 - 0.6 * e
    elif kind == "pop":
        state.scale *= max(ease("back", q), 0.0)
    elif kind == "blur":
        state.blur = max(state.blur, 1.0 - e)


def anim_state(cfg: LayerBase, t: float, duration: float, min_dim: float) -> AnimState:
    """Estado de animación de una capa en el instante t."""
    start = cfg.start if cfg.start is not None else 0.0
    end = cfg.end if cfg.end is not None else duration
    state = AnimState()
    if t < start or t > end:
        state.visible = False
        state.alpha = 0.0
        return state
    distance = cfg.slide_distance * min_dim
    if cfg.animate_in != "none" and cfg.in_duration > 0:
        q_in = (t - start) / cfg.in_duration
        if q_in < 1.0:
            _apply(cfg.animate_in, q_in, state, distance, cfg.easing)
    if cfg.animate_out != "none" and cfg.out_duration > 0:
        q_out = (end - t) / cfg.out_duration
        if q_out < 1.0:
            _apply(cfg.animate_out, q_out, state, -distance, cfg.easing)
    if state.alpha <= 0.002:
        state.visible = False
    return state
