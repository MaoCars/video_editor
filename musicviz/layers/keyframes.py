"""Interpolación de keyframes (posición, escala, opacidad, rotación)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..config import KEYFRAME_PROPS, LayerBase, PointKey, ScalarKey
from .animation import ease


@dataclass
class KeyState:
    position: Optional[tuple[float, float]] = None  # None = usar la posición fija de la capa
    scale: float = 1.0
    opacity: float = 1.0
    rotation: float = 0.0


def _interp(a, b, w: float):
    if isinstance(a, (tuple, list)):
        return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)
    return a + (b - a) * w


def evaluate(keys: Sequence[ScalarKey | PointKey], t: float):
    """Valor interpolado en t. Antes del primer keyframe vale el primero; después del último, el último."""
    if not keys:
        return None
    ks = sorted(keys, key=lambda k: k.time)
    if t <= ks[0].time:
        return ks[0].value
    if t >= ks[-1].time:
        return ks[-1].value
    for k0, k1 in zip(ks, ks[1:]):
        if k0.time <= t <= k1.time:
            span = max(k1.time - k0.time, 1e-6)
            w = ease(k1.easing, (t - k0.time) / span)
            return _interp(k0.value, k1.value, w)
    return ks[-1].value  # pragma: no cover


def key_state(cfg: LayerBase, t: float) -> KeyState:
    state = KeyState()
    if cfg.position_keys:
        state.position = tuple(evaluate(cfg.position_keys, t))  # type: ignore[arg-type]
    if cfg.scale_keys:
        state.scale = float(evaluate(cfg.scale_keys, t))
    if cfg.opacity_keys:
        state.opacity = min(max(float(evaluate(cfg.opacity_keys, t)), 0.0), 1.0)
    if cfg.rotation_keys:
        state.rotation = float(evaluate(cfg.rotation_keys, t))
    return state


def has_keys(cfg: LayerBase) -> bool:
    return any(getattr(cfg, f"{p}_keys") for p in KEYFRAME_PROPS)


def current_value(cfg: LayerBase, prop: str, t: float):
    """Valor que tendría la propiedad en t (interpolado si hay keyframes; si no, el valor base)."""
    keys = getattr(cfg, f"{prop}_keys")
    if keys:
        v = evaluate(keys, t)
        return tuple(v) if prop == "position" else float(v)
    if prop == "position":
        return tuple(cfg.position)
    if prop == "rotation":
        return float(getattr(cfg, "rotation", 0.0))
    return 1.0
