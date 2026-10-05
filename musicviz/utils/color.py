"""Utilidades de color: parseo de hex, gradientes y conversión a OpenCV."""
from __future__ import annotations

import colorsys
from functools import lru_cache
from typing import Sequence

import numpy as np

RGBA = tuple[float, float, float, float]

_NAMED = {
    "white": "#ffffff",
    "black": "#000000",
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "yellow": "#ffff00",
    "orange": "#ff8800",
    "purple": "#8800ff",
    "pink": "#ff4fa3",
}


@lru_cache(maxsize=1024)
def parse_color(value: str) -> RGBA:
    """Convierte '#rgb', '#rrggbb', '#rrggbbaa' o un nombre a RGBA en [0, 1]."""
    s = value.strip().lower()
    s = _NAMED.get(s, s)
    if not s.startswith("#"):
        raise ValueError(f"Color inválido: {value!r}")
    h = s[1:]
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 6:
        h += "ff"
    if len(h) != 8:
        raise ValueError(f"Color inválido: {value!r}")
    r, g, b, a = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4, 6))
    return (r, g, b, a)


def gradient(colors: Sequence[str], n: int) -> np.ndarray:
    """Devuelve (n, 4) float32 interpolando linealmente la lista de colores."""
    if n <= 0:
        return np.zeros((0, 4), np.float32)
    stops = np.array([parse_color(c) for c in colors], dtype=np.float32)
    if len(stops) == 1:
        return np.repeat(stops, n, axis=0)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    out = np.empty((n, 4), np.float32)
    for c in range(4):
        out[:, c] = np.interp(t, xs, stops[:, c])
    return out


def color_at(colors: Sequence[str], t: float) -> RGBA:
    """Color interpolado en la posición t ∈ [0, 1] del gradiente."""
    stops = [parse_color(c) for c in colors]
    if len(stops) == 1:
        return stops[0]
    t = min(max(float(t), 0.0), 1.0)
    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    a, b = stops[i], stops[i + 1]
    return tuple(a[k] * (1 - f) + b[k] * f for k in range(4))  # type: ignore[return-value]


def to_cv(rgba: Sequence[float], alpha: float = 1.0) -> tuple[int, int, int, int]:
    """RGBA float [0,1] -> tupla de enteros 0..255 para dibujar en buffers RGBA."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    a = (rgba[3] if len(rgba) > 3 else 1.0) * alpha
    return (
        int(round(min(max(r, 0.0), 1.0) * 255)),
        int(round(min(max(g, 0.0), 1.0) * 255)),
        int(round(min(max(b, 0.0), 1.0) * 255)),
        int(round(min(max(a, 0.0), 1.0) * 255)),
    )


def shift_hue(rgba: Sequence[float], degrees: float) -> RGBA:
    """Rota el tono de un color (grados)."""
    h, s, v = colorsys.rgb_to_hsv(rgba[0], rgba[1], rgba[2])
    h = (h + degrees / 360.0) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, rgba[3] if len(rgba) > 3 else 1.0)


def brighten(rgba: Sequence[float], factor: float) -> RGBA:
    """Multiplica el valor (brillo) de un color, saturando en 1."""
    return (
        min(rgba[0] * factor, 1.0),
        min(rgba[1] * factor, 1.0),
        min(rgba[2] * factor, 1.0),
        rgba[3] if len(rgba) > 3 else 1.0,
    )


def gradient_lut(colors: Sequence[str], n: int = 256) -> list[tuple[int, int, int]]:
    """Gradiente como lista de tuplas RGB enteras (rápido para dibujar con OpenCV)."""
    g = gradient(colors, n)
    return [(int(r * 255 + 0.5), int(gg * 255 + 0.5), int(b * 255 + 0.5)) for r, gg, b, _ in g]


def with_alpha(rgb: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    a = int(alpha * 255 + 0.5)
    return (rgb[0], rgb[1], rgb[2], 0 if a < 0 else 255 if a > 255 else a)


def gradient_from_stops(stops: np.ndarray, n: int = 256) -> np.ndarray:
    """Como `gradient` pero a partir de paradas RGBA ya numéricas (k, 4)."""
    stops = np.asarray(stops, dtype=np.float32)
    if len(stops) == 1:
        return np.repeat(stops, n, axis=0)
    xs = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = np.empty((n, 4), np.float32)
    for c in range(4):
        out[:, c] = np.interp(t, xs, stops[:, c])
    return out


def lut_to_int(lut: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)) for r, g, b, _ in lut]
