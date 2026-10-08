"""Cajas aproximadas de las capas en coordenadas relativas (para seleccionar y arrastrar en la vista previa)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config import LayerConfig, ProjectConfig
from ..layers.keyframes import current_value

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 relativos


@lru_cache(maxsize=64)
def _image_aspect(path: str) -> float:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        return w / max(h, 1)
    except Exception:  # noqa: BLE001
        return 1.0


@lru_cache(maxsize=256)
def _text_size_px(text: str, font: Optional[str], px: int, letter_spacing: float, align: str, line_spacing: float, max_w: Optional[float], stroke: float, upper: bool) -> tuple[int, int]:
    from ..layers.text import find_font, render_text_rgba

    f = find_font(font, max(px, 4))
    sprite = render_text_rgba(text.upper() if upper else text, f, "#ffffff", letter_spacing, align, line_spacing, max_w, stroke)
    return sprite.shape[1], sprite.shape[0]


def _anchor_box(x: float, y: float, w: float, h: float, anchor: str) -> Box:
    ax = 0.0 if "left" in anchor else 1.0 if "right" in anchor else 0.5
    ay = 0.0 if "top" in anchor else 1.0 if "bottom" in anchor else 0.5
    x0 = x - ax * w
    y0 = y - ay * h
    return (x0, y0, x0 + w, y0 + h)


def layer_box(layer: LayerConfig, project: ProjectConfig, t: float) -> Optional[Box]:
    """Caja (x0, y0, x1, y1) en coordenadas relativas que ocupa la capa en el instante t."""
    W, H = project.output.width, project.output.height
    md = min(W, H)
    px, py = current_value(layer, "position", t)
    scale = current_value(layer, "scale", t) if layer.scale_keys else 1.0
    kind = layer.type
    if kind == "text":
        px_size = max(int(layer.size * H), 4)
        max_w = layer.max_width * W if layer.max_width else None
        tw, th = _text_size_px(layer.text, layer.font, px_size, layer.letter_spacing * H / 1080, layer.align, layer.line_spacing, max_w, layer.stroke_width * H / 1080, layer.uppercase)
        w, h = tw * scale / W, th * scale / H
        v = {"top": "top", "bottom": "bottom", "middle": ""}[layer.valign]
        a = {"left": "left", "right": "right", "center": ""}[layer.align]
        anchor = "_".join(p for p in (v, a) if p) or "center"
        return _anchor_box(px, py, w, h, anchor)
    if kind == "image":
        aspect = 1.0 if layer.shape in ("square", "circle") else (layer.aspect or _image_aspect(layer.file))
        long_side = layer.size * md * scale
        bw, bh = (long_side, long_side / aspect) if aspect >= 1 else (long_side * aspect, long_side)
        return _anchor_box(px, py, bw / W, bh / H, layer.anchor)
    if kind == "circle":
        r = (layer.radius + layer.length) * md * scale * (1.0 + layer.pulse)
        return (px - r / W, py - r / H, px + r / W, py + r / H)
    if kind == "bars":
        h = layer.height * scale
        return (px - layer.width / 2, py - h, px + layer.width / 2, py + (h if layer.mirror else 0.02))
    if kind == "waveform":
        if layer.style == "circular":
            r = (layer.radius * md + layer.amplitude * H * scale) 
            return (px - r / W, py - r / H, px + r / W, py + r / H)
        a = layer.amplitude * scale
        return (px - layer.width / 2, py - a, px + layer.width / 2, py + a)
    if kind == "progress":
        return (px - layer.width / 2, py - 0.02, px + layer.width / 2, py + 0.02)
    if kind == "particles":
        r = layer.emitter_radius * md if layer.emitter == "ring" else 0.08 * md
        return (px - r / W, py - r / H, px + r / W, py + r / H)
    return None


def layer_rotation(layer: LayerConfig, t: float) -> float:
    """Ángulo (grados) con el que se dibuja la capa en t: rotación propia + velocidad + keyframes."""
    angle = float(current_value(layer, "rotation", t)) if layer.rotation_keys else 0.0
    if layer.type in ("image", "circle"):
        angle += float(getattr(layer, "rotation", 0.0)) + float(getattr(layer, "rotation_speed", 0.0)) * t
    return angle


Poly = list[tuple[float, float]]


def layer_polygon(layer: LayerConfig, project: ProjectConfig, t: float) -> Optional[Poly]:
    """Las cuatro esquinas de la capa en coordenadas relativas, giradas como se dibuja (texto e imagen giran
    alrededor del centro de su caja). Para el resto de capas es el rectángulo de `layer_box`."""
    box = layer_box(layer, project, t)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    angle = layer_rotation(layer, t) if layer.type in ("text", "image") else 0.0
    if abs(angle) < 1e-3:
        return corners
    import math

    W, H = project.output.width, project.output.height
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    a = math.radians(-angle)  # el render gira en sentido horario positivo (eje y hacia abajo)
    ca, sa = math.cos(a), math.sin(a)
    out: Poly = []
    for px, py in corners:
        dx, dy = px * W - cx, py * H - cy
        out.append(((cx + dx * ca - dy * sa) / W, (cy + dx * sa + dy * ca) / H))
    return out


def point_in_polygon(x: float, y: float, poly: Poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        xa, ya = poly[i]
        xb, yb = poly[(i + 1) % n]
        if (ya > y) != (yb > y):
            xi = xa + (y - ya) * (xb - xa) / (yb - ya)
            if x < xi:
                inside = not inside
    return inside


def hit_layer(project: ProjectConfig, t: float, rx: float, ry: float) -> Optional[int]:
    """Índice de la capa visible más alta bajo el punto (rx, ry); None si no hay ninguna."""
    for i in range(len(project.layers) - 1, -1, -1):
        layer = project.layers[i]
        if not layer.enabled:
            continue
        if layer.start is not None and t < layer.start:
            continue
        if layer.end is not None and t > layer.end:
            continue
        poly = layer_polygon(layer, project, t)
        if poly and point_in_polygon(rx, ry, poly):
            return i
    return None


SIZE_FIELD = {"text": "size", "image": "size", "circle": "radius", "bars": "height", "waveform": "amplitude", "particles": "size", "progress": "width"}
ROTATION_FIELD = {"image": "rotation", "circle": "rotation"}
