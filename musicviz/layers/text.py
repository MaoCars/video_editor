"""Texto (título, artista) renderizado con Pillow, con pulso y glow."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import TextLayer
from ..render.canvas import Canvas, RenderContext, paste_rgba
from ..utils.color import parse_color
from .base import Layer

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def find_font(name: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    if name:
        candidates.append(name)
        if sys.platform.startswith("win"):
            candidates.append(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name))
            if not name.lower().endswith((".ttf", ".otf")):
                candidates.append(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name + ".ttf"))
    candidates += _FONT_CANDIDATES
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_text_rgba(text: str, font, color: str, letter_spacing: float = 0.0) -> np.ndarray:
    """Renderiza texto a un sprite RGBA uint8 ajustado al contenido."""
    rgba = tuple(int(c * 255) for c in parse_color(color))
    if letter_spacing <= 0:
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        l, t, r, b = probe.textbbox((0, 0), text, font=font)
        pad = 4
        img = Image.new("RGBA", (max(r - l, 1) + 2 * pad, max(b - t, 1) + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=rgba)
        return np.array(img)
    # Espaciado manual letra a letra
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    widths = [probe.textlength(ch, font=font) for ch in text]
    l, t, r, b = probe.textbbox((0, 0), text or " ", font=font)
    total = int(sum(widths) + letter_spacing * max(len(text) - 1, 0)) + 8
    img = Image.new("RGBA", (max(total, 1), max(b - t, 1) + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = 4.0
    for ch, wch in zip(text, widths):
        draw.text((x, 4 - t), ch, font=font, fill=rgba)
        x += wch + letter_spacing
    return np.array(img)


class TextOverlay(Layer[TextLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        text = cfg.text.upper() if cfg.uppercase else cfg.text
        px = max(int(cfg.size * ctx.height * (1.0 + abs(cfg.pulse) * 1.2)), 4)
        font = find_font(cfg.font, px)
        self.sprite = render_text_rgba(text, font, cfg.color, ctx.px(cfg.letter_spacing))
        self.base_scale = 1.0 / (1.0 + abs(cfg.pulse) * 1.2)
        self.cx, self.cy = ctx.rel(cfg.position)

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        scale = self.base_scale * (1.0 + cfg.pulse * frame.drive(cfg.pulse_trigger))
        sh, sw = self.sprite.shape[:2]
        sprite = cv2.resize(self.sprite, (max(int(sw * scale), 1), max(int(sh * scale), 1)), interpolation=cv2.INTER_AREA)
        layer = canvas.new_layer()
        paste_rgba(layer, sprite, self.cx, self.cy)
        self.composite(canvas, layer)
