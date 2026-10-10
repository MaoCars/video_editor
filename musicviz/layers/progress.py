"""Barra de progreso de la canción, opcionalmente con tiempo transcurrido / total."""
from __future__ import annotations

import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import ProgressLayer
from ..utils.imaging import cv2  # opcional: sólo lo usa el render CPU
from ..render.canvas import Canvas, RenderContext, paste_rgba
from ..utils.color import parse_color, to_cv
from .base import Layer
from .text import find_font, render_text_rgba


def fmt_time(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


class ProgressBar(Layer[ProgressLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        self.duration = features.duration
        self.cx, self.cy = ctx.rel(cfg.position)
        self.w = cfg.width * ctx.width
        self.thick = max(int(round(ctx.px(cfg.thickness))), 1)
        self.col = to_cv(parse_color(cfg.color))
        self.bg = to_cv(parse_color(cfg.bg_color))
        self.font = find_font(cfg.font, max(int(ctx.px(26)), 8)) if cfg.show_time else None

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        layer = canvas.new_layer()
        x0 = int(round(self.cx - self.w / 2))
        x1 = int(round(self.cx + self.w / 2))
        y = int(round(self.cy))
        cv2.line(layer, (x0, y), (x1, y), self.bg, self.thick, cv2.LINE_AA)
        xp = int(round(x0 + (x1 - x0) * frame.progress))
        if xp > x0:
            cv2.line(layer, (x0, y), (xp, y), self.col, self.thick, cv2.LINE_AA)
        cv2.circle(layer, (xp, y), self.thick + 2, self.col, -1, cv2.LINE_AA)
        if self.font is not None:
            gap = self.ctx.px(14)
            left = render_text_rgba(fmt_time(frame.time), self.font, cfg.color)
            right = render_text_rgba(fmt_time(self.duration), self.font, cfg.color)
            paste_rgba(layer, left, x0 - left.shape[1] / 2 - gap, y)
            paste_rgba(layer, right, x1 + right.shape[1] / 2 + gap, y)
        self.composite(canvas, layer)
