"""Forma de onda (osciloscopio) lineal, rellena, en barras o circular."""
from __future__ import annotations

import math

import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import WaveformLayer
from ..utils.imaging import cv2  # opcional: sólo lo usa el render CPU
from ..render.canvas import Canvas, RenderContext
from ..utils.color import to_cv
from ..utils.mathx import moving_average
from .base import Layer


class Waveform(Layer[WaveformLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        self.features = features
        cfg = self.cfg
        self.colors_lut = self._lut_static[0]
        self.cx, self.cy = ctx.rel(cfg.position)
        self.area_w = cfg.width * ctx.width
        self.amp = cfg.amplitude * ctx.height
        self.thick = max(int(round(ctx.px(cfg.thickness))), 1)
        # Ventana de audio de ~1 frame y medio para que la onda "corra" con fluidez
        self.window = int(features.samples_per_frame * 1.5) + 2

    def _samples(self, frame: FrameFeatures) -> np.ndarray:
        cfg = self.cfg
        win = self.features.waveform_window(frame.time, max(self.window, cfg.samples))
        idx = np.linspace(0, len(win) - 1, cfg.samples).astype(np.int64)
        s = win[idx]
        if cfg.smooth > 1:
            s = moving_average(s, cfg.smooth)
        return s

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        self.colors_lut = self.luts(frame)[0]
        self.cx, self.cy = self.key_position()
        layer = canvas.new_layer()
        s = self._samples(frame)
        n = len(s)
        amp = self.amp * max(self.keys.scale, 0.0)
        if cfg.style == "circular":
            r = cfg.radius * self.ctx.min_dim
            ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
            rr = r + s * amp
            xs = self.cx + np.cos(ang) * rr
            ys = self.cy + np.sin(ang) * rr
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(layer, [pts], True, to_cv(self.colors_lut[128]), self.thick, cv2.LINE_AA)
            self.composite(canvas, layer)
            return

        xs = self.cx - self.area_w / 2 + np.linspace(0, self.area_w, n)
        ys = self.cy - s * amp
        if cfg.style == "bars":
            step = self.area_w / n
            bw = max(int(step * 0.6), 1)
            for i in range(n):
                col = to_cv(self.colors_lut[int(i / max(n - 1, 1) * 255)])
                h = abs(float(s[i])) * amp
                x = int(round(xs[i]))
                cv2.rectangle(layer, (x - bw // 2, int(round(self.cy - h))), (x + bw // 2, int(round(self.cy + h))), col, -1)
        elif cfg.style == "filled":
            top = np.stack([xs, ys], axis=1)
            base_y = (self.cy + s * amp) if cfg.mirror else np.full(n, self.cy)
            bottom = np.stack([xs[::-1], base_y[::-1]], axis=1)
            poly = np.concatenate([top, bottom]).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(layer, [poly], to_cv(self.colors_lut[128], 0.85), cv2.LINE_AA)
        else:
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
            if len(cfg.colors) > 1:
                seg = max(n // 32, 1)
                for i in range(0, n - 1, seg):
                    col = to_cv(self.colors_lut[int(i / max(n - 1, 1) * 255)])
                    cv2.polylines(layer, [pts[i : i + seg + 1]], False, col, self.thick, cv2.LINE_AA)
            else:
                cv2.polylines(layer, [pts], False, to_cv(self.colors_lut[0]), self.thick, cv2.LINE_AA)
            if cfg.mirror:
                pts2 = np.stack([xs, self.cy + s * amp], axis=1).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(layer, [pts2], False, to_cv(self.colors_lut[-1]), self.thick, cv2.LINE_AA)
        self.composite(canvas, layer)
