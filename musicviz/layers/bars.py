"""Espectro de barras (estilo Monstercat / NCS) con espejo, simetría, gradientes y estilos."""
from __future__ import annotations

import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import BarsLayer
from ..utils.imaging import cv2  # opcional: sólo lo usa el render CPU
from ..render.canvas import Canvas, RenderContext
from ..utils.color import to_cv
from .base import Layer


def resample_bands(spectrum: np.ndarray, n: int) -> np.ndarray:
    """Interpola el espectro de N bandas a n bandas."""
    if n == len(spectrum):
        return spectrum
    x_old = np.linspace(0.0, 1.0, len(spectrum))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, spectrum).astype(np.float32)


def build_values(spectrum: np.ndarray, n: int, symmetric: bool) -> np.ndarray:
    """Devuelve n valores; si symmetric, graves en el centro y agudos en los bordes."""
    if not symmetric:
        return resample_bands(spectrum, n)
    half = (n + 1) // 2
    v = resample_bands(spectrum, half)
    left = v[::-1]
    right = v[: n - half] if n - half > 0 else v[:0]
    return np.concatenate([left, right]).astype(np.float32)


class Bars(Layer[BarsLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        self.n = cfg.bands or features.spectrum.shape[1]
        self.colors_lut = self._lut_static[0]
        self.area_w = cfg.width * ctx.width
        self.slot = self.area_w / self.n
        self.bar_w = max(self.slot * (1.0 - cfg.gap), 1.0)
        cx, cy = ctx.rel(cfg.position)
        self.x0 = cx - self.area_w / 2.0
        self.cy = cy
        self.base_max_h = cfg.height * ctx.height
        self.max_h = self.base_max_h
        self.seg_h = max(ctx.px(cfg.segment_height), 2.0)

    def _color(self, i: int, value: float) -> tuple[int, int, int, int]:
        t = i / max(self.n - 1, 1) if self.cfg.gradient == "index" else value
        return to_cv(self.colors_lut[int(np.clip(t, 0, 1) * 255)])

    def geometry(self, frame: FrameFeatures) -> dict:
        """Geometría compartida CPU/GPU: valores, centros x, alturas (px) y colores RGBA float (n,4)."""
        assert self.ctx is not None
        cfg = self.cfg
        self.colors_lut = self.luts(frame)[0]
        cx, self.cy = self.key_position()
        self.x0 = cx - self.area_w / 2.0
        self.max_h = self.base_max_h * max(self.keys.scale, 0.0)
        values = build_values(frame.spectrum, self.n, cfg.symmetric)
        if cfg.bass_boost > 0:
            values = np.clip(values * (1.0 + cfg.bass_boost * frame.bass), 0.0, 1.0)
        heights = np.maximum(values * self.max_h, cfg.min_height * self.ctx.height)
        xcs = self.x0 + self.slot * (np.arange(self.n) + 0.5)
        if cfg.gradient == "index":
            t = np.arange(self.n, dtype=np.float32) / max(self.n - 1, 1)
        else:
            t = np.clip(values, 0.0, 1.0)
        colors = self.colors_lut[np.clip(t * 255, 0, 255).astype(np.int32)]
        return {"values": values, "xc": xcs.astype(np.float32), "heights": heights.astype(np.float32), "colors": colors, "cy": self.cy, "bar_w": self.bar_w}

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        geo = self.geometry(frame)
        values = geo["values"]
        layer = canvas.new_layer()
        bw = self.bar_w
        radius = int(round(bw / 2)) if cfg.rounded else 0
        for i, v in enumerate(values):
            h = max(float(v) * self.max_h, cfg.min_height * self.ctx.height)
            xc = self.x0 + self.slot * (i + 0.5)
            xl, xr = int(round(xc - bw / 2)), int(round(xc + bw / 2))
            if xr <= xl:
                xr = xl + 1
            col = self._color(i, float(v))
            top = self.cy - h
            bottom = self.cy + h if cfg.mirror else self.cy
            if cfg.style == "dots":
                cv2.circle(layer, (int(round(xc)), int(round(top))), max(radius, 1), col, -1, cv2.LINE_AA)
                if cfg.mirror:
                    cv2.circle(layer, (int(round(xc)), int(round(bottom))), max(radius, 1), col, -1, cv2.LINE_AA)
                continue
            if cfg.style == "segments":
                y = self.cy
                while y - self.seg_h >= top - 1e-6:
                    cv2.rectangle(layer, (xl, int(round(y - self.seg_h * 0.7))), (xr, int(round(y))), col, -1)
                    if cfg.mirror:
                        yy = 2 * self.cy - y
                        cv2.rectangle(layer, (xl, int(round(yy))), (xr, int(round(yy + self.seg_h * 0.7))), col, -1)
                    y -= self.seg_h
                continue
            thickness = -1 if cfg.style == "solid" else max(int(round(self.ctx.px(2))), 1)
            if cfg.rounded and radius > 0 and cfg.style == "solid":
                # Rectángulo + círculos en los extremos (barra redondeada)
                t0 = int(round(top + radius))
                b0 = int(round(bottom - radius))
                if b0 > t0:
                    cv2.rectangle(layer, (xl, t0), (xr, b0), col, -1)
                cv2.circle(layer, (int(round(xc)), t0), radius, col, -1, cv2.LINE_AA)
                cv2.circle(layer, (int(round(xc)), max(b0, t0)), radius, col, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(layer, (xl, int(round(top))), (xr, int(round(bottom))), col, thickness)
        if cfg.baseline:
            col = to_cv(self.colors_lut[128], 0.6)
            y = int(round(self.cy))
            cv2.line(layer, (int(self.x0), y), (int(self.x0 + self.slot * self.n), y), col, max(int(self.ctx.px(2)), 1), cv2.LINE_AA)
        self.composite(canvas, layer)
