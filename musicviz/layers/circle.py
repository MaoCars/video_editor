"""Espectro circular (estilo Trap Nation): barras radiales, línea, relleno, puntos o rayos."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import CircleLayer
from ..render.canvas import Canvas, RenderContext
from ..utils.color import parse_color, to_cv, with_alpha
from .base import Layer
from .bars import build_values


class CircleSpectrum(Layer[CircleLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        self.n = cfg.bands or features.spectrum.shape[1]
        self.colors_lut, self.lut_int = self._lut_static
        self.cx, self.cy = ctx.rel(cfg.position)
        self.base_r = cfg.radius * ctx.min_dim
        self.max_len = cfg.length * ctx.min_dim
        self.thick = max(int(round(ctx.px(cfg.thickness))), 1)
        self.ring_thick = max(int(round(ctx.px(cfg.ring_thickness))), 1)
        self._ring_fixed = to_cv(parse_color(cfg.ring_color)) if cfg.ring_color else None
        self.ring_color = self._ring_fixed or to_cv(self.colors_lut[0])
        # Ángulos: con mirror el espectro se refleja (graves arriba, agudos abajo) → forma simétrica.
        self.angles = np.linspace(0.0, 2.0 * math.pi, self.n, endpoint=False, dtype=np.float32)
        self._angle_lut_cache: tuple[float, int, np.ndarray] | None = None

    def _angle_colors(self, rot: float) -> np.ndarray:
        """Imagen (H, W, 3) uint8 con el gradiente angular (para el estilo 'filled')."""
        lut_id = id(self.colors_lut)
        if self._angle_lut_cache is not None and abs(self._angle_lut_cache[0] - rot) < 1e-4 and self._angle_lut_cache[1] == lut_id:
            return self._angle_lut_cache[2]
        assert self.ctx is not None
        ys, xs = np.mgrid[0 : self.ctx.height, 0 : self.ctx.width].astype(np.float32)
        ang = (np.arctan2(ys - self.cy, xs - self.cx) - rot) % (2.0 * math.pi)
        t = ang / (2.0 * math.pi)
        if self.cfg.mirror:
            t = 1.0 - np.abs(2.0 * t - 1.0)
        rgb = (self.colors_lut[(np.clip(t, 0, 1) * 255).astype(np.int32)][..., :3] * 255).astype(np.uint8)
        self._angle_lut_cache = (rot, lut_id, rgb)
        return rgb

    def _values(self, frame: FrameFeatures) -> np.ndarray:
        cfg = self.cfg
        if cfg.mirror:
            # Primera mitad del círculo: graves→agudos; segunda mitad reflejada.
            half = self.n // 2
            v = build_values(frame.spectrum, half, False)
            vals = np.concatenate([v, v[::-1]])
            if len(vals) < self.n:
                vals = np.append(vals, vals[-1])
            return vals[: self.n]
        return build_values(frame.spectrum, self.n, False)

    def _color(self, i: int, value: float, fade: float = 1.0) -> tuple[int, int, int, int]:
        if self.cfg.gradient == "angle":
            t = i / max(self.n - 1, 1)
            if self.cfg.mirror:
                t = 1.0 - abs(2.0 * t - 1.0)  # simétrico
        else:
            t = value
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return with_alpha(self.lut_int[int(t * 255)], fade)

    def bar_colors(self, values: np.ndarray, fade: float) -> np.ndarray:
        """Color RGBA float (n,4) de cada barra según el gradiente (angle o value)."""
        n = self.n
        if self.cfg.gradient == "angle":
            t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
            if self.cfg.mirror:
                t = 1.0 - np.abs(2.0 * t - 1.0)
        else:
            t = np.clip(values, 0.0, 1.0)
        cols = self.colors_lut[np.clip(t * 255, 0, 255).astype(np.int32)].copy()
        cols[:, 3] *= fade
        return cols

    def geometry(self, frame: FrameFeatures) -> list[dict]:
        """Geometría por anillo (compartida CPU/GPU): radios, puntos interiores/exteriores y colores."""
        assert self.ctx is not None
        cfg = self.cfg
        self.colors_lut, self.lut_int = self.luts(frame)
        self.ring_color = self._ring_fixed or to_cv(self.colors_lut[0])
        self.cx, self.cy = self.key_position()
        values = self._values(frame)
        pulse = (1.0 + cfg.pulse * self.intensity * frame.drive(cfg.pulse_trigger)) * max(self.keys.scale, 0.0)
        rot = math.radians(cfg.rotation + cfg.rotation_speed * frame.time + self.keys.rotation)
        rings = []
        for ring_i in range(cfg.rings):
            spread = 1.0 + ring_i * cfg.ring_spread
            r0 = self.base_r * pulse * spread
            ring_rot = rot + ring_i * (math.pi / self.n) * 0.5
            fade = 1.0 / (1.0 + ring_i * 0.6)
            ang = self.angles + ring_rot
            cos, sin = np.cos(ang), np.sin(ang)
            lens = values * self.max_len * (1.0 - 0.25 * ring_i) * max(self.keys.scale, 0.0)
            inner = lens if cfg.inner else 0.0
            rings.append({
                "r0": float(r0), "rot": float(ring_rot), "fade": float(fade), "values": values,
                "x_in": self.cx + cos * (r0 - inner), "y_in": self.cy + sin * (r0 - inner),
                "x_out": self.cx + cos * (r0 + lens), "y_out": self.cy + sin * (r0 + lens),
                "colors": self.bar_colors(values, fade),
            })
        return rings

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        layer = canvas.new_layer()
        for ring in self.geometry(frame):
            r0, ring_rot, fade = ring["r0"], ring["rot"], ring["fade"]
            vals = ring["values"].tolist()
            x_in, y_in, x_out, y_out = ring["x_in"], ring["y_in"], ring["x_out"], ring["y_out"]
            xi_in, yi_in = np.rint(x_in).astype(int).tolist(), np.rint(y_in).astype(int).tolist()
            xi_out, yi_out = np.rint(x_out).astype(int).tolist(), np.rint(y_out).astype(int).tolist()
            cols = [to_cv(c) for c in ring["colors"]]

            if cfg.ring:
                cv2.circle(layer, (int(round(self.cx)), int(round(self.cy))), int(round(r0)), self.ring_color, self.ring_thick, cv2.LINE_AA)

            style = cfg.style
            if style in ("bars", "rays"):
                thick = self.thick if style == "bars" else max(self.thick // 2, 1)
                for i in range(self.n):
                    col = cols[i]
                    p0 = (xi_in[i], yi_in[i])
                    p1 = (xi_out[i], yi_out[i])
                    cv2.line(layer, p0, p1, col, thick, cv2.LINE_AA)
                    if cfg.rounded and thick > 2:
                        cv2.circle(layer, p1, thick // 2, col, -1, cv2.LINE_AA)
                        if cfg.inner:
                            cv2.circle(layer, p0, thick // 2, col, -1, cv2.LINE_AA)
            elif style == "dots":
                for i in range(self.n):
                    rad = max(int(round(self.thick * (0.6 + vals[i]))), 1)
                    cv2.circle(layer, (xi_out[i], yi_out[i]), rad, cols[i], -1, cv2.LINE_AA)
            else:  # line / filled
                pts_out = np.stack([x_out, y_out], axis=1).astype(np.int32).reshape(-1, 1, 2)
                mid_col = to_cv(self.colors_lut[128], fade)
                if style == "filled":
                    mask = np.zeros(layer.shape[:2], np.uint8)
                    cv2.fillPoly(mask, [pts_out], 255, cv2.LINE_AA)
                    if cfg.inner:
                        pts_in = np.stack([x_in, y_in], axis=1).astype(np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(mask, [pts_in], 0, cv2.LINE_AA)
                    else:
                        cv2.circle(mask, (int(round(self.cx)), int(round(self.cy))), int(round(r0)), 0, -1, cv2.LINE_AA)
                    if cfg.gradient == "angle" and len(cfg.colors) > 1:
                        colors = self._angle_colors(ring_rot)
                    else:
                        colors = np.empty(layer.shape[:2] + (3,), np.uint8)
                        colors[:] = mid_col[:3]
                    sel = mask > 0
                    layer[sel, :3] = colors[sel]
                    layer[..., 3] = np.maximum(layer[..., 3], (mask.astype(np.float32) * fade).astype(np.uint8))
                    if cfg.ring:
                        cv2.circle(layer, (int(round(self.cx)), int(round(self.cy))), int(round(r0)), self.ring_color, self.ring_thick, cv2.LINE_AA)
                else:
                    for i in range(self.n):
                        j = (i + 1) % self.n
                        cv2.line(layer, (xi_out[i], yi_out[i]), (xi_out[j], yi_out[j]), cols[i], self.thick, cv2.LINE_AA)
                    if cfg.inner:
                        for i in range(self.n):
                            j = (i + 1) % self.n
                            cv2.line(layer, (xi_in[i], yi_in[i]), (xi_in[j], yi_in[j]), cols[i], self.thick, cv2.LINE_AA)
        self.composite(canvas, layer)
