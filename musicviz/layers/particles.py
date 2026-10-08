"""Sistema de partículas reactivo al ritmo.

Diseñado sin estado: la posición de cada partícula es una función analítica del tiempo,
de modo que cualquier frame puede renderizarse de forma independiente (render en paralelo).

- Partículas "ambiente": flotan continuamente; su velocidad aumenta con la energía de la música
  (se usa la integral de la energía para que el movimiento sea continuo).
- Ráfagas ("bursts"): en cada beat/kick nacen partículas desde un emisor y viven `lifetime` segundos.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import ParticlesLayer
from ..render.canvas import Canvas, RenderContext
from ..utils.color import with_alpha
from .base import Layer


class Particles(Layer[ParticlesLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        self.features = features
        rng = np.random.default_rng(cfg.seed)
        w, h = ctx.width, ctx.height
        s = ctx.scale
        self.colors_lut = self._lut_static[1]
        self.cx, self.cy = ctx.rel(cfg.position)
        self.emit_r = cfg.emitter_radius * ctx.min_dim

        # ---- ambiente
        n = cfg.count
        self.amb_pos = rng.uniform([0, 0], [w, h], size=(n, 2)).astype(np.float32)
        self.amb_dir = self._directions(rng, n, self.amb_pos)
        self.amb_speed = (cfg.speed * s * rng.uniform(0.4, 1.6, n)).astype(np.float32)
        self.amb_size = np.clip(cfg.size + rng.normal(0, cfg.size_variance / 2, n), 0.6, None).astype(np.float32) * s
        self.amb_phase = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        self.amb_color = rng.integers(0, 256, n).tolist()

        # ---- ráfagas: una por beat/kick (prefiltradas por umbral)
        trig = features.kick_onsets if cfg.burst_trigger == "kick" else features.beats
        strength = features.kick if cfg.burst_trigger == "kick" else features.beat_env
        idx = np.nonzero(trig)[0]
        idx = idx[strength[idx] >= cfg.burst_threshold]
        self.burst_times = idx / features.fps
        m = cfg.burst
        nb = len(idx)
        self.burst_p0 = np.zeros((nb, m, 2), np.float32)
        self.burst_v = np.zeros((nb, m, 2), np.float32)
        self.burst_size = np.zeros((nb, m), np.float32)
        self.burst_color = np.zeros((nb, m), np.int64)
        self.burst_life = np.zeros((nb, m), np.float32)
        for b in range(nb):
            p0 = self._emit_positions(rng, m)
            d = self._directions(rng, m, p0, burst=True)
            speed = cfg.burst_speed * s * rng.uniform(0.3, 1.2, m) * (0.5 + 0.5 * float(strength[idx[b]]))
            self.burst_p0[b] = p0
            self.burst_v[b] = d * speed[:, None]
            self.burst_size[b] = np.clip(cfg.size + rng.normal(0, cfg.size_variance, m), 0.8, None) * s
            self.burst_color[b] = rng.integers(0, 256, m)
            self.burst_life[b] = cfg.lifetime * rng.uniform(0.5, 1.0, m)
        self.gravity = cfg.gravity * s

    # ---------------------------------------------------------------- helpers
    def _emit_positions(self, rng: np.random.Generator, m: int) -> np.ndarray:
        cfg = self.cfg
        assert self.ctx is not None
        w, h = self.ctx.width, self.ctx.height
        if cfg.emitter == "center":
            return np.repeat(np.array([[self.cx, self.cy]], np.float32), m, axis=0) + rng.normal(0, 2, (m, 2))
        if cfg.emitter == "ring":
            a = rng.uniform(0, 2 * math.pi, m)
            return np.stack([self.cx + np.cos(a) * self.emit_r, self.cy + np.sin(a) * self.emit_r], axis=1).astype(np.float32)
        if cfg.emitter == "bottom":
            return np.stack([rng.uniform(0, w, m), np.full(m, h + 4.0)], axis=1).astype(np.float32)
        if cfg.emitter == "top":
            return np.stack([rng.uniform(0, w, m), np.full(m, -4.0)], axis=1).astype(np.float32)
        return rng.uniform([0, 0], [w, h], size=(m, 2)).astype(np.float32)

    def _directions(self, rng: np.random.Generator, m: int, pos: np.ndarray, burst: bool = False) -> np.ndarray:
        d = self.cfg.direction
        if burst and self.cfg.emitter in ("center", "ring") and d not in ("in",):
            d = "out"
        if d == "random" or (burst and d in ("up", "down", "left", "right") and self.cfg.emitter == "screen"):
            a = rng.uniform(0, 2 * math.pi, m)
            return np.stack([np.cos(a), np.sin(a)], axis=1).astype(np.float32)
        base = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        if d in base:
            v = np.repeat(np.array([base[d]], np.float32), m, axis=0)
            jitter = rng.normal(0, 0.25, (m, 2)).astype(np.float32)
            v = v + jitter
        else:  # out / in
            v = pos - np.array([self.cx, self.cy], np.float32)
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(norm > 1e-3, v / np.maximum(norm, 1e-6), rng.normal(0, 1, (m, 2)))
            v = v + rng.normal(0, 0.15, (m, 2))
            if d == "in":
                v = -v
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        return v.astype(np.float32)

    def _draw(self, layer: np.ndarray, x: float, y: float, size: float, col, vel=None) -> None:
        cfg = self.cfg
        xi, yi = int(round(x)), int(round(y))
        r = max(int(round(size)), 1)
        if cfg.shape == "square":
            cv2.rectangle(layer, (xi - r, yi - r), (xi + r, yi + r), col, -1)
        elif cfg.shape == "streak" and vel is not None:
            vx, vy = vel
            cv2.line(layer, (xi, yi), (int(round(x - vx * 0.06)), int(round(y - vy * 0.06))), col, max(r, 1), cv2.LINE_AA)
        else:
            cv2.circle(layer, (xi, yi), r, col, -1, cv2.LINE_AA)

    # ---------------------------------------------------------------- geometría (compartida CPU/GPU)
    def batches(self, frame: FrameFeatures) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Partículas vivas en este frame: lista de (pos (m,2), tamaño (m,), alfa (m,), índice de color (m,), velocidad (m,2))."""
        assert self.ctx is not None
        cfg = self.cfg
        w, h = self.ctx.width, self.ctx.height
        t = frame.time
        inten = self.intensity
        size_k = (1.0 + cfg.react_size * frame.bass) * (0.75 + 0.25 * inten)
        alpha_k = min(inten, 1.0)
        out = []
        if cfg.count > 0:
            musical_t = t + cfg.energy_speed * float(self.features.energy_integral[frame.index])
            disp = self.amb_dir * (self.amb_speed * musical_t)[:, None]
            pos = self.amb_pos + disp
            if cfg.direction in ("out", "in"):
                period = max(self.ctx.min_dim * 0.9, 1.0)
                travel = (self.amb_speed * musical_t) % period
                pos = np.array([self.cx, self.cy], np.float32) + self.amb_dir * travel[:, None]
                if cfg.direction == "in":
                    pos = np.array([self.cx, self.cy], np.float32) + self.amb_dir * (period - travel)[:, None]
                pos += (self.amb_pos - np.array([w / 2, h / 2], np.float32)) * 0.15
            margin = 8.0
            pos = pos.copy()
            pos[:, 0] = (pos[:, 0] + margin) % (w + 2 * margin) - margin
            pos[:, 1] = (pos[:, 1] + margin) % (h + 2 * margin) - margin
            tw = 1.0 - cfg.twinkle * 0.5 * (1.0 + np.sin(self.amb_phase + t * 3.0 + self.amb_speed * 0.01))
            level = (0.55 + 0.45 * frame.rms) * alpha_k
            out.append((pos.astype(np.float32), (self.amb_size * size_k).astype(np.float32), (tw * level).astype(np.float32), np.asarray(self.amb_color), (self.amb_dir * self.amb_speed[:, None]).astype(np.float32)))
        if cfg.burst > 0 and len(self.burst_times):
            lo = int(np.searchsorted(self.burst_times, t - cfg.lifetime, side="left"))
            hi = int(np.searchsorted(self.burst_times, t, side="right"))
            for b in range(lo, hi):
                dt = t - self.burst_times[b]
                life = self.burst_life[b]
                alive = dt < life
                if not np.any(alive):
                    continue
                frac = np.clip(dt / np.maximum(life, 1e-6), 0.0, 1.0)
                p = self.burst_p0[b] + self.burst_v[b] * dt
                p[:, 1] += 0.5 * self.gravity * dt * dt
                drag = 1.0 / (1.0 + dt * 1.5)
                p = self.burst_p0[b] + (p - self.burst_p0[b]) * drag
                alpha = (1.0 - frac) ** 1.5 * alpha_k
                sizes = self.burst_size[b] * (1.0 - 0.6 * frac) * size_k
                on = alive & (p[:, 0] >= -20) & (p[:, 1] >= -20) & (p[:, 0] <= w + 20) & (p[:, 1] <= h + 20)
                if not np.any(on):
                    continue
                out.append((p[on].astype(np.float32), sizes[on].astype(np.float32), alpha[on].astype(np.float32), self.burst_color[b][on], (self.burst_v[b][on] * drag).astype(np.float32)))
        return out

    # ---------------------------------------------------------------- render
    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        layer = canvas.new_layer()
        lut = self.colors_lut = self.luts(frame)[1]
        for pos, sizes, alphas, cols, vel in self.batches(frame):
            xs, ys = pos[:, 0].tolist(), pos[:, 1].tolist()
            sz, al, ci = sizes.tolist(), alphas.tolist(), cols.tolist()
            for i in range(len(xs)):
                self._draw(layer, xs[i], ys[i], sz[i], with_alpha(lut[ci[i]], al[i]), vel[i])
        self.composite(canvas, layer)
