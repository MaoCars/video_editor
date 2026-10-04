"""Efecto glitch digital: desplazamiento de bloques, separación RGB, scanlines, ruido e inversión."""
from __future__ import annotations

import numpy as np

from ..audio.analysis import FrameFeatures
from ..config import GlitchEffect
from .base import Effect


class Glitch(Effect[GlitchEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0.02:
            return img
        assert self.ctx is not None
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed * 31337 + frame.index)
        if rng.random() > cfg.probability:
            return img
        h, w = img.shape[:2]
        out = img.copy()
        k = min(k, 1.5)

        # --- bloques horizontales desplazados
        if cfg.block_shift > 0 and cfg.blocks > 0:
            n = max(int(round(cfg.blocks * min(k, 1.0))), 1)
            for _ in range(n):
                bh = int(rng.integers(max(h // 60, 2), max(h // 8, 4)))
                y0 = int(rng.integers(0, max(h - bh, 1)))
                dx = int(rng.normal(0, 0.08 * w) * cfg.block_shift * k)
                if dx != 0:
                    out[y0 : y0 + bh] = np.roll(img[y0 : y0 + bh], dx, axis=1)
                if cfg.invert > 0 and rng.random() < cfg.invert:
                    if out.shape[2] == 4:
                        # premultiplicado: invertir el color sólo donde hay cobertura (a·(1-c) = a - a·c)
                        out[y0 : y0 + bh, :, :3] = out[y0 : y0 + bh, :, 3:4] - out[y0 : y0 + bh, :, :3]
                    else:
                        out[y0 : y0 + bh] = 1.0 - out[y0 : y0 + bh]

        # --- separación de canales RGB
        if cfg.rgb_split > 0:
            s = int(round(self.ctx.px(6.0) * cfg.rgb_split * k))
            if s > 0:
                dr = int(rng.integers(-s, s + 1))
                db = int(rng.integers(-s, s + 1))
                dy = int(rng.integers(-max(s // 3, 1), max(s // 3, 1) + 1))
                out[..., 0] = np.roll(out[..., 0], (dy, dr), axis=(0, 1))
                out[..., 2] = np.roll(out[..., 2], (-dy, db), axis=(0, 1))

        # --- scanlines parpadeantes
        if cfg.scanlines > 0:
            sp = int(rng.integers(2, 5))
            offset = int(rng.integers(0, sp))
            rows = ((np.arange(h) + offset) % sp == 0).astype(np.float32)
            out *= (1.0 - rows * cfg.scanlines * min(k, 1.0) * 0.6)[:, None, None]

        # --- ruido digital en franjas
        if cfg.noise > 0:
            bands = int(rng.integers(1, 4))
            for _ in range(bands):
                bh = int(rng.integers(1, max(h // 40, 2)))
                y0 = int(rng.integers(0, max(h - bh, 1)))
                out[y0 : y0 + bh] += rng.uniform(-1, 1, (bh, w, 1)).astype(np.float32) * cfg.noise * k
        return out
