"""Imagen / logo superpuesto con pulso al ritmo, rotación, máscara circular y vibración."""
from __future__ import annotations

import cv2
import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import ImageLayer
from ..render.canvas import Canvas, RenderContext, paste_rgba
from .background import load_image_rgba
from .base import Layer


class ImageOverlay(Layer[ImageLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        img = load_image_rgba(cfg.file)
        target = cfg.scale * ctx.min_dim
        h, w = img.shape[:2]
        k = target / max(h, w)
        # Sobredimensionamos un poco para que el pulso no pierda nitidez
        k_hi = k * (1.0 + abs(cfg.pulse) * 1.2)
        self.sprite = cv2.resize(img, (max(int(w * k_hi), 1), max(int(h * k_hi), 1)), interpolation=cv2.INTER_AREA if k_hi < 1 else cv2.INTER_CUBIC)
        self.base_scale = k / k_hi
        if cfg.circle_mask:
            sh, sw = self.sprite.shape[:2]
            mask = np.zeros((sh, sw), np.uint8)
            cv2.circle(mask, (sw // 2, sh // 2), min(sh, sw) // 2, 255, -1, cv2.LINE_AA)
            self.sprite[..., 3] = (self.sprite[..., 3].astype(np.uint16) * mask // 255).astype(np.uint8)
        self.cx, self.cy = ctx.rel(cfg.position)
        self.rng_seed = 99

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        scale = self.base_scale * (1.0 + cfg.pulse * frame.drive(cfg.pulse_trigger))
        angle = cfg.rotation + cfg.rotation_speed * frame.time
        sh, sw = self.sprite.shape[:2]
        if abs(angle) > 1e-3:
            diag = int(np.ceil(np.hypot(sh, sw) * scale)) + 2
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), -angle, scale)
            M[0, 2] += diag / 2 - sw / 2
            M[1, 2] += diag / 2 - sh / 2
            sprite = cv2.warpAffine(self.sprite, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        else:
            nw, nh = max(int(round(sw * scale)), 1), max(int(round(sh * scale)), 1)
            sprite = cv2.resize(self.sprite, (nw, nh), interpolation=cv2.INTER_AREA)
        cx, cy = self.cx, self.cy
        if cfg.shake > 0:
            rng = np.random.default_rng(self.rng_seed + frame.index)
            k = frame.kick * cfg.shake * self.ctx.scale
            cx += float(rng.uniform(-k, k))
            cy += float(rng.uniform(-k, k))
        layer = canvas.new_layer()
        paste_rgba(layer, sprite, cx, cy)
        self.composite(canvas, layer)
