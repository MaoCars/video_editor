"""Imagen / portada / logo: encaje con recorte, formas (cuadrada, redonda, esquinas redondeadas),
anclaje en 9 posiciones, borde, sombra, pulso al ritmo, rotación y vibración."""
from __future__ import annotations

import math

import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import ImageLayer
from ..render.canvas import Canvas, RenderContext, anchor_center, paste_rgba, rounded_rect_mask
from ..utils.color import parse_color
from ..utils.imaging import blur_u8, cv2, ellipse_ring, fit_into_box, load_image_rgba
from .base import Layer


class ImageOverlay(Layer[ImageLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        img = load_image_rgba(cfg.file)
        h, w = img.shape[:2]
        aspect = 1.0 if cfg.shape in ("square", "circle") else (cfg.aspect or (w / h))
        long_side = cfg.size * ctx.min_dim
        # Sobredimensionamos un poco para que el pulso no pierda nitidez
        over = 1.0 + abs(cfg.pulse) * 1.2
        if aspect >= 1.0:
            bw, bh = long_side * over, long_side * over / aspect
        else:
            bw, bh = long_side * over * aspect, long_side * over
        bw_i, bh_i = max(int(round(bw)), 2), max(int(round(bh)), 2)
        sprite = fit_into_box(img, bw_i, bh_i, cfg.fit, cfg.focus)
        radius = ctx.px(cfg.corner_radius) * over
        mask = rounded_rect_mask(bw_i, bh_i, radius, cfg.shape)
        if cfg.shape in ("circle", "rounded"):
            sprite[..., 3] = (sprite[..., 3].astype(np.uint16) * mask // 255).astype(np.uint8)
        if cfg.border > 0:
            bpx = max(int(round(ctx.px(cfg.border) * over)), 1)
            col = tuple(int(c * 255) for c in parse_color(cfg.border_color))
            ring = np.zeros((bh_i, bw_i, 4), np.uint8)
            if cfg.shape == "circle":
                ring = ellipse_ring(bw_i, bh_i, bpx, col)
            else:
                outer = mask
                inner = rounded_rect_mask(max(bw_i - 2 * bpx, 1), max(bh_i - 2 * bpx, 1), max(radius - bpx, 0), cfg.shape)
                band = outer.copy()
                band[bpx : bpx + inner.shape[0], bpx : bpx + inner.shape[1]] = np.minimum(band[bpx : bpx + inner.shape[0], bpx : bpx + inner.shape[1]], 255 - inner)
                ring[..., :3] = col[:3]
                ring[..., 3] = (band.astype(np.uint16) * col[3] // 255).astype(np.uint8)
            paste_rgba(sprite, ring, bw_i / 2, bh_i / 2)
        self.sprite = sprite
        self.base_scale = 1.0 / over
        self.shadow_sprite = None
        if cfg.shadow > 0:
            blur = ctx.px(cfg.shadow_blur) * over
            pad = int(blur * 3) + 4
            sh = np.zeros((bh_i + 2 * pad, bw_i + 2 * pad, 4), np.uint8)
            sh[pad : pad + bh_i, pad : pad + bw_i, 3] = sprite[..., 3]
            if blur > 0.5:
                sh[..., 3] = blur_u8(sh[..., 3], blur)
            sh[..., 3] = (sh[..., 3].astype(np.float32) * min(cfg.shadow, 1.0)).astype(np.uint8)
            self.shadow_sprite = sh
        self.x, self.y = ctx.rel(cfg.position)

    def _transform(self, sprite: np.ndarray, scale: float, angle: float) -> np.ndarray:
        sh, sw = sprite.shape[:2]
        if abs(angle) > 1e-3:
            diag = int(np.ceil(np.hypot(sh, sw) * scale)) + 2
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), -angle, scale)
            M[0, 2] += diag / 2 - sw / 2
            M[1, 2] += diag / 2 - sh / 2
            return cv2.warpAffine(sprite, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        nw, nh = max(int(round(sw * scale)), 1), max(int(round(sh * scale)), 1)
        return cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    def placement(self, frame: FrameFeatures) -> tuple[float, float, float, float, float]:
        """(cx, cy, escala, ángulo en grados, desenfoque 0..1) del sprite en este frame (compartido CPU/GPU)."""
        assert self.ctx is not None
        cfg = self.cfg
        anim = self.anim
        keys = self.keys
        scale = self.base_scale * (1.0 + cfg.pulse * frame.drive(cfg.pulse_trigger)) * anim.scale * max(keys.scale, 0.0)
        angle = cfg.rotation + cfg.rotation_speed * frame.time + keys.rotation
        kx, ky = self.key_position()
        x, y = kx + anim.dx, ky + anim.dy
        if cfg.shake > 0:
            rng = np.random.default_rng(99 + frame.index)
            k = frame.kick * cfg.shake * self.ctx.scale
            x += float(rng.uniform(-k, k))
            y += float(rng.uniform(-k, k))
        # El anclaje se calcula con el tamaño sin rotar para que la imagen no "salte" al girar
        bw, bh = self.sprite.shape[1] * scale, self.sprite.shape[0] * scale
        cx, cy = anchor_center(x, y, bw, bh, cfg.anchor)
        return cx, cy, scale, angle, anim.blur

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        cx, cy, scale, angle, blur = self.placement(frame)
        sprite = self._transform(self.sprite, scale, angle)
        if blur > 0:
            sprite = cv2.GaussianBlur(sprite, (0, 0), max(blur * self.ctx.px(20), 0.3))
        layer = canvas.new_layer()
        if self.shadow_sprite is not None:
            shadow = self._transform(self.shadow_sprite, scale, angle)
            ox, oy = self.ctx.px(cfg.shadow_offset[0]), self.ctx.px(cfg.shadow_offset[1])
            paste_rgba(layer, shadow, cx + ox, cy + oy)
        paste_rgba(layer, sprite, cx, cy)
        self.composite(canvas, layer)
