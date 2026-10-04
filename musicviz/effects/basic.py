"""Efectos: bloom, aberración cromática, shake/zoom, viñeta, color, pixelado, strobe,
kaleidoscopio, desenfoque radial, scanlines y grano."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import (
    BloomEffect,
    ChromaticEffect,
    ColorEffect,
    FilmGrainEffect,
    KaleidoEffect,
    PixelateEffect,
    RadialBlurEffect,
    ScanlinesEffect,
    ShakeEffect,
    StrobeEffect,
    VignetteEffect,
)
from ..render.canvas import RenderContext, fast_blur
from ..utils.color import parse_color
from .base import Effect


def _zoom(img: np.ndarray, factor: float, angle: float = 0.0, dx: float = 0.0, dy: float = 0.0, border=cv2.BORDER_REFLECT) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, factor)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)


class Bloom(Effect[BloomEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0:
            return img
        assert self.ctx is not None
        cfg = self.cfg
        h, w = img.shape[:2]
        down = 4
        small = cv2.resize(img, (max(w // down, 1), max(h // down, 1)), interpolation=cv2.INTER_AREA)
        bright = cv2.max(small - np.float32(cfg.threshold), 0.0)
        bright *= np.float32(1.0 / max(1.0 - cfg.threshold, 1e-3))
        bright = cv2.GaussianBlur(bright, (0, 0), max(self.ctx.px(cfg.radius) / down, 0.5))
        halo = cv2.resize(bright, (w, h), interpolation=cv2.INTER_LINEAR)
        halo *= np.float32(cfg.strength * k)
        img += halo
        return img


class Chromatic(Effect[ChromaticEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0:
            return img
        assert self.ctx is not None
        amt = self.ctx.px(self.cfg.amount) * k
        h, w = img.shape[:2]
        f = amt / (w / 2.0)
        channels = list(cv2.split(img))  # canales contiguos (mucho más rápido que img[..., i])
        channels[0] = _zoom(channels[0], 1.0 + f)
        channels[2] = _zoom(channels[2], max(1.0 - f, 0.5))
        return cv2.merge(channels)


class Shake(Effect[ShakeEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0.001:
            return img
        assert self.ctx is not None
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed * 100003 + frame.index)
        amp = self.ctx.px(cfg.amount) * k
        dx, dy = rng.uniform(-amp, amp, 2)
        ang = rng.uniform(-1, 1) * cfg.rotation * k
        zoom = 1.0 + cfg.zoom * k
        return _zoom(img, zoom, ang, dx, dy)


class Vignette(Effect[VignetteEffect]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        xs = (np.arange(ctx.width, dtype=np.float32) - ctx.width / 2) / (ctx.width / 2)
        ys = (np.arange(ctx.height, dtype=np.float32) - ctx.height / 2) / (ctx.height / 2)
        r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / math.sqrt(2.0)
        soft = max(cfg.softness, 0.01)
        m = np.clip((r - (1.0 - soft)) / soft, 0.0, 1.0)
        m = m * m * (3 - 2 * m)
        self.mask = (1.0 - m * cfg.strength)[..., None].astype(np.float32)

    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0:
            return img
        mask = self.mask if k >= 1.0 else 1.0 - (1.0 - self.mask) * k
        img *= mask
        return img


class ColorGrade(Effect[ColorEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0:
            return img
        cfg = self.cfg
        alpha = img[..., 3:] if img.shape[2] == 4 else None
        out = np.ascontiguousarray(img[..., :3]) if alpha is not None else img
        hue = cfg.hue_speed * frame.time + cfg.hue_react * frame.rms * k
        if abs(hue) > 1e-3 or cfg.saturation != 1.0:
            hsv = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0] + hue) % 360.0
            if cfg.saturation != 1.0:
                hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + (cfg.saturation - 1.0) * k), 0, 1)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if cfg.contrast != 1.0:
            c = 1.0 + (cfg.contrast - 1.0) * k
            out = (out - 0.5) * c + 0.5
        if cfg.brightness != 0.0:
            out = out + cfg.brightness * k
        if cfg.gamma != 1.0:
            out = np.power(np.clip(out, 0, 1), 1.0 / cfg.gamma)
        if cfg.posterize > 1:
            levels = float(cfg.posterize)
            out = np.floor(np.clip(out, 0, 1) * levels) / (levels - 1)
        if alpha is not None:
            return np.concatenate([out, alpha], axis=2)
        return out


class Pixelate(Effect[PixelateEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0.05:
            return img
        assert self.ctx is not None
        size = max(int(round(self.ctx.px(self.cfg.size) * k)), 1)
        if size <= 1:
            return img
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(w // size, 1), max(h // size, 1)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class Strobe(Effect[StrobeEffect]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        self.color = np.array(parse_color(self.cfg.color)[:3], np.float32)

    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = min(self.drive(frame), 1.0)
        if k <= 0.01:
            return img
        if img.shape[2] == 4:
            out = img.copy()
            out[..., :3] = img[..., :3] * (1.0 - k) + self.color * k
            out[..., 3:] = np.maximum(img[..., 3:], k)
            return out
        return img * (1.0 - k) + self.color * k


class Kaleido(Effect[KaleidoEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0:
            return img
        cfg = self.cfg
        h, w = img.shape[:2]
        out = img.copy()
        if cfg.segments == 4 or cfg.axis == "horizontal":
            half = w // 2
            out[:, w - half :] = out[:, :half][:, ::-1]
        if cfg.segments == 4 or cfg.axis == "vertical":
            half = h // 2
            out[h - half :, :] = out[:half, :][::-1]
        if k < 1.0:
            out = img * (1.0 - k) + out * k
        return out


class RadialBlur(Effect[RadialBlurEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0.01:
            return img
        cfg = self.cfg
        n = max(int(cfg.samples), 2)
        acc = img.copy()
        for i in range(1, n):
            z = 1.0 + cfg.amount * k * (i / (n - 1))
            acc += _zoom(img, z)
        return acc / n


class Scanlines(Effect[ScanlinesEffect]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        sp = max(int(self.cfg.spacing), 2)
        rows = (np.arange(ctx.height) % sp == 0).astype(np.float32)
        self.mask = (1.0 - rows * self.cfg.darkness)[:, None, None]

    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0:
            return img
        mask = self.mask if k >= 1.0 else 1.0 - (1.0 - self.mask) * k
        img *= mask
        return img


class FilmGrain(Effect[FilmGrainEffect]):
    def apply(self, img: np.ndarray, frame: FrameFeatures) -> np.ndarray:
        k = self.drive(frame)
        if k <= 0:
            return img
        rng = np.random.default_rng(self.cfg.seed * 7919 + frame.index)
        h, w = img.shape[:2]
        noise = rng.standard_normal((h // 2, w // 2, 1), dtype=np.float32) * (self.cfg.amount * k)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        if img.shape[2] == 4:
            img[..., :3] += noise * img[..., 3:]  # sólo donde hay contenido (no ensucia la transparencia)
            return img
        return img + noise
