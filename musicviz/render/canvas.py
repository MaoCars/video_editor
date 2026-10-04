"""Lienzo float32 RGB y composición de capas RGBA (uint8) con glow y modos de fusión."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def fast_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque gaussiano rápido: para sigmas grandes reduce la imagen antes de filtrar."""
    if sigma <= 0.5:
        return img
    h, w = img.shape[:2]
    factor = 1
    while sigma / factor > 3.0 and min(h, w) // (factor * 2) >= 8:
        factor *= 2
    if factor == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, (max(w // factor, 1), max(h // factor, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


@dataclass
class RenderContext:
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def scale(self) -> float:
        """Factor para medidas expresadas 'a 1080p'."""
        return self.height / 1080.0

    @property
    def min_dim(self) -> int:
        return min(self.width, self.height)

    def px(self, value: float) -> float:
        return value * self.scale

    def rel(self, pos: tuple[float, float]) -> tuple[float, float]:
        return pos[0] * self.width, pos[1] * self.height


class Canvas:
    """Imagen float32 RGB en [0,1] sobre la que se componen capas."""

    def __init__(self, width: int, height: int, track_alpha: bool = False):
        self.width = width
        self.height = height
        self.img = np.zeros((height, width, 3), np.float32)
        # Cobertura alfa (sólo en modo transparente). El RGB se guarda premultiplicado.
        self.alpha: Optional[np.ndarray] = np.zeros((height, width, 1), np.float32) if track_alpha else None

    def clear(self) -> None:
        self.img[:] = 0.0
        if self.alpha is not None:
            self.alpha[:] = 0.0

    def new_layer(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 4), np.uint8)

    def fill(self, rgb: np.ndarray) -> None:
        self.img[:] = rgb
        if self.alpha is not None:
            self.alpha[:] = 1.0

    @staticmethod
    def _roi_of(alpha: np.ndarray, pad: int, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        x, y, w, h = cv2.boundingRect(alpha)
        if w == 0 or h == 0:
            return None
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, width)
        y1 = min(y + h + pad, height)
        return x0, y0, x1, y1

    def composite(
        self,
        layer: np.ndarray,
        opacity: float = 1.0,
        blend: str = "normal",
        glow: float = 0.0,
        glow_radius: float = 24.0,
    ) -> None:
        """Compone una capa RGBA uint8 sobre el lienzo.

        glow: intensidad de un halo aditivo (desenfoque de la capa). glow_radius en px.
        """
        if opacity <= 0.0:
            return
        alpha8 = cv2.extractChannel(layer, 3)  # copia contigua: boundingRect es mucho más rápido
        pad = int(glow_radius * 3) + 2 if glow > 0 else 1
        roi = self._roi_of(alpha8, pad, self.width, self.height)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        sub = layer[y0:y1, x0:x1]
        # Trabajar con arrays contiguos de 3 canales (las vistas [..., :3] son lentas en numpy/OpenCV)
        premult = cv2.cvtColor(sub, cv2.COLOR_RGBA2RGB).astype(np.float32)
        a = alpha8[y0:y1, x0:x1].astype(np.float32)[..., None]
        a *= np.float32(opacity / 255.0)
        premult *= a
        premult *= np.float32(1.0 / 255.0)
        dst = self.img[y0:y1, x0:x1]

        if blend == "normal":
            # dst = dst*(1-a) + rgb*a  ->  dst += premult - dst*a
            tmp = dst * a
            dst -= tmp
            dst += premult
        elif blend == "add":
            dst += premult
        elif blend == "screen":
            # 1 - (1-dst)*(1-p) = dst + p - dst*p
            tmp = dst * premult
            dst += premult
            dst -= tmp
        else:
            raise ValueError(f"Modo de fusión desconocido: {blend}")

        halo = None
        if glow > 0.0 and glow_radius > 0.0:
            halo = fast_blur(premult, glow_radius)
            halo *= np.float32(glow * 1.6)
            dst += halo

        if self.alpha is not None:
            da = self.alpha[y0:y1, x0:x1]
            da += a * (1.0 - da)  # unión de coberturas (válido para los tres modos de fusión)
            if halo is not None:
                np.maximum(da, np.clip(halo.max(axis=2, keepdims=True), 0.0, 1.0), out=da)

    def composite_rgb(self, rgb: np.ndarray, alpha: np.ndarray, blend: str = "normal") -> None:
        """Compone una imagen float RGB con máscara alpha float (H,W,1) del tamaño del lienzo."""
        if blend == "normal":
            self.img *= 1.0 - alpha
            self.img += rgb * alpha
        elif blend == "add":
            self.img += rgb * alpha
        elif blend == "screen":
            self.img[:] = 1.0 - (1.0 - self.img) * (1.0 - rgb * alpha)

    def to_uint8(self) -> np.ndarray:
        return float_to_uint8(self.img)


def float_to_uint8(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] -> uint8 con saturación (rápido vía OpenCV)."""
    img = np.clip(img, 0.0, 1.0)
    return cv2.convertScaleAbs(img, alpha=255.0)


def premultiplied_to_rgba8(img4: np.ndarray) -> np.ndarray:
    """(H,W,4) float con RGB premultiplicado + cobertura -> RGBA uint8 con alfa directo.

    El alfa final es el máximo entre la cobertura y la luz añadida (glow/bloom), de modo que los halos
    quedan semitransparentes en lugar de recortarse.
    """
    rgb = np.clip(img4[..., :3], 0.0, None)
    alpha = np.maximum(np.clip(img4[..., 3:4], 0.0, 1.0), np.clip(rgb.max(axis=2, keepdims=True), 0.0, 1.0))
    straight = np.where(alpha > 1e-4, rgb / np.maximum(alpha, 1e-4), 0.0)
    out = np.empty(img4.shape[:2] + (4,), np.uint8)
    out[..., :3] = cv2.convertScaleAbs(np.clip(straight, 0.0, 1.0), alpha=255.0)
    out[..., 3] = cv2.convertScaleAbs(alpha[..., 0], alpha=255.0)
    return out


def paste_rgba(layer: np.ndarray, sprite: np.ndarray, cx: float, cy: float) -> None:
    """Pega un sprite RGBA uint8 centrado en (cx, cy) dentro de una capa RGBA (alpha 'over')."""
    h, w = sprite.shape[:2]
    H, W = layer.shape[:2]
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + w, W), min(y0 + h, H)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    src = sprite[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)].astype(np.float32)
    dst = layer[dy0:dy1, dx0:dx1].astype(np.float32)
    sa = src[..., 3:4] / 255.0
    da = dst[..., 3:4] / 255.0
    out_a = sa + da * (1.0 - sa)
    safe = np.where(out_a > 0, out_a, 1.0)
    out_rgb = (src[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / safe
    layer[dy0:dy1, dx0:dx1, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    layer[dy0:dy1, dx0:dx1, 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)


def anchor_center(x: float, y: float, w: float, h: float, anchor: str) -> tuple[float, float]:
    """Centro de un sprite w×h cuyo punto `anchor` debe quedar en (x, y)."""
    ax = 0.5
    ay = 0.5
    if "left" in anchor:
        ax = 0.0
    elif "right" in anchor:
        ax = 1.0
    if "top" in anchor:
        ay = 0.0
    elif "bottom" in anchor:
        ay = 1.0
    return x + (0.5 - ax) * w, y + (0.5 - ay) * h


def rounded_rect_mask(w: int, h: int, radius: float, shape: str = "rounded") -> np.ndarray:
    """Máscara uint8 (h, w): rectángulo, rectángulo redondeado o elipse."""
    mask = np.zeros((h, w), np.uint8)
    if shape == "circle":
        cv2.ellipse(mask, (w // 2, h // 2), (max(w // 2 - 1, 1), max(h // 2 - 1, 1)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        return mask
    r = int(max(min(radius, w / 2, h / 2), 0))
    if shape != "rounded" or r <= 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return mask


def over_checkerboard(rgba: np.ndarray, cell: int = 16) -> np.ndarray:
    """Compone una imagen RGBA uint8 sobre un tablero gris (para previsualizar transparencias)."""
    h, w = rgba.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    board = np.where(((ys // cell) + (xs // cell)) % 2 == 0, 200, 150).astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgba[..., :3].astype(np.float32) * a + board[..., None] * (1.0 - a)
    return out.astype(np.uint8)
