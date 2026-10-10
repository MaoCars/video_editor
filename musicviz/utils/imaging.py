"""Operaciones de imagen con Pillow y NumPy, sin OpenCV.

OpenCV sólo hace falta para el backend CPU (dibujo de formas con antialiasing, composición por capas). El
backend GPU, la interfaz y la línea de comandos funcionan sin él, así el ejecutable puede ir sin OpenCV.
Las funciones de aquí las usan ambos backends en la fase de preparación (cargar, encajar, enmascarar,
sombras...), de modo que el resultado CPU y GPU sigue partiendo de los mismos sprites.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

try:  # opcional: acelera algunas operaciones y es imprescindible para el backend CPU
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - depende del entorno
    cv2 = None

HAS_CV2 = cv2 is not None
_SS = 4  # factor de supermuestreo para máscaras con antialiasing


def require_cv2(what: str = "El backend CPU") -> None:
    if cv2 is None:
        raise RuntimeError(
            f"{what} necesita OpenCV, que no está instalado. Instálalo con `pip install opencv-python-headless` "
            "(o `pip install musicviz[cpu]`) o usa el backend GPU (output.backend: gpu / auto)."
        )


# ---------------------------------------------------------------- carga y guardado
def load_image_rgba(path: str) -> np.ndarray:
    """Imagen RGBA uint8 (respeta la orientación EXIF)."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return np.array(im.convert("RGBA"))
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}") from exc


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Guarda RGB o RGBA uint8 (PNG con transparencia si tiene alfa)."""
    Image.fromarray(np.ascontiguousarray(img)).save(str(path))


# ---------------------------------------------------------------- conversiones
def to_uint8(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] -> uint8 con saturación y redondeo."""
    if cv2 is not None:
        return cv2.convertScaleAbs(np.clip(img, 0.0, 1.0), alpha=255.0)
    out = np.multiply(np.clip(img, 0.0, 1.0), 255.0, dtype=np.float32)
    np.add(out, 0.5, out=out)
    return out.astype(np.uint8)


def scale_uint8(img: np.ndarray, factor: float) -> np.ndarray:
    """uint8 * factor (0..1) -> uint8, con redondeo (oscurecer una imagen)."""
    if cv2 is not None:
        return cv2.convertScaleAbs(img, alpha=factor)
    out = np.multiply(img, np.float32(factor), dtype=np.float32)
    np.add(out, 0.5, out=out)
    return out.astype(np.uint8)


# ---------------------------------------------------------------- escalado
def _pil_resample(k: float):
    return Image.BOX if k < 1.0 else Image.BICUBIC


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Escala una imagen uint8 (1, 3 o 4 canales) o float32 (1 o 3 canales) a width x height."""
    width, height = max(int(width), 1), max(int(height), 1)
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    k = min(width / w, height / h)
    if cv2 is not None:
        inter = cv2.INTER_AREA if k < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(img, (width, height), interpolation=inter)
    resample = _pil_resample(k)
    if img.dtype == np.uint8:
        return np.array(Image.fromarray(np.ascontiguousarray(img)).resize((width, height), resample))
    if img.ndim == 2:
        return np.array(Image.fromarray(np.ascontiguousarray(img, dtype=np.float32), "F").resize((width, height), resample), np.float32)
    chans = [np.array(Image.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32), "F").resize((width, height), resample), np.float32) for c in range(img.shape[2])]
    return np.stack(chans, axis=2)


def fit_image(img: np.ndarray, width: int, height: int, mode: str, focus: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Encaja una imagen al lienzo completo: cover recorta alrededor de `focus`, contain deja bandas, stretch deforma."""
    h, w = img.shape[:2]
    if mode == "stretch":
        return resize(img, width, height)
    scale = max(width / w, height / h) if mode == "cover" else min(width / w, height / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = resize(img, nw, nh)
    out = np.zeros((height, width) + img.shape[2:], img.dtype)
    fx = min(max(focus[0], 0.0), 1.0) if mode == "cover" else 0.5
    fy = min(max(focus[1], 0.0), 1.0) if mode == "cover" else 0.5
    x0 = int(round((width - nw) * fx))
    y0 = int(round((height - nh) * fy))
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dw, dh = min(nw - sx0, width - dx0), min(nh - sy0, height - dy0)
    out[dy0 : dy0 + dh, dx0 : dx0 + dw] = resized[sy0 : sy0 + dh, sx0 : sx0 + dw]
    return out


def fit_into_box(img: np.ndarray, bw: int, bh: int, fit: str, focus: tuple[float, float]) -> np.ndarray:
    """Encaja una imagen RGBA en una caja bw×bh. cover recorta alrededor de `focus`; contain rellena con transparente."""
    h, w = img.shape[:2]
    if fit == "cover":
        k = max(bw / w, bh / h)
        nw, nh = max(int(round(w * k)), bw), max(int(round(h * k)), bh)
        resized = resize(img, nw, nh)
        x0 = int(round((nw - bw) * min(max(focus[0], 0.0), 1.0)))
        y0 = int(round((nh - bh) * min(max(focus[1], 0.0), 1.0)))
        return np.ascontiguousarray(resized[y0 : y0 + bh, x0 : x0 + bw])
    k = min(bw / w, bh / h)
    nw, nh = max(int(round(w * k)), 1), max(int(round(h * k)), 1)
    resized = resize(img, nw, nh)
    out = np.zeros((bh, bw, 4), np.uint8)
    x0, y0 = (bw - nw) // 2, (bh - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


# ---------------------------------------------------------------- máscaras y formas (con antialiasing por supermuestreo)
def _downsample_mask(big: Image.Image, w: int, h: int) -> np.ndarray:
    return np.array(big.resize((w, h), Image.BOX))


def rounded_rect_mask(w: int, h: int, radius: float, shape: str = "rounded") -> np.ndarray:
    """Máscara uint8 (h, w): rectángulo, rectángulo redondeado o elipse."""
    w, h = max(int(w), 1), max(int(h), 1)
    if shape == "circle":
        big = Image.new("L", (w * _SS, h * _SS), 0)
        ImageDraw.Draw(big).ellipse([0, 0, w * _SS - 1, h * _SS - 1], fill=255)
        return _downsample_mask(big, w, h)
    r = int(max(min(radius, w / 2, h / 2), 0))
    mask = np.full((h, w), 255, np.uint8)
    if shape != "rounded" or r <= 0:
        return mask
    big = Image.new("L", (w * _SS, h * _SS), 0)
    ImageDraw.Draw(big).rounded_rectangle([0, 0, w * _SS - 1, h * _SS - 1], radius=r * _SS, fill=255)
    return _downsample_mask(big, w, h)


def ellipse_ring(w: int, h: int, thickness: int, color: tuple[int, int, int, int]) -> np.ndarray:
    """Anillo elíptico RGBA uint8 (borde de una imagen redonda) con antialiasing."""
    w, h = max(int(w), 1), max(int(h), 1)
    t = max(int(thickness), 1)
    big = Image.new("L", (w * _SS, h * _SS), 0)
    ImageDraw.Draw(big).ellipse([0, 0, w * _SS - 1, h * _SS - 1], outline=255, width=t * _SS)
    band = _downsample_mask(big, w, h)
    ring = np.zeros((h, w, 4), np.uint8)
    ring[..., :3] = color[:3]
    ring[..., 3] = (band.astype(np.uint16) * color[3] // 255).astype(np.uint8)
    return ring


# ---------------------------------------------------------------- desenfoques
def blur_u8(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque gaussiano de un canal uint8 (sombras)."""
    if sigma <= 0.3:
        return channel
    if cv2 is not None:
        return cv2.GaussianBlur(channel, (0, 0), sigma)
    return np.array(Image.fromarray(np.ascontiguousarray(channel)).filter(ImageFilter.GaussianBlur(sigma)))


def _gaussian_1d(sigma: float) -> np.ndarray:
    radius = max(int(np.ceil(sigma * 3.0)), 1)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1, dtype=np.float32) / np.float32(sigma)) ** 2)
    return (k / k.sum()).astype(np.float32)


def _blur_f32(img: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque gaussiano de una imagen float32 (separable en NumPy cuando no hay OpenCV; Pillow no
    filtra imágenes en coma flotante)."""
    if cv2 is not None:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    k = _gaussian_1d(sigma)
    r = len(k) // 2
    out = np.ascontiguousarray(img, dtype=np.float32)
    for axis in (1, 0):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            acc += w * padded[tuple(sl)]
        out = acc
    return out


def fast_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque gaussiano rápido de una imagen float32: para sigmas grandes reduce la imagen antes de filtrar."""
    if sigma <= 0.5:
        return img
    h, w = img.shape[:2]
    factor = 1
    while sigma / factor > 3.0 and min(h, w) // (factor * 2) >= 8:
        factor *= 2
    if factor == 1:
        return _blur_f32(img, sigma)
    small = resize(img, max(w // factor, 1), max(h // factor, 1))
    small = _blur_f32(small, sigma / factor)
    if cv2 is not None:
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return resize(small, w, h)
