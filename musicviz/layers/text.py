"""Texto (título, artista) renderizado con Pillow: fuentes del sistema, alineación, ajuste de líneas,
contorno, sombra, caja de fondo, pulso y glow."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import TextLayer
from ..render.canvas import Canvas, RenderContext, anchor_center, paste_rgba, rounded_rect_mask
from ..utils.color import parse_color
from .base import Layer

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _font_dirs() -> list[Path]:
    dirs: list[Path] = []
    if sys.platform.startswith("win"):
        dirs.append(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
        dirs.append(Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Microsoft/Windows/Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=1)
def available_fonts() -> dict[str, str]:
    """Fuentes instaladas en el sistema: {nombre de archivo sin extensión: ruta}."""
    found: dict[str, str] = {}
    for d in _font_dirs():
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".ttf", ".otf", ".ttc") and p.stem not in found:
                found[p.stem] = str(p)
    return dict(sorted(found.items(), key=lambda kv: kv[0].lower()))


def resolve_font_path(name: Optional[str]) -> Optional[str]:
    """Convierte un nombre ('Arial', 'arialbd', 'Montserrat-Bold') o una ruta en una ruta de archivo."""
    if not name:
        return None
    p = Path(name).expanduser()
    if p.is_file():
        return str(p)
    fonts = available_fonts()
    lower = {k.lower(): v for k, v in fonts.items()}
    key = name.lower().removesuffix(".ttf").removesuffix(".otf")
    if key in lower:
        return lower[key]
    for k, v in lower.items():  # coincidencia parcial: "montserrat" -> "Montserrat-Regular"
        if key.replace(" ", "") in k.replace(" ", "").replace("-", "").replace("_", ""):
            return v
    return None


def find_font(name: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    resolved = resolve_font_path(name)
    if resolved:
        candidates.append(resolved)
    candidates += _FONT_CANDIDATES
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text_width(font, text: str, letter_spacing: float) -> float:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    if letter_spacing <= 0:
        return probe.textlength(text, font=font)
    return sum(probe.textlength(ch, font=font) for ch in text) + letter_spacing * max(len(text) - 1, 0)


def wrap_text(text: str, font, max_width: Optional[float], letter_spacing: float = 0.0) -> list[str]:
    """Divide en líneas respetando saltos manuales y, si hay max_width, por palabras."""
    lines: list[str] = []
    for para in text.replace("\\n", "\n").split("\n"):
        if not max_width or max_width <= 0:
            lines.append(para)
            continue
        words = para.split(" ")
        cur = ""
        for word in words:
            cand = word if not cur else cur + " " + word
            if _text_width(font, cand, letter_spacing) <= max_width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines or [""]


def render_text_rgba(
    text: str,
    font,
    color: str,
    letter_spacing: float = 0.0,
    align: str = "center",
    line_spacing: float = 1.15,
    max_width: Optional[float] = None,
    stroke_width: float = 0.0,
    stroke_color: str = "#000000",
) -> np.ndarray:
    """Renderiza texto (multilínea) a un sprite RGBA uint8 ajustado al contenido."""
    rgba = tuple(int(c * 255) for c in parse_color(color))
    stroke = tuple(int(c * 255) for c in parse_color(stroke_color))
    sw = int(round(stroke_width))
    lines = wrap_text(text, font, max_width, letter_spacing)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (10, 3)
    line_h = (ascent + descent) * line_spacing
    widths = [_text_width(font, ln, letter_spacing) for ln in lines]
    pad = 4 + sw
    W = int(max(widths) if widths else 1) + 2 * pad
    H = int(line_h * (len(lines) - 1) + ascent + descent) + 2 * pad
    img = Image.new("RGBA", (max(W, 1), max(H, 1)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, (ln, wd) in enumerate(zip(lines, widths)):
        y = pad + i * line_h
        if align == "left":
            x = float(pad)
        elif align == "right":
            x = W - pad - wd
        else:
            x = (W - wd) / 2
        if letter_spacing <= 0:
            draw.text((x, y), ln, font=font, fill=rgba, stroke_width=sw, stroke_fill=stroke if sw else None)
        else:
            for ch in ln:
                draw.text((x, y), ch, font=font, fill=rgba, stroke_width=sw, stroke_fill=stroke if sw else None)
                x += probe.textlength(ch, font=font) + letter_spacing
    return np.array(img)


class TextOverlay(Layer[TextLayer]):
    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        super().prepare(ctx, features)
        cfg = self.cfg
        text = cfg.text.upper() if cfg.uppercase else cfg.text
        over = 1.0 + abs(cfg.pulse) * 1.2
        px = max(int(cfg.size * ctx.height * over), 4)
        font = find_font(cfg.font, px)
        max_w = cfg.max_width * ctx.width * over if cfg.max_width else None
        sprite = render_text_rgba(
            text, font, cfg.color, ctx.px(cfg.letter_spacing) * over, cfg.align, cfg.line_spacing, max_w,
            ctx.px(cfg.stroke_width) * over, cfg.stroke_color,
        )
        if cfg.box_color:
            pad = int(round(ctx.px(cfg.box_padding) * over))
            h, w = sprite.shape[:2]
            box = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)
            col = parse_color(cfg.box_color)
            mask = rounded_rect_mask(box.shape[1], box.shape[0], ctx.px(cfg.box_radius) * over, "rounded")
            box[..., :3] = tuple(int(c * 255) for c in col[:3])
            box[..., 3] = (mask.astype(np.uint16) * int(col[3] * 255) // 255).astype(np.uint8)
            paste_rgba(box, sprite, box.shape[1] / 2, box.shape[0] / 2)
            sprite = box
        self.sprite = sprite
        self.shadow_sprite = None
        if cfg.shadow > 0:
            blur = ctx.px(cfg.shadow_blur) * over
            pad = int(blur * 3) + 4
            h, w = sprite.shape[:2]
            sh = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)
            sh[pad : pad + h, pad : pad + w, 3] = sprite[..., 3]
            if blur > 0.5:
                sh[..., 3] = cv2.GaussianBlur(sh[..., 3], (0, 0), blur)
            sh[..., 3] = (sh[..., 3].astype(np.float32) * min(cfg.shadow, 1.0)).astype(np.uint8)
            self.shadow_sprite = sh
        self.base_scale = 1.0 / over
        self.x, self.y = ctx.rel(cfg.position)
        v = {"top": "top", "bottom": "bottom", "middle": ""}[cfg.valign]
        h = {"left": "left", "right": "right", "center": ""}[cfg.align]
        self.anchor = "_".join(p for p in (v, h) if p) or "center"

    def render(self, canvas: Canvas, frame: FrameFeatures) -> None:
        assert self.ctx is not None
        cfg = self.cfg
        anim = self.anim
        keys = self.keys
        scale = self.base_scale * (1.0 + cfg.pulse * frame.drive(cfg.pulse_trigger)) * anim.scale * max(keys.scale, 0.0)
        sh, sw = self.sprite.shape[:2]
        w, h = max(int(sw * scale), 1), max(int(sh * scale), 1)
        sprite = cv2.resize(self.sprite, (w, h), interpolation=cv2.INTER_AREA)
        if anim.blur > 0:
            sprite = cv2.GaussianBlur(sprite, (0, 0), max(anim.blur * self.ctx.px(14), 0.3))
        x, y = self.key_position()
        cx, cy = anchor_center(x + anim.dx, y + anim.dy, w, h, self.anchor)
        layer = canvas.new_layer()
        if self.shadow_sprite is not None:
            ssh, ssw = self.shadow_sprite.shape[:2]
            shadow = cv2.resize(self.shadow_sprite, (max(int(ssw * scale), 1), max(int(ssh * scale), 1)), interpolation=cv2.INTER_AREA)
            if abs(keys.rotation) > 1e-3:
                shadow = rotate_rgba(shadow, keys.rotation)
            paste_rgba(layer, shadow, cx + self.ctx.px(cfg.shadow_offset[0]), cy + self.ctx.px(cfg.shadow_offset[1]))
        if abs(keys.rotation) > 1e-3:
            sprite = rotate_rgba(sprite, keys.rotation)
        paste_rgba(layer, sprite, cx, cy)
        self.composite(canvas, layer)


def rotate_rgba(sprite: np.ndarray, angle: float) -> np.ndarray:
    """Gira un sprite RGBA alrededor de su centro ampliando el lienzo para no recortarlo."""
    sh, sw = sprite.shape[:2]
    diag = int(np.ceil(np.hypot(sh, sw))) + 2
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), -angle, 1.0)
    M[0, 2] += diag / 2 - sw / 2
    M[1, 2] += diag / 2 - sh / 2
    return cv2.warpAffine(sprite, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
