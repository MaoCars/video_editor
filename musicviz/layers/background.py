"""Fondo: color sólido, gradiente lineal/radial o imagen, con pulso y reacción a la energía."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import BackgroundConfig
from ..render.canvas import Canvas, RenderContext, fast_blur
from ..utils.color import gradient, parse_color
from ..utils.lookahead import Lookahead


def load_image_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        img = img.copy()
    img[..., :3] = img[..., 2::-1]  # BGR -> RGB
    return img


def fit_image(img: np.ndarray, width: int, height: int, mode: str, focus: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    h, w = img.shape[:2]
    if mode == "stretch":
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    scale = max(width / w, height / h) if mode == "cover" else min(width / w, height / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    out = np.zeros((height, width, img.shape[2]), img.dtype)
    fx = min(max(focus[0], 0.0), 1.0) if mode == "cover" else 0.5
    fy = min(max(focus[1], 0.0), 1.0) if mode == "cover" else 0.5
    x0 = int(round((width - nw) * fx))
    y0 = int(round((height - nh) * fy))
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dw, dh = min(nw - sx0, width - dx0), min(nh - sy0, height - dy0)
    out[dy0 : dy0 + dh, dx0 : dx0 + dw] = resized[sy0 : sy0 + dh, sx0 : sx0 + dw]
    return out


class _VideoFrame:
    """Frame de video ya encajado al lienzo, en uint8 y/o float32 (cada versión se deriva de la otra al pedirla)."""

    __slots__ = ("_u8", "_f32")

    def __init__(self, u8: np.ndarray | None, f32: np.ndarray | None):
        self._u8 = u8
        self._f32 = f32

    @property
    def u8(self) -> np.ndarray:
        if self._u8 is None:
            assert self._f32 is not None
            self._u8 = cv2.convertScaleAbs(np.clip(self._f32, 0, 1), alpha=255.0)
        return self._u8

    @property
    def f32(self) -> np.ndarray:
        if self._f32 is None:
            assert self._u8 is not None
            self._f32 = np.multiply(self._u8, np.float32(1.0 / 255.0), dtype=np.float32)
        return self._f32


class VideoSource:
    """Lee frames de un video por tiempo. Lee en secuencia cuando los frames son consecutivos
    (lo habitual, porque cada proceso renderiza bloques de frames seguidos) y busca sólo si hay saltos."""

    MAX_SKIP = 8  # saltos hacia delante hasta este tamaño se leen de seguido; más lejos se busca

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video de fondo: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.n_frames <= 0:
            self.n_frames = self._count_frames()
        if self.n_frames <= 0:
            raise ValueError(f"El video de fondo no tiene frames legibles: {path}")
        self.duration = self.n_frames / self.fps
        self._last_index = -1
        self._last_frame: np.ndarray | None = None

    def _count_frames(self) -> int:
        n = 0
        while True:
            ok = self.cap.grab()
            if not ok:
                break
            n += 1
        self.cap.release()
        self.cap = cv2.VideoCapture(self.path)
        return n

    def frame_at(self, t: float, loop: bool) -> np.ndarray:
        """Frame RGB uint8 en el segundo t del video (con bucle o congelando el último frame)."""
        if loop:
            t = t % self.duration
        idx = int(t * self.fps)
        idx = min(max(idx, 0), self.n_frames - 1)
        if idx == self._last_index and self._last_frame is not None:
            return self._last_frame
        gap = idx - self._last_index
        if 1 < gap <= self.MAX_SKIP:
            # saltos cortos (video a más fps que el proyecto, velocidad > 1): descartar frames sin decodificar del todo
            for _ in range(gap - 1):
                if not self.cap.grab():
                    break
        elif gap != 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        if not ok:
            # Fin inesperado (p. ej. recuento de frames impreciso): volver al principio o repetir el último
            if self._last_frame is not None and not loop:
                return self._last_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self.cap.read()
            if not ok:
                raise RuntimeError(f"No se pudo leer el video de fondo: {self.path}")
            idx = 0
        self._last_index = idx
        self._last_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._last_frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass


class Background:
    def __init__(self, cfg: BackgroundConfig):
        self.cfg = cfg
        self.base: np.ndarray | None = None
        self.ctx: RenderContext | None = None
        self.video: VideoSource | None = None
        self._video_cache: tuple[int, _VideoFrame] | None = None
        self._lookahead: Lookahead[int, _VideoFrame] | None = None
        self._n_frames = 0
        self.wants_float = True  # el backend CPU compone en float; el GPU lo pone a False y usa el uint8
        self._grad_index: np.ndarray | None = None  # posición 0..255 de cada píxel en el gradiente
        self._section_cache: tuple[bytes, np.ndarray] | None = None

    def prepare(self, ctx: RenderContext, features: AudioFeatures) -> None:
        self.ctx = ctx
        w, h = ctx.width, ctx.height
        cfg = self.cfg
        if cfg.type == "solid":
            c = parse_color(cfg.color)
            base = np.empty((h, w, 3), np.float32)
            base[:] = c[:3]
        elif cfg.type == "gradient":
            ang = math.radians(cfg.angle)
            dx, dy = math.cos(ang), math.sin(ang)
            xs = (np.arange(w, dtype=np.float32) / max(w - 1, 1)) - 0.5
            ys = (np.arange(h, dtype=np.float32) / max(h - 1, 1)) - 0.5
            proj = xs[None, :] * dx + ys[:, None] * dy
            denom = abs(dx) + abs(dy)
            t = np.clip(proj / max(denom, 1e-6) + 0.5, 0.0, 1.0)
            lut = gradient(cfg.colors, 256)[:, :3]
            self._grad_index = (t * 255).astype(np.uint8)
            base = lut[self._grad_index]
        elif cfg.type == "radial":
            xs = (np.arange(w, dtype=np.float32) - w / 2) / (w / 2)
            ys = (np.arange(h, dtype=np.float32) - h / 2) / (h / 2)
            r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / math.sqrt(2.0)
            lut = gradient(cfg.colors, 256)[:, :3]
            self._grad_index = (np.clip(r, 0, 1) * 255).astype(np.uint8)
            base = lut[self._grad_index]
        elif cfg.type == "image":
            if not cfg.image:
                raise ValueError("background.type=image requiere background.image")
            img = load_image_rgba(cfg.image)
            img = fit_image(img, w, h, cfg.image_fit, cfg.focus)
            base = img[..., :3].astype(np.float32) / 255.0
            if cfg.blur > 0:
                base = fast_blur(base, ctx.px(cfg.blur))
        elif cfg.type == "video":
            if not cfg.video:
                raise ValueError("background.type=video requiere background.video")
            self.video = VideoSource(cfg.video)
            self._lookahead = Lookahead(self._decode_video_frame)
            self._n_frames = ctx.n_frames
            base = np.zeros((h, w, 3), np.float32)
        else:  # pragma: no cover
            raise ValueError(cfg.type)
        if cfg.darken > 0 and cfg.type != "video":
            base = base * (1.0 - cfg.darken)
        self.base = np.ascontiguousarray(base, dtype=np.float32)

    def _decode_video_frame(self, index: int) -> "_VideoFrame":
        """Decodifica, encaja y procesa el frame de video que corresponde al frame `index` del proyecto."""
        assert self.video is not None and self.ctx is not None
        cfg = self.cfg
        t = cfg.video_start + (index / self.ctx.fps) * cfg.video_speed
        rgb = self.video.frame_at(t, cfg.video_loop)
        fitted = fit_image(rgb, self.ctx.width, self.ctx.height, cfg.image_fit, cfg.focus)
        if cfg.blur > 0:
            img = np.multiply(fitted, np.float32(1.0 / 255.0), dtype=np.float32)
            img = fast_blur(img, self.ctx.px(cfg.blur))
            if cfg.darken > 0:
                img *= np.float32(1.0 - cfg.darken)
            return _VideoFrame(u8=None, f32=img)
        if cfg.darken > 0:
            fitted = cv2.convertScaleAbs(fitted, alpha=1.0 - cfg.darken)
        vf = _VideoFrame(u8=fitted, f32=None)
        if self.wants_float:
            vf.f32  # noqa: B018 - se calcula ya, en el hilo auxiliar, para que el principal no lo haga
        return vf

    def _video_frame(self, frame: FrameFeatures) -> "_VideoFrame":
        """Frame de video del instante actual. El siguiente se decodifica ya en un hilo auxiliar mientras
        el hilo principal dibuja las capas y codifica: en un render secuencial casi siempre está listo."""
        assert self._lookahead is not None
        if self._video_cache is not None and self._video_cache[0] == frame.index:
            return self._video_cache[1]
        vf = self._lookahead.get(frame.index)
        self._video_cache = (frame.index, vf)
        if frame.index + 1 < self._n_frames:
            self._lookahead.schedule(frame.index + 1)
        return vf

    def close(self) -> None:
        """Detiene el hilo de decodificación y libera el video."""
        if self._lookahead is not None:
            self._lookahead.close()
            self._lookahead = None
        if self.video is not None:
            self.video.release()
            self.video = None

    def _section_base(self, stops: np.ndarray) -> np.ndarray:
        """Fondo sólido/gradiente/radial recoloreado con la paleta de la sección (con caché)."""
        assert self.base is not None
        key = stops.tobytes()
        if self._section_cache is not None and self._section_cache[0] == key:
            return self._section_cache[1]
        cfg = self.cfg
        if self._grad_index is not None:
            xs = np.linspace(0, 1, len(stops))
            lut = np.stack([np.interp(np.linspace(0, 1, 256), xs, stops[:, c]) for c in range(3)], axis=1).astype(np.float32)
            base = lut[self._grad_index]
        elif cfg.type == "solid":
            base = np.empty_like(self.base)
            base[:] = stops[0, :3]
        else:
            return self.base  # imagen / video: la paleta de sección no se aplica
        if cfg.darken > 0:
            base *= np.float32(1.0 - cfg.darken)
        base = np.ascontiguousarray(base, dtype=np.float32)
        self._section_cache = (key, base)
        return base

    def source(self, frame: FrameFeatures, section_colors: np.ndarray | None = None) -> np.ndarray:
        """Imagen base float32 RGB del frame (video, fondo recoloreado por sección o base fija)."""
        assert self.base is not None
        if self.video is not None:
            return self._video_frame(frame).f32
        if section_colors is not None:
            return self._section_base(section_colors)
        return self.base

    def source_u8(self, frame: FrameFeatures, section_colors: np.ndarray | None = None) -> np.ndarray:
        """Como `source` pero RGB uint8 (lo que sube el backend GPU como textura, sin pasar por float)."""
        if self.video is not None:
            return self._video_frame(frame).u8
        src = self.source(frame, section_colors)
        return cv2.convertScaleAbs(np.clip(src, 0, 1), alpha=255.0)

    def motion(self, frame: FrameFeatures) -> tuple[float, float, float, float, float]:
        """(zoom, dx, dy, ángulo en grados, ganancia de brillo) del frame (compartido CPU/GPU)."""
        assert self.ctx is not None
        cfg = self.cfg
        intensity = frame.intensity
        zoom = cfg.zoom
        if cfg.pulse > 0:
            zoom *= 1.0 + cfg.pulse * intensity * frame.drive(cfg.pulse_trigger)
        dx = dy = 0.0
        angle = 0.0
        if cfg.shake > 0 or cfg.shake_rotation > 0:
            k = frame.drive(cfg.shake_trigger) * intensity
            if k > 0.001:
                rng = np.random.default_rng(4242 + frame.index)
                amp = self.ctx.px(cfg.shake) * k
                dx, dy = (float(v) for v in rng.uniform(-amp, amp, 2))
                angle = float(rng.uniform(-1, 1)) * cfg.shake_rotation * k
        gain = 1.0 + cfg.react * frame.drive(cfg.react_trigger) if cfg.react > 0 else 1.0
        return float(zoom), dx, dy, angle, float(gain)

    def render(self, canvas: Canvas, frame: FrameFeatures, section_colors: np.ndarray | None = None) -> None:
        assert self.base is not None and self.ctx is not None
        cfg = self.cfg
        img = self.source(frame, section_colors)
        zoom, dx, dy, angle, gain = self.motion(frame)
        if abs(zoom - 1.0) > 0.0005 or abs(dx) > 0.3 or abs(dy) > 0.3 or abs(angle) > 0.01:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
            M[0, 2] += dx
            M[1, 2] += dy
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if gain != 1.0:
            img = img * gain
        canvas.fill(img)
