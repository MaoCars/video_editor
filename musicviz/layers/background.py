"""Fondo: color sólido, gradiente lineal/radial o imagen, con pulso y reacción a la energía."""
from __future__ import annotations

import math
import re
import subprocess

import numpy as np

from ..audio.analysis import AudioFeatures, FrameFeatures
from ..config import BackgroundConfig
from ..render.canvas import Canvas, RenderContext
from ..utils.color import gradient, parse_color
from ..utils.imaging import cv2, fast_blur, fit_image, load_image_rgba, scale_uint8, to_uint8  # noqa: F401 - reexportados
from ..utils.lookahead import Lookahead
from ..utils.tools import find_tool


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
            self._u8 = to_uint8(self._f32)
        return self._u8

    @property
    def f32(self) -> np.ndarray:
        if self._f32 is None:
            assert self._u8 is not None
            self._f32 = np.multiply(self._u8, np.float32(1.0 / 255.0), dtype=np.float32)
        return self._f32


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?Video:.*")
_SIZE_RE = re.compile(r"[ ,](\d{2,5})x(\d{2,5})(?:[ ,\[]|$)")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")
_TBR_RE = re.compile(r"([\d.]+)\s*tbr")
_ROT_RE = re.compile(r"rotate\s*:\s*(-?\d+)|rotation of (-?[\d.]+) degrees")
_FRAMES_RE = re.compile(r"frame=\s*(\d+)")


def _ffmpeg() -> str:
    ff = find_tool("ffmpeg")
    if not ff:
        raise RuntimeError("Para usar un video de fondo hace falta ffmpeg (junto a la aplicación o en el PATH).")
    return ff


def probe_video(path: str) -> dict:
    """Metadatos de un video leyendo la cabecera que imprime `ffmpeg -i`: fps, duración, tamaño y rotación."""
    ff = _ffmpeg()
    res = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", path], capture_output=True, text=True, errors="ignore")
    err = res.stderr
    m = _VIDEO_RE.search(err)
    if m is None:
        raise FileNotFoundError(f"No se pudo abrir el video de fondo (ffmpeg no encuentra una pista de video): {path}")
    line = m.group(0)
    size = _SIZE_RE.search(line)
    if size is None:
        raise ValueError(f"No se pudo leer el tamaño del video de fondo: {path}")
    w, h = int(size.group(1)), int(size.group(2))
    fps_m = _FPS_RE.search(line) or _TBR_RE.search(line)
    fps = float(fps_m.group(1)) if fps_m else 30.0
    if not fps or fps <= 0 or fps > 1000:
        fps = 30.0
    rot = _ROT_RE.search(err)
    if rot:
        deg = abs(float(rot.group(1) or rot.group(2)))
        if round(deg) % 180 == 90:
            w, h = h, w
    dur_m = _DURATION_RE.search(err)
    duration = None
    if dur_m:
        duration = int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))
    if not duration or duration <= 0:
        # Sin duración en la cabecera (p. ej. algunos GIF/WebM): contar frames decodificando una vez
        cnt = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", path, "-map", "0:v:0", "-an", "-f", "null", "-"], capture_output=True, text=True, errors="ignore")
        frames = [int(x) for x in _FRAMES_RE.findall(cnt.stderr)]
        if not frames or frames[-1] <= 0:
            raise ValueError(f"El video de fondo no tiene frames legibles: {path}")
        duration = frames[-1] / fps
    return {"width": w, "height": h, "fps": fps, "duration": float(duration)}


class VideoSource:
    """Lee frames RGB de un video decodificándolo con ffmpeg (por tubería, sin OpenCV).

    Lee en secuencia cuando los frames son consecutivos (lo habitual) y relanza ffmpeg con `-ss` sólo si hay saltos.
    La salida se fuerza a la cadencia `fps` del video, así el frame k empieza en el segundo k / fps.
    """

    MAX_SKIP = 8  # saltos hacia delante hasta este tamaño se leen de seguido; más lejos se busca

    def __init__(self, path: str):
        self.path = path
        self.ffmpeg = _ffmpeg()
        info = probe_video(path)
        self.width, self.height = info["width"], info["height"]
        self.fps = info["fps"]
        self.duration = info["duration"]
        self.n_frames = max(int(round(self.duration * self.fps)), 1)
        self._frame_bytes = self.width * self.height * 3
        self._proc: subprocess.Popen | None = None
        self._next_index = -1  # índice del próximo frame que entregará la tubería
        self._last_index = -1
        self._last_frame: np.ndarray | None = None

    # ------------------------------------------------------------ tubería ffmpeg
    def _start(self, index: int) -> None:
        self._stop()
        t = max(index, 0) / self.fps
        cmd = [
            self.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "2",
            "-ss", f"{t:.6f}", "-i", self.path, "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"scale={self.width}:{self.height}", "-r", f"{self.fps:.6f}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, bufsize=0)
        self._next_index = max(index, 0)

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass

    def _read(self) -> np.ndarray | None:
        assert self._proc is not None and self._proc.stdout is not None
        n = self._frame_bytes
        chunks = []
        got = 0
        while got < n:
            chunk = self._proc.stdout.read(n - got)
            if not chunk:
                return None  # fin del video
            chunks.append(chunk)
            got += len(chunk)
        self._next_index += 1
        data = b"".join(chunks) if len(chunks) > 1 else chunks[0]
        return np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)

    # ------------------------------------------------------------ API
    def frame_at(self, t: float, loop: bool) -> np.ndarray:
        """Frame RGB uint8 en el segundo t del video (con bucle o congelando el último frame)."""
        if loop:
            t = t % self.duration
        idx = int(t * self.fps)
        idx = min(max(idx, 0), self.n_frames - 1)
        if idx == self._last_index and self._last_frame is not None:
            return self._last_frame
        gap = idx - self._next_index if self._proc is not None else None
        if gap is None or gap < 0 or gap > self.MAX_SKIP:
            self._start(idx)
        else:
            for _ in range(gap):  # saltos cortos: descartar frames ya decodificados
                if self._read() is None:
                    break
        frame = self._read()
        if frame is None:
            # Fin inesperado (duración imprecisa): volver al principio o repetir el último
            if self._last_frame is not None and not loop:
                return self._last_frame
            self._start(0)
            frame = self._read()
            if frame is None:
                raise RuntimeError(f"No se pudo leer el video de fondo: {self.path}")
            idx = 0
        self._last_index = idx
        self._last_frame = frame
        return frame

    def release(self) -> None:
        self._stop()


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
        if cfg.blur > 0 and self.wants_float:  # el backend GPU desenfoca en un shader (gpu_blur_sigma)
            img = np.multiply(fitted, np.float32(1.0 / 255.0), dtype=np.float32)
            img = fast_blur(img, self.ctx.px(cfg.blur))
            if cfg.darken > 0:
                img *= np.float32(1.0 - cfg.darken)
            return _VideoFrame(u8=None, f32=img)
        if cfg.darken > 0:
            fitted = scale_uint8(fitted, 1.0 - cfg.darken)
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

    def gpu_blur_sigma(self) -> float:
        """Sigma (px) con el que el backend GPU debe desenfocar el video de fondo (0 = nada que hacer)."""
        if self.video is None or self.ctx is None or self.cfg.blur <= 0 or self.wants_float:
            return 0.0
        return float(self.ctx.px(self.cfg.blur))

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
        return to_uint8(src)

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
