"""Exportación a video con ffmpeg (NVENC si hay GPU NVIDIA, libx264 como respaldo)."""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from ..audio.analysis import AudioFeatures
from ..config import ProjectConfig
from .engine import default_workers, iter_frames, output_size


class ExportError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise ExportError(
            "No se encontró ffmpeg. Instálalo y agrégalo al PATH (en Windows: `winget install Gyan.FFmpeg` "
            "o descarga desde https://www.gyan.dev/ffmpeg/builds/)."
        )
    return ff


@lru_cache(maxsize=None)
def available_encoders() -> set[str]:
    try:
        out = subprocess.run([ffmpeg_path(), "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False).stdout.decode(errors="ignore")
    except ExportError:
        return set()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(("V", " V")) and parts[0][0] in "V ":
            names.add(parts[1])
    return names


@lru_cache(maxsize=None)
def encoder_works(name: str) -> bool:
    """Comprueba codificando un clip minúsculo (detecta NVENC sin GPU disponible, drivers, etc.)."""
    if name not in available_encoders():
        return False
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=128x128:r=30:d=0.2", "-c:v", name, "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def choose_codec(requested: str) -> str:
    if requested != "auto":
        if not encoder_works(requested):
            raise ExportError(f"El encoder {requested} no está disponible o no funciona en este equipo.")
        return requested
    for candidate in ("h264_nvenc", "libx264"):
        if encoder_works(candidate):
            return candidate
    raise ExportError("ffmpeg no tiene ningún encoder H.264 utilizable (h264_nvenc / libx264).")


def codec_args(codec: str, bitrate: str, preset: Optional[str]) -> list[str]:
    if codec in ("h264_nvenc", "hevc_nvenc"):
        args = ["-c:v", codec, "-preset", preset or "p5", "-tune", "hq", "-rc", "vbr", "-b:v", bitrate, "-maxrate", _scale_bitrate(bitrate, 1.5), "-bufsize", _scale_bitrate(bitrate, 2.0), "-spatial-aq", "1", "-temporal-aq", "1", "-bf", "3"]
        if codec == "h264_nvenc":
            args += ["-profile:v", "high"]
        return args
    if codec == "libx264":
        return ["-c:v", "libx264", "-preset", preset or "medium", "-crf", "18", "-maxrate", bitrate, "-bufsize", _scale_bitrate(bitrate, 2.0)]
    if codec == "libx265":
        return ["-c:v", "libx265", "-preset", preset or "medium", "-crf", "20", "-tag:v", "hvc1"]
    return ["-c:v", codec, "-b:v", bitrate]


def _scale_bitrate(bitrate: str, factor: float) -> str:
    s = bitrate.strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    try:
        value = float(s) * mult * factor
    except ValueError:
        return bitrate
    return f"{int(value)}"


@dataclass
class ExportResult:
    path: Path
    frames: int
    seconds: float
    codec: str
    width: int
    height: int

    @property
    def fps_rendered(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def export_video(
    project: ProjectConfig,
    features: AudioFeatures,
    output: Optional[str | Path] = None,
    scale: float = 1.0,
    start: float = 0.0,
    duration: Optional[float] = None,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ExportResult:
    """Renderiza todos los frames y los codifica con ffmpeg junto con el audio.

    start/duration: recorte del video respecto al audio analizado (segundos).
    scale: factor de resolución (0.5 = mitad, para pruebas rápidas).
    """
    out_path = Path(output or project.output.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = output_size(project, scale)
    fps = project.output.fps
    codec = choose_codec(project.output.codec)
    workers = default_workers() if workers is None else max(int(workers), 1)
    if project.output.workers and workers == default_workers():
        workers = project.output.workers

    first = int(round(start * fps))
    total_available = features.n_frames - first
    if total_available <= 0:
        raise ExportError("El inicio indicado está fuera de la duración del audio.")
    n = total_available if duration is None else min(int(round(duration * fps)), total_available)
    indices = range(first, first + n)
    clip_seconds = n / fps

    audio_start = project.audio.start + start
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-ss", f"{audio_start:.6f}", "-t", f"{clip_seconds:.6f}", "-i", str(project.audio.file),
        "-map", "0:v:0", "-map", "1:a:0",
        *codec_args(codec, project.output.bitrate, project.output.preset),
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-b:a", project.output.audio_bitrate,
        "-movflags", "+faststart", "-shortest",
        str(out_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    done = 0
    try:
        for data in iter_frames(project, features, width, height, indices, workers=workers):
            proc.stdin.write(data)
            done += 1
            if progress and (done % 10 == 0 or done == n):
                progress(done, n)
    except BrokenPipeError as exc:
        _, err = proc.communicate()
        raise ExportError(f"ffmpeg cerró la entrada: {err.decode(errors='ignore').strip()}") from exc
    _, err = proc.communicate()  # cierra stdin y espera a que ffmpeg termine
    if proc.returncode != 0:
        raise ExportError(f"ffmpeg falló (código {proc.returncode}): {err.decode(errors='ignore').strip()}")
    return ExportResult(path=out_path, frames=done, seconds=time.perf_counter() - t0, codec=codec, width=width, height=height)
