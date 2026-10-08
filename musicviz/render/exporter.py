"""Exportación a video con ffmpeg (NVENC si hay GPU NVIDIA, libx264 como respaldo)."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from ..audio.analysis import AudioFeatures
from ..config import ALPHA_CODECS, ProjectConfig
from .engine import default_workers, iter_frames, output_size


class ExportError(RuntimeError):
    pass


class ExportCancelled(ExportError):
    """El render fue cancelado por el usuario (se elimina el archivo parcial)."""


def _bundled_dirs() -> list[Path]:
    """Carpetas donde la app empaquetada (PyInstaller) o una instalación portátil pueden llevar ffmpeg."""
    import sys

    dirs: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        dirs += [exe_dir, exe_dir / "bin", exe_dir / "ffmpeg"]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            dirs += [Path(meipass), Path(meipass) / "bin"]
    root = Path(__file__).resolve().parents[2]
    dirs += [root / "bin", root / "ffmpeg"]
    return dirs


def find_tool(name: str) -> Optional[str]:
    """Busca ffmpeg junto a la app (carpeta bin/) y, si no, en el PATH."""
    exe = f"{name}.exe" if os.name == "nt" else name
    for d in _bundled_dirs():
        candidate = d / exe
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def ffmpeg_path() -> str:
    ff = find_tool("ffmpeg")
    if not ff:
        raise ExportError(
            "No se encontró ffmpeg. Instálalo y agrégalo al PATH (en Windows: `winget install Gyan.FFmpeg` "
            "o descarga desde https://www.gyan.dev/ffmpeg/builds/), o copia ffmpeg.exe junto a la aplicación."
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


_ENCODER_OF = {"prores_4444": "prores_ks", "qtrle": "qtrle", "vp9_alpha": "libvpx-vp9", "png_sequence": "png"}


def choose_codec(requested: str, transparent: bool = False) -> str:
    if transparent:
        if requested == "auto":
            requested = "prores_4444"
        if requested not in ALPHA_CODECS:
            raise ExportError(f"El códec {requested} no conserva transparencia. Usa uno de: {', '.join(ALPHA_CODECS)} (o codec: auto).")
        if not encoder_works(_ENCODER_OF[requested]):
            raise ExportError(f"Tu ffmpeg no tiene el encoder {_ENCODER_OF[requested]} necesario para {requested}.")
        return requested
    if requested in ALPHA_CODECS:
        raise ExportError(f"El códec {requested} sólo tiene sentido con output.transparent: true.")
    if requested != "auto":
        if not encoder_works(requested):
            raise ExportError(f"El encoder {requested} no está disponible o no funciona en este equipo.")
        return requested
    for candidate in ("h264_nvenc", "libx264"):
        if encoder_works(candidate):
            return candidate
    raise ExportError("ffmpeg no tiene ningún encoder H.264 utilizable (h264_nvenc / libx264).")


def output_path_for(path: Path, codec: str) -> Path:
    """Ajusta la extensión al contenedor que exige el códec (.mov para ProRes/QTRLE, .webm para VP9,
    carpeta para la secuencia PNG)."""
    if codec in ("prores_4444", "qtrle"):
        return path.with_suffix(".mov")
    if codec == "vp9_alpha":
        return path.with_suffix(".webm")
    if codec == "png_sequence":
        return path.with_suffix("") if path.suffix.lower() in (".mp4", ".mov", ".webm", ".png", ".mkv") else path
    if path.suffix.lower() not in (".mp4", ".mov", ".mkv"):
        return path.with_suffix(".mp4")
    return path


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
    if codec == "prores_4444":
        return ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0", "-qscale:v", preset or "9"]
    if codec == "qtrle":
        return ["-c:v", "qtrle", "-pix_fmt", "argb"]
    if codec == "vp9_alpha":
        return ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", bitrate, "-deadline", "good", "-cpu-used", preset or "2", "-row-mt", "1", "-auto-alt-ref", "0"]
    if codec == "png_sequence":
        return ["-c:v", "png", "-pix_fmt", "rgba"]
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


class _PipeWriter:
    """Escribe los frames en la entrada de ffmpeg desde un hilo, con una cola corta, para que el render
    no se detenga cada vez que el codificador tarda en vaciar la tubería."""

    def __init__(self, pipe, depth: int = 4):
        self.pipe = pipe
        self.queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=depth)
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(target=self._run, name="ffmpeg-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            data = self.queue.get()
            if data is None:
                break
            if self.error is not None:
                continue  # tras un fallo se vacía la cola para no bloquear al productor
            try:
                self.pipe.write(data)
            except Exception as exc:  # noqa: BLE001
                self.error = exc

    def write(self, data: bytes) -> None:
        if self.error is not None:
            raise self.error
        self.queue.put(data)

    def close(self) -> None:
        """Termina de escribir lo encolado y relanza el error del hilo, si lo hubo."""
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error

    def abort(self) -> None:
        """Detiene el hilo descartando lo que quede en la cola (cancelación o error)."""
        if not self.thread.is_alive():
            return
        self.error = self.error or ExportCancelled("escritura abortada")
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)


def export_video(
    project: ProjectConfig,
    features: AudioFeatures,
    output: Optional[str | Path] = None,
    scale: float = 1.0,
    start: float = 0.0,
    duration: Optional[float] = None,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> ExportResult:
    """Renderiza todos los frames y los codifica con ffmpeg junto con el audio.

    start/duration: recorte del video respecto al audio analizado (segundos).
    scale: factor de resolución (0.5 = mitad, para pruebas rápidas).
    cancel: evento que, al activarse, detiene el render, cierra ffmpeg y borra el archivo parcial
            (se lanza ExportCancelled).
    """
    transparent = bool(project.output.transparent)
    codec = choose_codec(project.output.codec, transparent)
    out_path = output_path_for(Path(output or project.output.path), codec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = output_size(project, scale)
    fps = project.output.fps
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
    pix_in = "rgba" if transparent else "rgb24"
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
    ]
    if codec == "png_sequence":
        out_path.mkdir(parents=True, exist_ok=True)
        target = str(out_path / "frame_%06d.png")
        cmd += ["-map", "0:v:0", *codec_args(codec, project.output.bitrate, project.output.preset), "-start_number", "0", target]
    else:
        audio_codec = ["-c:a", "libopus", "-b:a", "192k"] if codec == "vp9_alpha" else ["-c:a", "aac", "-b:a", project.output.audio_bitrate]
        cmd += [
            "-ss", f"{audio_start:.6f}", "-t", f"{clip_seconds:.6f}", "-i", str(project.audio.file),
            "-map", "0:v:0", "-map", "1:a:0",
            *codec_args(codec, project.output.bitrate, project.output.preset),
        ]
        if not transparent:
            cmd += ["-pix_fmt", "yuv420p"]
        cmd += ["-r", str(fps), *audio_codec, "-shortest"]
        if out_path.suffix.lower() in (".mp4", ".mov"):
            cmd += ["-movflags", "+faststart"]
        cmd += [str(out_path)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    done = 0
    cancelled = False
    frames = iter_frames(project, features, width, height, indices, workers=workers)
    writer = _PipeWriter(proc.stdin)
    try:
        for data in frames:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            writer.write(data)
            done += 1
            if progress and (done % 10 == 0 or done == n):
                progress(done, n)
        writer.close()
    except (BrokenPipeError, OSError) as exc:
        frames.close()
        writer.abort()
        _, err = proc.communicate()
        raise ExportError(f"ffmpeg cerró la entrada: {err.decode(errors='ignore').strip()}") from exc
    except BaseException:  # error de render: no dejar ffmpeg ni el hilo escritor colgados
        proc.kill()
        writer.abort()
        proc.communicate()
        raise
    finally:
        frames.close()  # termina el pool de procesos si quedó a medias
    if cancelled:
        proc.kill()
        writer.abort()
        proc.communicate()
        try:
            if out_path.is_dir():
                shutil.rmtree(out_path, ignore_errors=True)
            else:
                out_path.unlink()
        except OSError:
            pass
        raise ExportCancelled("Render cancelado")
    _, err = proc.communicate()  # cierra stdin y espera a que ffmpeg termine
    if proc.returncode != 0:
        raise ExportError(f"ffmpeg falló (código {proc.returncode}): {err.decode(errors='ignore').strip()}")
    return ExportResult(path=out_path, frames=done, seconds=time.perf_counter() - t0, codec=codec, width=width, height=height)
