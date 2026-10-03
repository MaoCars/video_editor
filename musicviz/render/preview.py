"""Vista previa en ventana (OpenCV) con reproducción de audio opcional (sounddevice)."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..audio.analysis import AudioFeatures
from ..config import ProjectConfig
from .engine import Scene, output_size


def preview(project: ProjectConfig, features: AudioFeatures, scale: float = 0.5, start: float = 0.0, with_audio: bool = True) -> None:
    import cv2

    if not hasattr(cv2, "imshow"):
        raise RuntimeError("Esta instalación de OpenCV no tiene soporte de ventanas (opencv-python-headless). Instala opencv-python.")
    w, h = output_size(project, scale)
    scene = Scene(project, features, w, h)
    fps = features.fps
    index = int(round(start * fps))
    stream = None
    if with_audio:
        try:
            import sounddevice as sd

            stream = sd.OutputStream(samplerate=features.sr, channels=1, dtype="float32")
            stream.start()
        except Exception:  # noqa: BLE001 - audio opcional
            stream = None
    name = f"musicviz preview - {project.name} (ESC salir, ESPACIO pausa, ←/→ ±5s)"
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    paused = False
    t_wall = time.perf_counter()
    t_media = index / fps
    audio_pos = int(t_media * features.sr)
    try:
        while index < features.n_frames:
            frame = scene.render(index)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(name, bgr)
            if stream is not None and not paused:
                end = min(audio_pos + int(features.sr / fps), len(features.waveform))
                stream.write(np.ascontiguousarray(features.waveform[audio_pos:end]).reshape(-1, 1))
                audio_pos = end
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                paused = not paused
                t_wall = time.perf_counter()
                t_media = index / fps
            elif key in (81, 2424832):  # ←
                index = max(index - int(5 * fps), 0)
                t_wall, t_media, audio_pos = time.perf_counter(), index / fps, int(index / fps * features.sr)
                continue
            elif key in (83, 2555904):  # →
                index = min(index + int(5 * fps), features.n_frames - 1)
                t_wall, t_media, audio_pos = time.perf_counter(), index / fps, int(index / fps * features.sr)
                continue
            if paused:
                continue
            if stream is None:
                elapsed = time.perf_counter() - t_wall
                index = int((t_media + elapsed) * fps)
            else:
                index += 1
                audio_pos = int(index / fps * features.sr)
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        cv2.destroyAllWindows()
