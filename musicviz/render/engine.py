"""Motor de render: construye la escena a partir del proyecto y renderiza frames (en paralelo)."""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Iterable, Iterator, Optional

import cv2
import numpy as np

from ..audio.analysis import AudioFeatures, analyze
from ..config import ProjectConfig
from ..effects import build_effect
from ..layers import Background, build_layer
from .canvas import Canvas, RenderContext, float_to_uint8


class Scene:
    """Fondo + capas + efectos preparados para una resolución concreta."""

    def __init__(self, project: ProjectConfig, features: AudioFeatures, width: int, height: int):
        self.project = project
        self.features = features
        self.ctx = RenderContext(width=width, height=height, fps=features.fps, n_frames=features.n_frames)
        self.background = Background(project.background)
        self.layers = [build_layer(cfg) for cfg in project.layers if cfg.enabled]
        self.effects = [build_effect(cfg) for cfg in project.effects if cfg.enabled]
        self.background.prepare(self.ctx, features)
        for layer in self.layers:
            layer.prepare(self.ctx, features)
        for effect in self.effects:
            effect.prepare(self.ctx, features)
        self.canvas = Canvas(width, height)

    def render_float(self, index: int) -> np.ndarray:
        frame = self.features.frame(index)
        canvas = self.canvas
        self.background.render(canvas, frame)
        for layer in self.layers:
            layer.render(canvas, frame)
        img = canvas.img
        for effect in self.effects:
            img = effect.apply(img, frame)
        return img

    def render(self, index: int) -> np.ndarray:
        """Frame RGB uint8 (H, W, 3)."""
        return float_to_uint8(self.render_float(index))


def output_size(project: ProjectConfig, scale: float = 1.0) -> tuple[int, int]:
    w = int(round(project.output.width * scale))
    h = int(round(project.output.height * scale))
    return max(w - w % 2, 2), max(h - h % 2, 2)


def analyze_project(project: ProjectConfig) -> AudioFeatures:
    return analyze(project.audio, fps=float(project.output.fps))


def default_workers() -> int:
    cpus = os.cpu_count() or 2
    # Cada proceso usa ~150-300 MB a 1080p; con 8 GB de RAM es prudente no pasar de 6-8.
    return max(1, min(cpus - 2, 8))


# --------------------------------------------------------------------------- workers

_SCENE: Optional[Scene] = None


def _init_worker(project_dict: dict, features: AudioFeatures, width: int, height: int) -> None:
    global _SCENE
    cv2.setNumThreads(1)
    project = ProjectConfig.model_validate(project_dict)
    _SCENE = Scene(project, features, width, height)


def _render_worker(index: int) -> bytes:
    assert _SCENE is not None
    return _SCENE.render(index).tobytes()


def iter_frames(
    project: ProjectConfig,
    features: AudioFeatures,
    width: int,
    height: int,
    indices: Iterable[int],
    workers: int = 1,
) -> Iterator[bytes]:
    """Genera frames RGB24 crudos (bytes) en orden, usando `workers` procesos."""
    indices = list(indices)
    if workers <= 1 or len(indices) < 8:
        scene = Scene(project, features, width, height)
        for i in indices:
            yield scene.render(i).tobytes()
        return
    ctx = mp.get_context("spawn")
    project_dict = project.model_dump(mode="json")
    chunk = max(1, min(16, len(indices) // (workers * 4)))
    with ctx.Pool(workers, initializer=_init_worker, initargs=(project_dict, features, width, height)) as pool:
        for data in pool.imap(_render_worker, indices, chunksize=chunk):
            yield data


def render_frame_image(project: ProjectConfig, features: AudioFeatures, time: float, scale: float = 1.0) -> np.ndarray:
    """Renderiza un único frame (RGB uint8) en el instante `time` (segundos)."""
    w, h = output_size(project, scale)
    scene = Scene(project, features, w, h)
    index = int(round(time * features.fps))
    return scene.render(index)
