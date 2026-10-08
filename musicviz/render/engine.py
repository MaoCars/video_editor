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
from .canvas import Canvas, RenderContext, float_to_uint8, premultiplied_to_rgba8
from .sections import SectionTimeline, resolve_sections


class SceneBase:
    """Fondo + capas + efectos preparados para una resolución concreta (común a los backends CPU y GPU)."""

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
        self.transparent = bool(project.output.transparent)
        self.timeline = SectionTimeline(resolve_sections(project, features))

    @property
    def channels(self) -> int:
        return 4 if self.transparent else 3

    def frame_state(self, index: int):
        """FrameFeatures con el estado de sección aplicado, y la sección activa."""
        frame = self.features.frame(index)
        section = self.timeline.state_at(frame.time)
        frame.palette = section.palette
        frame.intensity = section.intensity
        frame.section = section.name
        return frame, section

    def active_layers(self, frame, section):
        for layer in self.layers:
            if not section.allows_layer(layer.cfg.name, layer.cfg.type):
                continue
            if layer.cfg.sections is not None and section.name not in layer.cfg.sections:
                continue
            if layer.begin_frame(frame):
                yield layer

    def active_effects(self, section):
        for effect in self.effects:
            if not section.allows_effect(effect.cfg.name, effect.cfg.type):
                continue
            if effect.cfg.sections is not None and section.name not in effect.cfg.sections:
                continue
            yield effect

    def render(self, index: int) -> np.ndarray:  # pragma: no cover - lo implementan los backends
        raise NotImplementedError

    def render_many(self, indices: Iterable[int]) -> Iterator[np.ndarray]:
        """Renderiza varios frames en orden. Los backends pueden solapar trabajo entre frames consecutivos."""
        for i in indices:
            yield self.render(i)

    def close(self) -> None:
        """Libera recursos (hilo de decodificación de video, contexto GL...)."""
        self.background.close()


class Scene(SceneBase):
    """Backend CPU (NumPy + OpenCV)."""

    def __init__(self, project: ProjectConfig, features: AudioFeatures, width: int, height: int):
        super().__init__(project, features, width, height)
        self.canvas = Canvas(width, height, track_alpha=self.transparent)

    def render_float(self, index: int) -> np.ndarray:
        """Frame float32: (H, W, 3) RGB o, en modo transparente, (H, W, 4) RGB premultiplicado + alfa."""
        frame, section = self.frame_state(index)
        canvas = self.canvas
        if self.transparent:
            canvas.clear()
        else:
            self.background.render(canvas, frame, section.background)
        for layer in self.active_layers(frame, section):
            layer.render(canvas, frame)
        img = canvas.img
        if self.transparent:
            img = np.concatenate([img, canvas.alpha], axis=2)
        for effect in self.active_effects(section):
            img = effect.apply(img, frame)
        return img

    def render(self, index: int) -> np.ndarray:
        """Frame uint8: RGB (H, W, 3) o RGBA (H, W, 4) si el proyecto es transparente."""
        img = self.render_float(index)
        if self.transparent:
            return premultiplied_to_rgba8(img)
        return float_to_uint8(img)


def resolve_backend(requested: str) -> str:
    """'gpu' o 'cpu' según lo pedido y la disponibilidad de OpenGL."""
    if requested == "cpu":
        return "cpu"
    from .gpu import gpu_available

    if gpu_available():
        return "gpu"
    if requested == "gpu":
        raise RuntimeError("El backend GPU no está disponible (no se pudo crear un contexto OpenGL). Usa backend: cpu.")
    return "cpu"


def make_scene(project: ProjectConfig, features: AudioFeatures, width: int, height: int, backend: Optional[str] = None) -> SceneBase:
    """Crea la escena con el backend indicado (None = el del proyecto)."""
    chosen = resolve_backend(backend or project.output.backend)
    if chosen == "gpu":
        from .gpu.scene import GpuScene

        return GpuScene(project, features, width, height)
    return Scene(project, features, width, height)


def output_size(project: ProjectConfig, scale: float = 1.0) -> tuple[int, int]:
    w = int(round(project.output.width * scale))
    h = int(round(project.output.height * scale))
    return max(w - w % 2, 2), max(h - h % 2, 2)


def analyze_project(project: ProjectConfig, use_cache: bool = True) -> AudioFeatures:
    """Analiza el audio del proyecto, reutilizando la caché en disco si el archivo y los parámetros no cambiaron."""
    from ..audio import cache
    from ..audio.analysis import DEFAULT_SR

    fps = float(project.output.fps)
    if use_cache:
        cached = cache.load(project.audio, fps, DEFAULT_SR)
        if cached is not None:
            return cached
    features = analyze(project.audio, fps=fps)
    if use_cache:
        cache.store(project.audio, fps, DEFAULT_SR, features)
    return features


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
    _SCENE = Scene(project, features, width, height)  # los procesos auxiliares siempre usan CPU


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
    """Genera frames crudos (bytes RGB24, o RGBA en modo transparente) en orden.

    Con backend GPU el render es secuencial en este proceso (la GPU hace el trabajo); con CPU se
    reparte entre `workers` procesos.
    """
    indices = list(indices)
    backend = resolve_backend(project.output.backend)
    if backend == "gpu" or workers <= 1 or len(indices) < 8:
        scene = make_scene(project, features, width, height, backend)
        try:
            for img in scene.render_many(indices):
                yield img.tobytes()
        finally:
            scene.close()
        return
    ctx = mp.get_context("spawn")
    project_dict = project.model_dump(mode="json")
    chunk = max(1, min(16, len(indices) // (workers * 4)))
    with ctx.Pool(workers, initializer=_init_worker, initargs=(project_dict, features, width, height)) as pool:
        for data in pool.imap(_render_worker, indices, chunksize=chunk):
            yield data


def render_frame_image(project: ProjectConfig, features: AudioFeatures, time: float, scale: float = 1.0) -> np.ndarray:
    """Renderiza un único frame (RGB uint8, o RGBA si el proyecto es transparente) en el instante `time`."""
    w, h = output_size(project, scale)
    scene = make_scene(project, features, w, h)
    try:
        index = int(round(time * features.fps))
        return scene.render(index)
    finally:
        scene.close()
