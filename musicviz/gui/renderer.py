"""Hilo de render de la interfaz: un solo hilo con contexto OpenGL persistente y escena en caché.

Evita crear un contexto por cada vista previa (20-100 ms en Windows) y reconstruir la escena
(sprites de texto, partículas...) cuando sólo cambia el instante de tiempo.
"""
from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable, Optional

from ..audio.analysis import AudioFeatures
from ..config import ProjectConfig
from ..render.engine import SceneBase, analyze_project, make_scene, output_size


class RenderService:
    def __init__(self, on_error: Optional[Callable[[str], None]] = None):
        self._jobs: "queue.Queue[tuple[Optional[str], Callable[[RenderService], Any]]]" = queue.Queue()
        self._on_error = on_error
        self._scene: Optional[SceneBase] = None
        self._scene_key: Optional[tuple] = None
        self._features: Optional[AudioFeatures] = None
        self._features_key: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="musicviz-render", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ cola
    def submit(self, fn: Callable[["RenderService"], Any], key: Optional[str] = None) -> None:
        """Encola un trabajo. Si `key` coincide con trabajos pendientes, los sustituye (sólo importa el último)."""
        if key is not None:
            self._drop_pending(key)
        self._jobs.put((key, fn))

    def _drop_pending(self, key: str) -> None:
        kept = []
        try:
            while True:
                item = self._jobs.get_nowait()
                if item[0] != key:
                    kept.append(item)
        except queue.Empty:
            pass
        for item in kept:
            self._jobs.put(item)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                key, fn = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                fn(self)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                if self._on_error:
                    self._on_error(str(exc))
        self._release_scene()

    def shutdown(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ recursos (sólo desde el hilo de render)
    def features_for(self, project: ProjectConfig, on_analyzing: Optional[Callable[[], None]] = None, on_ready: Optional[Callable[[AudioFeatures], None]] = None) -> AudioFeatures:
        key = project.audio.model_dump_json() + f"|{project.output.fps}"
        if self._features is None or key != self._features_key:
            if on_analyzing:
                on_analyzing()
            self._features = analyze_project(project)
            self._features_key = key
            self._release_scene()
            if on_ready:
                on_ready(self._features)
        return self._features

    @property
    def features(self) -> Optional[AudioFeatures]:
        return self._features

    def scene_for(self, project: ProjectConfig, scale: float) -> SceneBase:
        w, h = output_size(project, scale)
        key = (project.model_dump_json(), w, h, id(self._features))
        if self._scene is None or key != self._scene_key:
            self._release_scene()
            self._scene = make_scene(project, self.features_for(project), w, h)
            self._scene_key = key
        return self._scene

    def invalidate_scene(self) -> None:
        self._scene_key = None

    def _release_scene(self) -> None:
        if self._scene is not None:
            try:
                self._scene.close()
            except Exception:  # noqa: BLE001
                pass
            self._scene = None
            self._scene_key = None
