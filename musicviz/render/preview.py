"""Vista previa en ventana (Tkinter) con reproducción de audio opcional (sounddevice).

No necesita la versión de OpenCV con ventanas: dibuja los frames en un lienzo de Tkinter.
"""
from __future__ import annotations

import time

import numpy as np

from ..audio.analysis import AudioFeatures
from ..config import ProjectConfig
from .canvas import over_checkerboard
from .engine import make_scene, output_size


def preview(project: ProjectConfig, features: AudioFeatures, scale: float = 0.5, start: float = 0.0, with_audio: bool = True) -> None:
    import tkinter as tk

    from PIL import Image, ImageTk

    w, h = output_size(project, scale)
    scene = make_scene(project, features, w, h)
    fps = features.fps
    state = {"index": int(round(start * fps)), "paused": False, "t_wall": time.perf_counter(), "t_media": start, "audio_pos": 0, "stream": None}
    if with_audio:
        try:
            import sounddevice as sd

            state["stream"] = sd.OutputStream(samplerate=features.sr, channels=1, dtype="float32")
            state["stream"].start()
            state["audio_pos"] = int(start * features.sr)
        except Exception:  # noqa: BLE001 - audio opcional
            state["stream"] = None

    root = tk.Tk()
    root.title(f"musicviz preview - {project.name} (ESC salir, ESPACIO pausa, ←/→ ±5s)")
    canvas = tk.Canvas(root, width=w, height=h, bg="black", highlightthickness=0)
    canvas.pack()
    photo = {"img": None}

    def seek(delta: float) -> None:
        state["index"] = int(min(max(state["index"] + delta * fps, 0), features.n_frames - 1))
        state["t_wall"], state["t_media"] = time.perf_counter(), state["index"] / fps
        state["audio_pos"] = int(state["t_media"] * features.sr)

    def toggle_pause(_=None) -> None:
        state["paused"] = not state["paused"]
        state["t_wall"], state["t_media"] = time.perf_counter(), state["index"] / fps

    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("<space>", toggle_pause)
    root.bind("<Left>", lambda e: seek(-5))
    root.bind("<Right>", lambda e: seek(5))

    def tick() -> None:
        if state["index"] >= features.n_frames:
            root.destroy()
            return
        frame = scene.render(state["index"])
        if frame.shape[2] == 4:
            frame = over_checkerboard(frame)
        photo["img"] = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo["img"])
        stream = state["stream"]
        if stream is not None and not state["paused"]:
            end = min(state["audio_pos"] + int(features.sr / fps), len(features.waveform))
            stream.write(np.ascontiguousarray(features.waveform[state["audio_pos"] : end]).reshape(-1, 1))
            state["audio_pos"] = end
        if not state["paused"]:
            if stream is None:
                state["index"] = int((state["t_media"] + (time.perf_counter() - state["t_wall"])) * fps)
            else:
                state["index"] += 1
        root.after(1, tick)

    root.after(1, tick)
    try:
        root.mainloop()
    finally:
        if state["stream"] is not None:
            state["stream"].stop()
            state["stream"].close()
        scene.close()
