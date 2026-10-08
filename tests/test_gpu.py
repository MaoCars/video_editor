"""Backend GPU: disponibilidad, paridad con el backend CPU y exportación."""
import os
import shutil

import numpy as np
import pytest

from musicviz.config import ProjectConfig
from musicviz.presets import load_preset
from musicviz.render.engine import Scene, iter_frames, make_scene, resolve_backend
from musicviz.render.gpu import gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="sin contexto OpenGL disponible")

W, H = 320, 180


def _project(audio_cfg, **extra):
    data = {"audio": audio_cfg.model_dump(), "output": {"width": W, "height": H, "fps": 30}}
    data.update(extra)
    return ProjectConfig.model_validate(data)


def _diff(a, b):
    if a.shape[2] == 4:
        from musicviz.render.canvas import over_checkerboard

        a, b = over_checkerboard(a), over_checkerboard(b)
    d = np.abs(a.astype(int) - b.astype(int))
    return d.mean(), (d.max(axis=2) > 60).mean() * 100


def test_backend_resolution(monkeypatch):
    assert resolve_backend("auto") == "gpu"
    assert resolve_backend("cpu") == "cpu"
    assert resolve_backend("gpu") == "gpu"


@pytest.mark.parametrize("preset", ["trap_nation", "monstercat", "ncs", "dnb_glitch", "auto_sections", "spectrum_only"])
def test_presets_match_cpu(audio_cfg, features, preset):
    project = load_preset(preset, audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    cpu = Scene(project, features, W, H)
    gpu = make_scene(project, features, W, H, "gpu")
    try:
        assert type(gpu).__name__ == "GpuScene"
        for idx in (10, 60, 120):
            mean, big = _diff(cpu.render(idx), gpu.render(idx))
            assert mean < 6.0 and big < 6.0, (preset, idx, mean, big)
    finally:
        gpu.close()


def test_layers_and_effects_match_cpu(audio_cfg, features, tmp_path):
    import cv2

    photo = tmp_path / "p.png"
    img = np.zeros((90, 160, 3), np.uint8)
    img[:, :80] = (255, 60, 0)
    img[:, 80:] = (0, 200, 255)
    cv2.imwrite(str(photo), img)
    layers = [
        {"type": "bars", "colors": ["#ffffff", "#00ffff"], "glow": 0.3, "mirror": True, "symmetric": True},
        {"type": "circle", "colors": ["#ff2a6d", "#ffd166"], "rings": 2, "glow": 0.5, "style": "filled", "position": [0.3, 0.5], "radius": 0.15},
        {"type": "waveform", "style": "filled", "colors": ["#ff0000", "#00ff00"], "mirror": True, "position": [0.5, 0.85]},
        {"type": "particles", "count": 40, "burst": 20, "shape": "streak", "glow": 0.4},
        {"type": "image", "file": str(photo), "shape": "rounded", "size": 0.3, "border": 3, "shadow": 0.5, "position": [0.8, 0.3], "rotation": 15},
        {"type": "text", "text": "GPU", "size": 0.15, "stroke_width": 2, "shadow": 0.5, "position": [0.8, 0.75], "rotation_keys": [{"time": 0, "value": -10}]},
        {"type": "progress", "show_time": True},
    ]
    effects = [{"type": "bloom"}, {"type": "chromatic", "trigger": "always", "amount": 4}, {"type": "shake", "trigger": "always", "rotation": 2},
               {"type": "color", "hue_speed": 10, "saturation": 1.2}, {"type": "vignette"}, {"type": "scanlines"}, {"type": "glitch", "trigger": "always", "invert": 0.5, "noise": 0}]
    project = _project(audio_cfg, background={"type": "image", "image": str(photo), "pulse": 0.1, "shake": 10}, layers=layers, effects=effects)
    cpu = Scene(project, features, W, H)
    gpu = make_scene(project, features, W, H, "gpu")
    try:
        mean, big = _diff(cpu.render(80), gpu.render(80))
        assert mean < 6.0 and big < 6.0, (mean, big)
    finally:
        gpu.close()


def test_transparent_gpu(audio_cfg, features):
    project = load_preset("spectrum_only", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    gpu = make_scene(project, features, W, H, "gpu")
    try:
        img = gpu.render(60)
        assert img.shape == (H, W, 4) and img[2, 2, 3] == 0 and img[..., 3].max() > 200
    finally:
        gpu.close()


def test_iter_frames_gpu_sequential(audio_cfg, features):
    project = load_preset("minimal", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    project.output.backend = "gpu"
    frames = list(iter_frames(project, features, W, H, range(0, 6), workers=4))
    assert len(frames) == 6 and len(frames[0]) == W * H * 3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no disponible")
def test_export_with_gpu_backend(audio_cfg, features, tmp_path):
    from musicviz.render.exporter import encoder_works, export_video

    if not encoder_works("libx264") and not encoder_works("h264_nvenc"):
        pytest.skip("sin encoder H.264")
    project = load_preset("trap_nation", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    project.output.backend = "gpu"
    result = export_video(project, features, output=tmp_path / "gpu.mp4", duration=0.5, workers=1)
    assert result.path.exists() and result.frames == 15


def test_no_gpu_env_forces_cpu(audio_cfg, features, monkeypatch):
    monkeypatch.setenv("MUSICVIZ_NO_GPU", "1")
    gpu_available.cache_clear()
    try:
        assert resolve_backend("auto") == "cpu"
        with pytest.raises(RuntimeError):
            resolve_backend("gpu")
    finally:
        gpu_available.cache_clear()
