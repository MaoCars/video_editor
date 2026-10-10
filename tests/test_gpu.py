"""Backend GPU: disponibilidad, paridad con el backend CPU y exportación."""
import os
import shutil

import numpy as np
import pytest

from musicviz.config import ProjectConfig
from musicviz.presets import load_preset
from musicviz.render.engine import Scene, iter_frames, make_scene, resolve_backend
from musicviz.render.gpu import gpu_available
from musicviz.utils.imaging import HAS_CV2, resize, save_image

pytestmark = pytest.mark.skipif(not gpu_available(), reason="sin contexto OpenGL disponible")

W, H = 320, 180
needs_cpu = pytest.mark.skipif(not HAS_CV2, reason="la paridad con el backend CPU necesita OpenCV")


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
    assert resolve_backend("gpu") == "gpu"
    if HAS_CV2:
        assert resolve_backend("cpu") == "cpu"
    else:
        with pytest.raises(RuntimeError, match="OpenCV"):
            resolve_backend("cpu")


@needs_cpu
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


@needs_cpu
def test_layers_and_effects_match_cpu(audio_cfg, features, tmp_path):
    photo = tmp_path / "p.png"
    img = np.zeros((90, 160, 3), np.uint8)
    img[:, :80] = (0, 60, 255)
    img[:, 80:] = (255, 200, 0)
    save_image(photo, img)
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
        if HAS_CV2:
            assert resolve_backend("auto") == "cpu"
        else:
            with pytest.raises(RuntimeError, match="OpenCV"):
                resolve_backend("auto")
        with pytest.raises(RuntimeError):
            resolve_backend("gpu")
    finally:
        gpu_available.cache_clear()


def test_render_many_matches_render(audio_cfg, features, tmp_path):
    """La lectura asíncrona con PBOs (doble búfer) devuelve exactamente los mismos frames y en orden."""
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg no disponible")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x36:r=30:d=2", "-pix_fmt", "yuv420p", str(clip)], check=True)
    layers = [{"type": "bars", "glow": 0.3}, {"type": "particles", "count": 30}, {"type": "text", "text": "PBO", "position": [0.5, 0.2]}]
    effects = [{"type": "bloom"}, {"type": "glitch", "trigger": "always"}]
    for transparent in (False, True):
        project = _project(audio_cfg, background={"type": "video", "video": str(clip), "pulse": 0.1}, layers=layers, effects=effects)
        project.output.transparent = transparent
        gpu = make_scene(project, features, W, H, "gpu")
        try:
            idx = [0, 1, 2, 3, 4, 30, 31, 32]
            many = list(gpu.render_many(idx))
            single = [gpu.render(i) for i in idx]
            assert len(many) == len(idx) and many[0].shape == (H, W, 4 if transparent else 3)
            assert all(np.array_equal(a, b) for a, b in zip(many, single))
            assert list(gpu.render_many([7]))[0].shape == many[0].shape
        finally:
            gpu.close()


# ---------------------------------------------------------------- reproductor OpenGL
def test_screen_blit_keeps_orientation_and_aspect(audio_cfg, features):
    """Lo que se dibuja en pantalla es el frame final (misma orientación) con bandas negras si cambia la proporción."""
    from musicviz.render.gpu import GL_LOCK
    from musicviz.render.player import ScreenBlit

    layers = [{"type": "text", "text": "ARRIBA", "position": [0.5, 0.15], "size": 0.2}, {"type": "bars", "colors": ["#ff0000"]}]
    project = _project(audio_cfg, background={"type": "gradient", "colors": ["#000040", "#400000"]}, layers=layers)
    gpu = make_scene(project, features, W, H, "gpu")
    try:
        with GL_LOCK:
            blit = ScreenBlit(gpu.gl, gpu)
            tw, th = 480, 180  # más ancho que 16:9: bandas a los lados de 80 px
            tex = gpu.gl.texture((tw, th), 4, dtype="f1")
            fbo = gpu.gl.framebuffer(color_attachments=[tex])
            ref = gpu.render(40)
            blit.draw(fbo, tw, th)
            # fbo.read devuelve las filas de abajo arriba (convención GL, igual que la pantalla): se invierte para comparar
            shown = np.frombuffer(fbo.read(components=3, dtype="f1"), np.uint8).reshape(th, tw, 3)[::-1]
            fbo.release()
            tex.release()
            blit.release()
        assert shown[:, :80].max() == 0 and shown[:, -80:].max() == 0  # bandas negras
        inner = shown[:, 80:-80]
        assert inner.shape == ref.shape
        d = np.abs(inner.astype(int) - ref.astype(int))
        assert d.mean() < 2.0, d.mean()  # misma imagen, no invertida (el texto arriba sigue arriba)
        assert np.abs(inner.astype(int) - ref[::-1].astype(int)).mean() > d.mean() + 5
        # destino más alto que el frame: bandas arriba y abajo y el frame escalado
        with GL_LOCK:
            tex = gpu.gl.texture((160, 180), 4, dtype="f1")
            fbo = gpu.gl.framebuffer(color_attachments=[tex])
            blit2 = ScreenBlit(gpu.gl, gpu)
            blit2.draw(fbo, 160, 180)
            tall = np.frombuffer(fbo.read(components=3, dtype="f1"), np.uint8).reshape(180, 160, 3)[::-1]
            fbo.release()
            tex.release()
            blit2.release()
        assert tall[:40].max() == 0 and tall[-40:].max() == 0
        small = resize(ref, 160, 90)
        assert np.abs(tall[45:135].astype(int) - small.astype(int)).mean() < 12
    finally:
        gpu.close()


def test_window_size_fits_monitor():
    from musicviz.render.player import window_size

    assert window_size(1920, 1080, 2560, 1440) == (1920, 1080)
    assert window_size(1920, 1080, 1366, 768) == (1229, 691)
    assert window_size(1080, 1920, 1920, 1080) == (547, 972)


def test_player_window_runs_to_end(audio_cfg, features):
    """Abre la ventana glfw de verdad (bajo Xvfb en CI) y reproduce los últimos 0.3 s hasta cerrarse sola."""
    import subprocess
    import sys

    pytest.importorskip("glfw")
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        pytest.skip("sin pantalla")
    project = load_preset("minimal", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    code = (
        "import sys, json\n"
        "from musicviz.config import ProjectConfig\n"
        "from musicviz.render.engine import analyze_project\n"
        "from musicviz.render.player import run_player\n"
        f"p = ProjectConfig.model_validate_json({project.model_dump_json()!r})\n"
        "f = analyze_project(p)\n"
        "run_player(p, f, start=f.duration - 0.3, with_audio=False)\n"
        "print('PLAYER_OK')\n"
    )
    env = dict(os.environ, MUSICVIZ_NO_CACHE="1")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    if res.returncode != 0 and not sys.platform.startswith("linux"):
        pytest.skip(f"ventana OpenGL no disponible en este runner: {res.stderr[-300:]}")
    assert res.returncode == 0 and "PLAYER_OK" in res.stdout, res.stderr[-2000:]
