import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="estas pruebas usan el backend CPU (OpenCV)")

from musicviz.config import ProjectConfig
from musicviz.presets import load_preset
from musicviz.render.canvas import Canvas, paste_rgba
from musicviz.render.engine import Scene, iter_frames, render_frame_image
from musicviz.utils.color import color_at, gradient, parse_color, to_cv

W, H = 320, 180


def _project(audio_cfg, **extra):
    data = {"audio": audio_cfg.model_dump(), "output": {"width": W, "height": H, "fps": 30}}
    data.update(extra)
    return ProjectConfig.model_validate(data)


def test_color_utils():
    assert parse_color("#ff0000") == (1.0, 0.0, 0.0, 1.0)
    assert parse_color("#0f0") == (0.0, 1.0, 0.0, 1.0)
    assert parse_color("#00000080")[3] == pytest.approx(128 / 255)
    g = gradient(["#000000", "#ffffff"], 3)
    assert np.allclose(g[1, :3], 0.5)
    assert color_at(["#000000", "#ffffff"], 0.5)[0] == pytest.approx(0.5)
    assert to_cv((1, 0.5, 0, 1), 0.5) == (255, 128, 0, 128)


def test_canvas_blend_modes():
    c = Canvas(8, 8)
    layer = c.new_layer()
    layer[:, :] = (255, 0, 0, 128)
    c.composite(layer, blend="normal")
    assert np.allclose(c.img[0, 0], [128 / 255, 0, 0], atol=1e-2)
    c.composite(layer, blend="add")
    assert c.img[0, 0, 0] > 0.9
    c2 = Canvas(8, 8)
    c2.img[:] = 0.5
    c2.composite(layer, blend="screen")
    assert 0.5 < c2.img[0, 0, 0] < 1.0 and c2.img[0, 0, 1] == pytest.approx(0.5)


def test_paste_rgba_clipping():
    layer = np.zeros((10, 10, 4), np.uint8)
    sprite = np.full((6, 6, 4), 255, np.uint8)
    paste_rgba(layer, sprite, 0, 0)  # parcialmente fuera
    assert layer[0, 0, 3] == 255 and layer[5, 5, 3] == 0
    paste_rgba(layer, sprite, 50, 50)  # totalmente fuera: no falla


@pytest.mark.parametrize("preset", ["trap_nation", "monstercat", "ncs", "dnb_glitch", "minimal"])
def test_presets_render_frame(audio_cfg, features, preset):
    project = load_preset(preset, audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    scene = Scene(project, features, W, H)
    for i in (0, 45, features.n_frames - 1):
        img = scene.render(i)
        assert img.shape == (H, W, 3) and img.dtype == np.uint8
    # la escena no debe estar vacía en un frame con música
    assert scene.render(60).mean() > 2


def test_all_layer_and_effect_types(audio_cfg, features, tmp_path):
    import cv2

    logo = tmp_path / "logo.png"
    cv2.imwrite(str(logo), np.full((40, 40, 4), 255, np.uint8))
    layers = [
        {"type": "bars", "style": "segments", "symmetric": True, "gradient": "height"},
        {"type": "bars", "style": "outline", "rounded": False, "mirror": False, "baseline": True},
        {"type": "bars", "style": "dots"},
        {"type": "circle", "style": "line", "inner": True, "rings": 2},
        {"type": "circle", "style": "filled", "mirror": False},
        {"type": "circle", "style": "dots"},
        {"type": "circle", "style": "rays", "gradient": "value"},
        {"type": "waveform", "style": "filled", "mirror": True},
        {"type": "waveform", "style": "circular"},
        {"type": "waveform", "style": "bars"},
        {"type": "waveform", "style": "line", "colors": ["#f00", "#0f0"], "mirror": True},
        {"type": "particles", "count": 20, "burst": 5, "emitter": "center", "direction": "out", "shape": "streak"},
        {"type": "particles", "count": 20, "burst": 5, "emitter": "bottom", "direction": "up", "shape": "square", "gravity": 50},
        {"type": "particles", "count": 10, "burst": 3, "emitter": "ring", "direction": "in", "burst_trigger": "kick"},
        {"type": "image", "file": str(logo), "circle_mask": True, "rotation_speed": 30, "shake": 5},
        {"type": "text", "text": "Hola", "letter_spacing": 2, "uppercase": True},
        {"type": "progress", "show_time": True},
    ]
    effects = [
        {"type": "glitch", "trigger": "always", "invert": 0.5},
        {"type": "bloom"},
        {"type": "chromatic"},
        {"type": "shake", "trigger": "always", "rotation": 1},
        {"type": "vignette"},
        {"type": "color", "hue_speed": 30, "saturation": 1.2, "contrast": 1.1, "brightness": 0.05, "gamma": 1.2, "posterize": 8},
        {"type": "pixelate", "trigger": "always"},
        {"type": "strobe", "trigger": "always"},
        {"type": "kaleido", "segments": 4},
        {"type": "radial_blur", "trigger": "always"},
        {"type": "scanlines"},
        {"type": "grain"},
        {"type": "bloom", "start": 100.0},  # fuera de ventana temporal: no se aplica
    ]
    project = _project(audio_cfg, background={"type": "image", "image": str(logo), "blur": 2, "pulse": 0.1, "react": 0.2}, layers=layers, effects=effects)
    scene = Scene(project, features, W, H)
    img = scene.render(50)
    assert img.shape == (H, W, 3)
    assert np.isfinite(img).all()


def test_background_types(audio_cfg, features):
    for bg in ({"type": "solid", "color": "#102030"}, {"type": "gradient", "angle": 45}, {"type": "radial"}):
        scene = Scene(_project(audio_cfg, background=bg), features, W, H)
        img = scene.render(0)
        assert img.shape == (H, W, 3)
    solid = Scene(_project(audio_cfg, background={"type": "solid", "color": "#102030"}), features, W, H).render(0)
    assert tuple(solid[0, 0]) == (0x10, 0x20, 0x30)


def test_render_is_deterministic(audio_cfg, features):
    project = load_preset("dnb_glitch", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    a = Scene(project, features, W, H).render(70)
    b = Scene(project, features, W, H).render(70)
    assert np.array_equal(a, b)


def test_iter_frames_parallel_matches_serial(audio_cfg, features):
    project = load_preset("minimal", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    idx = list(range(0, 24))
    serial = list(iter_frames(project, features, W, H, idx, workers=1))
    parallel = list(iter_frames(project, features, W, H, idx, workers=2))
    assert serial == parallel


def test_render_frame_image_scale(audio_cfg, features):
    project = load_preset("minimal", audio_cfg.file)
    project.output.width, project.output.height = 640, 360
    img = render_frame_image(project, features, 1.0, scale=0.5)
    assert img.shape == (180, 320, 3)
