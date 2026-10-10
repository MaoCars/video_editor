import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="estas pruebas usan el backend CPU (OpenCV)")

from musicviz.config import PointKey, ProjectConfig, ScalarKey, TextLayer
from musicviz.layers.keyframes import current_value, evaluate, has_keys, key_state
from musicviz.render.engine import Scene

W, H = 320, 180


def _project(audio_cfg, **extra):
    data = {"audio": audio_cfg.model_dump(), "output": {"width": W, "height": H, "fps": 30}}
    data.update(extra)
    return ProjectConfig.model_validate(data)


def test_evaluate_scalar_and_point():
    keys = [ScalarKey(time=2, value=0.0, easing="linear"), ScalarKey(time=4, value=1.0, easing="linear"), ScalarKey(time=1, value=0.0)]
    assert evaluate(keys, 0.0) == 0.0  # antes del primero (orden por tiempo)
    assert evaluate(keys, 3.0) == pytest.approx(0.5)
    assert evaluate(keys, 10.0) == 1.0
    assert evaluate([], 1.0) is None
    pk = [PointKey(time=0, value=(0, 0)), PointKey(time=2, value=(1, 0.5), easing="linear")]
    assert evaluate(pk, 1.0) == pytest.approx((0.5, 0.25))
    eased = [ScalarKey(time=0, value=0), ScalarKey(time=1, value=1, easing="ease_in")]
    assert evaluate(eased, 0.5) == pytest.approx(0.25)


def test_key_state_and_current_value():
    cfg = TextLayer(type="text", text="x", position=(0.2, 0.3), rotation_keys=[ScalarKey(time=0, value=0), ScalarKey(time=2, value=90, easing="linear")], opacity_keys=[ScalarKey(time=0, value=2.0)])
    st = key_state(cfg, 1.0)
    assert st.position is None and st.rotation == pytest.approx(45) and st.opacity == 1.0 and st.scale == 1.0
    assert has_keys(cfg) and not has_keys(TextLayer(type="text", text="x"))
    assert current_value(cfg, "position", 1.0) == (0.2, 0.3)
    assert current_value(cfg, "rotation", 1.0) == pytest.approx(45)
    assert current_value(cfg, "scale", 1.0) == 1.0


def _col_center(img, x_rel):
    x = int(x_rel * W)
    return img[:, max(x - 4, 0) : x + 4].max()


def test_position_keyframes_move_image(audio_cfg, features, tmp_path):
    import cv2

    photo = tmp_path / "p.png"
    cv2.imwrite(str(photo), np.full((40, 40, 3), 255, np.uint8))
    layer = {"type": "image", "file": str(photo), "size": 0.2, "pulse": 0, "position_keys": [{"time": 1, "value": [0.1, 0.5]}, {"time": 3, "value": [0.9, 0.5], "easing": "linear"}]}
    scene = Scene(_project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=[layer]), features, W, H)
    at1 = scene.render(30)
    at2 = scene.render(60)
    at3 = scene.render(90)
    assert _col_center(at1, 0.1) > 200 and _col_center(at1, 0.9) < 10
    assert _col_center(at2, 0.5) > 200
    assert _col_center(at3, 0.9) > 200 and _col_center(at3, 0.1) < 10


def test_scale_opacity_rotation_keyframes(audio_cfg, features):
    base = {"type": "text", "text": "HOLA", "size": 0.25, "pulse": 0}
    big = dict(base, scale_keys=[{"time": 0, "value": 2.0}])
    gone = dict(base, opacity_keys=[{"time": 0, "value": 1.0}, {"time": 2, "value": 0.0, "easing": "linear"}])
    rot = dict(base, rotation_keys=[{"time": 0, "value": 90}])
    bg = {"type": "solid", "color": "#000000"}
    normal = Scene(_project(audio_cfg, background=bg, layers=[base]), features, W, H).render(30)
    scaled = Scene(_project(audio_cfg, background=bg, layers=[big]), features, W, H).render(30)
    assert (scaled > 100).sum() > (normal > 100).sum() * 1.5
    fading = Scene(_project(audio_cfg, background=bg, layers=[gone]), features, W, H)
    assert fading.render(30).max() < 200 and fading.render(30).max() > 60  # a mitad: ~50 %
    assert fading.render(90).max() == 0  # opacidad 0: la capa no se dibuja
    rotated = Scene(_project(audio_cfg, background=bg, layers=[rot]), features, W, H).render(30)
    # girado 90°: la tinta se extiende más en vertical que en horizontal
    ys, xs = np.nonzero(rotated.max(axis=2) > 100)
    assert np.ptp(ys) > np.ptp(xs)


def test_circle_and_bars_follow_keys(audio_cfg, features):
    bg = {"type": "solid", "color": "#000000"}
    circle = {"type": "circle", "colors": ["#ffffff"], "radius": 0.15, "pulse": 0, "glow": 0, "position_keys": [{"time": 0, "value": [0.25, 0.5]}], "scale_keys": [{"time": 0, "value": 0.5}]}
    img = Scene(_project(audio_cfg, background=bg, layers=[circle]), features, W, H).render(30)
    ys, xs = np.nonzero(img.max(axis=2) > 100)
    assert abs(xs.mean() - 0.25 * W) < 8 and np.ptp(xs) < 0.3 * W  # centrado a la izquierda y a mitad de tamaño
    bars = {"type": "bars", "colors": ["#ffffff"], "glow": 0, "min_height": 0.3, "opacity_keys": [{"time": 0, "value": 0.25}]}
    dim = Scene(_project(audio_cfg, background=bg, layers=[bars]), features, W, H).render(30)
    assert 40 < dim.max() < 90


def test_keyframe_lists_hidden_from_forms():
    from musicviz.gui.fields import field_specs

    names = [s.name for s in field_specs(TextLayer)]
    assert "position_keys" not in names and "opacity_keys" not in names
