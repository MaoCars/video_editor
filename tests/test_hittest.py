import pytest

from musicviz.config import ProjectConfig
from musicviz.gui.hittest import hit_layer, layer_box
from musicviz.layers.background import fit_image
import numpy as np


def _project(**extra):
    data = {"audio": {"file": "a.wav"}, "output": {"width": 1920, "height": 1080}}
    data.update(extra)
    return ProjectConfig.model_validate(data)


def test_boxes_follow_position_anchor_and_keys():
    p = _project(layers=[
        {"type": "circle", "radius": 0.2, "length": 0.1, "pulse": 0, "position": [0.3, 0.5]},
        {"type": "text", "text": "HOLA", "size": 0.1, "align": "left", "valign": "top", "position": [0.1, 0.1]},
        {"type": "image", "file": "x.png", "size": 0.5, "shape": "square", "anchor": "right", "position": [1.0, 0.5], "position_keys": [{"time": 0, "value": [0.5, 0.5]}, {"time": 2, "value": [1.0, 0.5], "easing": "linear"}]},
    ])
    cx0, cy0, cx1, cy1 = layer_box(p.layers[0], p, 0.0)
    assert (cx0 + cx1) / 2 == pytest.approx(0.3) and (cy0 + cy1) / 2 == pytest.approx(0.5)
    tx0, ty0, tx1, ty1 = layer_box(p.layers[1], p, 0.0)
    assert tx0 == pytest.approx(0.1) and ty0 == pytest.approx(0.1) and tx1 > tx0 and ty1 > ty0
    ix0, iy0, ix1, iy1 = layer_box(p.layers[2], p, 2.0)
    assert ix1 == pytest.approx(1.0) and (iy0 + iy1) / 2 == pytest.approx(0.5)
    assert layer_box(p.layers[2], p, 1.0)[2] == pytest.approx(0.75)  # a mitad del keyframe


def test_hit_layer_topmost_and_windows():
    p = _project(layers=[
        {"type": "bars", "position": [0.5, 0.5], "width": 0.9, "height": 0.4},
        {"type": "circle", "radius": 0.15, "length": 0.05, "pulse": 0, "position": [0.5, 0.5]},
        {"type": "text", "text": "X", "position": [0.5, 0.5], "start": 10},
    ])
    assert hit_layer(p, 0.0, 0.5, 0.5) == 1  # el círculo está encima de las barras; el texto aún no existe
    assert hit_layer(p, 12.0, 0.5, 0.5) == 2
    assert hit_layer(p, 0.0, 0.1, 0.5) == 0
    assert hit_layer(p, 0.0, 0.5, 0.02) is None
    p.layers[1].enabled = False
    assert hit_layer(p, 0.0, 0.5, 0.5) == 0


def test_fit_image_focus():
    img = np.zeros((100, 400, 4), np.uint8)
    img[..., 3] = 255
    img[:, :200, 0] = 255  # mitad izquierda roja
    left = fit_image(img, 100, 100, "cover", (0.0, 0.5))
    right = fit_image(img, 100, 100, "cover", (1.0, 0.5))
    assert left[50, 50, 0] == 255 and right[50, 50, 0] == 0
