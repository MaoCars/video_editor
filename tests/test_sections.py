"""Secciones (manuales y automáticas), paletas, intensidad, pertenencia y shake/zoom del fondo."""
import numpy as np
import pytest

from musicviz.config import ProjectConfig, SectionConfig
from musicviz.render.engine import Scene
from musicviz.render.sections import SectionTimeline, detect_sections, resolve_sections

W, H = 320, 180


def _project(audio_cfg, **extra):
    data = {"audio": audio_cfg.model_dump(), "output": {"width": W, "height": H, "fps": 30}}
    data.update(extra)
    return ProjectConfig.model_validate(data)


def _center(img):
    return img[H // 2 - 2 : H // 2 + 2, W // 2 - 2 : W // 2 + 2].reshape(-1, 3).mean(axis=0)


def test_detect_sections_covers_song(features):
    found = detect_sections(features, ProjectConfig.model_validate({"audio": {"file": "x"}}).auto_sections)
    assert found and found[0].start == 0.0
    assert found[-1].end == pytest.approx(features.duration, abs=0.05)
    for a, b in zip(found, found[1:]):
        assert a.end == pytest.approx(b.start)  # sin huecos ni solapes
    assert all(s.name in ("calm", "build", "drop") and s.palette for s in found)


def test_detect_sections_min_length(features):
    cfg = ProjectConfig.model_validate({"audio": {"file": "x"}, "auto_sections": {"min_length": 100}}).auto_sections
    assert len(detect_sections(features, cfg)) == 1  # todo se funde en una sola


def test_resolve_and_blend(audio_cfg, features):
    project = _project(
        audio_cfg,
        sections=[
            {"name": "a", "start": 0, "palette": ["#ff0000"], "intensity": 0.5, "background_colors": ["#000000", "#000000"]},
            {"name": "b", "start": 2, "palette": ["#0000ff"], "intensity": 1.5, "transition": 1.0, "effects": ["glitch"], "layers": ["circle", "titulo"]},
        ],
    )
    tl = SectionTimeline(resolve_sections(project, features))
    assert [s.end for s in tl.sections] == [2.0, pytest.approx(features.duration)]
    a = tl.state_at(1.0)
    assert a.name == "a" and a.intensity == 0.5 and np.allclose(a.palette[0, :3], [1, 0, 0])
    mid = tl.state_at(2.5)  # mitad de la transición (smoothstep(0.5) = 0.5)
    assert mid.name == "b" and mid.intensity == pytest.approx(1.0)
    assert mid.palette[0, 0] == pytest.approx(0.5, abs=0.02) and mid.palette[0, 2] == pytest.approx(0.5, abs=0.02)
    b = tl.state_at(4.0)
    assert b.intensity == 1.5 and np.allclose(b.palette[0, :3], [0, 0, 1])
    assert b.allows_effect(None, "glitch") and not b.allows_effect(None, "bloom")
    assert b.allows_layer("titulo", "text") and b.allows_layer(None, "circle") and not b.allows_layer(None, "bars")
    assert a.allows_effect(None, "bloom")  # sin lista = todo permitido
    assert not SectionTimeline([])  # sin secciones


def test_palette_colors_follow_sections(audio_cfg, features):
    layers = [{"type": "bars", "colors": ["palette"], "height": 0.45, "width": 0.3, "min_height": 0.4, "mirror": True, "gap": 0.0, "glow": 0}]
    sections = [
        {"name": "rojo", "start": 0, "palette": ["#ff0000"], "transition": 0},
        {"name": "azul", "start": 2, "palette": ["#0000ff"], "transition": 0},
    ]
    scene = Scene(_project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=layers, sections=sections), features, W, H)
    red = _center(scene.render(int(1.0 * 30)))
    blue = _center(scene.render(int(3.0 * 30)))
    assert red[0] > 200 and red[2] < 30
    assert blue[2] > 200 and blue[0] < 30
    # sin secciones, [palette] cae a blanco
    plain = Scene(_project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=layers), features, W, H)
    assert _center(plain.render(30)).min() > 200


def test_section_background_colors_and_intensity(audio_cfg, features):
    sections = [{"name": "s", "start": 0, "background_colors": ["#00ff00", "#00ff00"], "intensity": 0.0}]
    project = _project(audio_cfg, background={"type": "gradient", "colors": ["#ff0000", "#ff0000"]}, layers=[], sections=sections)
    img = Scene(project, features, W, H).render(10)
    assert _center(img)[1] > 200 and _center(img)[0] < 30
    # el fondo sólido también se recolorea con la sección
    solid = _project(audio_cfg, background={"type": "solid", "color": "#404040"}, layers=[], sections=sections)
    assert _center(Scene(solid, features, W, H).render(10))[1] > 200
    # intensidad 0 anula los efectos
    quiet = [{"name": "s", "start": 0, "intensity": 0.0}]
    project_fx = _project(audio_cfg, background={"type": "solid", "color": "#404040"}, layers=[], sections=quiet, effects=[{"type": "strobe", "trigger": "always", "intensity": 1.0}])
    assert _center(Scene(project_fx, features, W, H).render(10)).max() < 80


def test_membership_by_sections_field(audio_cfg, features):
    sections = [{"name": "calm", "start": 0, "transition": 0}, {"name": "drop", "start": 2, "transition": 0}]
    layers = [{"type": "bars", "colors": ["#ffffff"], "min_height": 0.4, "gap": 0.0, "sections": ["drop"], "glow": 0}]
    effects = [{"type": "strobe", "trigger": "always", "intensity": 1.0, "sections": ["calm"]}]
    scene = Scene(_project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=layers, sections=sections, effects=effects), features, W, H)
    calm = scene.render(int(1.0 * 30))
    drop = scene.render(int(3.0 * 30))
    assert calm.min() > 200  # strobe blanco activo en calm
    assert drop[2, 2].max() < 10 and _center(drop).max() > 200  # en drop: barras, sin strobe


def test_auto_sections_render(audio_cfg, features):
    from musicviz.presets import load_preset

    project = load_preset("auto_sections", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = W, H, 30
    scene = Scene(project, features, W, H)
    assert scene.timeline
    img = scene.render(60)
    assert img.shape == (H, W, 3) and img.mean() > 3


def test_background_pulse_and_shake(audio_cfg, features):
    kick_frame = int(np.argmax(features.kick))
    base = {"type": "gradient", "colors": ["#000000", "#ffffff"], "angle": 0}
    still = Scene(_project(audio_cfg, background=base, layers=[]), features, W, H).render(kick_frame)
    zoomed = Scene(_project(audio_cfg, background={**base, "pulse": 0.3, "pulse_trigger": "kick"}, layers=[]), features, W, H).render(kick_frame)
    shaken = Scene(_project(audio_cfg, background={**base, "shake": 40, "shake_trigger": "kick", "shake_rotation": 5}, layers=[]), features, W, H).render(kick_frame)
    fixed = Scene(_project(audio_cfg, background={**base, "zoom": 1.5}, layers=[]), features, W, H).render(kick_frame)
    assert np.abs(zoomed.astype(int) - still.astype(int)).mean() > 2
    assert np.abs(shaken.astype(int) - still.astype(int)).mean() > 2
    assert np.abs(fixed.astype(int) - still.astype(int)).mean() > 2
    # la vibración es determinista por frame
    again = Scene(_project(audio_cfg, background={**base, "shake": 40, "shake_trigger": "kick", "shake_rotation": 5}, layers=[]), features, W, H).render(kick_frame)
    assert np.array_equal(shaken, again)


def test_section_config_validation():
    with pytest.raises(Exception):
        SectionConfig(start=0, palette=["rojo"])
    s = SectionConfig(name="x", start=10)
    assert s.end is None and s.transition == 0.5


def test_sections_switch(audio_cfg, features):
    layers = [{"type": "bars", "colors": ["palette"], "height": 0.45, "width": 0.3, "min_height": 0.4, "mirror": True, "gap": 0.0, "glow": 0}]
    sections = [{"name": "rojo", "start": 0, "palette": ["#ff0000"], "transition": 0}]
    on = _project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=layers, sections=sections)
    off = _project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=layers, sections=sections, sections_enabled=False)
    assert _center(Scene(on, features, W, H).render(30))[0] > 200 and _center(Scene(on, features, W, H).render(30))[2] < 30
    assert _center(Scene(off, features, W, H).render(30)).min() > 200  # desactivadas: blanco, pero siguen guardadas
    assert off.sections and not resolve_sections(off, features)
