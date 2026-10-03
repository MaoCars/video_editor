import pytest
import yaml
from pydantic import ValidationError

from musicviz.config import BarsLayer, GlitchEffect, ProjectConfig
from musicviz.presets import load_preset, preset_names, preset_yaml


def test_presets_load_and_validate():
    names = preset_names()
    assert {"trap_nation", "monstercat", "ncs", "dnb_glitch", "minimal"} <= set(names)
    for name in names:
        project = load_preset(name, "song.mp3", "out.mp4")
        assert project.audio.file == "song.mp3"
        assert project.output.path == "out.mp4"
        # el YAML con comentarios también debe seguir siendo válido
        data = yaml.safe_load(preset_yaml(name, "song.mp3", "out.mp4"))
        ProjectConfig.model_validate(data)


def test_discriminated_unions():
    project = ProjectConfig.model_validate(
        {
            "audio": {"file": "a.wav"},
            "layers": [{"type": "bars", "colors": ["#fff"]}],
            "effects": [{"type": "glitch", "trigger": "kick"}],
        }
    )
    assert isinstance(project.layers[0], BarsLayer)
    assert isinstance(project.effects[0], GlitchEffect)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"audio": {"file": "a.wav"}, "layers": [{"type": "bars", "colour": "#fff"}]})


def test_bad_color_rejected():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"audio": {"file": "a.wav"}, "layers": [{"type": "bars", "colors": ["rojo"]}]})


def test_load_resolves_relative_paths(tmp_path):
    (tmp_path / "song.wav").write_bytes(b"")
    proj = tmp_path / "p.yaml"
    proj.write_text(yaml.safe_dump({"audio": {"file": "song.wav"}, "output": {"path": "out/v.mp4", "width": 1281}}), encoding="utf-8")
    cfg = ProjectConfig.load(proj)
    assert cfg.audio.file == str(tmp_path / "song.wav")
    assert cfg.output.path == str(tmp_path / "out" / "v.mp4")
    assert cfg.output.width == 1280  # se fuerza a par


def test_yaml_roundtrip():
    project = load_preset("trap_nation", "song.mp3")
    again = ProjectConfig.model_validate(yaml.safe_load(project.to_yaml()))
    assert again == project
