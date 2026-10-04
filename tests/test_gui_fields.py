import pytest

from musicviz.config import AudioConfig, BarsLayer, CircleLayer, GlitchEffect, OutputConfig, ParticlesLayer, TextLayer, get_args_of_union, EffectConfig, LayerConfig
from musicviz.gui.fields import apply_value, field_specs, format_value, parse_value, replace_submodel


def _spec(cls, name):
    return next(s for s in field_specs(cls) if s.name == name)


def test_kinds_detected():
    assert _spec(BarsLayer, "rounded").kind == "bool"
    assert _spec(BarsLayer, "bands").kind == "int" and _spec(BarsLayer, "bands").optional
    assert _spec(BarsLayer, "height").kind == "float"
    assert _spec(BarsLayer, "style").kind == "choice" and "segments" in _spec(BarsLayer, "style").choices
    assert _spec(BarsLayer, "colors").kind == "colors"
    assert _spec(BarsLayer, "position").kind == "pair"
    assert _spec(BarsLayer, "type").readonly
    assert _spec(CircleLayer, "ring_color").kind == "color" and _spec(CircleLayer, "ring_color").optional
    assert _spec(TextLayer, "text").kind == "str"
    assert _spec(AudioConfig, "smoothing").kind == "model"
    assert _spec(OutputConfig, "path").kind == "file"
    assert "file" not in [s.name for s in field_specs(AudioConfig, exclude=("file",))]


def test_labels_spanish():
    assert _spec(BarsLayer, "opacity").label == "Opacidad"
    assert _spec(GlitchEffect, "rgb_split").label == "Separación RGB"


def test_parse_and_format_roundtrip():
    bars = BarsLayer(type="bars")
    spec = _spec(BarsLayer, "position")
    assert format_value((0.5, 0.25), spec) == "0.5, 0.25"
    assert parse_value("0.1, 0.9", spec) == (0.1, 0.9)
    with pytest.raises(ValueError):
        parse_value("0.1", spec)
    cspec = _spec(BarsLayer, "colors")
    assert parse_value("#fff, #000", cspec) == ["#fff", "#000"]
    assert format_value(["#fff", "#000"], cspec) == "#fff, #000"
    assert parse_value("", _spec(BarsLayer, "bands")) is None
    assert parse_value("12", _spec(BarsLayer, "bands")) == 12
    assert parse_value("True", _spec(BarsLayer, "mirror")) is True
    assert format_value(0.1 + 0.2, _spec(BarsLayer, "height")) == "0.3"
    assert bars.position == (0.5, 0.5)


def test_apply_value_validates():
    bars = BarsLayer(type="bars")
    new = apply_value(bars, _spec(BarsLayer, "gap"), "0.5")
    assert new.gap == 0.5 and bars.gap != 0.5  # inmutable: devuelve copia
    with pytest.raises(ValueError):
        apply_value(bars, _spec(BarsLayer, "gap"), "2")  # fuera de rango (le=0.95)
    with pytest.raises(ValueError):
        apply_value(bars, _spec(BarsLayer, "colors"), "rojo")
    with pytest.raises(ValueError):
        apply_value(bars, _spec(BarsLayer, "height"), "abc")
    assert apply_value(bars, _spec(BarsLayer, "style"), "dots").style == "dots"


def test_replace_submodel():
    audio = AudioConfig(file="a.wav")
    new = replace_submodel(audio, "smoothing", audio.smoothing.model_copy(update={"attack": 0.9}))
    assert new.smoothing.attack == 0.9


def test_union_members_have_type_literal():
    for union in (LayerConfig, EffectConfig):
        classes = get_args_of_union(union)
        assert len(classes) >= 7
        for cls in classes:
            assert "type" in cls.model_fields
    assert ParticlesLayer in get_args_of_union(LayerConfig)
