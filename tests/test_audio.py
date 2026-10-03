import numpy as np

from musicviz.audio.analysis import analyze, load_audio
from musicviz.config import AudioConfig
from tests.conftest import SR


def test_features_shapes(features):
    assert features.n_frames == int(np.ceil(features.duration * 30))
    assert features.spectrum.shape == (features.n_frames, 32)
    assert features.spectrum.min() >= 0.0 and features.spectrum.max() <= 1.0
    for name in ("bass", "mid", "treble", "rms", "onset", "beat_env", "kick", "drop"):
        arr = getattr(features, name)
        assert arr.shape == (features.n_frames,), name
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0), name


def test_beats_match_bpm(features):
    # 120 BPM en 6 s con kicks desde 0.5 s -> 11 kicks; la detección de graves debe acercarse.
    kicks = features.kick_times
    assert 9 <= len(kicks) <= 13, kicks
    expected = np.arange(0.5, 6.0, 0.5)
    # cada kick real debería tener un kick detectado a menos de 50 ms
    hits = sum(np.min(np.abs(kicks - e)) < 0.05 for e in expected)
    assert hits >= len(expected) - 2
    assert 110 <= features.bpm <= 130 or 55 <= features.bpm <= 65 or 230 <= features.bpm <= 250


def test_frame_accessor_and_drive(features):
    f = features.frame(10)
    assert f.index == 10 and abs(f.time - 10 / 30) < 1e-9
    assert f.drive("always") == 1.0
    assert 0.0 <= f.drive("bass", 0.5) <= 1.0
    assert f.drive("beat") == f.beat_env
    # índices fuera de rango se recortan
    assert features.frame(10_000).index == features.n_frames - 1


def test_waveform_window(features):
    win = features.waveform_window(1.0, 256)
    assert win.shape == (256,)
    edge = features.waveform_window(-1.0, 128)
    assert np.all(edge == 0)


def test_load_audio_roundtrip(audio_cfg, waveform):
    data = load_audio(audio_cfg.file, sr=SR)
    assert abs(len(data) - len(waveform)) < SR * 0.01
    assert np.corrcoef(data[: len(waveform)], waveform[: len(data)])[0, 1] > 0.99


def test_normalization_modes(audio_cfg, waveform):
    for mode in ("hybrid", "per_band", "global"):
        cfg = audio_cfg.model_copy(update={"normalize": mode})
        f = analyze(cfg, fps=30.0, waveform=waveform, sr=SR)
        assert f.spectrum.max() > 0.3
