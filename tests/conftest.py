import numpy as np
import pytest

from musicviz.audio.analysis import analyze
from musicviz.config import AudioConfig

SR = 22050


def synth_track(duration: float = 6.0, bpm: float = 120.0, sr: int = SR) -> np.ndarray:
    """Pista sintética: kicks en cada beat + tono constante + hi-hats."""
    t = np.arange(int(sr * duration)) / sr
    sig = 0.05 * np.sin(2 * np.pi * 440 * t)
    beat = 60.0 / bpm
    for b in np.arange(0.5, duration, beat):
        i = int(b * sr)
        n = int(0.2 * sr)
        tt = np.arange(n) / sr
        kick = np.sin(2 * np.pi * np.cumsum(np.linspace(140, 45, n)) / sr) * np.exp(-tt * 20)
        seg = sig[i : i + n]
        seg += kick[: len(seg)] * 0.8
    rng = np.random.default_rng(0)
    for b in np.arange(0.25, duration, beat / 2):
        i = int(b * sr)
        n = int(0.03 * sr)
        seg = sig[i : i + n]
        seg += rng.standard_normal(len(seg)) * np.exp(-np.arange(len(seg)) / sr * 100) * 0.1
    return np.tanh(sig).astype(np.float32)


@pytest.fixture(scope="session")
def waveform():
    return synth_track()


@pytest.fixture(scope="session")
def audio_cfg(tmp_path_factory, waveform):
    import soundfile as sf

    path = tmp_path_factory.mktemp("audio") / "track.wav"
    sf.write(str(path), waveform, SR)
    return AudioConfig(file=str(path), bands=32)


@pytest.fixture(scope="session")
def features(audio_cfg, waveform):
    return analyze(audio_cfg, fps=30.0, waveform=waveform, sr=SR)
