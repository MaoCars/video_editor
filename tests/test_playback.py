"""Audio de la vista previa: diagnóstico y `musicviz check --require audio`."""
import numpy as np
from typer.testing import CliRunner

from musicviz.audio import playback
from musicviz.cli import app as cli_app


def test_sounddevice_is_installed():
    """sounddevice es dependencia obligatoria: sin él la vista previa y la ventana GL no suenan."""
    sd = playback.load_sounddevice()
    assert hasattr(sd, "play") and hasattr(sd, "stop")


def test_audio_status_and_messages(monkeypatch):
    ok, txt = playback.audio_status()
    assert ok and txt.startswith("sounddevice")

    def missing():
        raise ImportError("No module named 'sounddevice'")

    monkeypatch.setattr(playback, "load_sounddevice", missing)
    assert playback.audio_status() == (False, "sounddevice no instalado (pip install sounddevice)")
    assert "falta el módulo" in playback.describe_error(ImportError("x"))
    assert "PortAudio" in playback.describe_error(OSError("PortAudio library not found"))
    assert "salida de sonido" in playback.describe_error(RuntimeError("Invalid device"))


def test_play_slices_from_start(monkeypatch):
    calls = []

    class Fake:
        def stop(self):
            calls.append("stop")

        def play(self, data, sr):
            calls.append((len(data), sr))

    monkeypatch.setattr(playback, "load_sounddevice", lambda: Fake())
    playback.play(np.zeros(1000, np.float32), 100, start_s=2.5)
    assert calls == ["stop", (750, 100)]


def test_check_require_audio(monkeypatch):
    runner = CliRunner()
    assert runner.invoke(cli_app, ["check", "--require", "audio"]).exit_code == 0

    def missing():
        raise ImportError("No module named 'sounddevice'")

    monkeypatch.setattr(playback, "load_sounddevice", missing)
    res = runner.invoke(cli_app, ["check", "--require", "audio"])
    assert res.exit_code == 1 and "Faltan componentes obligatorios" in res.output
