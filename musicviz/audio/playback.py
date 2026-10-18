"""Audio de la vista previa (sounddevice + PortAudio): disponibilidad y mensajes claros cuando falla."""
from __future__ import annotations

from typing import Optional


def load_sounddevice():
    """Devuelve el módulo sounddevice, o lanza ImportError/OSError si no está o falta la librería PortAudio."""
    import sounddevice as sd  # noqa: PLC0415 - import diferido: carga una DLL/so al importarse

    return sd


def audio_status() -> tuple[bool, str]:
    """(hay audio, descripción) para `musicviz check` y el diálogo de entorno."""
    try:
        sd = load_sounddevice()
    except ImportError:
        return False, "sounddevice no instalado (pip install sounddevice)"
    except OSError as exc:  # sounddevice instalado pero sin la librería PortAudio
        return False, f"falta la librería de audio PortAudio: {exc}"
    try:
        dev = sd.query_devices(kind="output")
        return True, f"sounddevice {sd.__version__} · salida: {dev['name']}"
    except Exception as exc:  # noqa: BLE001 - p. ej. sin tarjeta de sonido o en un servidor
        return True, f"sounddevice {sd.__version__} · sin dispositivo de salida ({exc})"


def describe_error(exc: BaseException) -> str:
    """Mensaje para la barra de estado cuando no se puede reproducir el audio de la vista previa."""
    if isinstance(exc, ImportError):
        return "Reproduciendo sin audio: falta el módulo sounddevice (pip install sounddevice)"
    if isinstance(exc, OSError) and "portaudio" in str(exc).lower():
        return "Reproduciendo sin audio: falta la librería de audio PortAudio"
    return f"Reproduciendo sin audio: no se pudo abrir la salida de sonido ({exc})"


def play(waveform, sr: int, start_s: float = 0.0):
    """Empieza a reproducir `waveform` desde `start_s`. Devuelve el módulo sounddevice (para sd.stop())."""
    sd = load_sounddevice()
    sd.stop()
    sd.play(waveform[int(max(start_s, 0.0) * sr):], sr)
    return sd


def stop(sd: Optional[object] = None) -> None:
    try:
        (sd or load_sounddevice()).stop()
    except Exception:  # noqa: BLE001
        pass
