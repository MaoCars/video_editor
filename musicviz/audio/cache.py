"""Caché en disco del análisis de audio (evita recalcular 1-3 s por canción al abrir proyectos)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import AudioConfig
from .analysis import AudioFeatures

CACHE_VERSION = 1
MAX_ENTRIES = 24
_ARRAYS = ("band_freqs", "spectrum", "bass", "mid", "treble", "rms", "onset", "beats", "beat_env", "kick", "kick_onsets", "drop", "energy_integral")


def cache_dir() -> Path:
    override = os.environ.get("MUSICVIZ_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "musicviz" / "analysis"


def cache_key(cfg: AudioConfig, fps: float, sr: int) -> Optional[str]:
    try:
        st = os.stat(cfg.file)
    except OSError:
        return None
    payload = {"v": CACHE_VERSION, "file": os.path.abspath(cfg.file), "size": st.st_size, "mtime": int(st.st_mtime), "fps": fps, "sr": sr, "cfg": cfg.model_dump(mode="json", exclude={"file"})}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load(cfg: AudioConfig, fps: float, sr: int) -> Optional[AudioFeatures]:
    if os.environ.get("MUSICVIZ_NO_CACHE"):
        return None
    key = cache_key(cfg, fps, sr)
    if key is None:
        return None
    path = cache_dir() / f"{key}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            arrays = {name: data[name] for name in _ARRAYS}
            waveform = data["waveform"].astype(np.float32) / 32767.0
        path.touch()  # LRU por fecha de acceso
        return AudioFeatures(sr=meta["sr"], fps=meta["fps"], n_frames=meta["n_frames"], duration=meta["duration"], bpm=meta["bpm"], waveform=waveform, **arrays)
    except Exception:  # noqa: BLE001 - caché corrupta: se ignora y se recalcula
        return None


def store(cfg: AudioConfig, fps: float, sr: int, features: AudioFeatures) -> None:
    if os.environ.get("MUSICVIZ_NO_CACHE"):
        return
    key = cache_key(cfg, fps, sr)
    if key is None:
        return
    try:
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        meta = {"sr": features.sr, "fps": features.fps, "n_frames": features.n_frames, "duration": features.duration, "bpm": features.bpm}
        arrays = {name: getattr(features, name) for name in _ARRAYS}
        wave16 = np.clip(features.waveform * 32767.0, -32768, 32767).astype(np.int16)
        tmp = d / f"{key}.tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(meta)), waveform=wave16, **arrays)
        tmp.replace(d / f"{key}.npz")
        _prune(d)
    except Exception:  # noqa: BLE001 - la caché nunca debe romper el análisis
        pass


def _prune(d: Path) -> None:
    files = sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[MAX_ENTRIES:]:
        try:
            old.unlink()
        except OSError:
            pass


def clear() -> int:
    d = cache_dir()
    n = 0
    for f in d.glob("*.npz") if d.is_dir() else []:
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n
