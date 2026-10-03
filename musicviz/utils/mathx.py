"""Pequeñas utilidades numéricas compartidas."""
from __future__ import annotations

import numpy as np


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 == edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def attack_release(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Suavizado asimétrico por frame (sube rápido con `attack`, baja lento con `release`).

    x: (n_frames, ...) valores. attack/release ∈ (0, 1]: fracción del salto que se
    recorre en cada frame. attack=1 sigue la señal exactamente al subir.
    """
    attack = float(np.clip(attack, 1e-4, 1.0))
    release = float(np.clip(release, 1e-4, 1.0))
    out = np.empty_like(x, dtype=np.float32)
    if len(x) == 0:
        return out
    prev = np.asarray(x[0], dtype=np.float32)
    out[0] = prev
    for i in range(1, len(x)):
        cur = x[i]
        coef = np.where(cur > prev, attack, release).astype(np.float32)
        prev = prev + coef * (cur - prev)
        out[i] = prev
    return out


def decay_envelope(triggers: np.ndarray, fps: float, tau: float) -> np.ndarray:
    """Envolvente que salta a `triggers[i]` (si es mayor) y decae exponencialmente con constante tau (s)."""
    k = float(np.exp(-1.0 / max(fps * tau, 1e-6)))
    out = np.zeros(len(triggers), np.float32)
    prev = 0.0
    for i in range(len(triggers)):
        prev = max(prev * k, float(triggers[i]))
        out[i] = prev
    return out


def normalize_percentile(x: np.ndarray, lo_p: float = 2.0, hi_p: float = 99.0, axis=None) -> np.ndarray:
    """Normaliza a [0,1] usando percentiles como piso y techo (robusto a picos)."""
    lo = np.percentile(x, lo_p, axis=axis, keepdims=axis is not None)
    hi = np.percentile(x, hi_p, axis=axis, keepdims=axis is not None)
    rng = np.maximum(hi - lo, 1e-6)
    return np.clip((x - lo) / rng, 0.0, 1.0).astype(np.float32)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    if window == 1:
        return x.astype(np.float32)
    kernel = np.ones(window, np.float32) / window
    pad = window // 2
    xp = np.pad(x.astype(np.float32), (pad, window - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)
