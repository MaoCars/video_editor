"""Análisis de audio: carga, espectro por frame, bandas, energía, beats, kicks y drops."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..config import AudioConfig
from ..utils.mathx import attack_release, decay_envelope, moving_average, normalize_percentile

DEFAULT_SR = 44100


# --------------------------------------------------------------------------- carga


def _ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def load_audio(path: str | Path, sr: int = DEFAULT_SR, start: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    """Carga cualquier formato soportado por ffmpeg como mono float32 en [-1, 1].

    Si ffmpeg no está disponible usa soundfile (wav/flac/ogg/mp3 según libsndfile).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de audio: {path}")
    ff = _ffmpeg_path()
    if ff:
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if start > 0:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", str(path)]
        if duration is not None:
            cmd += ["-t", f"{duration:.6f}"]
        cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            raise RuntimeError(f"ffmpeg no pudo decodificar {path.name}: {proc.stderr.decode(errors='ignore').strip()}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    import soundfile as sf  # fallback

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(sr), int(file_sr))
        mono = resample_poly(mono, sr // g, file_sr // g).astype(np.float32)
    s0 = int(start * sr)
    s1 = len(mono) if duration is None else min(len(mono), s0 + int(duration * sr))
    return np.ascontiguousarray(mono[s0:s1], dtype=np.float32)


def probe_duration(path: str | Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        return float(proc.stdout.decode().strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------- estructuras


@dataclass
class FrameFeatures:
    """Características de audio de un frame concreto (todo normalizado a [0, 1])."""

    index: int
    time: float
    spectrum: np.ndarray
    bass: float
    mid: float
    treble: float
    rms: float
    onset: float
    is_beat: bool
    beat_env: float
    kick: float
    drop: float
    progress: float
    # Estado de la sección activa (lo rellena el motor de render antes de dibujar)
    palette: Optional[np.ndarray] = None  # (n, 4) RGBA float
    intensity: float = 1.0
    section: str = ""

    def drive(self, trigger: str, threshold: float = 0.5) -> float:
        """Valor 0..1 que controla un efecto según el tipo de disparador."""
        if trigger == "always":
            return 1.0
        if trigger == "beat":
            return self.beat_env
        if trigger == "kick":
            return self.kick
        if trigger == "drop":
            return self.drop
        value = {"bass": self.bass, "energy": self.rms, "treble": self.treble}.get(trigger, 0.0)
        if threshold >= 1.0:
            return 0.0
        return float(np.clip((value - threshold) / (1.0 - threshold), 0.0, 1.0))


@dataclass
class AudioFeatures:
    sr: int
    fps: float
    n_frames: int
    duration: float
    bpm: float
    band_freqs: np.ndarray
    spectrum: np.ndarray  # (n_frames, n_bands)
    bass: np.ndarray
    mid: np.ndarray
    treble: np.ndarray
    rms: np.ndarray
    onset: np.ndarray
    beats: np.ndarray  # bool
    beat_env: np.ndarray
    kick: np.ndarray
    kick_onsets: np.ndarray  # bool
    drop: np.ndarray
    energy_integral: np.ndarray  # ∫ rms dt (segundos "musicales")
    waveform: np.ndarray = field(repr=False)  # mono float32

    @property
    def samples_per_frame(self) -> float:
        return self.sr / self.fps

    @property
    def beat_times(self) -> np.ndarray:
        return np.nonzero(self.beats)[0] / self.fps

    @property
    def kick_times(self) -> np.ndarray:
        return np.nonzero(self.kick_onsets)[0] / self.fps

    def frame(self, i: int) -> FrameFeatures:
        i = int(np.clip(i, 0, self.n_frames - 1))
        return FrameFeatures(
            index=i,
            time=i / self.fps,
            spectrum=self.spectrum[i],
            bass=float(self.bass[i]),
            mid=float(self.mid[i]),
            treble=float(self.treble[i]),
            rms=float(self.rms[i]),
            onset=float(self.onset[i]),
            is_beat=bool(self.beats[i]),
            beat_env=float(self.beat_env[i]),
            kick=float(self.kick[i]),
            drop=float(self.drop[i]),
            progress=i / max(self.n_frames - 1, 1),
        )

    def waveform_window(self, t: float, n_samples: int) -> np.ndarray:
        """Ventana de `n_samples` muestras centrada en el tiempo t (rellena con ceros en bordes)."""
        center = int(round(t * self.sr))
        start = center - n_samples // 2
        out = np.zeros(n_samples, np.float32)
        s0 = max(start, 0)
        s1 = min(start + n_samples, len(self.waveform))
        if s1 > s0:
            out[s0 - start : s1 - start] = self.waveform[s0:s1]
        return out


# --------------------------------------------------------------------------- análisis


def _band_matrix(n_bins: int, sr: int, fft_size: int, n_bands: int, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Matriz (n_bands, n_bins) que agrupa bins FFT en bandas log-espaciadas."""
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    fmax = min(fmax, sr / 2.0)
    fmin = max(fmin, freqs[1])
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    W = np.zeros((n_bands, n_bins), np.float32)
    for k in range(n_bands):
        lo, hi = edges[k], edges[k + 1]
        idx = np.nonzero((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            # Banda más estrecha que un bin: interpolación lineal entre los dos bins vecinos.
            j = int(np.searchsorted(freqs, centers[k]))
            j = int(np.clip(j, 1, n_bins - 1))
            f = (centers[k] - freqs[j - 1]) / max(freqs[j] - freqs[j - 1], 1e-9)
            W[k, j - 1] = 1.0 - f
            W[k, j] = f
        else:
            W[k, idx] = 1.0 / len(idx)
    return W, centers.astype(np.float32)


def _frame_matrix(signal: np.ndarray, n_frames: int, hop: float, fft_size: int) -> np.ndarray:
    """Devuelve (n_frames, fft_size) con ventanas centradas en i*hop."""
    half = fft_size // 2
    padded = np.pad(signal, (half, fft_size), mode="constant")
    starts = np.round(np.arange(n_frames) * hop).astype(np.int64)
    idx = starts[:, None] + np.arange(fft_size)[None, :]
    return padded[idx]


def _pick_peaks(strength: np.ndarray, fps: float, sensitivity: float, min_interval: float) -> np.ndarray:
    """Detección de picos: máximo local por encima de la media móvil + umbral adaptativo."""
    n = len(strength)
    beats = np.zeros(n, bool)
    if n < 3:
        return beats
    win = max(int(fps * 0.6), 3)
    local_mean = moving_average(strength, win)
    local_std = np.sqrt(np.maximum(moving_average(strength**2, win) - local_mean**2, 0.0))
    thresh = local_mean + (0.35 / max(sensitivity, 1e-3)) * (local_std + 0.08)
    min_gap = max(int(round(min_interval * fps)), 1)
    last = -min_gap
    for i in range(1, n - 1):
        s = strength[i]
        if s >= strength[i - 1] and s > strength[i + 1] and s > thresh[i] and i - last >= min_gap:
            beats[i] = True
            last = i
    return beats


def estimate_bpm(onset: np.ndarray, fps: float, bpm_min: float = 60.0, bpm_max: float = 200.0) -> float:
    """Tempo por autocorrelación de la envolvente de onsets."""
    x = onset - onset.mean()
    if len(x) < int(fps * 2) or not np.any(x):
        return 0.0
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= max(ac[0], 1e-9)
    lag_min = int(fps * 60.0 / bpm_max)
    lag_max = min(int(fps * 60.0 / bpm_min), n - 1)
    if lag_max <= lag_min:
        return 0.0
    seg = ac[lag_min:lag_max]
    # Preferencia suave por tempos cercanos a 120-150 BPM (típico en EDM) para evitar octavas erróneas.
    lags = np.arange(lag_min, lag_max)
    bpms = 60.0 * fps / lags
    weight = np.exp(-0.5 * ((np.log2(bpms / 135.0)) / 0.9) ** 2)
    best = int(np.argmax(seg * weight))
    return float(60.0 * fps / lags[best])


def analyze(cfg: AudioConfig, fps: float, waveform: Optional[np.ndarray] = None, sr: int = DEFAULT_SR) -> AudioFeatures:
    """Analiza el audio del proyecto y devuelve características por frame de video."""
    if waveform is None:
        waveform = load_audio(cfg.file, sr=sr, start=cfg.start, duration=cfg.duration)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if peak > 0:
        waveform = waveform / peak  # normaliza el pico a 1

    duration = len(waveform) / sr
    n_frames = max(int(np.ceil(duration * fps)), 1)
    hop = sr / fps
    fft_size = int(cfg.fft_size)
    frames = _frame_matrix(waveform, n_frames, hop, fft_size)
    window = np.hanning(fft_size).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)  # (n_frames, n_bins)
    spec *= 2.0 / fft_size
    n_bins = spec.shape[1]
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

    # ---- bandas log-espaciadas
    W, centers = _band_matrix(n_bins, sr, fft_size, cfg.bands, cfg.fmin, cfg.fmax)
    bands_lin = spec @ W.T  # (n_frames, n_bands)
    bands_db = 20.0 * np.log10(bands_lin + 1e-6)
    if cfg.normalize == "per_band":
        bands_norm = normalize_percentile(bands_db, 5.0, 99.5, axis=0)
    else:
        ceil_global = float(np.percentile(bands_db, 99.5))
        if cfg.normalize == "hybrid":
            band_ceil = np.percentile(bands_db, 99.0, axis=0)
            bands_db = bands_db + np.maximum(ceil_global - band_ceil, 0.0) * cfg.tilt
        floor = ceil_global - cfg.dynamic_range
        bands_norm = np.clip((bands_db - floor) / max(ceil_global - floor, 1e-6), 0.0, 1.0).astype(np.float32)
    bands_norm = np.clip(bands_norm * cfg.gain, 0.0, 1.0) ** float(cfg.gamma)
    if cfg.spatial_smoothing > 0 and cfg.bands > 2:
        bands_norm = gaussian_filter1d(bands_norm, cfg.spatial_smoothing, axis=1, mode="nearest").astype(np.float32)
    spectrum = attack_release(bands_norm, cfg.smoothing.attack, cfg.smoothing.release)

    # ---- energía por rangos
    def band_energy(lo: float, hi: float) -> np.ndarray:
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            return np.zeros(n_frames, np.float32)
        return np.sqrt(np.mean(spec[:, m] ** 2, axis=1)).astype(np.float32)

    bass_raw = band_energy(20.0, 160.0)
    mid_raw = band_energy(160.0, 2500.0)
    treble_raw = band_energy(2500.0, min(16000.0, sr / 2))
    bass = attack_release(normalize_percentile(bass_raw, 1.0, 99.0), 0.8, cfg.smoothing.release)
    mid = attack_release(normalize_percentile(mid_raw, 1.0, 99.0), 0.8, cfg.smoothing.release)
    treble = attack_release(normalize_percentile(treble_raw, 1.0, 99.0), 0.8, cfg.smoothing.release)

    rms_raw = np.sqrt(np.mean(frames**2, axis=1)).astype(np.float32)
    rms = attack_release(normalize_percentile(rms_raw, 1.0, 99.5), 0.8, cfg.smoothing.release)

    # ---- onsets (flujo espectral) y beats
    log_spec = np.log1p(spec * 200.0)
    flux = np.sum(np.maximum(np.diff(log_spec, axis=0, prepend=log_spec[:1]), 0.0), axis=1)
    onset = normalize_percentile(moving_average(flux, max(int(fps / 30), 1)), 5.0, 99.5)
    beats = _pick_peaks(onset, fps, cfg.beat_sensitivity, cfg.min_beat_interval)
    beat_env = decay_envelope(beats.astype(np.float32), fps, 0.18)

    # ---- kick: onsets sólo en graves (ideal para "golpes" de bass en dubstep/dnb)
    low = freqs < 150.0
    low_log = np.log1p(spec[:, low] * 400.0)
    kick_flux = np.sum(np.maximum(np.diff(low_log, axis=0, prepend=low_log[:1]), 0.0), axis=1)
    kick_strength = normalize_percentile(moving_average(kick_flux, max(int(fps / 30), 1)), 5.0, 99.5)
    kick_onsets = _pick_peaks(kick_strength, fps, cfg.beat_sensitivity, cfg.min_beat_interval)
    kick_amp = np.where(kick_onsets, np.clip(bass_raw / max(np.percentile(bass_raw, 99.0), 1e-6), 0.3, 1.0), 0.0)
    kick = decay_envelope(kick_amp.astype(np.float32), fps, 0.22)

    # ---- drop: energía a corto plazo muy por encima de la media a largo plazo
    short = moving_average(rms_raw, max(int(fps * 0.5), 1))
    long_ = moving_average(rms_raw, max(int(fps * 6.0), 1))
    drop = np.clip((short - long_) / max(np.percentile(rms_raw, 99.0), 1e-6) * 2.5, 0.0, 1.0).astype(np.float32)
    drop = attack_release(drop, 0.6, 0.05)

    energy_integral = np.cumsum(rms) / fps

    bpm = estimate_bpm(onset, fps)

    return AudioFeatures(
        sr=sr,
        fps=fps,
        n_frames=n_frames,
        duration=duration,
        bpm=bpm,
        band_freqs=centers,
        spectrum=spectrum.astype(np.float32),
        bass=bass,
        mid=mid,
        treble=treble,
        rms=rms,
        onset=onset,
        beats=beats,
        beat_env=beat_env,
        kick=kick,
        kick_onsets=kick_onsets,
        drop=drop,
        energy_integral=energy_integral.astype(np.float32),
        waveform=waveform,
    )
