"""Secciones de la canción: resolución de la lista (manual o automática) y estado por frame."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..audio.analysis import AudioFeatures
from ..config import DEFAULT_INTENSITIES, DEFAULT_PALETTES, AutoSectionsConfig, ProjectConfig, SectionConfig
from ..utils.color import gradient
from ..utils.mathx import moving_average, smoothstep


@dataclass
class ResolvedSection:
    name: str
    start: float
    end: float
    palette: np.ndarray  # (n, 4) float RGBA
    background: Optional[np.ndarray]  # (n, 4) o None
    intensity: float
    effects: Optional[set[str]]
    layers: Optional[set[str]]
    transition: float


@dataclass
class SectionState:
    name: str = ""
    palette: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    intensity: float = 1.0
    effects: Optional[set[str]] = None
    layers: Optional[set[str]] = None
    blend: float = 1.0  # 0..1 progreso de la transición (1 = sección plena)

    def allows_effect(self, name: Optional[str], type_: str) -> bool:
        return self.effects is None or name in self.effects or type_ in self.effects

    def allows_layer(self, name: Optional[str], type_: str) -> bool:
        return self.layers is None or name in self.layers or type_ in self.layers


def _stops(colors: list[str], n: int) -> np.ndarray:
    return gradient(colors, n)


def _blend_palettes(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    n = max(len(a), len(b), 2)
    if len(a) != n:
        a = _resample(a, n)
    if len(b) != n:
        b = _resample(b, n)
    return (a * (1.0 - w) + b * w).astype(np.float32)


def _resample(stops: np.ndarray, n: int) -> np.ndarray:
    xs = np.linspace(0, 1, len(stops))
    xn = np.linspace(0, 1, n)
    return np.stack([np.interp(xn, xs, stops[:, c]) for c in range(4)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- detección automática


def detect_sections(features: AudioFeatures, cfg: AutoSectionsConfig) -> list[SectionConfig]:
    """Clasifica la canción en tramos calm / build / drop según la energía suavizada."""
    fps = features.fps
    n = features.n_frames
    if n < int(fps * 2):
        return [SectionConfig(name="drop", start=0.0, end=features.duration)]
    loud = moving_average(features.rms, max(int(fps * 2.5), 1))
    hi = float(np.percentile(loud, 92))
    lo = float(np.percentile(loud, 25))
    span = max(hi - lo, 1e-6)
    norm = np.clip((loud - lo) / span, 0.0, 1.0)
    drop_boost = moving_average(features.drop, max(int(fps * 1.0), 1))
    score = np.clip(norm + 0.35 * drop_boost, 0.0, 1.0)
    labels = np.where(score > 0.62, 2, np.where(score > 0.3, 1, 0))  # 0 calm, 1 build, 2 drop

    # Segmentos consecutivos
    segs: list[list] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            segs.append([int(labels[start]), start, i])
            start = i
    # Fundir segmentos cortos con el vecino más parecido
    min_len = int(cfg.min_length * fps)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, (lab, s0, s1) in enumerate(segs):
            if s1 - s0 < min_len:
                if k > 0 and (k == len(segs) - 1 or abs(segs[k - 1][0] - lab) <= abs(segs[k + 1][0] - lab)):
                    segs[k - 1][2] = s1
                else:
                    segs[k + 1][1] = s0
                del segs[k]
                changed = True
                break
    # Unir vecinos con la misma etiqueta
    merged: list[list] = []
    for seg in segs:
        if merged and merged[-1][0] == seg[0]:
            merged[-1][2] = seg[2]
        else:
            merged.append(list(seg))
    names = {0: "calm", 1: "build", 2: "drop"}
    out: list[SectionConfig] = []
    for lab, s0, s1 in merged:
        name = names[lab]
        out.append(
            SectionConfig(
                name=name,
                start=round(s0 / fps, 2),
                end=round(min(s1 / fps, features.duration), 2),
                palette=list(cfg.palettes.get(name, DEFAULT_PALETTES[name])),
                background_colors=list(cfg.background_colors[name]) if name in cfg.background_colors else None,
                intensity=float(cfg.intensities.get(name, DEFAULT_INTENSITIES[name])),
                transition=cfg.transition,
            )
        )
    return out


# --------------------------------------------------------------------------- resolución


def resolve_sections(project: ProjectConfig, features: AudioFeatures) -> list[ResolvedSection]:
    if not project.sections_enabled:
        return []
    raw = detect_sections(features, project.auto_sections) if project.sections == "auto" else list(project.sections)
    raw = sorted(raw, key=lambda s: s.start)
    out: list[ResolvedSection] = []
    prev_palette: Optional[np.ndarray] = None
    for i, sec in enumerate(raw):
        end = sec.end
        if end is None:
            end = raw[i + 1].start if i + 1 < len(raw) else features.duration
        end = max(end, sec.start)
        if sec.palette:
            palette = _stops(sec.palette, max(len(sec.palette), 2))
        elif sec.name in project.auto_sections.palettes:
            palette = _stops(project.auto_sections.palettes[sec.name], 2)
        elif prev_palette is not None:
            palette = prev_palette
        else:
            palette = _stops(["#ffffff"], 2)
        prev_palette = palette
        bg = _stops(sec.background_colors, max(len(sec.background_colors), 2)) if sec.background_colors else None
        out.append(
            ResolvedSection(
                name=sec.name or f"seccion_{i + 1}",
                start=sec.start,
                end=end,
                palette=palette,
                background=bg,
                intensity=sec.intensity,
                effects=set(sec.effects) if sec.effects is not None else None,
                layers=set(sec.layers) if sec.layers is not None else None,
                transition=sec.transition,
            )
        )
    return out


class SectionTimeline:
    def __init__(self, sections: list[ResolvedSection]):
        self.sections = sections

    def __bool__(self) -> bool:
        return bool(self.sections)

    def index_at(self, t: float) -> int:
        idx = -1
        for i, s in enumerate(self.sections):
            if t >= s.start:
                idx = i
        return idx

    def state_at(self, t: float) -> SectionState:
        if not self.sections:
            return SectionState()
        i = self.index_at(t)
        if i < 0:
            first = self.sections[0]
            return SectionState(first.name, first.palette, first.background, first.intensity, first.effects, first.layers, 1.0)
        cur = self.sections[i]
        if t > cur.end and i == len(self.sections) - 1:
            pass  # después de la última: se mantiene
        state = SectionState(cur.name, cur.palette, cur.background, cur.intensity, cur.effects, cur.layers, 1.0)
        if i > 0 and cur.transition > 0 and t - cur.start < cur.transition:
            prev = self.sections[i - 1]
            w = smoothstep(0.0, 1.0, (t - cur.start) / cur.transition)
            state.blend = w
            state.palette = _blend_palettes(prev.palette, cur.palette, w)
            if cur.background is not None and prev.background is not None:
                state.background = _blend_palettes(prev.background, cur.background, w)
            state.intensity = prev.intensity * (1.0 - w) + cur.intensity * w
        return state
