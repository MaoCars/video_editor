"""Presets incluidos (archivos YAML con `{audio}` como marcador del archivo de audio)."""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from ..config import ProjectConfig


def preset_names() -> list[str]:
    files = resources.files(__package__)
    return sorted(p.name[:-5] for p in files.iterdir() if p.name.endswith(".yaml"))


def preset_text(name: str) -> str:
    path = resources.files(__package__) / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"Preset desconocido: {name!r}. Disponibles: {', '.join(preset_names())}")
    return path.read_text(encoding="utf-8")


def preset_description(name: str) -> str:
    lines = [l[1:].strip() for l in preset_text(name).splitlines() if l.startswith("#")]
    return " ".join(lines)


def load_preset(name: str, audio: str, output: str | None = None) -> ProjectConfig:
    data = yaml.safe_load(preset_text(name).replace("{audio}", audio))
    if output:
        data.setdefault("output", {})["path"] = output
    return ProjectConfig.model_validate(data)


def preset_yaml(name: str, audio: str, output: str | None = None) -> str:
    """Texto YAML del preset con el audio sustituido (conserva los comentarios)."""
    text = preset_text(name).replace("{audio}", audio)
    if output:
        text = text.replace("output:\n", f"output:\n  path: {output}\n", 1)
    return text
