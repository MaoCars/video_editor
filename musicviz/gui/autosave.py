"""Autoguardado del proyecto abierto para recuperarlo si la aplicación se cierra sin guardar."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..config import ProjectConfig
from .recent import config_dir


def autosave_path() -> Path:
    return config_dir() / "autosave.yaml"


def meta_path() -> Path:
    return config_dir() / "autosave.json"


def write(project: ProjectConfig, original: Optional[Path]) -> None:
    """Guarda una copia del proyecto y de dónde venía (ruta original, si la tenía)."""
    path = autosave_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    project.save(tmp)
    tmp.replace(path)
    meta_path().write_text(json.dumps({"original": str(original) if original else None, "time": time.time()}), encoding="utf-8")


def pending() -> Optional[dict]:
    """Información del autoguardado pendiente ({"original", "time", "path"}) o None si no hay ninguno."""
    path = autosave_path()
    if not path.exists():
        return None
    info = {"original": None, "time": path.stat().st_mtime}
    try:
        info.update(json.loads(meta_path().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    info["path"] = path
    return info


def load() -> ProjectConfig:
    return ProjectConfig.load(autosave_path())


def clear() -> None:
    for p in (autosave_path(), meta_path(), autosave_path().with_suffix(".tmp")):
        try:
            p.unlink()
        except OSError:
            pass
