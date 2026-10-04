"""Lista de proyectos recientes (archivo JSON en la carpeta de configuración del usuario)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MAX_RECENT = 10


def config_dir() -> Path:
    override = os.environ.get("MUSICVIZ_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "musicviz"


def recent_file() -> Path:
    return config_dir() / "recent.json"


def load_recent() -> list[Path]:
    try:
        data = json.loads(recent_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[Path] = []
    for item in data if isinstance(data, list) else []:
        p = Path(str(item))
        if p not in out:
            out.append(p)
    return out[:MAX_RECENT]


def _save(items: list[Path]) -> None:
    try:
        recent_file().parent.mkdir(parents=True, exist_ok=True)
        recent_file().write_text(json.dumps([str(p) for p in items[:MAX_RECENT]], ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def add_recent(path: str | Path) -> list[Path]:
    p = Path(path).resolve()
    items = [q for q in load_recent() if q != p]
    items.insert(0, p)
    _save(items)
    return items[:MAX_RECENT]


def remove_recent(path: str | Path) -> list[Path]:
    p = Path(path)
    items = [q for q in load_recent() if q != p and q != p.resolve()]
    _save(items)
    return items


def clear_recent() -> None:
    _save([])
