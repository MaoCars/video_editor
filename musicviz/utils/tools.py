"""Localización de herramientas externas (ffmpeg) junto a la app empaquetada o en el PATH."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def bundled_dirs() -> list[Path]:
    """Carpetas donde la app empaquetada (PyInstaller) o una instalación portátil pueden llevar ffmpeg."""
    dirs: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        dirs += [exe_dir, exe_dir / "bin", exe_dir / "ffmpeg"]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            dirs += [Path(meipass), Path(meipass) / "bin"]
    root = Path(__file__).resolve().parents[2]
    dirs += [root / "bin", root / "ffmpeg"]
    return dirs


def find_tool(name: str) -> Optional[str]:
    """Busca ffmpeg junto a la app (carpeta bin/) y, si no, en el PATH."""
    exe = f"{name}.exe" if os.name == "nt" else name
    for d in bundled_dirs():
        candidate = d / exe
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)
