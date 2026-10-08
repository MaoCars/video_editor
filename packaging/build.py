"""Construye la aplicación empaquetada con PyInstaller y, opcionalmente, incluye ffmpeg.

    python packaging/build.py            # dist/musicviz/ con los ejecutables
    python packaging/build.py --ffmpeg   # además descarga ffmpeg (Windows) y lo deja junto al ejecutable
    python packaging/build.py --zip      # y comprime dist/musicviz en dist/musicviz-<version>-<so>.zip
"""
from __future__ import annotations

import argparse
import io
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "musicviz"
FFMPEG_WIN_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"


def version() -> str:
    text = (ROOT / "musicviz" / "__init__.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip('"').strip("'")
    return "0.0.0"


def build() -> None:
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    shutil.rmtree(DIST, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(ROOT / "packaging" / "musicviz.spec")], cwd=ROOT, check=True)
    for extra in ("README.md", "examples"):
        src = ROOT / extra
        if src.is_dir():
            shutil.copytree(src, DIST / extra, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, DIST / extra)


def add_ffmpeg() -> None:
    bin_dir = DIST / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        print("Descargando ffmpeg para Windows...")
        data = urllib.request.urlopen(FFMPEG_WIN_URL, timeout=600).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                name = member.rsplit("/", 1)[-1]
                if name in ("ffmpeg.exe", "ffprobe.exe"):
                    (bin_dir / name).write_bytes(zf.read(member))
                    print("  incluido", name)
        (bin_dir / "LICENCIA-ffmpeg.txt").write_text("ffmpeg (build GPL de BtbN/FFmpeg-Builds): https://github.com/BtbN/FFmpeg-Builds — licencia GPL.\n", encoding="utf-8")
    else:
        for tool in ("ffmpeg", "ffprobe"):
            path = shutil.which(tool)
            if path:
                shutil.copy2(path, bin_dir / tool)
                print("  copiado", tool, "desde", path)


def make_zip() -> Path:
    tag = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system(), platform.system().lower())
    out = ROOT / "dist" / f"musicviz-{version()}-{tag}-x64"
    archive = shutil.make_archive(str(out), "zip", root_dir=DIST.parent, base_dir=DIST.name)
    print("ZIP:", archive)
    return Path(archive)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg", action="store_true", help="Incluir ffmpeg/ffprobe junto a la aplicación")
    ap.add_argument("--zip", action="store_true", help="Comprimir el resultado")
    args = ap.parse_args()
    build()
    if args.ffmpeg:
        add_ffmpeg()
    if args.zip:
        make_zip()
    print("Listo:", DIST)
