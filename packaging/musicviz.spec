# -*- mode: python ; coding: utf-8 -*-
"""Spec de PyInstaller: genera la carpeta dist/musicviz con musicviz-gui(.exe) y musicviz(.exe).

Uso:  pyinstaller packaging/musicviz.spec   (desde la raíz del repositorio)
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
block_cipher = None

datas = collect_data_files("musicviz", includes=["presets/*.yaml"])
datas += collect_data_files("moderngl")
datas += collect_data_files("glcontext")
hiddenimports = (
    collect_submodules("musicviz")
    + collect_submodules("glcontext")
    + ["moderngl", "glcontext", "PIL.ImageTk", "PIL._tkinter_finder", "soundfile", "yaml"]
)
excludes = ["tests", "pytest", "matplotlib", "IPython", "jupyter", "notebook", "pandas", "sklearn", "torch", "tensorflow", "scipy", "tkinter.test", "unittest", "pydoc_data", "lib2to3", "setuptools", "pkg_resources", "distutils"]

common = dict(
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    cipher=block_cipher,
    noarchive=False,
)

a_gui = Analysis([str(ROOT / "packaging" / "launcher_gui.py")], **common)
a_cli = Analysis([str(ROOT / "packaging" / "launcher_cli.py")], **common)
MERGE((a_gui, "musicviz-gui", "musicviz-gui"), (a_cli, "musicviz", "musicviz"))

pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)
exe_gui = EXE(
    pyz_gui, a_gui.scripts, [],
    exclude_binaries=True, name="musicviz-gui", debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, icon=str(ROOT / "packaging" / "icon.ico") if (ROOT / "packaging" / "icon.ico").exists() else None,
)
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)
exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True, name="musicviz", debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True,
)
coll = COLLECT(
    exe_gui, a_gui.binaries, a_gui.zipfiles, a_gui.datas,
    exe_cli, a_cli.binaries, a_cli.zipfiles, a_cli.datas,
    strip=False, upx=False, name="musicviz",
)
