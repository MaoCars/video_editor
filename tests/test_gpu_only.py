"""El backend GPU, la consola y las utilidades funcionan sin OpenCV (el ejecutable va sin él).

Se ejecuta en un subproceso con `cv2` bloqueado, así que vale aunque OpenCV esté instalado en el entorno.
"""
import os
import shutil
import subprocess
import sys

import pytest

from musicviz.render.gpu import gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="sin contexto OpenGL disponible")

W, H = 320, 180


def _run(code: str, timeout: float = 240.0) -> subprocess.CompletedProcess:
    prelude = "import sys\nsys.modules['cv2'] = None  # simula un entorno sin OpenCV\n"
    env = dict(os.environ, MUSICVIZ_NO_CACHE="1")
    return subprocess.run([sys.executable, "-c", prelude + code], capture_output=True, text=True, timeout=timeout, env=env)


def test_gpu_render_without_opencv(audio_cfg, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg no disponible")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x36:r=30:d=2", "-pix_fmt", "yuv420p", str(clip)], check=True)
    photo = tmp_path / "p.png"
    code = f"""
import numpy as np
from PIL import Image
Image.fromarray(np.full((40, 60, 3), (255, 80, 0), np.uint8)).save({str(photo)!r})
from musicviz.config import ProjectConfig
from musicviz.render.engine import analyze_project, make_scene, render_frame_image, resolve_backend
from musicviz.utils.imaging import HAS_CV2, save_image
assert not HAS_CV2
layers = [
    {{"type": "bars", "glow": 0.3}}, {{"type": "circle", "glow": 0.4}}, {{"type": "particles", "count": 30}},
    {{"type": "image", "file": {str(photo)!r}, "shape": "circle", "size": 0.3, "border": 3, "shadow": 0.5, "position": [0.8, 0.3]}},
    {{"type": "image", "file": {str(photo)!r}, "shape": "rounded", "size": 0.2, "border": 2, "position": [0.2, 0.3]}},
    {{"type": "text", "text": "SIN CV2", "size": 0.12, "shadow": 0.6, "box_color": "#202020", "position": [0.5, 0.8]}},
    {{"type": "progress", "show_time": True}},
]
effects = [{{"type": "bloom"}}, {{"type": "glitch", "trigger": "always"}}, {{"type": "chromatic", "trigger": "always"}}]
p = ProjectConfig.model_validate({{"audio": {{"file": {str(audio_cfg.file)!r}}}, "output": {{"width": {W}, "height": {H}, "fps": 30}},
    "background": {{"type": "video", "video": {str(clip)!r}, "blur": 3, "darken": 0.2, "pulse": 0.1}}, "layers": layers, "effects": effects}})
assert resolve_backend("auto") == "gpu"
f = analyze_project(p)
sc = make_scene(p, f, {W}, {H}, "gpu")
frames = list(sc.render_many([0, 1, 2, 3, 40, 41]))
assert all(fr.shape == ({H}, {W}, 3) for fr in frames)
assert frames[0].mean() > 5 and not np.array_equal(frames[0], frames[4])  # el video avanza
sc.close()
# imagen de fondo + proyecto transparente
p2 = p.model_copy(update={{"background": p.background.model_copy(update={{"type": "image", "image": {str(photo)!r}, "blur": 2}}), "layers": [], "effects": []}})
img = render_frame_image(p2, f, 1.0)
assert img.shape == ({H}, {W}, 3) and img[..., 0].mean() > 200 and img[..., 2].mean() < 30, img.reshape(-1, 3).mean(axis=0)  # fondo naranja
p3 = p.model_copy(deep=True); p3.output.transparent = True
rgba = render_frame_image(p3, f, 1.0)
assert rgba.shape == ({H}, {W}, 4) and rgba[..., 3].min() == 0 and rgba[..., 3].max() > 200
save_image({str(tmp_path / 'snap.png')!r}, rgba)
assert Image.open({str(tmp_path / 'snap.png')!r}).mode == "RGBA"
# el backend CPU avisa claramente
try:
    resolve_backend("cpu")
except RuntimeError as exc:
    assert "OpenCV" in str(exc)
else:
    raise AssertionError("resolve_backend('cpu') debería fallar sin OpenCV")
print("GPU_ONLY_OK")
"""
    res = _run(code)
    assert res.returncode == 0 and "GPU_ONLY_OK" in res.stdout, res.stderr[-3000:]


def test_video_source_without_opencv(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg no disponible")
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x36:r=30:d=1", "-f", "lavfi", "-i", "color=green:s=64x36:r=30:d=1",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    code = f"""
from musicviz.layers.background import VideoSource, probe_video
info = probe_video({str(clip)!r})
assert info["width"] == 64 and info["height"] == 36 and abs(info["fps"] - 30) < 0.01 and abs(info["duration"] - 2.0) < 0.05, info
src = VideoSource({str(clip)!r})
assert src.n_frames == 60
red = src.frame_at(0.5, loop=True); green = src.frame_at(1.5, loop=True)
assert red.shape == (36, 64, 3) and red[0, 0, 0] > 150 and red[0, 0, 1] < 80, red[0, 0]
assert green[0, 0, 1] > 100 and green[0, 0, 0] < 80, green[0, 0]
assert src.frame_at(2.5, loop=True)[0, 0, 0] > 150          # bucle
assert src.frame_at(5.0, loop=False)[0, 0, 1] > 100         # sin bucle: se congela el último
seq = [src.frame_at(i / 10.0, loop=False) for i in range(20)]  # saltos cortos (3 frames) sin relanzar ffmpeg
assert all(f[0, 0, 0] > 150 for f in seq[:10]) and all(f[0, 0, 1] > 100 for f in seq[10:])
src.release()
print("VIDEO_OK")
"""
    res = _run(code, timeout=120)
    assert res.returncode == 0 and "VIDEO_OK" in res.stdout, res.stderr[-3000:]


def test_check_command_without_opencv():
    res = _run("from musicviz.cli import app\nfrom typer.testing import CliRunner\nr = CliRunner().invoke(app, ['check'])\nassert r.exit_code == 0, r.output\nprint(r.output)")
    assert res.returncode == 0, res.stderr[-2000:]
    assert "Backend CPU" in res.stdout and "no instalado" in res.stdout
