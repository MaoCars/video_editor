import shutil
import subprocess

import pytest

from musicviz.presets import load_preset
from musicviz.render.exporter import _scale_bitrate, choose_codec, codec_args, encoder_works, export_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no disponible")


def test_bitrate_scaling():
    assert _scale_bitrate("16M", 1.5) == "24000000"
    assert _scale_bitrate("320k", 2.0) == "640000"
    assert _scale_bitrate("raro", 2.0) == "raro"


def test_codec_args_contain_encoder():
    assert "h264_nvenc" in codec_args("h264_nvenc", "16M", None)
    assert "libx264" in codec_args("libx264", "16M", "fast")


def test_export_short_clip(audio_cfg, features, tmp_path):
    if not encoder_works("libx264") and not encoder_works("h264_nvenc"):
        pytest.skip("sin encoder H.264")
    project = load_preset("trap_nation", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps = 320, 180, 30
    out = tmp_path / "clip.mp4"
    calls = []
    result = export_video(project, features, output=out, start=1.0, duration=1.0, workers=1, progress=lambda d, t: calls.append((d, t)))
    assert out.exists() and out.stat().st_size > 1000
    assert result.frames == 30 and calls[-1] == (30, 30)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "csv=p=0", str(out)],
        stdout=subprocess.PIPE, check=True,
    ).stdout.decode()
    assert "video,320,180" in probe and "audio" in probe
    assert choose_codec("auto") in ("h264_nvenc", "libx264")
