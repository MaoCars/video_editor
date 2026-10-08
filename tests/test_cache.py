import numpy as np
import pytest

from musicviz.audio import cache
from musicviz.config import ProjectConfig
from musicviz.render.engine import analyze_project


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSICVIZ_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MUSICVIZ_NO_CACHE", raising=False)


def test_cache_roundtrip(audio_cfg):
    project = ProjectConfig.model_validate({"audio": audio_cfg.model_dump(), "output": {"fps": 30}})
    first = analyze_project(project)
    files = list(cache.cache_dir().glob("*.npz"))
    assert len(files) == 1
    second = analyze_project(project)
    assert second.n_frames == first.n_frames and second.bpm == first.bpm
    assert np.allclose(second.spectrum, first.spectrum) and np.array_equal(second.beats, first.beats)
    assert np.abs(second.waveform - first.waveform).max() < 1e-3  # int16 en disco
    # cambiar un parámetro invalida la caché
    project.audio.bands = 16
    third = analyze_project(project)
    assert third.spectrum.shape[1] == 16 and len(list(cache.cache_dir().glob("*.npz"))) == 2
    assert cache.clear() == 2


def test_cache_disabled(audio_cfg, monkeypatch):
    monkeypatch.setenv("MUSICVIZ_NO_CACHE", "1")
    project = ProjectConfig.model_validate({"audio": audio_cfg.model_dump(), "output": {"fps": 30}})
    analyze_project(project)
    assert not list(cache.cache_dir().glob("*.npz")) if cache.cache_dir().is_dir() else True


def test_corrupt_cache_is_ignored(audio_cfg):
    project = ProjectConfig.model_validate({"audio": audio_cfg.model_dump(), "output": {"fps": 30}})
    analyze_project(project)
    for f in cache.cache_dir().glob("*.npz"):
        f.write_bytes(b"basura")
    assert analyze_project(project).n_frames > 0


def test_render_service_reuses_scene(audio_cfg, features):
    from musicviz.gui.renderer import RenderService

    project = ProjectConfig.model_validate({"audio": audio_cfg.model_dump(), "output": {"width": 160, "height": 90, "fps": 30}, "layers": [{"type": "bars"}]})
    svc = RenderService()
    seen = {}

    def job(s):
        s.features_for(project)
        a = s.scene_for(project, 1.0)
        b = s.scene_for(project, 1.0)
        seen["same"] = a is b
        project2 = project.model_copy(deep=True)
        project2.layers[0] = project2.layers[0].model_copy(update={"height": 0.1})
        c = s.scene_for(project2, 1.0)
        seen["rebuilt"] = c is not a
        seen["img"] = c.render(10).shape

    import time

    svc.submit(job)
    for _ in range(200):
        if "img" in seen:
            break
        time.sleep(0.05)
    svc.shutdown()
    assert seen == {"same": True, "rebuilt": True, "img": (90, 160, 3)}
