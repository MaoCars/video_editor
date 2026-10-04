from pathlib import Path

import pytest

from musicviz.gui import recent


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSICVIZ_CONFIG_DIR", str(tmp_path / "cfg"))


def test_add_load_remove_clear(tmp_path):
    assert recent.load_recent() == []
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    recent.add_recent(a)
    recent.add_recent(b)
    recent.add_recent(a)  # vuelve al principio sin duplicarse
    items = recent.load_recent()
    assert items == [a.resolve(), b.resolve()]
    recent.remove_recent(b)
    assert recent.load_recent() == [a.resolve()]
    recent.clear_recent()
    assert recent.load_recent() == []


def test_max_entries(tmp_path):
    for i in range(15):
        recent.add_recent(tmp_path / f"p{i}.yaml")
    items = recent.load_recent()
    assert len(items) == recent.MAX_RECENT
    assert items[0].name == "p14.yaml"


def test_corrupt_file_is_ignored():
    recent.recent_file().parent.mkdir(parents=True, exist_ok=True)
    recent.recent_file().write_text("{no es json", encoding="utf-8")
    assert recent.load_recent() == []
