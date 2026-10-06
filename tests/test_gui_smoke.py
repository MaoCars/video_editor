"""Prueba de humo de la interfaz gráfica. Se omite si no hay tkinter o no hay pantalla
(en Linux ejecuta `xvfb-run -a pytest`)."""
import time

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app(audio_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("MUSICVIZ_CONFIG_DIR", str(tmp_path / "cfg"))
    from musicviz.gui.app import App
    from musicviz.presets import load_preset

    project = load_preset("trap_nation", audio_cfg.file, str(tmp_path / "out.mp4"))
    project.output.width, project.output.height, project.output.fps = 320, 180, 30
    yaml_path = tmp_path / "proyecto.yaml"
    project.save(yaml_path)
    try:
        application = App(str(yaml_path))
    except tk.TclError as exc:
        pytest.skip(f"sin pantalla disponible: {exc}")
    yield application
    application.dirty = False
    application._stop_play()
    application.destroy()


def _pump(app, condition, timeout=90.0):
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        app.update()
        time.sleep(0.02)
    return condition()


def test_gui_loads_previews_and_edits(app, tmp_path):
    from musicviz.config import GlitchEffect
    from musicviz.gui.recent import load_recent

    assert app.project.output.path.endswith("out.mp4")  # la ruta de salida no se pierde al cargar
    assert _pump(app, lambda: app._photo is not None), "la vista previa no se generó"
    assert app._features is not None and app._features.n_frames > 0
    assert len(load_recent()) == 1

    # edición de listas
    n_eff = len(app.project.effects)
    app._add_item("effects", GlitchEffect)
    assert len(app.project.effects) == n_eff + 1 and app.project.effects[-1].type == "glitch"
    app._dup_item("effects")
    app._move_item("effects", -1)
    app._del_item("effects")
    assert len(app.project.effects) == n_eff + 1
    assert app.dirty

    # cambio de un campo mediante el formulario → la vista previa se vuelve a pedir
    app.nb.select(1)
    app.layers_list.selection_clear(0, "end")
    app.layers_list.selection_set(1)
    app._show_item_form("layers")
    app._photo = None
    app._set_item("layers", 1, app.project.layers[1].model_copy(update={"radius": 0.3}))
    assert app.project.layers[1].radius == 0.3
    assert _pump(app, lambda: app._photo is not None)

    # editor de colores: quitar y añadir muestras sin pasar por el selector
    from musicviz.gui.app import ColorList

    def find_colorlist(widget):
        if isinstance(widget, ColorList):
            return widget
        for child in widget.winfo_children():
            found = find_colorlist(child)
            if found is not None:
                return found
        return None

    app.update()
    cl = find_colorlist(app.layers_form.inner)
    assert cl is not None and len(cl.get()) >= 2
    n = len(cl.get())
    cl._remove(0)
    assert len(app.project.layers[1].colors) == n - 1
    cl._emit(cl.get() + ["#123456"])
    assert app.project.layers[1].colors[-1] == "#123456"
    cl._reverse()
    assert app.project.layers[1].colors[0] == "#123456"
    cl.var.set("palette")
    cl._from_text()
    assert app.project.layers[1].colors == ["palette"]

    # timeline: saltar, seleccionar, arrastrar bordes y mover
    from types import SimpleNamespace

    tl = app.timeline
    app.update()
    assert tl.project is app.project and tl.features is not None and tl._rows
    x_mid = tl._t2x(tl.duration / 2)
    tl._on_press(SimpleNamespace(x=x_mid, y=5))
    tl._on_release(SimpleNamespace(x=x_mid, y=5))
    assert abs(app.time_var.get() - tl.duration / 2) < 0.1
    row_y = 18 + 46 + 16 + 20 * 2 + 10  # tercera fila (texto)
    app.project.layers[2] = app.project.layers[2].model_copy(update={"start": 1.0, "end": 4.0})
    app._refresh_timeline()
    xa = tl._t2x(4.0)
    tl._on_press(SimpleNamespace(x=xa, y=row_y))  # borde final
    assert tl._drag and tl._drag["what"] == "edge_end" and tl.selected == ("layers", 2)
    tl._on_drag(SimpleNamespace(x=tl._t2x(5.0), y=row_y))
    tl._on_release(SimpleNamespace(x=tl._t2x(5.0), y=row_y))
    assert abs(app.project.layers[2].end - 5.0) < 0.1
    xm = tl._t2x(3.0)
    tl._on_press(SimpleNamespace(x=xm, y=row_y))  # mover
    tl._on_drag(SimpleNamespace(x=tl._t2x(3.5), y=row_y))
    tl._on_release(SimpleNamespace(x=tl._t2x(3.5), y=row_y))
    assert abs(app.project.layers[2].start - 1.5) < 0.1 and abs(app.project.layers[2].end - 5.5) < 0.1
    # keyframes: añadir sin diálogo (directo al modelo), sub-fila en la timeline, mover y borrar
    from musicviz.config import PointKey, ScalarKey

    app._set_keys(2, "opacity", [ScalarKey(time=1.0, value=1.0), ScalarKey(time=3.0, value=0.2)])
    app.update()
    key_rows = [r for r in tl._rows if r["kind"] == "keys"]
    assert len(key_rows) == 1 and key_rows[0]["prop"] == "opacity"
    ky = 18 + 46 + 16 + 20 * tl._rows.index(key_rows[0]) + 10
    kx = tl._t2x(3.0)
    hit = tl._hit(kx, ky)
    assert hit and hit["what"] == "key" and hit["k"] == 1
    tl._on_press(SimpleNamespace(x=kx, y=ky))
    tl._on_drag(SimpleNamespace(x=tl._t2x(4.0), y=ky))
    tl._on_release(SimpleNamespace(x=tl._t2x(4.0), y=ky))
    assert abs(app.project.layers[2].opacity_keys[1].time - 4.0) < 0.1
    tl._on_right(SimpleNamespace(x=tl._t2x(4.0), y=ky))
    assert len(app.project.layers[2].opacity_keys) == 1
    assert app.project.layers[2].opacity_keys[0].time == 1.0
    tl._on_wheel(SimpleNamespace(x=x_mid, y=row_y, delta=120, num=None))  # zoom
    assert tl.view1 - tl.view0 < tl.duration
    app.timeline_visible.set(False)
    app._toggle_timeline()
    app.timeline_visible.set(True)
    app._toggle_timeline()

    # vista previa interactiva: seleccionar, arrastrar el texto, rueda y flechas
    app.update()
    assert app._disp is not None
    ox, oy, dw, dh = app._disp
    txt = app.project.layers[2]
    assert txt.position == (0.5, 0.9) and not txt.position_keys
    x, y = ox + 0.5 * dw, oy + 0.9 * dh
    app._pv_press(SimpleNamespace(x=x, y=y, state=0))
    assert app._sel_layer == 2 and app._pdrag and app._pdrag["kind"] == "layer"
    app._pv_drag(SimpleNamespace(x=x, y=y - 0.4 * dh))
    app._pv_release(SimpleNamespace(x=x, y=y - 0.4 * dh))
    assert abs(app.project.layers[2].position[1] - 0.5) < 0.02 and abs(app.project.layers[2].position[0] - 0.5) < 0.01
    size0 = app.project.layers[2].size
    app._pv_wheel(SimpleNamespace(delta=120, num=None))
    assert app.project.layers[2].size > size0
    app._pv_arrow(SimpleNamespace(keysym="Right", state=1))
    assert app.project.layers[2].position[0] > 0.5
    # con keyframes de posición, arrastrar crea/actualiza el keyframe del instante actual
    app._set_keys(2, "position", [PointKey(time=0.0, value=(0.5, 0.5))])
    app.time_var.set(2.0)
    app._pv_press(SimpleNamespace(x=ox + 0.5 * dw, y=oy + 0.5 * dh, state=0))
    app._pv_drag(SimpleNamespace(x=ox + 0.8 * dw, y=oy + 0.5 * dh))
    app._pv_release(SimpleNamespace(x=ox + 0.8 * dw, y=oy + 0.5 * dh))
    keys = app.project.layers[2].position_keys
    assert len(keys) == 2 and abs(keys[1].time - 2.0) < 0.01 and abs(keys[1].value[0] - 0.8) < 0.02
    # clic en zona vacía selecciona el fondo; rueda = zoom del fondo
    app._pv_press(SimpleNamespace(x=ox + 0.03 * dw, y=oy + 0.03 * dh, state=0))
    assert app._sel_bg and app._sel_layer is None
    app._pv_wheel(SimpleNamespace(delta=120, num=None))
    assert app.project.background.zoom > 1.0

    # reproducción durante un instante
    app._start_play()
    _pump(app, lambda: app.time_var.get() > 0.3, timeout=20)
    app._stop_play()
    assert app.time_var.get() > 0.3

    # guardar y volver a cargar
    app.project_path = tmp_path / "guardado.yaml"
    assert app._save()
    assert (tmp_path / "guardado.yaml").exists()
    assert not app.dirty


def test_gui_render_and_cancel(app, tmp_path, monkeypatch):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg no disponible")
    from tkinter import messagebox

    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    assert _pump(app, lambda: app._features is not None)
    app.r_duration.set("3")
    app.r_scale.set("0.5")
    app._start_render()
    assert app._rendering
    assert _pump(app, lambda: app.progress["value"] >= 10, timeout=60)
    app._cancel_render()
    assert _pump(app, lambda: not app._rendering, timeout=60)
    assert "cancelado" in app.status_var.get().lower()
    assert not (tmp_path / "out.mp4").exists()

    # render corto completo
    app.r_duration.set("0.5")
    app.r_start.set("1")
    app._start_render()
    assert _pump(app, lambda: not app._rendering, timeout=120)
    assert (tmp_path / "out.mp4").exists()
