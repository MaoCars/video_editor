"""Imagen (recorte, formas, anclaje), texto (alineación, ajuste, fuentes) y salida transparente."""
import shutil
import subprocess

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="estas pruebas usan el backend CPU (OpenCV)")

from musicviz.config import ImageLayer, ProjectConfig, TextLayer
from musicviz.layers.image import fit_into_box
from musicviz.layers.text import available_fonts, find_font, render_text_rgba, resolve_font_path, wrap_text
from musicviz.presets import load_preset
from musicviz.render.canvas import anchor_center, over_checkerboard, rounded_rect_mask
from musicviz.render.engine import Scene

W, H = 320, 180


def _project(audio_cfg, **extra):
    data = {"audio": audio_cfg.model_dump(), "output": {"width": W, "height": H, "fps": 30}}
    data.update(extra)
    return ProjectConfig.model_validate(data)


@pytest.fixture
def photo(tmp_path):
    img = np.zeros((120, 200, 3), np.uint8)
    img[:, :100] = (255, 0, 0)  # izquierda azul (BGR), derecha verde
    img[:, 100:] = (0, 255, 0)
    path = tmp_path / "foto.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


# ---------------------------------------------------------------- helpers


def test_anchor_center():
    assert anchor_center(100, 100, 40, 20, "center") == (100, 100)
    assert anchor_center(100, 100, 40, 20, "left") == (120, 100)
    assert anchor_center(100, 100, 40, 20, "right") == (80, 100)
    assert anchor_center(100, 100, 40, 20, "top_left") == (120, 110)
    assert anchor_center(100, 100, 40, 20, "bottom_right") == (80, 90)


def test_masks():
    sq = rounded_rect_mask(40, 40, 0, "square")
    assert sq.min() == 255
    circ = rounded_rect_mask(40, 40, 0, "circle")
    assert circ[20, 20] == 255 and circ[0, 0] == 0
    rnd = rounded_rect_mask(40, 40, 12, "rounded")
    assert rnd[20, 20] == 255 and rnd[0, 0] == 0 and rnd[0, 20] == 255


def test_fit_into_box_cover_and_contain():
    img = np.zeros((100, 200, 4), np.uint8)
    img[..., 3] = 255
    img[:, :100, 0] = 255
    cover_left = fit_into_box(img, 50, 50, "cover", (0.0, 0.5))
    cover_right = fit_into_box(img, 50, 50, "cover", (1.0, 0.5))
    assert cover_left.shape == (50, 50, 4) and cover_left[25, 25, 0] == 255
    assert cover_right[25, 25, 0] == 0
    contain = fit_into_box(img, 50, 50, "contain", (0.5, 0.5))
    assert contain.shape == (50, 50, 4)
    assert contain[2, 25, 3] == 0 and contain[25, 25, 3] == 255  # bandas transparentes arriba/abajo


# ---------------------------------------------------------------- imagen


@pytest.mark.parametrize("shape", ["original", "square", "circle", "rounded"])
def test_image_shapes_render(audio_cfg, features, photo, shape):
    layer = {"type": "image", "file": photo, "shape": shape, "size": 0.5, "border": 3, "shadow": 0.6}
    project = _project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=[layer])
    img = Scene(project, features, W, H).render(10)
    assert img.mean() > 5
    if shape in ("square", "circle"):
        # caja cuadrada: la zona lejos del centro en horizontal queda negra
        assert img[H // 2, 10].max() < 30


def test_image_anchor_positions(audio_cfg, features, photo):
    def render(anchor, pos):
        layer = {"type": "image", "file": photo, "shape": "square", "size": 0.4, "anchor": anchor, "position": pos, "pulse": 0}
        project = _project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=[layer])
        return Scene(project, features, W, H).render(0)

    left = render("left", (0.0, 0.5))
    right = render("right", (1.0, 0.5))
    assert left[H // 2, 5].max() > 100 and left[H // 2, W - 5].max() < 30
    assert right[H // 2, W - 5].max() > 100 and right[H // 2, 5].max() < 30
    top_left = render("top_left", (0.0, 0.0))
    assert top_left[3, 3].max() > 100 and top_left[H - 3, W - 3].max() < 30


def test_image_legacy_fields_still_load(photo):
    layer = ImageLayer.model_validate({"type": "image", "file": photo, "scale": 0.2, "circle_mask": True})
    assert layer.size == 0.2 and layer.shape == "circle"


# ---------------------------------------------------------------- texto


def test_fonts_listing_and_resolution():
    fonts = available_fonts()
    assert isinstance(fonts, dict)
    if fonts:
        name = next(iter(fonts))
        assert resolve_font_path(name) == fonts[name]
        assert resolve_font_path(name.upper()) == fonts[name]
    assert resolve_font_path("fuente-que-no-existe-xyz") is None
    assert find_font("fuente-que-no-existe-xyz", 20) is not None  # cae a la fuente por defecto


def test_wrap_and_align():
    font = find_font(None, 24)
    lines = wrap_text("uno dos tres cuatro cinco seis", font, 120)
    assert len(lines) > 1 and " ".join(lines) == "uno dos tres cuatro cinco seis"
    assert wrap_text("a\\nb", font, None) == ["a", "b"]  # salto manual escrito como \\n
    left = render_text_rgba("ab\nabcdef", font, "#ffffff", align="left")
    right = render_text_rgba("ab\nabcdef", font, "#ffffff", align="right")
    assert left.shape == right.shape
    h = left.shape[0] // 2
    # primera línea corta: en 'left' hay tinta a la izquierda, en 'right' a la derecha
    assert left[:h, : left.shape[1] // 3, 3].sum() > right[:h, : right.shape[1] // 3, 3].sum()
    stroked = render_text_rgba("A", font, "#ffffff", stroke_width=3, stroke_color="#ff0000")
    assert (stroked[..., 0] > 200).sum() > (stroked[..., 1] > 200).sum()  # el contorno rojo añade píxeles


def test_text_layer_features(audio_cfg, features):
    layers = [
        {"type": "text", "text": "Título largo que se parte en varias líneas", "align": "left", "valign": "top", "position": [0.05, 0.05], "max_width": 0.4, "size": 0.08, "stroke_width": 2, "shadow": 0.5, "box_color": "#00000080"},
        {"type": "text", "text": "Derecha", "align": "right", "valign": "bottom", "position": [0.98, 0.98], "size": 0.08},
    ]
    project = _project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=layers)
    img = Scene(project, features, W, H).render(5)
    assert img[: H // 2, : W // 2].max() > 100  # texto arriba-izquierda
    assert img[int(H * 0.85) :, int(W * 0.7) :].max() > 100  # texto abajo-derecha
    assert img[int(H * 0.4) : int(H * 0.6), int(W * 0.6) :].max() < 30  # centro-derecha vacío


# ---------------------------------------------------------------- transparencia


def test_transparent_render_has_alpha(audio_cfg, features):
    data = load_preset("spectrum_only", audio_cfg.file).model_dump()
    data["output"].update(width=W, height=H, fps=30)
    data["effects"] = [{"type": "bloom"}, {"type": "shake", "trigger": "always"}, {"type": "color", "saturation": 1.2}, {"type": "glitch", "trigger": "always", "invert": 1.0}, {"type": "chromatic"}, {"type": "grain"}, {"type": "pixelate", "trigger": "always", "size": 2}]
    project = ProjectConfig.model_validate(data)
    scene = Scene(project, features, W, H)
    img = scene.render(40)
    assert img.shape == (H, W, 4)
    assert img[2, 2, 3] == 0  # esquina transparente
    assert img[..., 3].max() > 200  # el espectro es opaco
    assert img[..., 3].mean() < 128
    board = over_checkerboard(img)
    assert board.shape == (H, W, 3)
    # el strobe sobre fondo transparente añade un velo con el alfa del flash
    data["effects"] = [{"type": "strobe", "trigger": "always", "intensity": 0.2}]
    flash = Scene(ProjectConfig.model_validate(data), features, W, H).render(40)
    assert 40 <= flash[2, 2, 3] <= 60


def test_transparent_ignores_background(audio_cfg, features):
    project = _project(audio_cfg, output={"width": W, "height": H, "fps": 30, "transparent": True}, background={"type": "solid", "color": "#ff0000"}, layers=[])
    img = Scene(project, features, W, H).render(0)
    assert img[..., 3].max() == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no disponible")
@pytest.mark.parametrize("codec,suffix,pix", [("prores_4444", ".mov", "yuva444p"), ("vp9_alpha", ".webm", "alpha_mode=1"), ("qtrle", ".mov", "argb")])
def test_export_alpha_codecs(audio_cfg, features, tmp_path, codec, suffix, pix):
    from musicviz.render.exporter import encoder_works, export_video

    enc = {"prores_4444": "prores_ks", "vp9_alpha": "libvpx-vp9", "qtrle": "qtrle"}[codec]
    if not encoder_works(enc):
        pytest.skip(f"sin encoder {enc}")
    project = load_preset("spectrum_only", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps, project.output.codec = 160, 90, 30, codec
    result = export_video(project, features, output=tmp_path / "alpha.mp4", duration=0.5, workers=1)
    assert result.path.suffix == suffix and result.path.exists()
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt:stream_tags=alpha_mode", str(result.path)], stdout=subprocess.PIPE, check=True).stdout.decode()
    assert pix in probe, probe


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no disponible")
def test_export_png_sequence(audio_cfg, features, tmp_path):
    from musicviz.render.exporter import export_video

    project = load_preset("spectrum_only", audio_cfg.file)
    project.output.width, project.output.height, project.output.fps, project.output.codec = 160, 90, 30, "png_sequence"
    result = export_video(project, features, output=tmp_path / "frames.mp4", duration=0.2, workers=1)
    pngs = sorted(result.path.glob("*.png"))
    assert len(pngs) == 6
    first = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
    assert first.shape == (90, 160, 4)


def test_codec_validation():
    from musicviz.render.exporter import ExportError, choose_codec, output_path_for
    from pathlib import Path

    with pytest.raises(ExportError):
        choose_codec("libx264", transparent=True)
    with pytest.raises(ExportError):
        choose_codec("prores_4444", transparent=False)
    assert output_path_for(Path("a.mp4"), "prores_4444") == Path("a.mov")
    assert output_path_for(Path("a.mp4"), "vp9_alpha") == Path("a.webm")
    assert output_path_for(Path("a.mp4"), "png_sequence") == Path("a")
    assert output_path_for(Path("a.mp4"), "libx264") == Path("a.mp4")


def test_gui_field_kinds_for_new_fields():
    from musicviz.gui.fields import field_specs

    kinds = {s.name: s.kind for s in field_specs(TextLayer)}
    assert kinds["font"] == "font" and kinds["stroke_color"] == "color" and kinds["box_color"] == "color"
    kinds = {s.name: s.kind for s in field_specs(ImageLayer)}
    assert kinds["anchor"] == "choice" and kinds["focus"] == "pair" and kinds["border_color"] == "color"


# ---------------------------------------------------------------- animaciones


def test_easing_bounds():
    from musicviz.layers.animation import ease

    for kind in ("linear", "ease_in", "ease_out", "ease_in_out", "back", "bounce"):
        assert ease(kind, 0.0) == pytest.approx(0.0, abs=1e-6)
        assert ease(kind, 1.0) == pytest.approx(1.0, abs=1e-6)
        assert ease(kind, -1) == pytest.approx(0.0, abs=1e-6) and ease(kind, 2) == pytest.approx(1.0, abs=1e-6)


def test_anim_state_windows_and_motion():
    from musicviz.layers.animation import anim_state

    cfg = TextLayer(type="text", text="x", start=2.0, end=6.0, animate_in="slide_left", in_duration=1.0, animate_out="zoom_out", out_duration=1.0, easing="linear", slide_distance=0.1)
    assert not anim_state(cfg, 1.0, 10.0, 1000).visible
    assert not anim_state(cfg, 7.0, 10.0, 1000).visible
    mid = anim_state(cfg, 4.0, 10.0, 1000)
    assert mid.visible and mid.is_identity
    half_in = anim_state(cfg, 2.5, 10.0, 1000)
    assert half_in.alpha == pytest.approx(0.5) and half_in.dx == pytest.approx(50.0) and half_in.dy == 0
    half_out = anim_state(cfg, 5.5, 10.0, 1000)
    assert half_out.alpha == pytest.approx(0.5) and half_out.scale == pytest.approx(1.3)
    # sin start/end: entra al principio y sale al final de la canción
    cfg2 = TextLayer(type="text", text="x", animate_in="fade", in_duration=1.0, animate_out="fade", out_duration=1.0, easing="linear")
    assert anim_state(cfg2, 0.25, 10.0, 1000).alpha == pytest.approx(0.25)
    assert anim_state(cfg2, 9.75, 10.0, 1000).alpha == pytest.approx(0.25)
    assert anim_state(cfg2, 5.0, 10.0, 1000).is_identity


@pytest.mark.parametrize("anim", ["fade", "slide_left", "slide_up", "zoom_in", "pop", "blur"])
def test_layers_animate_in_render(audio_cfg, features, photo, anim):
    layers = [
        {"type": "text", "text": "HOLA", "size": 0.3, "start": 1.0, "animate_in": anim, "in_duration": 1.0, "end": 4.0, "animate_out": "fade", "out_duration": 0.5},
        {"type": "image", "file": photo, "size": 0.3, "position": [0.95, 0.5], "anchor": "right", "start": 1.0, "animate_in": anim, "in_duration": 1.0, "pulse": 0},
        {"type": "bars", "start": 1.0, "animate_in": anim, "in_duration": 1.0, "end": 4.0, "colors": ["#ffffff"]},
    ]
    project = _project(audio_cfg, background={"type": "solid", "color": "#000000"}, layers=layers)
    scene = Scene(project, features, W, H)
    before = scene.render(int(0.5 * 30))
    assert before.max() < 5  # nada visible antes de start
    early = scene.render(int(1.1 * 30))
    full = scene.render(int(2.5 * 30))
    assert full.sum() > early.sum() > 0  # va apareciendo
    gone = scene.render(int(4.5 * 30))
    assert gone[:, : int(W * 0.55)].max() < 5  # el texto ya salió (la imagen sigue a la derecha)


# ---------------------------------------------------------------- video de fondo


@pytest.fixture
def clip(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg no disponible")
    path = tmp_path / "clip.mp4"
    # 2 s a 30 fps: rojo el primer segundo, verde el segundo
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x36:r=30:d=1", "-f", "lavfi", "-i", "color=green:s=64x36:r=30:d=1",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return str(path)


def test_video_source_sequential_and_seek(clip):
    from musicviz.layers.background import VideoSource

    src = VideoSource(clip)
    assert src.n_frames == 60 and src.fps == pytest.approx(30.0) and src.duration == pytest.approx(2.0)
    red = src.frame_at(0.5, loop=True)
    green = src.frame_at(1.5, loop=True)  # salto -> búsqueda
    assert red[0, 0, 0] > 150 and red[0, 0, 1] < 80
    assert green[0, 0, 1] > 100 and green[0, 0, 0] < 80
    assert src.frame_at(2.5, loop=True)[0, 0, 0] > 150  # bucle: vuelve al rojo
    assert src.frame_at(5.0, loop=False)[0, 0, 1] > 100  # sin bucle: se congela el último (verde)
    src.release()


def test_video_background_render(audio_cfg, features, clip):
    project = _project(audio_cfg, background={"type": "video", "video": clip, "video_loop": True, "darken": 0.0, "pulse": 0.02}, layers=[])
    scene = Scene(project, features, W, H)
    f_red = scene.render(int(0.5 * 30))
    f_green = scene.render(int(1.5 * 30))
    f_loop = scene.render(int(2.5 * 30))
    assert f_red[H // 2, W // 2, 0] > 150 and f_green[H // 2, W // 2, 1] > 100 and f_loop[H // 2, W // 2, 0] > 150
    # velocidad x2: a 0.75 s del proyecto ya va por el segundo verde del video
    project2 = _project(audio_cfg, background={"type": "video", "video": clip, "video_speed": 2.0}, layers=[])
    assert Scene(project2, features, W, H).render(int(0.75 * 30))[H // 2, W // 2, 1] > 100
    # frames en paralelo coinciden con los secuenciales
    from musicviz.render.engine import iter_frames

    idx = list(range(0, 24))
    assert list(iter_frames(project, features, W, H, idx, workers=1)) == list(iter_frames(project, features, W, H, idx, workers=2))


def test_video_source_short_gaps_read_through(clip):
    from musicviz.layers.background import VideoSource

    src = VideoSource(clip)
    # proyecto a 10 fps sobre un video a 30: saltos de 3 frames, se leen de seguido sin buscar
    seen = [src.frame_at(i / 10.0, loop=False) for i in range(20)]
    assert all(f[0, 0, 0] > 150 for f in seen[:10]) and all(f[0, 0, 1] > 100 for f in seen[10:])
    assert src._last_index == 57
    src.release()


def test_lookahead_matches_direct_calls():
    from musicviz.utils.lookahead import Lookahead

    calls = []

    def fn(k):
        calls.append(k)
        return k * 2

    la = Lookahead(fn)
    assert la.get(0) == 0
    la.schedule(1)
    assert la.get(1) == 2  # acierto: calculado en el hilo auxiliar
    la.schedule(2)
    assert la.get(5) == 10  # fallo: se descarta el 2 y se calcula el 5 aquí
    la.schedule(6)
    la.schedule(6)  # repetido: no se lanza dos veces
    assert la.get(6) == 12
    la.close()
    assert la.hits == 2 and la.misses == 2 and sorted(calls) == [0, 1, 2, 5, 6]


def test_video_prefetch_matches_and_releases(audio_cfg, features, clip):
    project = _project(audio_cfg, background={"type": "video", "video": clip, "video_loop": True, "blur": 2, "darken": 0.2}, layers=[])
    scene = Scene(project, features, W, H)
    direct = [Scene(project, features, W, H).render(i) for i in (0, 1, 2, 3, 40, 41)]
    many = list(scene.render_many([0, 1, 2, 3, 40, 41]))
    assert all(np.array_equal(a, b) for a, b in zip(direct, many))
    la = scene.background._lookahead
    assert la is not None and la.hits >= 3
    scene.close()
    assert scene.background.video is None and scene.background._lookahead is None
