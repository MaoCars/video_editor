"""Escena con backend GPU: dibuja capas y efectos con OpenGL (moderngl) reutilizando la geometría de las capas."""
from __future__ import annotations

import math
from typing import Optional

import cv2
import moderngl
import numpy as np

from ...audio.analysis import AudioFeatures, FrameFeatures
from ...config import ProjectConfig
from ...effects.basic import Bloom, Chromatic, ColorGrade, FilmGrain, Kaleido, Pixelate, RadialBlur, Scanlines, Shake, Strobe, Vignette
from ...effects.glitch import Glitch
from ...layers.bars import Bars
from ...layers.circle import CircleSpectrum
from ...layers.image import ImageOverlay
from ...layers.particles import Particles
from ...layers.progress import ProgressBar, fmt_time
from ...layers.text import TextOverlay, render_text_rgba
from ...layers.waveform import Waveform
from ...utils.color import parse_color
from ..engine import SceneBase
from . import GL_LOCK, create_context
from . import shaders as S

QUAD = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
INST_FLOATS = 10  # p0(2) p1(2) r(1) col(4) kind(1)


class GpuScene(SceneBase):
    def __init__(self, project: ProjectConfig, features: AudioFeatures, width: int, height: int, ctx: Optional[moderngl.Context] = None):
        super().__init__(project, features, width, height)
        self.gl = ctx or create_context()
        self._own_ctx = ctx is None
        with GL_LOCK:
            self._setup_gl(width, height)

    def _setup_gl(self, width: int, height: int) -> None:
        gl = self.gl
        self.w, self.h = width, height
        # ---- programas
        self.p_shapes = gl.program(vertex_shader=S.SHAPES_VS, fragment_shader=S.SHAPES_FS)
        self.p_mesh = gl.program(vertex_shader=S.MESH_VS, fragment_shader=S.MESH_FS)
        self.p_sprite = gl.program(vertex_shader=S.SPRITE_VS, fragment_shader=S.SPRITE_FS)
        fs = lambda src: gl.program(vertex_shader=S.FULLSCREEN_VS, fragment_shader=src)  # noqa: E731
        self.p_bg = fs(S.BACKGROUND_FS)
        self.p_comp = fs(S.COMPOSITE_FS)
        self.p_halo = fs(S.HALO_FS)
        self.p_copy = fs(S.COPY_FS)
        self.p_blur = fs(S.BLUR_FS)
        self.p_thr = fs(S.THRESHOLD_FS)
        self.p_final = fs(S.FINAL_FS)
        self.p_fx = {
            "chromatic": fs(S.CHROMATIC_FS), "shake": fs(S.SHAKE_FS), "vignette": fs(S.VIGNETTE_FS), "color": fs(S.COLOR_FS),
            "pixelate": fs(S.PIXELATE_FS), "strobe": fs(S.STROBE_FS), "kaleido": fs(S.KALEIDO_FS), "radial": fs(S.RADIAL_FS),
            "scanlines": fs(S.SCANLINES_FS), "grain": fs(S.GRAIN_FS), "glitch": fs(S.GLITCH_FS),
        }
        for prog in [self.p_shapes, self.p_mesh, self.p_sprite, self.p_bg, *self.p_fx.values()]:
            if "u_res" in prog:
                prog["u_res"].value = (float(width), float(height))
        # ---- geometría
        self.quad_vbo = gl.buffer(QUAD.tobytes())
        self.quad_vao = {}
        self.inst_cap = 1024
        self.inst_vbo = gl.buffer(reserve=self.inst_cap * INST_FLOATS * 4, dynamic=True)
        self.shapes_vao = gl.vertex_array(self.p_shapes, [(self.quad_vbo, "2f", "in_pos"), (self.inst_vbo, "2f 2f 1f 4f 1f/i", "in_p0", "in_p1", "in_r", "in_col", "in_kind")])
        self.mesh_cap = 4096
        self.mesh_vbo = gl.buffer(reserve=self.mesh_cap * 6 * 4, dynamic=True)
        self.mesh_vao = gl.vertex_array(self.p_mesh, [(self.mesh_vbo, "2f 4f", "in_pos", "in_col")])
        self.sprite_vao = gl.vertex_array(self.p_sprite, [(self.quad_vbo, "2f", "in_pos")])
        # ---- framebuffers
        self.canvas_fbo = self._fbo(width, height, "f2")
        self.ping_fbo = self._fbo(width, height, "f2")
        self.layer_fbo = self._fbo(width, height, "f2")
        self.final_fbo = self._fbo(width, height, "f1")
        self._small: dict[tuple[int, int], tuple[moderngl.Framebuffer, moderngl.Framebuffer]] = {}
        self._tex_cache: dict = {}
        self._bg_tex: Optional[moderngl.Texture] = None
        self._bg_key = None
        gl.disable(moderngl.DEPTH_TEST)
        gl.disable(moderngl.CULL_FACE)

    # ------------------------------------------------------------------ utilidades GL
    def _fbo(self, w: int, h: int, dtype: str) -> moderngl.Framebuffer:
        tex = self.gl.texture((w, h), 4, dtype=dtype)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        return self.gl.framebuffer(color_attachments=[tex])

    def _fs_vao(self, prog) -> moderngl.VertexArray:
        key = id(prog)
        if key not in self.quad_vao:
            self.quad_vao[key] = self.gl.vertex_array(prog, [(self.quad_vbo, "2f", "in_pos")])
        return self.quad_vao[key]

    def _fullscreen(self, prog, tex: moderngl.Texture, **uniforms) -> None:
        tex.use(0)
        prog["tex"].value = 0
        for k, v in uniforms.items():
            if k in prog:  # el compilador elimina los uniforms que un shader no usa
                prog[k].value = v
        self._fs_vao(prog).render(moderngl.TRIANGLE_STRIP)

    def _blend(self, mode: str) -> None:
        gl = self.gl
        gl.enable(moderngl.BLEND)
        if mode == "normal":
            gl.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        elif mode == "add":
            gl.blend_func = (moderngl.ONE, moderngl.ONE, moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        elif mode == "screen":
            gl.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_COLOR, moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

    def _upload(self, key, rgba: np.ndarray, mipmaps: bool = False) -> moderngl.Texture:
        tex = self._tex_cache.get(key)
        if tex is None:
            h, w = rgba.shape[:2]
            tex = self.gl.texture((w, h), rgba.shape[2], np.ascontiguousarray(rgba).tobytes())
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR if mipmaps else moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            if mipmaps:
                tex.build_mipmaps()
            self._tex_cache[key] = tex
        return tex

    def _draw_shapes(self, inst: np.ndarray) -> None:
        n = len(inst)
        if n == 0:
            return
        if n > self.inst_cap:
            self.inst_cap = int(n * 1.5)
            self.inst_vbo.orphan(self.inst_cap * INST_FLOATS * 4)
        self.inst_vbo.write(np.ascontiguousarray(inst, dtype=np.float32).tobytes())
        self.shapes_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)

    def _draw_mesh(self, verts: np.ndarray, mode=moderngl.TRIANGLE_STRIP) -> None:
        n = len(verts)
        if n < 3:
            return
        if n > self.mesh_cap:
            self.mesh_cap = int(n * 1.5)
            self.mesh_vbo.orphan(self.mesh_cap * 6 * 4)
        verts = np.array(verts, dtype=np.float32, copy=True)
        verts[:, :2] += 0.5  # misma convención de píxel que OpenCV
        self.mesh_vbo.write(np.ascontiguousarray(verts).tobytes())
        self.mesh_vao.render(mode, vertices=n)

    def _draw_sprite(self, tex: moderngl.Texture, cx: float, cy: float, w: float, h: float, angle_deg: float, alpha: float = 1.0, lod: float = 0.0) -> None:
        tex.use(0)
        p = self.p_sprite
        p["tex"].value = 0
        p["u_center"].value = (float(cx), float(cy))
        p["u_size"].value = (float(w), float(h))
        p["u_angle"].value = math.radians(angle_deg)
        p["u_alpha"].value = float(alpha)
        p["u_lod"].value = float(lod)
        self.sprite_vao.render(moderngl.TRIANGLE_STRIP)

    def _blurred(self, tex: moderngl.Texture, sigma: float) -> moderngl.Texture:
        """Versión desenfocada (a resolución reducida) de una textura, equivalente a canvas.fast_blur."""
        w, h = self.w, self.h
        f = 1
        while sigma / f > 3.0 and min(w, h) // (f * 2) >= 8:
            f *= 2
        sw, sh = max(w // f, 1), max(h // f, 1)
        if (sw, sh) not in self._small:
            self._small[(sw, sh)] = (self._fbo(sw, sh, "f2"), self._fbo(sw, sh, "f2"))
        a, b = self._small[(sw, sh)]
        self.gl.disable(moderngl.BLEND)
        a.use()
        self._fullscreen(self.p_copy, tex)
        b.use()
        self._fullscreen(self.p_blur, a.color_attachments[0], u_step=(1.0 / sw, 0.0), u_sigma=max(sigma / f, 0.3))
        a.use()
        self._fullscreen(self.p_blur, b.color_attachments[0], u_step=(0.0, 1.0 / sh), u_sigma=max(sigma / f, 0.3))
        return a.color_attachments[0]

    @staticmethod
    def _inst(p0, p1, r, col, kind) -> np.ndarray:
        """Construye el array de instancias (n, 10) a partir de arrays/escalares."""
        p0 = np.asarray(p0, np.float32).reshape(-1, 2)
        n = len(p0)
        p1 = np.asarray(p1, np.float32).reshape(-1, 2) if np.ndim(p1) > 1 or (np.ndim(p1) == 1 and len(p1) == 2 and n == 1) else np.broadcast_to(np.asarray(p1, np.float32).reshape(-1, 2), (n, 2))
        if p1.shape[0] != n:
            p1 = np.broadcast_to(p1, (n, 2))
        r = np.broadcast_to(np.asarray(r, np.float32).reshape(-1), (n,))
        col = np.asarray(col, np.float32)
        col = np.broadcast_to(col.reshape(-1, 4), (n, 4))
        kind = np.broadcast_to(np.asarray(kind, np.float32).reshape(-1), (n,))
        # OpenCV sitúa el píxel entero (x, y) donde OpenGL tiene su centro en (x+0.5, y+0.5)
        return np.concatenate([p0 + 0.5, p1 + 0.5, r[:, None], col, kind[:, None]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ fondo
    def _draw_background(self, frame: FrameFeatures, section_colors) -> None:
        bg = self.background
        src = bg.source(frame, section_colors)
        key = ("video", frame.index) if bg.video is not None else ("sec", section_colors.tobytes() if section_colors is not None else b"")
        if self._bg_tex is None or key != self._bg_key:
            data = cv2.convertScaleAbs(np.clip(src, 0, 1), alpha=255.0)
            if self._bg_tex is None:
                self._bg_tex = self.gl.texture((self.w, self.h), 3, np.ascontiguousarray(data).tobytes())
                self._bg_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            else:
                self._bg_tex.write(np.ascontiguousarray(data).tobytes())
            self._bg_key = key
        zoom, dx, dy, angle, gain = bg.motion(frame)
        self.gl.disable(moderngl.BLEND)
        self._fullscreen(self.p_bg, self._bg_tex, u_zoom=float(zoom), u_angle=math.radians(angle), u_shift=(float(dx), float(dy)), u_gain=float(gain))

    # ------------------------------------------------------------------ capas
    def _composite_layer(self, layer) -> None:
        cfg = layer.cfg
        opacity = cfg.opacity * layer.anim.alpha * layer.keys.opacity
        if opacity <= 0:
            return
        tex = self.layer_fbo.color_attachments[0]
        self.canvas_fbo.use()
        self._blend(cfg.blend)
        self._fullscreen(self.p_comp, tex, u_opacity=float(opacity))
        glow = cfg.glow * layer.intensity
        radius = self.ctx.px(cfg.glow_radius)
        if glow > 0 and radius > 0:
            small = self._blurred(tex, radius)
            self.canvas_fbo.use()
            self._blend("add")
            self._fullscreen(self.p_halo, small, u_k=float(glow * 1.6 * opacity))

    def _draw_layer(self, layer, frame: FrameFeatures) -> None:
        self.layer_fbo.use()
        self.layer_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._blend("normal")
        if isinstance(layer, Bars):
            self._draw_bars(layer, frame)
        elif isinstance(layer, CircleSpectrum):
            self._draw_circle(layer, frame)
        elif isinstance(layer, Waveform):
            self._draw_waveform(layer, frame)
        elif isinstance(layer, Particles):
            self._draw_particles(layer, frame)
        elif isinstance(layer, (TextOverlay, ImageOverlay)):
            self._draw_sprite_layer(layer, frame)
        elif isinstance(layer, ProgressBar):
            self._draw_progress(layer, frame)
        else:  # pragma: no cover - tipos futuros
            return
        self._composite_layer(layer)

    def _draw_bars(self, layer: Bars, frame: FrameFeatures) -> None:
        cfg = layer.cfg
        geo = layer.geometry(frame)
        xc, hs, cols, cy, bw = geo["xc"], geo["heights"], geo["colors"], geo["cy"], geo["bar_w"]
        n = len(xc)
        top = cy - hs
        bottom = cy + hs if cfg.mirror else np.full(n, cy, np.float32)
        r = bw / 2.0 + 0.5  # OpenCV dibuja rectángulos inclusivos: 1 px más anchos
        inst = []
        if cfg.style == "dots":
            rd = max(round(bw / 2.0), 1) + 0.5
            p = np.stack([xc, top], 1)
            inst.append(self._inst(p, p, rd, cols, 0))
            if cfg.mirror:
                p = np.stack([xc, bottom], 1)
                inst.append(self._inst(p, p, rd, cols, 0))
        elif cfg.style == "segments":
            seg = layer.seg_h
            for i in range(n):
                y = cy
                while y - seg >= top[i] - 1e-6:
                    inst.append(self._inst([[xc[i], y - seg * 0.7 - 0.5]], [[xc[i], y + 0.5]], r, cols[i], 1))
                    if cfg.mirror:
                        yy = 2 * cy - y
                        inst.append(self._inst([[xc[i], yy - 0.5]], [[xc[i], yy + seg * 0.7 + 0.5]], r, cols[i], 1))
                    y -= seg
        elif cfg.style == "outline":
            t = max(round(self.ctx.px(2)), 1.0) / 2
            for i in range(n):
                xl, xr = xc[i] - r, xc[i] + r
                inst.append(self._inst([[xl, top[i]], [xr, top[i]], [xl, top[i]], [xl, bottom[i]]], [[xr, top[i]], [xr, bottom[i]], [xl, bottom[i]], [xr, bottom[i]]], t, cols[i], 1))
        else:
            if cfg.rounded and r >= 1.5:
                rr = r - 0.5
                p0 = np.stack([xc, top + rr], 1)
                p1 = np.stack([xc, np.maximum(bottom - rr, top + rr)], 1)
                inst.append(self._inst(p0, p1, r, cols, 0))
            else:
                inst.append(self._inst(np.stack([xc, top - 0.5], 1), np.stack([xc, bottom + 0.5], 1), r, cols, 1))
        if cfg.baseline:
            c = layer.colors_lut[128].copy()
            c[3] *= 0.6
            inst.append(self._inst([[layer.x0, cy]], [[layer.x0 + layer.slot * n, cy]], max(self.ctx.px(2), 1.0) / 2, c, 0))
        self._draw_shapes(np.concatenate(inst) if inst else np.zeros((0, INST_FLOATS), np.float32))

    def _ring_instances(self, cx: float, cy: float, r0: float, half: float, col) -> np.ndarray:
        n = max(int(r0 / 4), 48)
        ang = np.linspace(0, 2 * math.pi, n + 1)
        pts = np.stack([cx + np.cos(ang) * r0, cy + np.sin(ang) * r0], 1)
        return self._inst(pts[:-1], pts[1:], half, col, 0)

    def _draw_circle(self, layer: CircleSpectrum, frame: FrameFeatures) -> None:
        cfg = layer.cfg
        rings = layer.geometry(frame)
        cx, cy = layer.cx, layer.cy
        ring_col = np.array(layer.ring_color, np.float32) / 255.0
        inst = []
        meshes = []
        for ring in rings:
            r0, fade, vals, cols = ring["r0"], ring["fade"], ring["values"], ring["colors"]
            p_in = np.stack([ring["x_in"], ring["y_in"]], 1)
            p_out = np.stack([ring["x_out"], ring["y_out"]], 1)
            if cfg.ring and cfg.style != "filled":
                inst.append(self._ring_instances(cx, cy, r0, layer.ring_thick / 2, ring_col))
            style = cfg.style
            if style in ("bars", "rays"):
                thick = layer.thick if style == "bars" else max(layer.thick // 2, 1)
                kind = 0 if (cfg.rounded and thick > 2) else 1
                inst.append(self._inst(p_in, p_out, thick / 2, cols, kind))
            elif style == "dots":
                rad = np.maximum(np.rint(layer.thick * (0.6 + vals)), 1) + 0.5
                inst.append(self._inst(p_out, p_out, rad, cols, 0))
            elif style == "line":
                nxt = np.roll(p_out, -1, axis=0)
                inst.append(self._inst(p_out, nxt, layer.thick / 2, cols, 0))
                if cfg.inner:
                    inst.append(self._inst(p_in, np.roll(p_in, -1, axis=0), layer.thick / 2, cols, 0))
            else:  # filled
                n = len(p_out)
                if cfg.gradient == "angle" and len(cfg.colors) > 1:
                    vc = cols.copy()
                else:
                    vc = np.tile(layer.colors_lut[128], (n, 1))
                    vc[:, 3] = fade
                if cfg.inner:
                    inner = p_in
                else:
                    ang = layer.angles + ring["rot"]
                    inner = np.stack([cx + np.cos(ang) * r0, cy + np.sin(ang) * r0], 1)
                strip = np.empty((2 * (n + 1), 6), np.float32)
                strip[0::2, :2] = np.vstack([inner, inner[:1]])
                strip[1::2, :2] = np.vstack([p_out, p_out[:1]])
                strip[0::2, 2:] = np.vstack([vc, vc[:1]])
                strip[1::2, 2:] = np.vstack([vc, vc[:1]])
                meshes.append(strip)
                if cfg.ring:
                    inst.append(self._ring_instances(cx, cy, r0, layer.ring_thick / 2, ring_col))
        for m in meshes:
            self._draw_mesh(m)
        if inst:
            self._draw_shapes(np.concatenate(inst))

    def _draw_waveform(self, layer: Waveform, frame: FrameFeatures) -> None:
        cfg = layer.cfg
        lut = layer.colors_lut = layer.luts(frame)[0]
        layer.cx, layer.cy = layer.key_position()
        s = layer._samples(frame)
        n = len(s)
        amp = layer.amp * max(layer.keys.scale, 0.0)
        half = layer.thick / 2
        inst = []
        if cfg.style == "circular":
            r = cfg.radius * self.ctx.min_dim
            ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
            rr = r + s * amp
            pts = np.stack([layer.cx + np.cos(ang) * rr, layer.cy + np.sin(ang) * rr], 1)
            inst.append(self._inst(pts, np.roll(pts, -1, axis=0), half, lut[128], 0))
        else:
            xs = layer.cx - layer.area_w / 2 + np.linspace(0, layer.area_w, n)
            ys = layer.cy - s * amp
            idx = np.clip((np.arange(n) / max(n - 1, 1) * 255).astype(int), 0, 255)
            cols = lut[idx]
            if cfg.style == "bars":
                step = layer.area_w / n
                bw = max(step * 0.6, 1.0)
                hh = np.abs(s) * amp
                inst.append(self._inst(np.stack([xs, layer.cy - hh], 1), np.stack([xs, layer.cy + hh], 1), bw / 2, cols, 1))
            elif cfg.style == "filled":
                base_y = (layer.cy + s * amp) if cfg.mirror else np.full(n, layer.cy)
                c = lut[128].copy()
                c[3] *= 0.85
                strip = np.empty((2 * n, 6), np.float32)
                strip[0::2, :2] = np.stack([xs, ys], 1)
                strip[1::2, :2] = np.stack([xs, base_y], 1)
                strip[:, 2:] = c
                self._draw_mesh(strip)
            else:
                pts = np.stack([xs, ys], 1)
                seg_cols = cols if len(cfg.colors) > 1 else np.tile(lut[0], (n, 1))
                inst.append(self._inst(pts[:-1], pts[1:], half, seg_cols[:-1], 0))
                if cfg.mirror:
                    pts2 = np.stack([xs, layer.cy + s * amp], 1)
                    inst.append(self._inst(pts2[:-1], pts2[1:], half, lut[-1], 0))
        if inst:
            self._draw_shapes(np.concatenate(inst))

    def _draw_particles(self, layer: Particles, frame: FrameFeatures) -> None:
        cfg = layer.cfg
        lut = layer.luts(frame)[0]
        inst = []
        for pos, sizes, alphas, cols, vel in layer.batches(frame):
            c = lut[np.asarray(cols)].copy()
            c[:, 3] *= alphas
            r = np.maximum(np.rint(sizes), 1.0)
            if cfg.shape == "square":
                inst.append(self._inst(pos, pos, r + 0.5, c, 2))
            elif cfg.shape == "streak":
                inst.append(self._inst(pos, pos - vel * 0.06, np.maximum(r, 1.0) / 2, c, 0))
            else:
                inst.append(self._inst(pos, pos, r + 0.5, c, 0))
        if inst:
            self._draw_shapes(np.concatenate(inst))

    def _draw_sprite_layer(self, layer, frame: FrameFeatures) -> None:
        cfg = layer.cfg
        cx, cy, scale, angle, blur = layer.placement(frame)
        key = ("sprite", id(layer))
        tex = self._upload(key, layer.sprite, mipmaps=True)
        sh, sw = layer.sprite.shape[:2]
        lod = blur * 3.0
        if layer.shadow_sprite is not None:
            stex = self._upload(("shadow", id(layer)), layer.shadow_sprite, mipmaps=True)
            ssh, ssw = layer.shadow_sprite.shape[:2]
            ox, oy = self.ctx.px(cfg.shadow_offset[0]), self.ctx.px(cfg.shadow_offset[1])
            self._draw_sprite(stex, cx + ox, cy + oy, ssw * scale, ssh * scale, angle, 1.0, lod)
        self._draw_sprite(tex, cx, cy, sw * scale, sh * scale, angle, 1.0, lod)

    def _draw_progress(self, layer: ProgressBar, frame: FrameFeatures) -> None:
        cfg = layer.cfg
        x0 = layer.cx - layer.w / 2
        x1 = layer.cx + layer.w / 2
        y = layer.cy
        bg = np.array(layer.bg, np.float32) / 255.0
        col = np.array(layer.col, np.float32) / 255.0
        xp = x0 + (x1 - x0) * frame.progress
        inst = [self._inst([[x0, y]], [[x1, y]], layer.thick / 2, bg, 0)]
        if xp > x0:
            inst.append(self._inst([[x0, y]], [[xp, y]], layer.thick / 2, col, 0))
        inst.append(self._inst([[xp, y]], [[xp, y]], layer.thick + 2.5, col, 0))
        self._draw_shapes(np.concatenate(inst))
        if layer.font is not None:
            gap = self.ctx.px(14)
            for text, side in ((fmt_time(frame.time), -1), (fmt_time(layer.duration), 1)):
                key = ("ptext", id(layer), text)
                if key not in self._tex_cache:
                    self._tex_cache[key] = None
                    sprite = render_text_rgba(text, layer.font, cfg.color)
                    self._tex_cache[key] = (self._upload(key + ("tex",), sprite), sprite.shape[1], sprite.shape[0])
                tex, tw, th = self._tex_cache[key]
                cxx = (x0 - tw / 2 - gap) if side < 0 else (x1 + tw / 2 + gap)
                self._draw_sprite(tex, cxx, y, tw, th, 0.0)

    # ------------------------------------------------------------------ efectos
    def _pass(self, name: str, **uniforms) -> None:
        """Aplica un shader de pantalla completa de `cur` a `other` y los intercambia."""
        prog = self.p_fx[name]
        self._other.use()
        self.gl.disable(moderngl.BLEND)
        if "u_alpha" in prog:
            prog["u_alpha"].value = 1 if self.transparent else 0
        self._fullscreen(prog, self._cur.color_attachments[0], **uniforms)
        self._cur, self._other = self._other, self._cur

    def _apply_effect(self, effect, frame: FrameFeatures) -> None:
        k = effect.drive(frame)
        cfg = effect.cfg
        if isinstance(effect, Bloom):
            if k <= 0:
                return
            small = self._blurred_threshold(self._cur.color_attachments[0], cfg.threshold, self.ctx.px(cfg.radius))
            self._cur.use()
            self._blend("add")
            self._fullscreen(self.p_halo, small, u_k=float(cfg.strength * k))
        elif isinstance(effect, Chromatic):
            if k <= 0:
                return
            amt = self.ctx.px(cfg.amount) * k
            self._pass("chromatic", u_k=float(k), u_f=float(amt / (self.w / 2.0)))
        elif isinstance(effect, Shake):
            prm = effect.params(frame)
            if prm is None:
                return
            dx, dy, ang, zoom = prm
            self._pass("shake", u_k=float(k), u_zoom=float(zoom), u_angle=math.radians(ang), u_shift=(float(dx), float(dy)))
        elif isinstance(effect, Vignette):
            if k <= 0:
                return
            self._pass("vignette", u_k=float(min(k, 1.0)), u_strength=float(cfg.strength), u_soft=float(max(cfg.softness, 0.01)))
        elif isinstance(effect, ColorGrade):
            if k <= 0:
                return
            hue = cfg.hue_speed * frame.time + cfg.hue_react * frame.rms * k
            self._pass("color", u_k=float(k), u_hue=float(hue), u_sat=float(cfg.saturation), u_contrast=float(cfg.contrast), u_bright=float(cfg.brightness), u_gamma=float(cfg.gamma), u_poster=float(cfg.posterize))
        elif isinstance(effect, Pixelate):
            if k <= 0.05:
                return
            size = max(int(round(self.ctx.px(cfg.size) * k)), 1)
            if size <= 1:
                return
            self._pass("pixelate", u_k=float(k), u_size=float(size))
        elif isinstance(effect, Strobe):
            k = min(k, 1.0)
            if k <= 0.01:
                return
            self._pass("strobe", u_k=float(k), u_color=tuple(float(c) for c in effect.color))
        elif isinstance(effect, Kaleido):
            if k <= 0:
                return
            horiz = 1 if (cfg.segments == 4 or cfg.axis == "horizontal") else 0
            vert = 1 if (cfg.segments == 4 or cfg.axis == "vertical") else 0
            self._pass("kaleido", u_k=float(min(k, 1.0)), u_h=horiz, u_v=vert)
        elif isinstance(effect, RadialBlur):
            if k <= 0.01:
                return
            self._pass("radial", u_k=float(k), u_amount=float(cfg.amount), u_n=max(int(cfg.samples), 2))
        elif isinstance(effect, Scanlines):
            if k <= 0:
                return
            self._pass("scanlines", u_k=float(min(k, 1.0)), u_spacing=float(max(int(cfg.spacing), 2)), u_dark=float(cfg.darkness))
        elif isinstance(effect, FilmGrain):
            if k <= 0:
                return
            self._pass("grain", u_k=float(k), u_amount=float(cfg.amount), u_seed=float((cfg.seed * 7919 + frame.index) % 10007))
        elif isinstance(effect, Glitch):
            prm = effect.params(frame)
            if prm is None:
                return
            prog = self.p_fx["glitch"]
            blocks = np.zeros((32, 4), np.float32)
            nb = min(len(prm["blocks"]), 32)
            for i, (y0, bh, dx, inv) in enumerate(prm["blocks"][:nb]):
                blocks[i] = (y0, bh, dx, 1.0 if inv else 0.0)
            prog["u_blocks"].write(blocks.tobytes())
            noise = np.zeros((4, 4), np.float32)
            nn = min(len(prm["noise"]), 4)
            for i, (y0, bh, seed, strength) in enumerate(prm["noise"][:nn]):
                noise[i] = (y0, bh, float(seed % 9973), strength)
            prog["u_noise"].write(noise.tobytes())
            rgb = prm["rgb"] or (0, 0, 0)
            scan = prm["scan"] or (2, 0, 0.0)
            self._pass("glitch", u_k=float(prm["k"]), u_nblocks=nb, u_rgb=tuple(float(v) for v in rgb), u_has_rgb=1 if prm["rgb"] else 0, u_scan=tuple(float(v) for v in scan), u_has_scan=1 if prm["scan"] else 0, u_nnoise=nn)

    def _blurred_threshold(self, tex: moderngl.Texture, thr: float, radius: float) -> moderngl.Texture:
        w, h = self.w, self.h
        f = 4
        sw, sh = max(w // f, 1), max(h // f, 1)
        if (sw, sh) not in self._small:
            self._small[(sw, sh)] = (self._fbo(sw, sh, "f2"), self._fbo(sw, sh, "f2"))
        a, b = self._small[(sw, sh)]
        self.gl.disable(moderngl.BLEND)
        a.use()
        self._fullscreen(self.p_thr, tex, u_thr=float(thr))
        sigma = max(radius / f, 0.5)
        b.use()
        self._fullscreen(self.p_blur, a.color_attachments[0], u_step=(1.0 / sw, 0.0), u_sigma=sigma)
        a.use()
        self._fullscreen(self.p_blur, b.color_attachments[0], u_step=(0.0, 1.0 / sh), u_sigma=sigma)
        return a.color_attachments[0]

    # ------------------------------------------------------------------ frame
    def render(self, index: int) -> np.ndarray:
        with GL_LOCK:
            return self._render(index)

    def _render(self, index: int) -> np.ndarray:
        frame, section = self.frame_state(index)
        self.canvas_fbo.use()
        self.canvas_fbo.clear(0.0, 0.0, 0.0, 0.0)
        if not self.transparent:
            self._draw_background(frame, section.background)
        for layer in self.active_layers(frame, section):
            self._draw_layer(layer, frame)
        self._cur, self._other = self.canvas_fbo, self.ping_fbo
        for effect in self.active_effects(section):
            self._apply_effect(effect, frame)
        self.final_fbo.use()
        self.gl.disable(moderngl.BLEND)
        self._fullscreen(self.p_final, self._cur.color_attachments[0], u_alpha_mode=1 if self.transparent else 0)
        comps = 4 if self.transparent else 3
        data = self.final_fbo.read(components=comps, dtype="f1")
        return np.frombuffer(data, np.uint8).reshape(self.h, self.w, comps)

    def close(self) -> None:
        """Libera los recursos GL. Debe llamarse desde el hilo que creó la escena."""
        if getattr(self, "gl", None) is None:
            return
        with GL_LOCK:
            try:
                for tex in self._tex_cache.values():
                    if isinstance(tex, moderngl.Texture):
                        tex.release()
                    elif isinstance(tex, tuple) and tex and isinstance(tex[0], moderngl.Texture):
                        tex[0].release()
                if self._own_ctx:
                    self.gl.release()
            except Exception:  # noqa: BLE001
                pass
        self.gl = None
