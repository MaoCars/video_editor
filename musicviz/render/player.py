"""Reproductor OpenGL a resolución completa: la escena GPU se dibuja directamente en una ventana (glfw).

Se usa desde la interfaz (en un proceso aparte, para no mezclar el bucle de Tk con el de la ventana GL)
y desde la consola (`musicviz preview`). Si no hay OpenGL o glfw, la interfaz y la consola caen a la vista
previa en Tkinter (`render/preview.py`).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Optional

import numpy as np

from ..audio.analysis import AudioFeatures
from ..config import ProjectConfig

SCREEN_VS = """
#version 330
uniform vec2 u_scale;
in vec2 in_pos;
out vec2 uv;
void main() { uv = vec2(in_pos.x * 0.5 + 0.5, 0.5 - in_pos.y * 0.5); gl_Position = vec4(in_pos * u_scale, 0.0, 1.0); }
"""

SCREEN_FS = """
#version 330
uniform sampler2D tex; uniform int u_checker; uniform vec2 u_size;
in vec2 uv; out vec4 frag;
void main() {
    vec4 c = texture(tex, uv);
    if (u_checker == 1) {
        vec2 cell = floor(uv * u_size / 16.0);
        float k = mod(cell.x + cell.y, 2.0) < 0.5 ? 0.35 : 0.55;
        frag = vec4(mix(vec3(k), c.rgb, c.a), 1.0);
    } else {
        frag = vec4(c.rgb, 1.0);
    }
}
"""

QUAD = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")


class ScreenBlit:
    """Dibuja la textura final de la escena ajustada (con bandas) al tamaño de destino, sin invertir filas."""

    def __init__(self, gl, scene):
        import moderngl

        self.gl = gl
        self.scene = scene
        self.prog = gl.program(vertex_shader=SCREEN_VS, fragment_shader=SCREEN_FS)
        self.vbo = gl.buffer(QUAD.tobytes())
        self.vao = gl.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self.prog["u_checker"].value = 1 if scene.transparent else 0
        self.prog["u_size"].value = (float(scene.w), float(scene.h))
        self._blend = moderngl.BLEND
        self._strip = moderngl.TRIANGLE_STRIP

    def draw(self, fbo, width: int, height: int) -> None:
        """Dibuja la escena actual en `fbo` (de width x height píxeles) manteniendo la proporción."""
        if width <= 0 or height <= 0:
            return
        fbo.viewport = (0, 0, width, height)  # la ventana puede haber cambiado de tamaño
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        sx = min(1.0, (self.scene.w / self.scene.h) / (width / height))
        sy = min(1.0, (width / height) / (self.scene.w / self.scene.h))
        self.prog["u_scale"].value = (sx, sy)
        self.gl.disable(self._blend)
        tex = self.scene.final_fbo.color_attachments[0]
        tex.use(0)
        self.prog["tex"].value = 0
        self.vao.render(self._strip)

    def release(self) -> None:
        self.vao.release()
        self.vbo.release()
        self.prog.release()


class _Audio:
    """Reproducción de audio opcional (sounddevice) sincronizada por posición."""

    def __init__(self, features: AudioFeatures):
        self.features = features
        self.ok = False
        try:
            import sounddevice as sd

            self.sd = sd
            sd.check_output_settings(samplerate=features.sr)
            self.ok = True
        except Exception:  # noqa: BLE001
            self.sd = None

    def play_from(self, t: float) -> None:
        if not self.ok:
            return
        try:
            start = int(max(t, 0.0) * self.features.sr)
            self.sd.play(self.features.waveform[start:], self.features.sr)
        except Exception:  # noqa: BLE001
            self.ok = False

    def stop(self) -> None:
        if self.sd is not None:
            try:
                self.sd.stop()
            except Exception:  # noqa: BLE001
                pass


def window_size(width: int, height: int, screen_w: int, screen_h: int, margin: float = 0.9) -> tuple[int, int]:
    """Tamaño inicial de la ventana: la resolución del proyecto, reducida si no cabe en el monitor."""
    k = min(1.0, margin * screen_w / width, margin * screen_h / height)
    return max(int(round(width * k)), 64), max(int(round(height * k)), 36)


def run_player(
    project: ProjectConfig,
    features: AudioFeatures,
    start: float = 0.0,
    with_audio: bool = True,
    title: Optional[str] = None,
    loop: Optional[tuple[float, float]] = None,
) -> None:
    """Abre la ventana y reproduce hasta que el usuario la cierra (ESC/Q), ESPACIO pausa, ←/→ ±5 s,
    Inicio vuelve al principio, F alterna pantalla completa. Con `loop` = (a, b) repite ese tramo.
    Bloquea hasta cerrar la ventana."""
    import glfw
    import moderngl

    from .gpu import GL_LOCK
    from .gpu.scene import GpuScene

    if not glfw.init():
        raise RuntimeError("No se pudo iniciar glfw (¿sin pantalla?)")
    try:
        w, h = project.output.width, project.output.height
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor) if monitor else None
        ww, wh = window_size(w, h, mode.size.width, mode.size.height) if mode else (min(w, 1280), min(h, 720))
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.RESIZABLE, True)
        base_title = title or f"musicviz · {project.name} · {w}x{h}"
        win = glfw.create_window(ww, wh, base_title, None, None)
        if not win:
            raise RuntimeError("No se pudo crear la ventana OpenGL 3.3")
        glfw.make_context_current(win)
        glfw.swap_interval(1)
        with GL_LOCK:
            gl = moderngl.create_context(require=330)
        scene = GpuScene(project, features, w, h, ctx=gl)
        blit = ScreenBlit(gl, scene)
        audio = _Audio(features) if with_audio else None
        fps = features.fps
        state = {"t0": 0.0, "wall": 0.0, "paused": False, "fullscreen": False, "restore": (0, 0, ww, wh), "seek": None}

        def media_time() -> float:
            return state["t0"] if state["paused"] else state["t0"] + (time.perf_counter() - state["wall"])

        def set_time(t: float, resume_audio: bool = True) -> None:
            t = min(max(t, 0.0), max(features.duration - 1e-3, 0.0))
            state["t0"], state["wall"] = t, time.perf_counter()
            if audio is not None:
                audio.stop()
                if not state["paused"] and resume_audio:
                    audio.play_from(t)

        def toggle_pause() -> None:
            t = media_time()
            state["paused"] = not state["paused"]
            set_time(t)

        def toggle_fullscreen() -> None:
            if state["fullscreen"]:
                x, y, rw, rh = state["restore"]
                glfw.set_window_monitor(win, None, x, y, rw, rh, 0)
            else:
                x, y = glfw.get_window_pos(win)
                rw, rh = glfw.get_window_size(win)
                state["restore"] = (x, y, rw, rh)
                m = glfw.get_primary_monitor()
                vm = glfw.get_video_mode(m)
                glfw.set_window_monitor(win, m, 0, 0, vm.size.width, vm.size.height, vm.refresh_rate)
            state["fullscreen"] = not state["fullscreen"]

        def on_key(_win, key, _scancode, action, _mods):
            if action != glfw.PRESS:
                return
            if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                toggle_pause()
            elif key == glfw.KEY_LEFT:
                set_time(media_time() - 5.0)
            elif key == glfw.KEY_RIGHT:
                set_time(media_time() + 5.0)
            elif key == glfw.KEY_HOME:
                set_time(0.0)
            elif key == glfw.KEY_F:
                toggle_fullscreen()

        glfw.set_key_callback(win, on_key)
        set_time(start)
        last_index = -1
        last_title = 0.0
        try:
            while not glfw.window_should_close(win):
                t = media_time()
                if loop is not None and t >= loop[1]:
                    set_time(loop[0])
                    t = media_time()
                if t >= features.duration:
                    break
                index = int(t * fps)
                fw, fh = glfw.get_framebuffer_size(win)
                with GL_LOCK:
                    if index != last_index:
                        scene._render_to_final(index)
                        last_index = index
                    blit.draw(gl.screen, fw, fh)
                glfw.swap_buffers(win)
                glfw.poll_events()
                if time.perf_counter() - last_title > 0.25:
                    last_title = time.perf_counter()
                    mins, secs = divmod(int(t), 60)
                    glfw.set_window_title(win, f"{base_title} · {mins}:{secs:02d}{' · pausa' if state['paused'] else ''}")
                if state["paused"]:
                    time.sleep(0.01)
        finally:
            if audio is not None:
                audio.stop()
            with GL_LOCK:
                blit.release()
            scene.close()
            glfw.destroy_window(win)
    finally:
        glfw.terminate()


def player_available() -> bool:
    """True si glfw está instalado (la ventana además necesita pantalla y OpenGL 3.3)."""
    try:
        import glfw  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- proceso aparte (para la interfaz)
def _player_main(project_json: str, features: AudioFeatures, start: float, with_audio: bool, loop: Optional[tuple[float, float]]) -> None:
    project = ProjectConfig.model_validate_json(project_json)
    run_player(project, features, start=start, with_audio=with_audio, loop=loop)


def launch_player(
    project: ProjectConfig, features: AudioFeatures, start: float = 0.0, with_audio: bool = True, loop: Optional[tuple[float, float]] = None
) -> mp.Process:
    """Abre el reproductor en un proceso nuevo y devuelve el proceso (terminarlo cierra la ventana)."""
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_player_main, args=(project.model_dump_json(), features, start, with_audio, loop), name="musicviz-player", daemon=True)
    proc.start()
    return proc
