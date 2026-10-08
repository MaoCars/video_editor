"""Backend de render en GPU (OpenGL 3.3 vía moderngl)."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

# Un solo hilo usa OpenGL a la vez (los contextos son por hilo; la creación concurrente puede fallar
# con algunos drivers, p. ej. Mesa bajo Xvfb).
GL_LOCK = threading.RLock()


def create_context():
    """Crea un contexto OpenGL sin ventana. En Linux sin pantalla usa EGL (Mesa/llvmpipe si no hay GPU)."""
    import moderngl

    errors = []
    backends = [None]
    if os.name != "nt":
        # EGL (sin ventana) primero: no depende de la conexión X de la interfaz; GLX como respaldo.
        backends = ["egl", None]
    for backend in backends:
        try:
            with GL_LOCK:
                return moderngl.create_standalone_context(backend=backend, require=330) if backend else moderngl.create_standalone_context(require=330)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend or 'default'}: {exc}")
    raise RuntimeError("No se pudo crear un contexto OpenGL: " + " | ".join(errors))


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    if os.environ.get("MUSICVIZ_NO_GPU"):
        return False
    try:
        ctx = create_context()
        with GL_LOCK:
            ctx.release()
        return True
    except Exception:  # noqa: BLE001
        return False


def gpu_info() -> str:
    try:
        ctx = create_context()
        with GL_LOCK:
            try:
                return f"{ctx.info['GL_RENDERER']} (OpenGL {ctx.info['GL_VERSION']})"
            finally:
                ctx.release()
    except Exception as exc:  # noqa: BLE001
        return f"no disponible: {exc}"
