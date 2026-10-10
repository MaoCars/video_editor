"""Aplicación de escritorio (Tkinter) para crear y renderizar visualizers sin tocar YAML."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from pydantic import BaseModel

from ..audio.analysis import AudioFeatures
from ..config import KEYFRAME_PROPS, EffectConfig, LayerConfig, PointKey, ProjectConfig, ScalarKey, SectionConfig, get_args_of_union
from ..layers.keyframes import current_value
from ..layers.text import available_fonts
from ..presets import load_preset, preset_names
from ..render.canvas import over_checkerboard
from . import autosave
from .fields import FieldSpec, apply_value, field_specs, format_value, replace_submodel
from .recent import add_recent, clear_recent, load_recent, remove_recent
from .renderer import RenderService
from .hittest import ROTATION_FIELD, SIZE_FIELD, hit_layer, layer_polygon
from .timeline import Timeline

PREVIEW_SCALES = {"Baja (480p)": 480 / 1080, "Media (540p)": 0.5, "Alta (720p)": 720 / 1080}
AUDIO_TYPES = [("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.opus *.wma"), ("Todos", "*.*")]
IMAGE_TYPES = [("Imágenes", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Todos", "*.*")]
VIDEO_TYPES = [("Videos", "*.mp4 *.mov *.mkv *.webm *.avi"), ("Todos", "*.*")]


# --------------------------------------------------------------------------- utilidades


class ScrollFrame(ttk.Frame):
    """Frame con scroll vertical (contenido en `self.inner`)."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", lambda e: self._bind_wheel())
            w.bind("<Leave>", lambda e: self._unbind_wheel())

    def _bind_wheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        delta = -1 if (event.num == 4 or event.delta > 0) else 1
        self.canvas.yview_scroll(delta, "units")

    def clear(self):
        for child in self.inner.winfo_children():
            child.destroy()


def _swatch_color(value: str) -> Optional[str]:
    """Hex #rrggbb para pintar una muestra, o None si no es un color válido."""
    try:
        from ..utils.color import parse_color

        r, g, b, _ = parse_color(value)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    except (ValueError, TypeError):
        return None


class ColorList(ttk.Frame):
    """Editor de lista de colores: cuadro de texto + fila de muestras con botón para quitar cada color.

    Clic en una muestra cambia ese color; «+» añade uno; «palette» sigue la paleta de la sección.
    """

    def __init__(self, master, colors: list[str], on_change: Callable[[list[str]], None]):
        super().__init__(master)
        self.on_change = on_change
        self.var = tk.StringVar(value=", ".join(colors))
        self.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(self, textvariable=self.var)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", lambda e: self._from_text())
        self.entry.bind("<FocusOut>", lambda e: self._from_text())
        self.row = ttk.Frame(self)
        self.row.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._render()

    # -- estado
    def get(self) -> list[str]:
        return [p.strip() for p in self.var.get().replace(";", ",").split(",") if p.strip()]

    def set(self, colors: list[str]) -> None:
        self.var.set(", ".join(colors))
        self._render()

    def _emit(self, colors: list[str]) -> None:
        self.var.set(", ".join(colors))
        self._render()
        self.on_change(colors)

    def _from_text(self) -> None:
        self._render()
        self.on_change(self.get())

    # -- muestras
    def _render(self) -> None:
        for child in self.row.winfo_children():
            child.destroy()
        colors = self.get()
        for i, col in enumerate(colors):
            chip = ttk.Frame(self.row, padding=(0, 0, 4, 0))
            chip.pack(side="left")
            hexcol = _swatch_color(col)
            if col.lower() == "palette":
                sw = tk.Label(chip, text="palette", font=("TkDefaultFont", 8), relief="groove", padx=3)
            else:
                sw = tk.Label(chip, width=3, bg=hexcol or "#888888", relief="solid", bd=1, cursor="hand2", text="" if hexcol else "?")
            sw.pack(side="left")
            sw.bind("<Button-1>", lambda e, k=i: self._edit(k))
            tk.Button(chip, text="×", command=lambda k=i: self._remove(k), font=("TkDefaultFont", 8), bd=0, padx=2, cursor="hand2").pack(side="left")
        ttk.Button(self.row, text="+", width=2, command=self._add).pack(side="left")
        if len(colors) > 1:
            ttk.Button(self.row, text="⇄", width=2, command=self._reverse).pack(side="left", padx=(4, 0))

    def _pick(self, initial: Optional[str]) -> Optional[str]:
        try:
            _, hexcol = colorchooser.askcolor(color=initial, parent=self)
        except tk.TclError:
            _, hexcol = colorchooser.askcolor(parent=self)
        return hexcol

    def _add(self) -> None:
        colors = self.get()
        hexcol = self._pick(_swatch_color(colors[-1]) if colors else "#ffffff")
        if hexcol:
            self._emit([c for c in colors if c.lower() != "palette"] + [hexcol])

    def _edit(self, k: int) -> None:
        colors = self.get()
        if k >= len(colors):
            return
        if colors[k].lower() == "palette":
            return
        hexcol = self._pick(_swatch_color(colors[k]))
        if hexcol:
            colors[k] = hexcol
            self._emit(colors)

    def _remove(self, k: int) -> None:
        colors = self.get()
        if k < len(colors):
            del colors[k]
            self._emit(colors)

    def _reverse(self) -> None:
        self._emit(list(reversed(self.get())))


class ModelForm(ttk.Frame):
    """Formulario generado a partir de un modelo pydantic.

    on_change(new_model) se llama con una copia validada cada vez que el usuario cambia un campo.
    """

    def __init__(self, master, model: BaseModel, on_change: Callable[[BaseModel], None], exclude: tuple[str, ...] = (), status: Optional[Callable[[str], None]] = None):
        super().__init__(master)
        self.model = model
        self.on_change = on_change
        self.status = status or (lambda s: None)
        self._color_lists: dict[str, ColorList] = {}
        self.columnconfigure(1, weight=1)
        self._build(exclude)

    def _build(self, exclude):
        from ..config import EffectBase, LayerBase

        specs = field_specs(type(self.model), exclude)
        base_cls = LayerBase if isinstance(self.model, LayerBase) else EffectBase if isinstance(self.model, EffectBase) else None
        if base_cls is not None and type(self.model) is not base_cls:
            # Primero los campos propios del tipo (colores, radio, estilo...), luego los comunes plegados en un bloque
            common_names = set(base_cls.model_fields) - {"enabled", "opacity", "blend", "trigger", "threshold", "intensity"}
            own = [sp for sp in specs if sp.name not in common_names]
            common = [sp for sp in specs if sp.name in common_names]
            row = self._build_rows(own, 0)
            box = ttk.LabelFrame(self, text="Comunes: posición, tiempo, animación, secciones", padding=4)
            box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=4, pady=(8, 4))
            box.columnconfigure(1, weight=1)
            sub = ttk.Frame(box)
            sub.pack(fill="x")
            sub.columnconfigure(1, weight=1)
            self._build_rows(common, 0, parent=sub)
            return
        self._build_rows(specs, 0)

    def _build_rows(self, specs, row: int, parent=None) -> int:
        parent = parent or self
        for spec in specs:
            value = getattr(self.model, spec.name)
            if spec.kind == "model":
                box = ttk.LabelFrame(parent, text=spec.label)
                box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
                box.columnconfigure(0, weight=1)
                sub = ModelForm(box, value, lambda m, n=spec.name: self._set_sub(n, m), status=self.status)
                sub.pack(fill="x")
                row += 1
                continue
            ttk.Label(parent, text=spec.label).grid(row=row, column=0, sticky="w", padx=(6, 8), pady=2)
            if spec.kind == "bool":
                var = tk.BooleanVar(value=bool(value))
                w = ttk.Checkbutton(parent, variable=var, command=lambda s=spec, v=var: self._commit(s, v.get()))
                w.grid(row=row, column=1, sticky="w")
            elif spec.kind == "choice":
                var = tk.StringVar(value=str(value) if value is not None else "")
                values = [str(c) for c in spec.choices] + ([""] if spec.optional else [])
                w = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=16)
                w.grid(row=row, column=1, sticky="w")
                w.bind("<<ComboboxSelected>>", lambda e, s=spec, v=var: self._commit(s, v.get()))
            elif spec.kind == "str" and spec.readonly:
                ttk.Label(parent, text=str(value), font=("TkDefaultFont", 9, "bold")).grid(row=row, column=1, sticky="w")
            elif spec.kind == "font":
                var = tk.StringVar(value=str(value) if value else "")
                w = ttk.Combobox(parent, textvariable=var, values=[""] + list(available_fonts()), width=26)
                w.grid(row=row, column=1, sticky="ew")
                w.bind("<<ComboboxSelected>>", lambda e, s=spec, v=var: self._commit(s, v.get()))
                w.bind("<Return>", lambda e, s=spec, v=var: self._commit(s, v.get()))
                w.bind("<FocusOut>", lambda e, s=spec, v=var: self._commit(s, v.get()))
                ttk.Button(parent, text="…", width=3, command=lambda s=spec, v=var: self._pick_font_file(s, v)).grid(row=row, column=2, padx=2)
            elif spec.kind == "colors":
                widget = ColorList(parent, list(value or []), lambda cols, s=spec: self._commit_colors(s, cols))
                widget.grid(row=row, column=1, columnspan=2, sticky="ew", pady=(2, 4))
                self._color_lists[spec.name] = widget
            else:
                var = tk.StringVar(value=format_value(value, spec))
                entry = ttk.Entry(parent, textvariable=var, width=28)
                entry.grid(row=row, column=1, sticky="ew")
                entry.bind("<Return>", lambda e, s=spec, v=var, w=entry: self._commit(s, v.get(), w))
                entry.bind("<FocusOut>", lambda e, s=spec, v=var, w=entry: self._commit(s, v.get(), w))
                if spec.kind == "color":
                    side = ttk.Frame(parent)
                    side.grid(row=row, column=2, padx=2)
                    swatch = tk.Label(side, width=2, bg=_swatch_color(str(value)) or "#888888", relief="solid", bd=1)
                    swatch.pack(side="left", padx=(0, 2))
                    var.trace_add("write", lambda *a, v=var, sw=swatch: sw.configure(bg=_swatch_color(v.get()) or "#888888"))
                    ttk.Button(side, text="…", width=3, command=lambda s=spec, v=var, w=entry: self._pick_color(s, v, w, replace=True)).pack(side="left")
                elif spec.kind == "file":
                    ttk.Button(parent, text="…", width=3, command=lambda s=spec, v=var, w=entry: self._pick_file(s, v, w)).grid(row=row, column=2, padx=2)
            row += 1
        return row

    def _set_sub(self, name: str, sub: BaseModel):
        try:
            self.model = replace_submodel(self.model, name, sub)
        except Exception as exc:  # noqa: BLE001
            self.status(f"Valor inválido: {exc}")
            return
        self.on_change(self.model)

    def _commit(self, spec: FieldSpec, raw: Any, widget: Optional[tk.Widget] = None):
        try:
            new_model = apply_value(self.model, spec, raw)
        except ValueError as exc:
            self.status(f"{spec.label}: {exc}")
            if widget is not None:
                widget.configure(style="Error.TEntry")
            return
        if widget is not None:
            widget.configure(style="TEntry")
        if new_model == self.model:
            return
        self.model = new_model
        self.status("")
        self.on_change(new_model)

    def _commit_colors(self, spec: FieldSpec, colors: list[str]):
        try:
            new_model = apply_value(self.model, spec, ", ".join(colors))
        except ValueError as exc:
            self.status(f"{spec.label}: {exc}")
            return
        self.status("")
        if new_model == self.model:
            return
        self.model = new_model
        self.on_change(new_model)

    def _pick_color(self, spec: FieldSpec, var: tk.StringVar, widget: tk.Widget, replace: bool):
        initial = var.get().split(",")[-1].strip() or "#ffffff"
        try:
            rgb, hexcol = colorchooser.askcolor(color=initial[:7] if initial.startswith("#") else None, parent=self)
        except tk.TclError:
            rgb, hexcol = colorchooser.askcolor(parent=self)
        if not hexcol:
            return
        if replace or not var.get().strip():
            var.set(hexcol)
        else:
            var.set(var.get().rstrip(", ") + ", " + hexcol)
        self._commit(spec, var.get(), widget)

    def _pick_font_file(self, spec: FieldSpec, var: tk.StringVar):
        path = filedialog.askopenfilename(parent=self, title="Archivo de fuente", filetypes=[("Fuentes", "*.ttf *.otf *.ttc"), ("Todos", "*.*")])
        if path:
            var.set(path)
            self._commit(spec, path)

    def _pick_file(self, spec: FieldSpec, var: tk.StringVar, widget: tk.Widget):
        if spec.name == "path":
            path = filedialog.asksaveasfilename(parent=self, defaultextension=".mp4", filetypes=[("Video MP4", "*.mp4"), ("Todos", "*.*")])
        elif spec.name == "image":
            path = filedialog.askopenfilename(parent=self, filetypes=IMAGE_TYPES)
        elif spec.name == "video":
            path = filedialog.askopenfilename(parent=self, filetypes=VIDEO_TYPES)
        else:
            path = filedialog.askopenfilename(parent=self, filetypes=IMAGE_TYPES + AUDIO_TYPES)
        if path:
            var.set(path)
            self._commit(spec, path, widget)


# --------------------------------------------------------------------------- aplicación


class App(tk.Tk):
    def __init__(self, project_path: Optional[str] = None):
        super().__init__()
        self.title("musicviz — Music Visualizer")
        self.geometry("1380x900")
        self.minsize(1100, 700)
        style = ttk.Style(self)
        style.configure("Error.TEntry", fieldbackground="#ffd6d6")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))

        self.project: ProjectConfig = load_preset("trap_nation", "", "")
        self.project_path: Optional[Path] = None
        self.dirty = False
        self._features: Optional[AudioFeatures] = None
        self._features_key: Optional[str] = None
        self.renderer = RenderService(on_error=lambda msg: self._queue.put(("error", msg)))
        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._preview_busy = False
        self._preview_pending = False
        self._preview_after: Optional[str] = None
        self._playing = False
        self._player_proc = None  # reproductor OpenGL en proceso aparte
        self._play_thread: Optional[threading.Thread] = None
        self._rendering = False
        self._cancel_event = threading.Event()
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._suspend_traces = False
        self._undo: list[str] = []  # historial de deshacer/rehacer: instantáneas JSON del proyecto
        self._redo: list[str] = []
        self._snapshot = self.project.model_dump_json()
        self._autosave_after: Optional[str] = None
        self._settle_after: Optional[str] = None

        self._build_menu()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(60, self._poll_queue)
        if project_path:
            self._open_project(Path(project_path))
        else:
            self._refresh_all()
        self.after(300, self._offer_recovery)

    # ------------------------------------------------------------------ construcción UI
    def _build_menu(self):
        menubar = tk.Menu(self)
        m = tk.Menu(menubar, tearoff=0)
        m.add_command(label="Nuevo desde preset…", command=self._new_from_preset, accelerator="Ctrl+N")
        m.add_command(label="Abrir proyecto…", command=self._open_dialog, accelerator="Ctrl+O")
        self.recent_menu = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Recientes", menu=self.recent_menu)
        self._refresh_recent_menu()
        m.add_command(label="Guardar", command=self._save, accelerator="Ctrl+S")
        m.add_command(label="Guardar como…", command=self._save_as)
        m.add_separator()
        m.add_command(label="Salir", command=self._on_close)
        menubar.add_cascade(label="Archivo", menu=m)
        self.edit_menu = tk.Menu(menubar, tearoff=0)
        self.edit_menu.add_command(label="Deshacer", command=self._undo_cmd, accelerator="Ctrl+Z", state="disabled")
        self.edit_menu.add_command(label="Rehacer", command=self._redo_cmd, accelerator="Ctrl+Y", state="disabled")
        menubar.add_cascade(label="Editar", menu=self.edit_menu)
        h = tk.Menu(menubar, tearoff=0)
        h.add_command(label="Comprobar entorno (ffmpeg / NVENC)", command=self._check_env)
        h.add_command(label="Acerca de", command=lambda: messagebox.showinfo("musicviz", "musicviz — generador de videos music visualizer.\nEdita el proyecto a la izquierda, mira la vista previa a la derecha y pulsa Renderizar."))
        menubar.add_cascade(label="Ayuda", menu=h)
        self.config(menu=menubar)
        self.bind_all("<Control-n>", lambda e: self._new_from_preset())
        self.bind_all("<Control-o>", lambda e: self._open_dialog())
        self.bind_all("<Control-s>", lambda e: self._save())
        self.bind_all("<Control-z>", self._undo_cmd)
        self.bind_all("<Control-y>", self._redo_cmd)
        self.bind_all("<Control-Shift-Z>", self._redo_cmd)

    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill="x")
        ttk.Label(top, text="Audio:").grid(row=0, column=0, sticky="w")
        self.audio_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.audio_var, width=60).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Elegir…", command=self._pick_audio).grid(row=0, column=2)
        ttk.Label(top, text="Salida:").grid(row=0, column=3, sticky="w", padx=(16, 0))
        self.output_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.output_var, width=40).grid(row=0, column=4, sticky="ew", padx=4)
        ttk.Button(top, text="Elegir…", command=self._pick_output).grid(row=0, column=5)
        ttk.Label(top, text="Preset:").grid(row=0, column=6, padx=(16, 0))
        self.preset_var = tk.StringVar(value="trap_nation")
        ttk.Combobox(top, textvariable=self.preset_var, values=preset_names(), state="readonly", width=14).grid(row=0, column=7, padx=4)
        ttk.Button(top, text="Aplicar preset", command=self._apply_preset).grid(row=0, column=8)
        top.columnconfigure(1, weight=2)
        top.columnconfigure(4, weight=1)
        self.audio_var.trace_add("write", lambda *a: self._on_paths_changed())
        self.output_var.trace_add("write", lambda *a: self._on_paths_changed())

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        left = ttk.Frame(paned)
        paned.add(left, weight=0)
        self.nb = ttk.Notebook(left, width=520)
        self.nb.pack(fill="both", expand=True)
        self.tab_project = ScrollFrame(self.nb)
        self.tab_layers = ttk.Frame(self.nb)
        self.tab_effects = ttk.Frame(self.nb)
        self.tab_sections = ttk.Frame(self.nb)
        self.nb.add(self.tab_project, text="Proyecto")
        self.nb.add(self.tab_layers, text="Capas")
        self.nb.add(self.tab_effects, text="Efectos")
        self.nb.add(self.tab_sections, text="Secciones")
        self._build_list_tab(self.tab_layers, "layers")
        self._build_list_tab(self.tab_effects, "effects")
        self._build_list_tab(self.tab_sections, "sections")

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.preview = tk.Canvas(right, bg="#111118", highlightthickness=0, cursor="crosshair")
        self.preview.pack(fill="both", expand=True)
        self.preview.create_text(20, 20, anchor="nw", text="Vista previa\n\nElige un archivo de audio para empezar", fill="#aaa", font=("TkDefaultFont", 12), tags="hint")
        self._disp = None  # (ox, oy, dw, dh): rectángulo donde se dibuja la imagen
        self._sel_layer: Optional[int] = None  # capa seleccionada en la vista previa (None = fondo si _sel_bg)
        self._sel_bg = False
        self._pdrag: Optional[dict] = None
        self._quick_preview = False
        self.preview.bind("<Configure>", lambda e: self._redraw_preview())
        self.preview.bind("<ButtonPress-1>", self._pv_press)
        self.preview.bind("<B1-Motion>", self._pv_drag)
        self.preview.bind("<ButtonRelease-1>", self._pv_release)
        self.preview.bind("<MouseWheel>", self._pv_wheel)
        self.preview.bind("<Button-4>", self._pv_wheel)
        self.preview.bind("<Button-5>", self._pv_wheel)
        self.preview.bind("<Shift-MouseWheel>", lambda e: self._pv_wheel(e, shift=True))
        self.preview.bind("<Shift-Button-4>", lambda e: self._pv_wheel(e, shift=True))
        self.preview.bind("<Shift-Button-5>", lambda e: self._pv_wheel(e, shift=True))
        for key in ("<Left>", "<Right>", "<Up>", "<Down>", "<Shift-Left>", "<Shift-Right>", "<Shift-Up>", "<Shift-Down>"):
            self.preview.bind(key, self._pv_arrow)
        self.preview.bind("<Escape>", lambda e: self._pv_select(None, False))
        self.preview.bind("<Delete>", lambda e: self._pv_delete_selected())
        ctl = ttk.Frame(right, padding=(0, 6))
        ctl.pack(fill="x")
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(ctl, from_=0.0, to=1.0, variable=self.time_var, command=lambda v: self._on_time_drag())
        self.time_scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.time_label = ttk.Label(ctl, text="0:00 / 0:00", width=20)
        self.time_label.pack(side="left")
        self.play_btn = ttk.Button(ctl, text="▶ Reproducir", command=self._toggle_play, width=14)
        self.play_btn.pack(side="left", padx=4)
        self.window_btn = ttk.Button(ctl, text="⧉ Ventana GL", command=self._toggle_player, width=13)
        self.window_btn.pack(side="left", padx=(0, 4))
        ttk.Button(ctl, text="Actualizar", command=lambda: self._request_preview(force=True)).pack(side="left", padx=4)
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Auto", variable=self.auto_var).pack(side="left")
        ttk.Label(ctl, text="Calidad:").pack(side="left", padx=(12, 2))
        self.scale_var = tk.StringVar(value="Media (540p)")
        cb = ttk.Combobox(ctl, textvariable=self.scale_var, values=list(PREVIEW_SCALES), state="readonly", width=13)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._request_preview(force=True))

        tl_bar = ttk.Frame(right)
        tl_bar.pack(fill="x")
        self.timeline_visible = tk.BooleanVar(value=True)
        self.timeline_btn = ttk.Checkbutton(tl_bar, text="Timeline", variable=self.timeline_visible, command=self._toggle_timeline, style="Toolbutton")
        self.timeline_btn.pack(side="left")
        ttk.Label(tl_bar, text="Vista previa: clic selecciona · arrastrar mueve · rueda tamaño (Shift: giro) · flechas ajustan · fondo: arrastrar encuadre, rueda zoom", foreground="#777", font=("TkDefaultFont", 8)).pack(side="left", padx=10)
        self.timeline = Timeline(
            right, on_seek=self._tl_seek, on_select=self._tl_select, on_change=self._tl_change, on_section_change=self._tl_section_change,
            on_key_move=self._key_move, on_key_edit=self._key_edit, on_key_add=self._key_add, on_key_delete=self._key_delete,
        )
        self.timeline.pack(fill="x", pady=(2, 0))

        bottom = ttk.Frame(self, padding=(8, 4))
        bottom.pack(fill="x")
        self.render_btn = ttk.Button(bottom, text="🎬 Renderizar video", style="Accent.TButton", command=self._start_render)
        self.render_btn.pack(side="left")
        self.cancel_btn = ttk.Button(bottom, text="✖ Cancelar", command=self._cancel_render, state="disabled")
        self.cancel_btn.pack(side="left", padx=(4, 0))
        ttk.Label(bottom, text="Desde (s):").pack(side="left", padx=(16, 2))
        self.r_start = tk.StringVar(value="0")
        ttk.Entry(bottom, textvariable=self.r_start, width=7).pack(side="left")
        ttk.Label(bottom, text="Duración (s, vacío = todo):").pack(side="left", padx=(8, 2))
        self.r_duration = tk.StringVar(value="")
        ttk.Entry(bottom, textvariable=self.r_duration, width=7).pack(side="left")
        ttk.Label(bottom, text="Escala:").pack(side="left", padx=(8, 2))
        self.r_scale = tk.StringVar(value="1.0")
        ttk.Combobox(bottom, textvariable=self.r_scale, values=["0.25", "0.5", "0.75", "1.0"], state="readonly", width=5).pack(side="left")
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=320)
        self.progress.pack(side="left", padx=12, fill="x", expand=True)
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken", padding=(6, 2)).pack(fill="x", side="bottom")

    def _build_list_tab(self, tab: ttk.Frame, kind: str):
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        bar = ttk.Frame(tab, padding=4)
        bar.grid(row=0, column=0, sticky="ew")
        if kind == "sections":
            self.sections_enabled_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(bar, text="Usar secciones", variable=self.sections_enabled_var, command=self._toggle_sections).pack(side="left", padx=(0, 10))
            ttk.Button(bar, text="＋ Añadir sección", command=lambda: self._add_item("sections", SectionConfig)).pack(side="left")
            ttk.Button(bar, text="Detectar automáticamente", command=self._detect_sections).pack(side="left", padx=2)
            ttk.Button(bar, text="Modo auto", command=self._set_sections_auto).pack(side="left", padx=2)
        else:
            add_btn = ttk.Menubutton(bar, text="＋ Añadir")
            menu = tk.Menu(add_btn, tearoff=0)
            union = LayerConfig if kind == "layers" else EffectConfig
            for cls in get_args_of_union(union):
                tname = get_literal_default(cls)
                menu.add_command(label=tname, command=lambda c=cls, k=kind: self._add_item(k, c))
            add_btn["menu"] = menu
            add_btn.pack(side="left")
        ttk.Button(bar, text="Duplicar", command=lambda k=kind: self._dup_item(k)).pack(side="left", padx=2)
        ttk.Button(bar, text="Eliminar", command=lambda k=kind: self._del_item(k)).pack(side="left", padx=2)
        ttk.Button(bar, text="▲", width=3, command=lambda k=kind: self._move_item(k, -1)).pack(side="left", padx=2)
        ttk.Button(bar, text="▼", width=3, command=lambda k=kind: self._move_item(k, 1)).pack(side="left", padx=2)
        body = ttk.Panedwindow(tab, orient="vertical")
        body.grid(row=1, column=0, sticky="nsew")
        lb = tk.Listbox(body, height=7, exportselection=False, activestyle="none")
        lb.bind("<<ListboxSelect>>", lambda e, k=kind: self._show_item_form(k))
        body.add(lb, weight=0)
        form = ScrollFrame(body)
        body.add(form, weight=1)
        setattr(self, f"{kind}_list", lb)
        setattr(self, f"{kind}_form", form)
        if kind == "sections":
            ttk.Label(tab, text="Las capas con colores = palette siguen la paleta de la sección. Pon start/end en segundos; end vacío = hasta la siguiente.", wraplength=480, foreground="#555", padding=(6, 2)).grid(row=2, column=0, sticky="ew")

    # ------------------------------------------------------------------ listas (capas / efectos / secciones)
    def _items(self, kind: str) -> list:
        items = getattr(self.project, kind)
        if isinstance(items, str):  # sections: "auto"
            return []
        return items

    def _item_label(self, kind: str, i: int, item) -> str:
        if kind == "sections":
            end = f"{item.end:.1f}s" if item.end is not None else "…"
            return f"{i + 1}. {item.name or 'sección'}   {item.start:.1f}s → {end}   ×{item.intensity:g}"
        extra = ""
        if kind == "effects":
            extra = f"  ({item.trigger})"
        elif hasattr(item, "text"):
            extra = f"  “{item.text[:18]}”"
        if getattr(item, "sections", None):
            extra += f"  [{', '.join(item.sections)}]"
        flag = "" if item.enabled else "  [off]"
        return f"{i + 1}. {item.type}{extra}{flag}"

    def _detect_sections(self):
        if self._features is None:
            self._set_status("Espera a que termine el análisis del audio (o elige un archivo de audio)")
            return
        from ..render.sections import detect_sections

        found = detect_sections(self._features, self.project.auto_sections)
        if self.project.sections and self.project.sections != "auto":
            if not messagebox.askyesno("Detectar secciones", "Se reemplazarán las secciones actuales por las detectadas. ¿Continuar?", parent=self):
                return
        self.project.sections = found
        self._refresh_list("sections", select=0)
        self._changed()
        self._set_status(f"{len(found)} secciones detectadas; edita paletas e intensidades a tu gusto")

    def _toggle_sections(self):
        self.project.sections_enabled = bool(self.sections_enabled_var.get())
        self._refresh_list("sections")
        self._changed()
        self._set_status("Secciones activadas" if self.project.sections_enabled else "Secciones desactivadas: el diseño no cambia a lo largo de la canción (se conservan guardadas)")

    def _set_sections_auto(self):
        self.project.sections = "auto"
        self._refresh_list("sections")
        self._changed()
        self._set_status("Secciones en modo automático: se detectan al renderizar (ajustes en auto_sections del YAML)")

    # ------------------------------------------------------------------ estado / refresco
    def _set_status(self, text: str):
        self.status_var.set(text or "Listo")

    def _mark_dirty(self):
        self.dirty = True
        self._update_title()
        self._record_history()
        self._schedule_autosave()

    # ------------------------------------------------------------------ deshacer / rehacer
    def _record_history(self):
        """Guarda el estado anterior del proyecto si ha cambiado (una entrada por cambio, no por arrastre)."""
        cur = self.project.model_dump_json()
        if cur == self._snapshot:
            return
        self._undo.append(self._snapshot)
        del self._undo[:-100]
        self._redo.clear()
        self._snapshot = cur
        self._update_edit_menu()

    def _reset_history(self):
        self._undo.clear()
        self._redo.clear()
        self._snapshot = self.project.model_dump_json()
        self._update_edit_menu()

    def _update_edit_menu(self):
        if hasattr(self, "edit_menu"):
            self.edit_menu.entryconfigure(0, state="normal" if self._undo else "disabled")
            self.edit_menu.entryconfigure(1, state="normal" if self._redo else "disabled")

    def _undo_cmd(self, event=None):
        if isinstance(getattr(event, "widget", None), tk.Text):
            return  # los cuadros de texto tienen su propio deshacer
        if not self._undo:
            self._set_status("Nada que deshacer")
            return "break"
        self._redo.append(self.project.model_dump_json())
        self._restore_snapshot(self._undo.pop(), "Deshecho")
        return "break"

    def _redo_cmd(self, event=None):
        if isinstance(getattr(event, "widget", None), tk.Text):
            return
        if not self._redo:
            self._set_status("Nada que rehacer")
            return "break"
        self._undo.append(self.project.model_dump_json())
        self._restore_snapshot(self._redo.pop(), "Rehecho")
        return "break"

    def _restore_snapshot(self, snapshot: str, label: str):
        if self._playing:
            self._stop_play()
        self.project = ProjectConfig.model_validate_json(snapshot)
        self._snapshot = snapshot
        self.dirty = True
        self._update_edit_menu()
        self._refresh_all()
        self._schedule_autosave()
        self._set_status(f"{label} ({len(self._undo)} pasos atrás, {len(self._redo)} adelante)")

    # ------------------------------------------------------------------ autoguardado
    def _schedule_autosave(self, delay_ms: int = 3000):
        if self._autosave_after is not None:
            self.after_cancel(self._autosave_after)
        self._autosave_after = self.after(delay_ms, self._autosave_now)

    def _autosave_now(self):
        self._autosave_after = None
        if not self.dirty:
            return
        try:
            autosave.write(self.project, self.project_path)
        except Exception as exc:  # noqa: BLE001 - nunca debe molestar al usuario
            print(f"autoguardado: {exc}", file=sys.stderr)

    def _offer_recovery(self):
        """Al arrancar: si quedó un proyecto sin guardar de la sesión anterior, ofrece recuperarlo."""
        info = autosave.pending()
        if info is None:
            return
        if not self._ask_recover(info):
            autosave.clear()
            return
        try:
            project = autosave.load()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Recuperar proyecto", f"No se pudo leer el autoguardado:\n{exc}", parent=self)
            autosave.clear()
            return
        self.project = project
        self.project_path = Path(info["original"]) if info.get("original") else None
        self.dirty = True
        self._reset_history()
        self._refresh_all()
        self._set_status("Proyecto recuperado del autoguardado (sin guardar todavía)")

    def _ask_recover(self, info: dict) -> bool:
        when = time.strftime("%d/%m/%Y %H:%M", time.localtime(info.get("time", 0)))
        origin = f"\n\nVenía de: {info['original']}" if info.get("original") else "\n\n(Era un proyecto nuevo sin guardar)"
        return messagebox.askyesno("Recuperar proyecto", f"Hay un proyecto sin guardar de la sesión anterior ({when}).{origin}\n\n¿Quieres recuperarlo?", parent=self)

    def _update_title(self):
        name = self.project_path.name if self.project_path else "sin guardar"
        self.title(f"musicviz — {name}{' *' if self.dirty else ''}")

    def _refresh_all(self):
        self._suspend_traces = True
        try:
            self.audio_var.set(self.project.audio.file or "")
            self.output_var.set(self.project.output.path or "")
        finally:
            self._suspend_traces = False
        self._build_project_tab()
        self._refresh_list("layers")
        self._refresh_list("effects")
        self._refresh_list("sections")
        self._update_title()
        self._refresh_timeline()
        self._request_preview()

    def _build_project_tab(self):
        self.tab_project.clear()
        inner = self.tab_project.inner
        for title, model, exclude, setter in (
            ("Salida", self.project.output, ("path",), self._set_output),
            ("Audio / análisis", self.project.audio, ("file",), self._set_audio),
            ("Fondo", self.project.background, (), self._set_background),
        ):
            box = ttk.LabelFrame(inner, text=title, padding=4)
            box.pack(fill="x", padx=4, pady=4)
            ModelForm(box, model, setter, exclude=exclude, status=self._set_status).pack(fill="x")

    def _set_output(self, m):
        self.project.output = m
        self._changed()

    def _set_audio(self, m):
        self.project.audio = m
        self._changed()

    def _set_background(self, m):
        self.project.background = m
        self._changed()

    def _changed(self):
        self._mark_dirty()
        self._refresh_timeline()
        self._request_preview()

    def _on_paths_changed(self):
        if self._suspend_traces:
            return
        audio = self.audio_var.get().strip()
        out = self.output_var.get().strip()
        if audio != (self.project.audio.file or "") or out != (self.project.output.path or ""):
            self.project.audio.file = audio
            self.project.output.path = out
            self._mark_dirty()
            self._request_preview()

    def _refresh_list(self, kind: str, select: Optional[int] = None):
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        items = self._items(kind)
        cur = select if select is not None else (lb.curselection()[0] if lb.curselection() else 0)
        lb.delete(0, "end")
        if kind == "sections":
            self.sections_enabled_var.set(self.project.sections_enabled)
            if not self.project.sections_enabled:
                lb.insert("end", "(desactivadas: marca “Usar secciones” para aplicarlas)")
            elif self.project.sections == "auto":
                lb.insert("end", "(automático: se detectan al renderizar; pulsa “Detectar” para editarlas)")
        for i, item in enumerate(items):
            lb.insert("end", self._item_label(kind, i, item))
        if items:
            cur = min(cur, len(items) - 1)
            lb.selection_set(cur)
            lb.see(cur)
        self._show_item_form(kind)

    def _show_item_form(self, kind: str):
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        form: ScrollFrame = getattr(self, f"{kind}_form")
        form.clear()
        if kind in ("layers", "effects") and hasattr(self, "timeline") and lb.curselection():
            if self.timeline.selected != (kind, lb.curselection()[0]):
                self.timeline.set_selected(kind, lb.curselection()[0])
            if kind == "layers" and hasattr(self, "preview"):
                self._sel_layer, self._sel_bg = lb.curselection()[0], False
                self._draw_overlay()
        items = self._items(kind)
        if not lb.curselection() or not items:
            ttk.Label(form.inner, text="Añade un elemento con el botón ＋", padding=10).pack()
            return
        idx = lb.curselection()[0]
        ModelForm(form.inner, items[idx], lambda m, k=kind, i=idx: self._set_item(k, i, m), status=self._set_status).pack(fill="x", padx=4, pady=4)
        if kind == "layers":
            self._build_keyframe_panel(form.inner, idx)

    # ------------------------------------------------------------------ keyframes
    def _build_keyframe_panel(self, parent, idx: int):
        box = ttk.LabelFrame(parent, text="Keyframes (posición, escala, opacidad, rotación)", padding=4)
        box.pack(fill="x", padx=4, pady=(0, 8))
        box.columnconfigure(1, weight=1)
        layer = self.project.layers[idx]
        labels = {"position": "Posición", "scale": "Escala", "opacity": "Opacidad", "rotation": "Rotación"}
        for r, prop in enumerate(KEYFRAME_PROPS):
            keys = getattr(layer, f"{prop}_keys")
            ttk.Label(box, text=labels[prop]).grid(row=r, column=0, sticky="w", padx=(2, 8))
            ttk.Label(box, text=f"{len(keys)} keyframe{'s' if len(keys) != 1 else ''}", foreground="#555").grid(row=r, column=1, sticky="w")
            ttk.Button(box, text="＋ en el cursor", command=lambda p=prop, i=idx: self._key_add(i, p, round(float(self.time_var.get()), 2))).grid(row=r, column=2, padx=2)
            ttk.Button(box, text="Borrar todos", command=lambda p=prop, i=idx: self._keys_clear(i, p), state="normal" if keys else "disabled").grid(row=r, column=3, padx=2)
        ttk.Label(box, text="En la timeline: arrastra un rombo para moverlo, doble clic para editar valor y curva, clic derecho para borrarlo, doble clic en la fila para añadir.", wraplength=460, foreground="#777", font=("TkDefaultFont", 8)).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _set_keys(self, idx: int, prop: str, keys: list):
        layer = self.project.layers[idx]
        self.project.layers[idx] = layer.model_copy(update={f"{prop}_keys": sorted(keys, key=lambda k: k.time)})
        self._refresh_list("layers", select=idx)
        self._changed()

    def _key_add(self, idx: int, prop: str, t: float):
        layer = self.project.layers[idx]
        value = current_value(layer, prop, t)
        key = PointKey(time=t, value=value) if prop == "position" else ScalarKey(time=t, value=value)
        result = KeyDialog(self, prop, key, allow_delete=False).result
        if result is None:
            return
        keys = list(getattr(layer, f"{prop}_keys")) + [result]
        self._set_keys(idx, prop, keys)
        self.nb.select(1)

    def _key_edit(self, idx: int, prop: str, k: int):
        layer = self.project.layers[idx]
        keys = list(getattr(layer, f"{prop}_keys"))
        if k >= len(keys):
            return
        dlg = KeyDialog(self, prop, keys[k], allow_delete=True)
        if dlg.deleted:
            del keys[k]
        elif dlg.result is not None:
            keys[k] = dlg.result
        else:
            return
        self._set_keys(idx, prop, keys)

    def _key_move(self, idx: int, prop: str, k: int, t: float):
        layer = self.project.layers[idx]
        keys = list(getattr(layer, f"{prop}_keys"))
        if k < len(keys):
            keys[k] = keys[k].model_copy(update={"time": max(t, 0.0)})
            self._set_keys(idx, prop, keys)

    def _key_delete(self, idx: int, prop: str, k: int):
        layer = self.project.layers[idx]
        keys = list(getattr(layer, f"{prop}_keys"))
        if k < len(keys):
            del keys[k]
            self._set_keys(idx, prop, keys)

    def _keys_clear(self, idx: int, prop: str):
        if messagebox.askyesno("Borrar keyframes", f"¿Borrar todos los keyframes de {prop} de esta capa?", parent=self):
            self._set_keys(idx, prop, [])

    def _set_item(self, kind: str, idx: int, model):
        items = self._items(kind)
        items[idx] = model
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        lb.delete(idx)
        lb.insert(idx, self._item_label(kind, idx, model))
        lb.selection_set(idx)
        self._changed()

    def _selected(self, kind: str) -> Optional[int]:
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        return lb.curselection()[0] if lb.curselection() else None

    def _add_item(self, kind: str, cls):
        if kind == "sections":
            if isinstance(self.project.sections, str):
                self.project.sections = []
            self.project.sections.append(SectionConfig(name=f"seccion_{len(self.project.sections) + 1}", start=round(float(self.time_var.get()), 1)))
            self._refresh_list("sections", select=len(self.project.sections) - 1)
            self._changed()
            return
        tname = get_literal_default(cls)
        data = {"type": tname}
        if tname == "image":
            path = filedialog.askopenfilename(parent=self, title="Imagen / logo", filetypes=IMAGE_TYPES)
            if not path:
                return
            data["file"] = path
        if tname == "text":
            data["text"] = "Texto"
        item = cls.model_validate(data)
        items = self._items(kind)
        items.append(item)
        self._refresh_list(kind, select=len(items) - 1)
        self._changed()

    def _dup_item(self, kind: str):
        idx = self._selected(kind)
        if idx is None:
            return
        items = self._items(kind)
        if not items:
            return
        items.insert(idx + 1, items[idx].model_copy(deep=True))
        self._refresh_list(kind, select=idx + 1)
        self._changed()

    def _del_item(self, kind: str):
        idx = self._selected(kind)
        if idx is None:
            return
        items = self._items(kind)
        if not items:
            return
        del items[idx]
        if kind == "layers" and self._sel_layer is not None and self._sel_layer >= len(items):
            self._sel_layer = None
        self._refresh_list(kind, select=max(idx - 1, 0))
        self._changed()

    def _move_item(self, kind: str, delta: int):
        idx = self._selected(kind)
        if idx is None:
            return
        items = self._items(kind)
        if not items:
            return
        j = idx + delta
        if 0 <= j < len(items):
            items[idx], items[j] = items[j], items[idx]
            self._refresh_list(kind, select=j)
            self._changed()

    # ------------------------------------------------------------------ timeline
    def _toggle_timeline(self):
        if self.timeline_visible.get():
            self.timeline.pack(fill="x", pady=(2, 0))
            self._refresh_timeline()
        else:
            self.timeline.pack_forget()

    def _refresh_timeline(self):
        if self.timeline_visible.get():
            self.timeline.set_data(self.project, self._features)
            self.timeline.set_time(float(self.time_var.get()))

    def _tl_seek(self, t: float):
        self.time_var.set(t)
        self._update_time_label()
        if not self._playing:
            self._request_preview(force=True, quick=True)

    def _tl_select(self, kind: str, idx: int):
        self.nb.select(1 if kind == "layers" else 2)
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        lb.selection_clear(0, "end")
        lb.selection_set(idx)
        lb.see(idx)
        self._show_item_form(kind)
        self.timeline.set_selected(kind, idx)

    def _tl_change(self, kind: str, idx: int, start, end):
        items = self._items(kind)
        if idx >= len(items):
            return
        try:
            items[idx] = items[idx].model_copy(update={"start": start, "end": end})
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Tiempo inválido: {exc}")
            return
        self._refresh_list(kind, select=idx)
        self._changed()

    def _tl_section_change(self, idx: int, new_start: float):
        secs = self.project.sections
        if isinstance(secs, str) or idx <= 0 or idx >= len(secs):
            return
        secs[idx] = secs[idx].model_copy(update={"start": new_start})
        if secs[idx - 1].end is not None:
            secs[idx - 1] = secs[idx - 1].model_copy(update={"end": new_start})
        self._refresh_list("sections", select=idx)
        self._changed()

    # ------------------------------------------------------------------ recientes
    def _refresh_recent_menu(self):
        menu = self.recent_menu
        menu.delete(0, "end")
        items = load_recent()
        if not items:
            menu.add_command(label="(vacío)", state="disabled")
        for p in items:
            label = p.name if len(str(p)) < 60 else p.name
            menu.add_command(label=f"{label}   —   {p.parent}", command=lambda q=p: self._open_recent(q))
        menu.add_separator()
        menu.add_command(label="Limpiar lista", command=self._clear_recent, state="normal" if items else "disabled")

    def _open_recent(self, path: Path):
        if not path.exists():
            if messagebox.askyesno("No encontrado", f"El proyecto ya no existe:\n{path}\n\n¿Quitarlo de la lista?", parent=self):
                remove_recent(path)
                self._refresh_recent_menu()
            return
        if not self._confirm_discard():
            return
        self._open_project(path)

    def _clear_recent(self):
        clear_recent()
        self._refresh_recent_menu()

    def _remember(self, path: Path):
        add_recent(path)
        self._refresh_recent_menu()

    # ------------------------------------------------------------------ archivo
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        ans = messagebox.askyesnocancel("Cambios sin guardar", "¿Guardar los cambios del proyecto?", parent=self)
        if ans is None:
            return False
        if ans:
            return self._save()
        return True

    def _new_from_preset(self):
        if not self._confirm_discard():
            return
        audio, out = self.audio_var.get().strip(), self.output_var.get().strip()
        self.project = load_preset(self.preset_var.get(), audio, out)
        self.project_path = None
        self.dirty = False
        self._reset_history()
        autosave.clear()
        self._refresh_all()

    def _apply_preset(self):
        if self.project.layers or self.project.effects:
            if not messagebox.askyesno("Aplicar preset", "Se reemplazarán las capas y efectos actuales por los del preset. ¿Continuar?", parent=self):
                return
        audio, out = self.audio_var.get().strip(), self.output_var.get().strip()
        new = load_preset(self.preset_var.get(), audio, out)
        new.output.width, new.output.height, new.output.fps = self.project.output.width, self.project.output.height, self.project.output.fps
        self.project = new
        self._mark_dirty()
        self._refresh_all()

    def _open_dialog(self):
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(parent=self, filetypes=[("Proyecto musicviz", "*.yaml *.yml")])
        if path:
            self._open_project(Path(path))

    def _open_project(self, path: Path):
        try:
            self.project = ProjectConfig.load(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error al abrir", str(exc), parent=self)
            return
        self.project_path = path
        self.dirty = False
        self._reset_history()
        autosave.clear()
        self._remember(path)
        self._refresh_all()

    def _save(self) -> bool:
        if self.project_path is None:
            return self._save_as()
        try:
            self.project.save(self.project_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error al guardar", str(exc), parent=self)
            return False
        self.dirty = False
        self._update_title()
        autosave.clear()
        self._remember(self.project_path)
        self._set_status(f"Guardado en {self.project_path}")
        return True

    def _save_as(self) -> bool:
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".yaml", filetypes=[("Proyecto musicviz", "*.yaml")], initialfile=(self.project_path.name if self.project_path else "proyecto.yaml"))
        if not path:
            return False
        self.project_path = Path(path)
        return self._save()

    def _pick_audio(self):
        path = filedialog.askopenfilename(parent=self, title="Archivo de audio", filetypes=AUDIO_TYPES)
        if path:
            self.audio_var.set(path)
            if not self.output_var.get().strip():
                self.output_var.set(str(Path(path).with_suffix("")) + "_visualizer.mp4")

    def _pick_output(self):
        path = filedialog.asksaveasfilename(parent=self, title="Video de salida", defaultextension=".mp4", filetypes=[("Video MP4", "*.mp4")])
        if path:
            self.output_var.set(path)

    # ------------------------------------------------------------------ análisis / preview
    def _features_cache_key(self) -> str:
        return self.project.audio.model_dump_json() + f"|{self.project.output.fps}"

    def _audio_ready(self) -> bool:
        f = self.project.audio.file
        return bool(f) and Path(f).is_file()

    def _request_preview(self, force: bool = False, quick: bool = False):
        """Pide una vista previa. Con `quick` se renderiza a baja resolución (arrastres) y, al parar
        de arrastrar medio segundo, se vuelve a pedir a la calidad elegida."""
        if self._playing:
            return
        if not force and not self.auto_var.get():
            return
        if quick:
            self._quick_preview = True
            if self._settle_after is not None:
                self.after_cancel(self._settle_after)
            self._settle_after = self.after(500, self._settle_preview)
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
        self._preview_after = self.after(30 if quick else (150 if force else 600), self._run_preview)

    def _settle_preview(self):
        self._settle_after = None
        if self._pdrag is not None:
            return  # sigue arrastrando una capa: la liberación ya pide la vista completa
        self._quick_preview = False
        self._request_preview(force=True)

    def _run_preview(self):
        self._preview_after = None
        self._update_time_label()
        if not self._audio_ready():
            self._photo = None
            self.preview.delete("img", "overlay")
            self.preview.itemconfigure("hint", state="normal")
            return
        project = self.project.model_copy(deep=True)
        t = float(self.time_var.get())
        scale = min(PREVIEW_SCALES[self.scale_var.get()], 0.3) if self._quick_preview else PREVIEW_SCALES[self.scale_var.get()]
        self._preview_busy = True

        def job(svc: RenderService):
            try:
                feats = self._service_features(svc, project)
                scene = svc.scene_for(project, scale)
                img = scene.render(int(round(t * feats.fps)))
                self._queue.put(("preview", img))
            finally:
                self._queue.put(("preview_done", None))

        self.renderer.submit(job, key="preview")  # sólo importa la última vista previa pedida

    def _service_features(self, svc: RenderService, project: ProjectConfig) -> AudioFeatures:
        """Características del audio desde el hilo de render (analiza o usa la caché si hace falta)."""
        return svc.features_for(
            project,
            on_analyzing=lambda: self._queue.put(("status", "Analizando audio…")),
            on_ready=lambda feats: self._queue.put(("features", feats)),
        )

    def _get_features_for(self, project: ProjectConfig) -> AudioFeatures:
        """Características para hilos ajenos al de render (exportación): reutiliza las del servicio o analiza."""
        feats = self.renderer.features
        key = project.audio.model_dump_json() + f"|{project.output.fps}"
        if feats is not None and self._features_key == key:
            return feats
        from ..render.engine import analyze_project

        self._queue.put(("status", "Analizando audio…"))
        feats = analyze_project(project)
        self._queue.put(("features", feats))
        return feats

    def _show_image(self, img: np.ndarray):
        if img.shape[2] == 4:
            img = over_checkerboard(img)
        self._last_frame = img
        self._redraw_preview()

    def _redraw_preview(self):
        img = getattr(self, "_last_frame", None)
        if img is None:
            return
        cw = max(self.preview.winfo_width(), 320)
        ch = max(self.preview.winfo_height(), 180)
        h, w = img.shape[:2]
        k = min(cw / w, ch / h)
        dw, dh = max(int(w * k), 1), max(int(h * k), 1)
        pil = Image.fromarray(img)
        if (dw, dh) != (w, h):
            pil = pil.resize((dw, dh), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(pil)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self._disp = (ox, oy, dw, dh)
        self.preview.delete("img")
        self.preview.itemconfigure("hint", state="hidden")
        self.preview.create_image(ox, oy, anchor="nw", image=self._photo, tags="img")
        self.preview.tag_lower("img")
        self._draw_overlay()

    def _draw_overlay(self):
        self.preview.delete("overlay")
        if self._disp is None:
            return
        ox, oy, dw, dh = self._disp
        if self._sel_bg:
            self.preview.create_rectangle(ox + 1, oy + 1, ox + dw - 1, oy + dh - 1, outline="#ffd166", dash=(6, 4), width=2, tags="overlay")
            self.preview.create_text(ox + 8, oy + 8, anchor="nw", text="fondo · arrastra para encuadrar · rueda: zoom", fill="#ffd166", font=("TkDefaultFont", 9, "bold"), tags="overlay")
            return
        if self._sel_layer is None or self._sel_layer >= len(self.project.layers):
            return
        layer = self.project.layers[self._sel_layer]
        poly = layer_polygon(layer, self.project, float(self.time_var.get()))
        if not poly:
            return
        pts = [(ox + px * dw, oy + py * dh) for px, py in poly]
        flat = [c for pt in pts for c in pt]
        self.preview.create_polygon(*flat, outline="#ffffff", fill="", dash=(5, 3), width=2, tags="overlay")
        for cx, cy in pts:
            self.preview.create_rectangle(cx - 3, cy - 3, cx + 3, cy + 3, fill="#ffffff", outline="", tags="overlay")
        top_x = min(p[0] for p in pts)
        top_y = min(p[1] for p in pts)
        name = layer.name or layer.type
        extra = "  (keyframe)" if layer.position_keys else ""
        self.preview.create_text(top_x, max(top_y - 4, oy + 2), anchor="sw" if top_y - 4 > oy + 12 else "nw", text=name + extra, fill="#ffffff", font=("TkDefaultFont", 9, "bold"), tags="overlay")

    # ------------------------------------------------------------------ interacción en la vista previa
    def _pv_to_rel(self, x: float, y: float) -> Optional[tuple[float, float]]:
        if self._disp is None:
            return None
        ox, oy, dw, dh = self._disp
        return ((x - ox) / dw, (y - oy) / dh)

    def _pv_select(self, idx: Optional[int], bg: bool):
        self._sel_layer, self._sel_bg = idx, bg
        if idx is not None:
            self._tl_select("layers", idx)
        elif bg:
            self.nb.select(0)
        self._draw_overlay()

    def _pv_press(self, event):
        self.preview.focus_set()
        rel = self._pv_to_rel(event.x, event.y)
        if rel is None or not self._audio_ready():
            return
        rx, ry = rel
        t = float(self.time_var.get())
        idx = None if (event.state & 0x20000) else hit_layer(self.project, t, rx, ry)  # Alt fuerza el fondo
        inside = 0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0
        if idx is None and not inside:
            self._pv_select(None, False)
            return
        self._pv_select(idx, idx is None)
        if idx is not None:
            layer = self.project.layers[idx]
            from ..layers.keyframes import current_value

            self._pdrag = {"kind": "layer", "idx": idx, "rx": rx, "ry": ry, "orig": tuple(current_value(layer, "position", t)), "moved": False, "t": t}
        else:
            bg = self.project.background
            if bg.type in ("image", "video") and bg.image_fit == "cover":
                self._pdrag = {"kind": "bg", "rx": rx, "ry": ry, "orig": tuple(bg.focus), "moved": False}
            else:
                self._pdrag = None
                self._set_status("El fondo sólo se encuadra con tipo imagen o video en modo cover; usa la rueda para el zoom")

    def _pv_drag(self, event):
        d = self._pdrag
        rel = self._pv_to_rel(event.x, event.y)
        if not d or rel is None:
            return
        dx, dy = rel[0] - d["rx"], rel[1] - d["ry"]
        if abs(dx) < 0.002 and abs(dy) < 0.002 and not d["moved"]:
            return
        d["moved"] = True
        if d["kind"] == "layer":
            nx = min(max(d["orig"][0] + dx, -0.5), 1.5)
            ny = min(max(d["orig"][1] + dy, -0.5), 1.5)
            self._set_layer_position(d["idx"], (round(nx, 4), round(ny, 4)), d["t"], quick=True)
        else:
            # Arrastrar el fondo mueve el encuadre en sentido contrario al foco
            fx = min(max(d["orig"][0] - dx, 0.0), 1.0)
            fy = min(max(d["orig"][1] - dy, 0.0), 1.0)
            self.project.background = self.project.background.model_copy(update={"focus": (round(fx, 4), round(fy, 4))})
            self._quick_preview = True
            self._request_preview(force=True)
        self._draw_overlay()

    def _pv_release(self, event):
        d, self._pdrag = self._pdrag, None
        if not d or not d.get("moved"):
            return
        self._quick_preview = False
        if d["kind"] == "bg":
            self._build_project_tab()
        else:
            self._refresh_list("layers", select=d["idx"])
        self._changed()

    def _set_layer_position(self, idx: int, pos: tuple[float, float], t: float, quick: bool = False):
        """Mueve una capa: posición fija, o el keyframe de posición del instante actual (creándolo si no existe)."""
        layer = self.project.layers[idx]
        if layer.position_keys:
            keys = list(layer.position_keys)
            hit = next((k for k, key in enumerate(keys) if abs(key.time - t) < 0.05), None)
            if hit is None:
                keys.append(PointKey(time=round(t, 2), value=pos))
            else:
                keys[hit] = keys[hit].model_copy(update={"value": pos})
            self.project.layers[idx] = layer.model_copy(update={"position_keys": sorted(keys, key=lambda k: k.time)})
        else:
            self.project.layers[idx] = layer.model_copy(update={"position": pos})
        self._quick_preview = quick
        self._request_preview(force=True)
        if not quick:
            self._refresh_list("layers", select=idx)
            self._changed()

    def _pv_wheel(self, event, shift: bool = False):
        direction = 1 if (getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0) else -1
        factor = 1.05 if direction > 0 else 1 / 1.05
        if self._sel_bg:
            bg = self.project.background
            zoom = min(max(bg.zoom * factor, 1.0), 3.0)
            self.project.background = bg.model_copy(update={"zoom": round(zoom, 3)})
            self._build_project_tab()
            self._changed()
            self._set_status(f"Zoom del fondo: {zoom:.2f}")
            return
        if self._sel_layer is None or self._sel_layer >= len(self.project.layers):
            return
        layer = self.project.layers[self._sel_layer]
        if shift:
            field = ROTATION_FIELD.get(layer.type)
            if not field:
                self._set_status("Esta capa no tiene rotación (usa keyframes de rotación para el texto)")
                return
            value = round(getattr(layer, field) + direction * 5.0, 1)
        else:
            field = SIZE_FIELD.get(layer.type)
            if not field:
                return
            value = round(getattr(layer, field) * factor, 4)
        try:
            self.project.layers[self._sel_layer] = layer.model_copy(update={field: value})
        except Exception as exc:  # noqa: BLE001
            self._set_status(str(exc))
            return
        self._refresh_list("layers", select=self._sel_layer)
        self._changed()
        self._set_status(f"{layer.type}.{field} = {value}")

    def _pv_arrow(self, event):
        if self._sel_layer is None or self._sel_layer >= len(self.project.layers):
            return
        step_px = 10 if (event.state & 0x1) else 1
        dx = {"Left": -1, "Right": 1}.get(event.keysym, 0) * step_px / self.project.output.width
        dy = {"Up": -1, "Down": 1}.get(event.keysym, 0) * step_px / self.project.output.height
        from ..layers.keyframes import current_value

        t = float(self.time_var.get())
        px, py = current_value(self.project.layers[self._sel_layer], "position", t)
        self._set_layer_position(self._sel_layer, (round(px + dx, 4), round(py + dy, 4)), t)

    def _pv_delete_selected(self):
        if self._sel_layer is not None and self._sel_layer < len(self.project.layers):
            self.layers_list.selection_clear(0, "end")
            self.layers_list.selection_set(self._sel_layer)
            self._del_item("layers")
            self._pv_select(None, False)

    def _on_time_drag(self):
        self._update_time_label()
        if not self._playing:
            self._request_preview(force=True, quick=True)

    def _update_time_label(self):
        total = self._features.duration if self._features else 0.0
        if self.timeline_visible.get():
            self.timeline.set_time(float(self.time_var.get()))
        if hasattr(self, "preview"):
            self._draw_overlay()
        label = f"{_fmt(self.time_var.get())} / {_fmt(total)}"
        if self._features is not None and self.project.sections:
            try:
                from ..render.sections import SectionTimeline, resolve_sections

                name = SectionTimeline(resolve_sections(self.project, self._features)).state_at(float(self.time_var.get())).name
                if name:
                    label += f"  ·  {name}"
            except Exception:  # noqa: BLE001 - sección inválida a medio editar
                pass
        self.time_label.configure(text=label)

    # ------------------------------------------------------------------ reproducción
    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if not self._audio_ready():
            self._set_status("Elige un archivo de audio primero")
            return
        self._playing = True
        self.play_btn.configure(text="■ Detener")
        project = self.project.model_copy(deep=True)
        start = float(self.time_var.get())
        scale = min(PREVIEW_SCALES[self.scale_var.get()], 0.5)
        self.renderer.submit(lambda svc: self._play_job(svc, project, start, scale), key="play")

    def _stop_play(self):
        self._playing = False
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        self.play_btn.configure(text="▶ Reproducir")

    # ------------------------------------------------------------------ ventana OpenGL
    def _toggle_player(self):
        """Abre (o cierra) el reproductor OpenGL a resolución completa en un proceso aparte."""
        if self._player_proc is not None and self._player_proc.is_alive():
            self._close_player()
            return
        if not self._audio_ready():
            self._set_status("Elige un archivo de audio primero")
            return
        from ..render.gpu import gpu_available
        from ..render.player import player_available

        if not player_available():
            messagebox.showwarning("Falta glfw", "La ventana OpenGL necesita el paquete `glfw` (pip install glfw).", parent=self)
            return
        if not gpu_available():
            messagebox.showwarning("Sin OpenGL", "No hay un contexto OpenGL 3.3 disponible; usa el botón Reproducir.", parent=self)
            return
        if self._playing:
            self._stop_play()
        project = self.project.model_copy(deep=True)
        start = float(self.time_var.get())
        self.window_btn.configure(state="disabled")
        self.renderer.submit(lambda svc: self._player_job(svc, project, start), key="player")

    def _player_job(self, svc: RenderService, project: ProjectConfig, start: float):
        try:
            feats = self._service_features(svc, project)
            from ..render.player import launch_player

            proc = launch_player(project, feats, start=start)
            self._queue.put(("player_started", proc))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"Ventana OpenGL: {exc}"))
            self._queue.put(("player_ended", None))

    def _close_player(self):
        proc = self._player_proc
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        self._player_proc = None
        self.window_btn.configure(text="⧉ Ventana GL", state="normal")

    def _watch_player(self):
        proc = self._player_proc
        if proc is None:
            return
        if proc.is_alive():
            self.after(300, self._watch_player)
            return
        self._player_proc = None
        self.window_btn.configure(text="⧉ Ventana GL", state="normal")
        if proc.exitcode not in (0, None, -15):
            self._set_status(f"La ventana OpenGL se cerró con error (código {proc.exitcode}); revisa la consola.")

    def _play_job(self, svc: RenderService, project: ProjectConfig, start: float, scale: float):
        """Reproducción en el hilo de render: reutiliza la escena en caché y sincroniza con el reloj (y el audio)."""
        try:
            feats = self._service_features(svc, project)
            scene = svc.scene_for(project, scale)
            audio_ok = False
            try:
                import sounddevice as sd

                sd.play(feats.waveform[int(start * feats.sr) :], feats.sr)
                audio_ok = True
            except Exception:  # noqa: BLE001
                self._queue.put(("status", "Reproduciendo sin audio (instala `sounddevice` para oírlo)"))
            t0 = time.perf_counter()
            last_frame = -1
            while self._playing:
                t = start + (time.perf_counter() - t0)
                if t >= feats.duration:
                    break
                idx = int(t * feats.fps)
                if idx == last_frame:
                    time.sleep(0.002)
                    continue
                last_frame = idx
                img = scene.render(idx)
                self._queue.put(("play_frame", (img, t)))
                if not audio_ok:
                    time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"Reproducción: {exc}"))
            traceback.print_exc()
        finally:
            self._queue.put(("play_end", None))

    # ------------------------------------------------------------------ render
    def _start_render(self):
        if self._rendering:
            return
        if not self._audio_ready():
            messagebox.showwarning("Falta el audio", "Elige un archivo de audio válido.", parent=self)
            return
        if not self.output_var.get().strip():
            self._pick_output()
            if not self.output_var.get().strip():
                return
        try:
            start = float(self.r_start.get() or 0)
            duration = float(self.r_duration.get()) if self.r_duration.get().strip() else None
            scale = float(self.r_scale.get())
        except ValueError:
            messagebox.showwarning("Valores inválidos", "Revisa inicio, duración y escala.", parent=self)
            return
        if self._playing:
            self._stop_play()
        self._rendering = True
        self._cancel_event.clear()
        self.render_btn.configure(state="disabled", text="Renderizando…")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0, maximum=100)
        project = self.project.model_copy(deep=True)
        threading.Thread(target=self._render_worker, args=(project, start, duration, scale), daemon=True).start()

    def _cancel_render(self):
        if self._rendering and messagebox.askyesno("Cancelar render", "¿Cancelar el render en curso? Se borrará el archivo parcial.", parent=self):
            self._cancel_event.set()
            self.cancel_btn.configure(state="disabled")
            self._set_status("Cancelando…")

    def _render_worker(self, project: ProjectConfig, start: float, duration: Optional[float], scale: float):
        try:
            from ..render.exporter import ExportCancelled, export_video

            feats = self._get_features_for(project)
            self._queue.put(("status", "Renderizando… (puedes seguir editando, pero no cierres la ventana)"))
            result = export_video(
                project, feats, scale=scale, start=start, duration=duration,
                progress=lambda d, t: self._queue.put(("progress", (d, t))), cancel=self._cancel_event,
            )
            self._queue.put(("render_done", result))
        except ExportCancelled:
            self._queue.put(("render_cancelled", None))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("render_error", str(exc)))
            traceback.print_exc()

    # ------------------------------------------------------------------ cola de eventos
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "features":
                    self._features = payload
                    self._features_key = self.project.audio.model_dump_json() + f"|{self.project.output.fps}"
                    self.time_scale.configure(to=max(payload.duration, 0.1))
                    self._update_time_label()
                    self._refresh_timeline()
                    self._set_status(f"Audio: {payload.duration:.1f}s · BPM≈{payload.bpm:.0f} · {int(payload.beats.sum())} beats · {int(payload.kick_onsets.sum())} kicks")
                elif kind == "preview":
                    self._show_image(payload)
                elif kind == "preview_done":
                    self._preview_busy = False
                    if self._preview_pending:
                        self._preview_pending = False
                        self._request_preview(force=True)
                elif kind == "play_frame":
                    img, t = payload
                    self._show_image(img)
                    self.time_var.set(t)
                    self._update_time_label()
                elif kind == "play_end":
                    if self._playing:
                        self._stop_play()
                elif kind == "player_started":
                    self._player_proc = payload
                    self.window_btn.configure(text="■ Cerrar ventana", state="normal")
                    self._set_status("Reproductor OpenGL abierto (ESC cierra, ESPACIO pausa, ←/→ ±5 s, F pantalla completa)")
                    self.after(300, self._watch_player)
                elif kind == "player_ended":
                    self._player_proc = None
                    self.window_btn.configure(text="⧉ Ventana GL", state="normal")
                elif kind == "progress":
                    done, total = payload
                    self.progress.configure(maximum=total, value=done)
                    self._set_status(f"Renderizando… {done}/{total} frames ({100 * done / max(total, 1):.0f}%)")
                elif kind == "render_cancelled":
                    self._render_finished()
                    self.progress.configure(value=0)
                    self._set_status("Render cancelado (archivo parcial eliminado)")
                elif kind == "render_done":
                    self._render_finished()
                    self.progress.configure(value=self.progress["maximum"])
                    msg = f"Video listo:\n{payload.path}\n\n{payload.width}x{payload.height} · {payload.codec} · {payload.frames} frames en {payload.seconds:.0f}s"
                    self._set_status(f"Video listo: {payload.path}")
                    if messagebox.askyesno("Render terminado", msg + "\n\n¿Abrir la carpeta?", parent=self):
                        _open_folder(Path(payload.path).parent)
                elif kind == "render_error":
                    self._render_finished()
                    self._set_status("Error en el render")
                    messagebox.showerror("Error en el render", payload, parent=self)
                elif kind == "error":
                    self._set_status(payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _render_finished(self):
        self._rendering = False
        self.render_btn.configure(state="normal", text="🎬 Renderizar video")
        self.cancel_btn.configure(state="disabled")

    def _check_env(self):
        from ..render.exporter import ExportError, available_encoders, encoder_works, ffmpeg_path
        from ..render.gpu import gpu_info

        lines = [f"GPU (OpenGL): {gpu_info()}"]
        try:
            lines.append(f"ffmpeg: {ffmpeg_path()}")
            for enc in ("h264_nvenc", "hevc_nvenc", "libx264"):
                if enc in available_encoders():
                    lines.append(f"{enc}: {'funciona' if encoder_works(enc) else 'listado pero NO funciona (drivers NVIDIA?)'}")
                else:
                    lines.append(f"{enc}: no disponible")
        except ExportError as exc:
            lines.append(str(exc))
        lines.append(f"CPUs: {os.cpu_count()}")
        try:
            import sounddevice  # noqa: F401

            lines.append("sounddevice: instalado (audio en la vista previa)")
        except ImportError:
            lines.append("sounddevice: no instalado → `pip install sounddevice` para oír la vista previa")
        messagebox.showinfo("Entorno", "\n".join(lines), parent=self)

    def _on_close(self):
        if self._rendering:
            if not messagebox.askyesno("Render en curso", "Hay un render en curso. ¿Cancelarlo y salir?", parent=self):
                return
            self._cancel_event.set()
        if not self._confirm_discard():
            return
        self._stop_play()
        self._close_player()
        autosave.clear()
        self.renderer.shutdown()
        self.destroy()


class KeyDialog(tk.Toplevel):
    """Diálogo para crear o editar un keyframe (tiempo, valor, curva)."""

    EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out", "back", "bounce"]
    HINTS = {
        "position": "x, y relativos al lienzo (0..1). Ej.: 0.5, 0.5 es el centro.",
        "scale": "1 = tamaño configurado; 2 = el doble; 0 = invisible.",
        "opacity": "0 = transparente, 1 = opaco (se multiplica con la opacidad de la capa).",
        "rotation": "Grados; se suman a la rotación propia de la capa.",
    }

    def __init__(self, master, prop: str, key, allow_delete: bool):
        super().__init__(master)
        self.title(f"Keyframe de {prop}")
        self.resizable(False, False)
        self.result = None
        self.deleted = False
        self.prop = prop
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Tiempo (s):").grid(row=0, column=0, sticky="w", pady=2)
        self.t_var = tk.StringVar(value=f"{key.time:g}")
        ttk.Entry(frm, textvariable=self.t_var, width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(frm, text="Valor:").grid(row=1, column=0, sticky="w", pady=2)
        if prop == "position":
            self.v_var = tk.StringVar(value=f"{key.value[0]:g}, {key.value[1]:g}")
        else:
            self.v_var = tk.StringVar(value=f"{key.value:g}")
        ttk.Entry(frm, textvariable=self.v_var, width=16).grid(row=1, column=1, sticky="w")
        ttk.Label(frm, text="Curva de llegada:").grid(row=2, column=0, sticky="w", pady=2)
        self.e_var = tk.StringVar(value=key.easing)
        ttk.Combobox(frm, textvariable=self.e_var, values=self.EASINGS, state="readonly", width=14).grid(row=2, column=1, sticky="w")
        ttk.Label(frm, text=self.HINTS[prop], foreground="#666", wraplength=300, font=("TkDefaultFont", 8)).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 8))
        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, sticky="e")
        if allow_delete:
            ttk.Button(btns, text="Eliminar", command=self._delete).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancelar", command=self.destroy).pack(side="left", padx=4)
        ttk.Button(btns, text="Guardar", command=self._save).pack(side="left", padx=4)
        self.bind("<Return>", lambda e: self._save())
        self.bind("<Escape>", lambda e: self.destroy())
        self.transient(master)
        self.grab_set()
        self.wait_window(self)

    def _save(self):
        try:
            t = float(self.t_var.get())
            if self.prop == "position":
                parts = [float(p) for p in self.v_var.get().replace(";", ",").split(",")]
                if len(parts) != 2:
                    raise ValueError("Se esperan dos números")
                self.result = PointKey(time=t, value=(parts[0], parts[1]), easing=self.e_var.get())
            else:
                self.result = ScalarKey(time=t, value=float(self.v_var.get()), easing=self.e_var.get())
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("Valor inválido", str(exc), parent=self)
            return
        self.destroy()

    def _delete(self):
        self.deleted = True
        self.destroy()


def get_literal_default(cls) -> str:
    ann = cls.model_fields["type"].annotation
    from typing import get_args

    return get_args(ann)[0]


def _fmt(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _open_folder(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:  # noqa: BLE001
        pass


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    app = App(argv[0] if argv else None)
    app.mainloop()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
