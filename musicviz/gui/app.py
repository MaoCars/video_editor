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
from ..config import EffectConfig, LayerConfig, ProjectConfig, get_args_of_union
from ..presets import load_preset, preset_names
from .fields import FieldSpec, apply_value, field_specs, format_value, replace_submodel

PREVIEW_SCALES = {"Baja (480p)": 480 / 1080, "Media (540p)": 0.5, "Alta (720p)": 720 / 1080}
AUDIO_TYPES = [("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.opus *.wma"), ("Todos", "*.*")]
IMAGE_TYPES = [("Imágenes", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Todos", "*.*")]


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


class ModelForm(ttk.Frame):
    """Formulario generado a partir de un modelo pydantic.

    on_change(new_model) se llama con una copia validada cada vez que el usuario cambia un campo.
    """

    def __init__(self, master, model: BaseModel, on_change: Callable[[BaseModel], None], exclude: tuple[str, ...] = (), status: Optional[Callable[[str], None]] = None):
        super().__init__(master)
        self.model = model
        self.on_change = on_change
        self.status = status or (lambda s: None)
        self.columnconfigure(1, weight=1)
        self._build(exclude)

    def _build(self, exclude):
        row = 0
        for spec in field_specs(type(self.model), exclude):
            value = getattr(self.model, spec.name)
            if spec.kind == "model":
                box = ttk.LabelFrame(self, text=spec.label)
                box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
                box.columnconfigure(0, weight=1)
                sub = ModelForm(box, value, lambda m, n=spec.name: self._set_sub(n, m), status=self.status)
                sub.pack(fill="x")
                row += 1
                continue
            ttk.Label(self, text=spec.label).grid(row=row, column=0, sticky="w", padx=(6, 8), pady=2)
            if spec.kind == "bool":
                var = tk.BooleanVar(value=bool(value))
                w = ttk.Checkbutton(self, variable=var, command=lambda s=spec, v=var: self._commit(s, v.get()))
                w.grid(row=row, column=1, sticky="w")
            elif spec.kind == "choice":
                var = tk.StringVar(value=str(value) if value is not None else "")
                values = [str(c) for c in spec.choices] + ([""] if spec.optional else [])
                w = ttk.Combobox(self, textvariable=var, values=values, state="readonly", width=16)
                w.grid(row=row, column=1, sticky="w")
                w.bind("<<ComboboxSelected>>", lambda e, s=spec, v=var: self._commit(s, v.get()))
            elif spec.kind == "str" and spec.readonly:
                ttk.Label(self, text=str(value), font=("TkDefaultFont", 9, "bold")).grid(row=row, column=1, sticky="w")
            else:
                var = tk.StringVar(value=format_value(value, spec))
                entry = ttk.Entry(self, textvariable=var, width=28)
                entry.grid(row=row, column=1, sticky="ew")
                entry.bind("<Return>", lambda e, s=spec, v=var, w=entry: self._commit(s, v.get(), w))
                entry.bind("<FocusOut>", lambda e, s=spec, v=var, w=entry: self._commit(s, v.get(), w))
                if spec.kind == "color":
                    ttk.Button(self, text="…", width=3, command=lambda s=spec, v=var, w=entry: self._pick_color(s, v, w, replace=True)).grid(row=row, column=2, padx=2)
                elif spec.kind == "colors":
                    ttk.Button(self, text="+", width=3, command=lambda s=spec, v=var, w=entry: self._pick_color(s, v, w, replace=False)).grid(row=row, column=2, padx=2)
                elif spec.kind == "file":
                    ttk.Button(self, text="…", width=3, command=lambda s=spec, v=var, w=entry: self._pick_file(s, v, w)).grid(row=row, column=2, padx=2)
            row += 1

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

    def _pick_file(self, spec: FieldSpec, var: tk.StringVar, widget: tk.Widget):
        if spec.name == "path":
            path = filedialog.asksaveasfilename(parent=self, defaultextension=".mp4", filetypes=[("Video MP4", "*.mp4"), ("Todos", "*.*")])
        elif spec.name == "image":
            path = filedialog.askopenfilename(parent=self, filetypes=IMAGE_TYPES)
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
        self.geometry("1380x840")
        self.minsize(1100, 700)
        style = ttk.Style(self)
        style.configure("Error.TEntry", fieldbackground="#ffd6d6")
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))

        self.project: ProjectConfig = load_preset("trap_nation", "", "")
        self.project_path: Optional[Path] = None
        self.dirty = False
        self._features: Optional[AudioFeatures] = None
        self._features_key: Optional[str] = None
        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._preview_busy = False
        self._preview_pending = False
        self._preview_after: Optional[str] = None
        self._playing = False
        self._play_thread: Optional[threading.Thread] = None
        self._rendering = False
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._suspend_traces = False

        self._build_menu()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(60, self._poll_queue)
        if project_path:
            self._open_project(Path(project_path))
        else:
            self._refresh_all()

    # ------------------------------------------------------------------ construcción UI
    def _build_menu(self):
        menubar = tk.Menu(self)
        m = tk.Menu(menubar, tearoff=0)
        m.add_command(label="Nuevo desde preset…", command=self._new_from_preset, accelerator="Ctrl+N")
        m.add_command(label="Abrir proyecto…", command=self._open_dialog, accelerator="Ctrl+O")
        m.add_command(label="Guardar", command=self._save, accelerator="Ctrl+S")
        m.add_command(label="Guardar como…", command=self._save_as)
        m.add_separator()
        m.add_command(label="Salir", command=self._on_close)
        menubar.add_cascade(label="Archivo", menu=m)
        h = tk.Menu(menubar, tearoff=0)
        h.add_command(label="Comprobar entorno (ffmpeg / NVENC)", command=self._check_env)
        h.add_command(label="Acerca de", command=lambda: messagebox.showinfo("musicviz", "musicviz — generador de videos music visualizer.\nEdita el proyecto a la izquierda, mira la vista previa a la derecha y pulsa Renderizar."))
        menubar.add_cascade(label="Ayuda", menu=h)
        self.config(menu=menubar)
        self.bind_all("<Control-n>", lambda e: self._new_from_preset())
        self.bind_all("<Control-o>", lambda e: self._open_dialog())
        self.bind_all("<Control-s>", lambda e: self._save())

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
        self.nb.add(self.tab_project, text="Proyecto")
        self.nb.add(self.tab_layers, text="Capas")
        self.nb.add(self.tab_effects, text="Efectos")
        self._build_list_tab(self.tab_layers, "layers")
        self._build_list_tab(self.tab_effects, "effects")

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.preview_label = tk.Label(right, bg="#111118", text="Vista previa\n\nElige un archivo de audio para empezar", fg="#aaa", font=("TkDefaultFont", 12))
        self.preview_label.pack(fill="both", expand=True)
        ctl = ttk.Frame(right, padding=(0, 6))
        ctl.pack(fill="x")
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(ctl, from_=0.0, to=1.0, variable=self.time_var, command=lambda v: self._on_time_drag())
        self.time_scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.time_label = ttk.Label(ctl, text="0:00 / 0:00", width=12)
        self.time_label.pack(side="left")
        self.play_btn = ttk.Button(ctl, text="▶ Reproducir", command=self._toggle_play, width=14)
        self.play_btn.pack(side="left", padx=4)
        ttk.Button(ctl, text="Actualizar", command=lambda: self._request_preview(force=True)).pack(side="left", padx=4)
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Auto", variable=self.auto_var).pack(side="left")
        ttk.Label(ctl, text="Calidad:").pack(side="left", padx=(12, 2))
        self.scale_var = tk.StringVar(value="Media (540p)")
        cb = ttk.Combobox(ctl, textvariable=self.scale_var, values=list(PREVIEW_SCALES), state="readonly", width=13)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._request_preview(force=True))

        bottom = ttk.Frame(self, padding=(8, 4))
        bottom.pack(fill="x")
        self.render_btn = ttk.Button(bottom, text="🎬 Renderizar video", style="Accent.TButton", command=self._start_render)
        self.render_btn.pack(side="left")
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
        add_btn = ttk.Menubutton(bar, text="＋ Añadir")
        menu = tk.Menu(add_btn, tearoff=0)
        union = LayerConfig if kind == "layers" else EffectConfig
        for cls in get_args_of_union(union):
            tname = cls.model_fields["type"].default if cls.model_fields["type"].default is not None else get_literal_default(cls)
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

    # ------------------------------------------------------------------ estado / refresco
    def _set_status(self, text: str):
        self.status_var.set(text or "Listo")

    def _mark_dirty(self):
        self.dirty = True
        self._update_title()

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
        self._update_title()
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
        items = getattr(self.project, kind)
        cur = select if select is not None else (lb.curselection()[0] if lb.curselection() else 0)
        lb.delete(0, "end")
        for i, item in enumerate(items):
            extra = ""
            if kind == "effects":
                extra = f"  ({item.trigger})"
            elif hasattr(item, "text"):
                extra = f"  “{item.text[:18]}”"
            flag = "" if item.enabled else "  [off]"
            lb.insert("end", f"{i + 1}. {item.type}{extra}{flag}")
        if items:
            cur = min(cur, len(items) - 1)
            lb.selection_set(cur)
            lb.see(cur)
        self._show_item_form(kind)

    def _show_item_form(self, kind: str):
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        form: ScrollFrame = getattr(self, f"{kind}_form")
        form.clear()
        items = getattr(self.project, kind)
        if not lb.curselection() or not items:
            ttk.Label(form.inner, text="Añade una capa o efecto con el botón ＋", padding=10).pack()
            return
        idx = lb.curselection()[0]
        ModelForm(form.inner, items[idx], lambda m, k=kind, i=idx: self._set_item(k, i, m), status=self._set_status).pack(fill="x", padx=4, pady=4)

    def _set_item(self, kind: str, idx: int, model):
        items = getattr(self.project, kind)
        items[idx] = model
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        extra = f"  ({model.trigger})" if kind == "effects" else (f"  “{model.text[:18]}”" if hasattr(model, "text") else "")
        flag = "" if model.enabled else "  [off]"
        lb.delete(idx)
        lb.insert(idx, f"{idx + 1}. {model.type}{extra}{flag}")
        lb.selection_set(idx)
        self._changed()

    def _selected(self, kind: str) -> Optional[int]:
        lb: tk.Listbox = getattr(self, f"{kind}_list")
        return lb.curselection()[0] if lb.curselection() else None

    def _add_item(self, kind: str, cls):
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
        items = getattr(self.project, kind)
        items.append(item)
        self._refresh_list(kind, select=len(items) - 1)
        self._changed()

    def _dup_item(self, kind: str):
        idx = self._selected(kind)
        if idx is None:
            return
        items = getattr(self.project, kind)
        items.insert(idx + 1, items[idx].model_copy(deep=True))
        self._refresh_list(kind, select=idx + 1)
        self._changed()

    def _del_item(self, kind: str):
        idx = self._selected(kind)
        if idx is None:
            return
        items = getattr(self.project, kind)
        del items[idx]
        self._refresh_list(kind, select=max(idx - 1, 0))
        self._changed()

    def _move_item(self, kind: str, delta: int):
        idx = self._selected(kind)
        if idx is None:
            return
        items = getattr(self.project, kind)
        j = idx + delta
        if 0 <= j < len(items):
            items[idx], items[j] = items[j], items[idx]
            self._refresh_list(kind, select=j)
            self._changed()

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

    def _request_preview(self, force: bool = False):
        if self._playing:
            return
        if not force and not self.auto_var.get():
            return
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
        self._preview_after = self.after(150 if force else 600, self._run_preview)

    def _run_preview(self):
        self._preview_after = None
        self._update_time_label()
        if not self._audio_ready():
            self.preview_label.configure(image="", text="Vista previa\n\nElige un archivo de audio para empezar")
            self._photo = None
            return
        if self._preview_busy:
            self._preview_pending = True
            return
        self._preview_busy = True
        project = self.project.model_copy(deep=True)
        t = float(self.time_var.get())
        scale = PREVIEW_SCALES[self.scale_var.get()]
        threading.Thread(target=self._preview_worker, args=(project, t, scale), daemon=True).start()

    def _preview_worker(self, project: ProjectConfig, t: float, scale: float):
        try:
            from ..render.engine import render_frame_image

            feats = self._get_features_for(project)
            img = render_frame_image(project, feats, t, scale)
            self._queue.put(("preview", img))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"Vista previa: {exc}"))
            traceback.print_exc()
        finally:
            self._queue.put(("preview_done", None))

    def _get_features_for(self, project: ProjectConfig) -> AudioFeatures:
        key = project.audio.model_dump_json() + f"|{project.output.fps}"
        if self._features is None or key != self._features_key:
            from ..render.engine import analyze_project

            self._queue.put(("status", "Analizando audio…"))
            feats = analyze_project(project)
            self._features, self._features_key = feats, key
            self._queue.put(("features", feats))
        return self._features

    def _show_image(self, img: np.ndarray):
        lw = max(self.preview_label.winfo_width(), 320)
        lh = max(self.preview_label.winfo_height(), 180)
        h, w = img.shape[:2]
        k = min(lw / w, lh / h)
        pil = Image.fromarray(img)
        if abs(k - 1.0) > 0.01:
            pil = pil.resize((max(int(w * k), 1), max(int(h * k), 1)), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=self._photo, text="")

    def _on_time_drag(self):
        self._update_time_label()
        if not self._playing:
            self._request_preview(force=True)

    def _update_time_label(self):
        total = self._features.duration if self._features else 0.0
        self.time_label.configure(text=f"{_fmt(self.time_var.get())} / {_fmt(total)}")

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
        self._play_thread = threading.Thread(target=self._play_worker, args=(project, start, scale), daemon=True)
        self._play_thread.start()

    def _stop_play(self):
        self._playing = False
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        self.play_btn.configure(text="▶ Reproducir")

    def _play_worker(self, project: ProjectConfig, start: float, scale: float):
        try:
            from ..render.engine import Scene, output_size

            feats = self._get_features_for(project)
            w, h = output_size(project, scale)
            scene = Scene(project, feats, w, h)
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
        self.render_btn.configure(state="disabled", text="Renderizando…")
        self.progress.configure(value=0, maximum=100)
        project = self.project.model_copy(deep=True)
        threading.Thread(target=self._render_worker, args=(project, start, duration, scale), daemon=True).start()

    def _render_worker(self, project: ProjectConfig, start: float, duration: Optional[float], scale: float):
        try:
            from ..render.exporter import export_video

            feats = self._get_features_for(project)
            self._queue.put(("status", "Renderizando… (puedes seguir editando, pero no cierres la ventana)"))
            result = export_video(project, feats, scale=scale, start=start, duration=duration, progress=lambda d, t: self._queue.put(("progress", (d, t))))
            self._queue.put(("render_done", result))
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
                    self.time_scale.configure(to=max(payload.duration, 0.1))
                    self._update_time_label()
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
                elif kind == "progress":
                    done, total = payload
                    self.progress.configure(maximum=total, value=done)
                    self._set_status(f"Renderizando… {done}/{total} frames ({100 * done / max(total, 1):.0f}%)")
                elif kind == "render_done":
                    self._rendering = False
                    self.render_btn.configure(state="normal", text="🎬 Renderizar video")
                    self.progress.configure(value=self.progress["maximum"])
                    msg = f"Video listo:\n{payload.path}\n\n{payload.width}x{payload.height} · {payload.codec} · {payload.frames} frames en {payload.seconds:.0f}s"
                    self._set_status(f"Video listo: {payload.path}")
                    if messagebox.askyesno("Render terminado", msg + "\n\n¿Abrir la carpeta?", parent=self):
                        _open_folder(Path(payload.path).parent)
                elif kind == "render_error":
                    self._rendering = False
                    self.render_btn.configure(state="normal", text="🎬 Renderizar video")
                    self._set_status("Error en el render")
                    messagebox.showerror("Error en el render", payload, parent=self)
                elif kind == "error":
                    self._set_status(payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _check_env(self):
        from ..render.exporter import ExportError, available_encoders, encoder_works, ffmpeg_path

        lines = []
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
        if self._rendering and not messagebox.askyesno("Render en curso", "Hay un render en curso. ¿Salir de todos modos?", parent=self):
            return
        if not self._confirm_discard():
            return
        self._stop_play()
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
