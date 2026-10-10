"""Panel de línea de tiempo (Tkinter Canvas): forma de onda con beats, secciones, una fila por capa y
por efecto con sus ventanas de tiempo y animaciones, y un cursor sincronizado con la vista previa.

Interacción: clic en la regla o el audio = saltar; clic en una barra = seleccionar; arrastrar los
bordes de una barra = cambiar inicio/fin; arrastrar el centro = mover; arrastrar el límite entre dos
secciones = ajustarlo; rueda = zoom; Shift+rueda = desplazar.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import numpy as np

from ..audio.analysis import AudioFeatures
from ..config import KEYFRAME_PROPS, ProjectConfig

PROP_LABELS = {"position": "posición", "scale": "escala", "opacity": "opacidad", "rotation": "rotación"}
KEY_COL = "#ffd166"
KEY_R = 5

LABEL_W = 132
RULER_H = 18
AUDIO_H = 46
SECTION_H = 16
ROW_H = 20
EDGE_PX = 6
BG = "#1b1b24"
GRID = "#2d2d3a"
TEXT = "#d8d8e0"
WAVE = "#5a7fd6"
BEAT = "#6d6d80"
KICK = "#ff6b6b"
LOOP_COL = "#ffd166"
CURSOR = "#ff3b3b"
LAYER_COL = "#43a86f"
EFFECT_COL = "#d99a2b"
SELECT_COL = "#ffffff"
SECTION_COLORS = {"calm": "#3b6fd1", "build": "#8a3fc9", "drop": "#d6365a"}
OTHER_SECTION = "#5c5c6e"
ANIM_COL = "#8fd3ad"


class Timeline(ttk.Frame):
    def __init__(
        self,
        master,
        on_seek: Callable[[float], None],
        on_select: Callable[[str, int], None],
        on_change: Callable[[str, int, Optional[float], Optional[float]], None],
        on_section_change: Callable[[int, float], None],
        on_key_move: Optional[Callable[[int, str, int, float], None]] = None,
        on_key_edit: Optional[Callable[[int, str, int], None]] = None,
        on_key_add: Optional[Callable[[int, str, float], None]] = None,
        on_key_delete: Optional[Callable[[int, str, int], None]] = None,
        height: int = 190,
    ):
        super().__init__(master)
        self.on_seek, self.on_select, self.on_change, self.on_section_change = on_seek, on_select, on_change, on_section_change
        self.on_key_move, self.on_key_edit, self.on_key_add, self.on_key_delete = on_key_move, on_key_edit, on_key_add, on_key_delete
        self.loop: Optional[tuple[float, float]] = None
        self.project: Optional[ProjectConfig] = None
        self.features: Optional[AudioFeatures] = None
        self.duration = 60.0
        self.time = 0.0
        self.view0, self.view1 = 0.0, 60.0  # ventana visible (s)
        self.selected: Optional[tuple[str, int]] = None
        self._drag: Optional[dict] = None
        self._rows: list[dict] = []
        self._env: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._sections: list = []

        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, height=height)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self._hscroll)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        self.canvas.bind("<Button-3>", self._on_right)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self._on_wheel(e, pan=True))
        self.canvas.bind("<Shift-Button-4>", lambda e: self._on_wheel(e, pan=True))
        self.canvas.bind("<Shift-Button-5>", lambda e: self._on_wheel(e, pan=True))

    # ------------------------------------------------------------------ datos
    def set_data(self, project: ProjectConfig, features: Optional[AudioFeatures]) -> None:
        self.project = project
        if features is not self.features:
            self.features = features
            self._env = None
        new_duration = features.duration if features else self._guess_duration(project)
        if abs(new_duration - self.duration) > 1e-6:
            self.duration = max(new_duration, 1.0)
            self.view0, self.view1 = 0.0, self.duration
        self._sections = []
        if features is not None:
            try:
                from ..render.sections import resolve_sections

                self._sections = resolve_sections(project, features)
            except Exception:  # noqa: BLE001 - sección inválida a medio editar
                self._sections = []
        self.redraw()

    def _guess_duration(self, project: ProjectConfig) -> float:
        ends = [l.end for l in project.layers if l.end] + [e.end for e in project.effects if e.end]
        if project.sections and project.sections != "auto":
            ends += [s.end for s in project.sections if s.end] + [s.start for s in project.sections]
        return max(ends + [60.0])

    def set_time(self, t: float) -> None:
        self.time = t
        x = self._t2x(t)
        self.canvas.coords("cursor", x, 0, x, self._total_height())
        self.canvas.itemconfigure("cursor_label", text=_fmt(t))
        self.canvas.coords("cursor_label", min(max(x + 4, LABEL_W + 4), self.canvas.winfo_width() - 30), 2)

    def set_loop(self, rng: Optional[tuple[float, float]]) -> None:
        """Tramo A-B que se repite al reproducir (None = sin bucle)."""
        self.loop = rng
        self.redraw()

    def set_selected(self, kind: Optional[str], idx: Optional[int]) -> None:
        self.selected = (kind, idx) if kind is not None and idx is not None else None
        self.redraw()

    # ------------------------------------------------------------------ geometría
    def _plot_w(self) -> float:
        return max(self.canvas.winfo_width() - LABEL_W - 2, 50)

    def _t2x(self, t: float) -> float:
        span = max(self.view1 - self.view0, 1e-6)
        return LABEL_W + (t - self.view0) / span * self._plot_w()

    def _x2t(self, x: float) -> float:
        span = self.view1 - self.view0
        t = self.view0 + (x - LABEL_W) / self._plot_w() * span
        return min(max(t, 0.0), self.duration)

    def _total_height(self) -> int:
        return RULER_H + AUDIO_H + SECTION_H + ROW_H * max(len(self._rows), 1) + 4

    # ------------------------------------------------------------------ dibujo
    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self.project is None:
            c.create_text(10, 10, anchor="nw", text="Timeline: carga un audio para ver las pistas", fill=TEXT)
            return
        self._rows = []
        for i, l in enumerate(self.project.layers):
            self._rows.append({"kind": "layers", "idx": i, "item": l})
            for prop in KEYFRAME_PROPS:
                if getattr(l, f"{prop}_keys"):
                    self._rows.append({"kind": "keys", "idx": i, "prop": prop, "item": l})
        self._rows += [{"kind": "effects", "idx": i, "item": e} for i, e in enumerate(self.project.effects)]
        W = c.winfo_width()
        H = self._total_height()
        c.configure(scrollregion=(0, 0, W, H))
        self._draw_ruler(W)
        self._draw_audio(W)
        self._draw_sections(W)
        if self.loop is not None and self.features is not None:
            xa, xb = self._t2x(self.loop[0]), self._t2x(self.loop[1])
            c.create_rectangle(xa, 0, xb, H, fill=LOOP_COL, stipple="gray25", outline="", tags="loop")
            for x, label in ((xa, "A"), (xb, "B")):
                c.create_line(x, 0, x, H, fill=LOOP_COL, dash=(3, 3), tags="loop")
                c.create_text(x + 3, RULER_H - 2, anchor="sw", text=label, fill=LOOP_COL, font=("TkDefaultFont", 8, "bold"), tags="loop")
        y = RULER_H + AUDIO_H + SECTION_H
        for row in self._rows:
            if row["kind"] == "keys":
                self._draw_key_row(row, y, W)
            else:
                self._draw_row(row, y, W)
            y += ROW_H
        c.create_rectangle(0, 0, LABEL_W, H, fill="#15151c", outline="")  # columna de etiquetas opaca
        self._draw_labels()
        x = self._t2x(self.time)
        c.create_line(x, 0, x, H, fill=CURSOR, width=2, tags="cursor")
        c.create_text(x + 4, 2, anchor="nw", text=_fmt(self.time), fill=CURSOR, font=("TkDefaultFont", 8, "bold"), tags="cursor_label")
        self._update_hbar()

    def _draw_labels(self) -> None:
        c = self.canvas
        c.create_text(6, RULER_H + AUDIO_H / 2, anchor="w", text="Audio", fill=TEXT, font=("TkDefaultFont", 9, "bold"))
        c.create_text(6, RULER_H + AUDIO_H + SECTION_H / 2, anchor="w", text="Secciones", fill=TEXT, font=("TkDefaultFont", 8))
        y = RULER_H + AUDIO_H + SECTION_H
        for row in self._rows:
            item = row["item"]
            if row["kind"] == "keys":
                c.create_text(18, y + ROW_H / 2, anchor="w", text="↳ " + PROP_LABELS[row["prop"]], fill=KEY_COL, font=("TkDefaultFont", 7))
                y += ROW_H
                continue
            name = item.name or item.type
            prefix = "◆ " if row["kind"] == "layers" else "✦ "
            fill = TEXT if item.enabled else "#777"
            bold = self.selected == (row["kind"], row["idx"])
            c.create_text(6, y + ROW_H / 2, anchor="w", text=(prefix + name)[:20], fill=SELECT_COL if bold else fill, font=("TkDefaultFont", 8, "bold" if bold else "normal"))
            y += ROW_H

    def _draw_ruler(self, W: int) -> None:
        c = self.canvas
        c.create_rectangle(LABEL_W, 0, W, RULER_H, fill="#22222e", outline="")
        span = self.view1 - self.view0
        step = _nice_step(span / max(self._plot_w() / 70, 1))
        t = np.floor(self.view0 / step) * step
        H = self._total_height()
        while t <= self.view1 + 1e-9:
            x = self._t2x(t)
            if x >= LABEL_W:
                c.create_line(x, 0, x, H, fill=GRID)
                c.create_text(x + 2, RULER_H - 2, anchor="sw", text=_fmt(t, step < 1), fill=TEXT, font=("TkDefaultFont", 7))
            t += step

    def _envelope(self, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
        """Mín/máx de la onda por columna visible."""
        assert self.features is not None
        wf = self.features.waveform
        sr = self.features.sr
        s0 = int(max(self.view0, 0) * sr)
        s1 = int(min(self.view1, self.duration) * sr)
        seg = wf[s0:max(s1, s0 + 1)]
        if len(seg) < n_cols:
            seg = np.pad(seg, (0, n_cols - len(seg)))
        cut = (len(seg) // n_cols) * n_cols
        blocks = seg[:cut].reshape(n_cols, -1)
        return blocks.min(axis=1), blocks.max(axis=1)

    def _draw_audio(self, W: int) -> None:
        c = self.canvas
        y0 = RULER_H
        y1 = RULER_H + AUDIO_H
        c.create_rectangle(LABEL_W, y0, W, y1, fill="#191922", outline="")
        if self.features is None:
            c.create_text(LABEL_W + 8, (y0 + y1) / 2, anchor="w", text="(analizando audio…)", fill="#777")
            return
        n_cols = max(int(self._plot_w()), 2)
        lo, hi = self._envelope(n_cols)
        mid = (y0 + y1) / 2
        amp = (y1 - y0) / 2 - 2
        pts = []
        xs = LABEL_W + np.arange(n_cols)
        for x, h in zip(xs, hi):
            pts += [x, mid - h * amp]
        for x, l in zip(xs[::-1], lo[::-1]):
            pts += [x, mid - l * amp]
        c.create_polygon(*pts, fill=WAVE, outline="")
        # beats y kicks (sólo si hay sitio para distinguirlos)
        span = self.view1 - self.view0
        if span / max(self._plot_w(), 1) < 0.05:
            for t in self.features.beat_times:
                if self.view0 <= t <= self.view1:
                    x = self._t2x(t)
                    c.create_line(x, y1 - 6, x, y1, fill=BEAT)
            for t in self.features.kick_times:
                if self.view0 <= t <= self.view1:
                    x = self._t2x(t)
                    c.create_line(x, y1 - 12, x, y1, fill=KICK)

    def _draw_sections(self, W: int) -> None:
        c = self.canvas
        y0 = RULER_H + AUDIO_H
        y1 = y0 + SECTION_H
        c.create_rectangle(LABEL_W, y0, W, y1, fill="#20202a", outline="")
        if not self._sections or not self.project or not self.project.sections_enabled:
            return
        for i, sec in enumerate(self._sections):
            xa, xb = self._t2x(sec.start), self._t2x(sec.end)
            if xb < LABEL_W or xa > W:
                continue
            col = SECTION_COLORS.get(sec.name, OTHER_SECTION)
            c.create_rectangle(max(xa, LABEL_W), y0 + 1, min(xb, W), y1 - 1, fill=col, outline="", tags=("section", f"section{i}"))
            if xb - xa > 40:
                c.create_text((max(xa, LABEL_W) + min(xb, W)) / 2, (y0 + y1) / 2, text=f"{sec.name} ×{sec.intensity:g}", fill="white", font=("TkDefaultFont", 7))
            if i > 0 and self.project.sections != "auto":
                c.create_line(xa, y0, xa, y1, fill="white", width=2, tags=("secbound", f"secbound{i}"))

    def _draw_row(self, row: dict, y: int, W: int) -> None:
        c = self.canvas
        item = row["item"]
        kind = row["kind"]
        y0, y1 = y + 3, y + ROW_H - 3
        start = item.start if item.start is not None else 0.0
        end = item.end if item.end is not None else self.duration
        xa, xb = self._t2x(start), self._t2x(end)
        selected = self.selected == (kind, row["idx"])
        base = LAYER_COL if kind == "layers" else EFFECT_COL
        fill = base if item.enabled else "#444450"
        c.create_rectangle(LABEL_W, y, W, y + ROW_H, fill="#1b1b24", outline=GRID)
        c.create_rectangle(max(xa, LABEL_W), y0, min(xb, W), y1, fill=fill, outline=SELECT_COL if selected else "", width=2 if selected else 0, tags=(f"bar_{kind}_{row['idx']}",))
        # animaciones de entrada / salida (capas)
        if kind == "layers":
            if item.animate_in != "none" and item.in_duration > 0:
                c.create_rectangle(max(xa, LABEL_W), y0, min(self._t2x(start + item.in_duration), xb), y1, fill=ANIM_COL, outline="", stipple="gray50")
            if item.animate_out != "none" and item.out_duration > 0:
                c.create_rectangle(max(self._t2x(end - item.out_duration), xa), y0, min(xb, W), y1, fill=ANIM_COL, outline="", stipple="gray50")
        # tramos donde la sección no lo permite
        if self._sections and self.project is not None and self.project.sections_enabled:
            for sec in self._sections:
                allowed = (item.sections is None or sec.name in item.sections) and (
                    (sec.layers is None or item.name in sec.layers or item.type in sec.layers) if kind == "layers" else (sec.effects is None or item.name in sec.effects or item.type in sec.effects)
                )
                if not allowed:
                    sa, sb = max(self._t2x(sec.start), xa, LABEL_W), min(self._t2x(sec.end), xb, W)
                    if sb > sa:
                        c.create_rectangle(sa, y0, sb, y1, fill="#000000", outline="", stipple="gray75")
        if xb - xa > 60:
            label = item.type if kind == "effects" else (item.name or item.type)
            c.create_text(max(xa, LABEL_W) + 4, (y0 + y1) / 2, anchor="w", text=label, fill="#111", font=("TkDefaultFont", 7))

    def _draw_key_row(self, row: dict, y: int, W: int) -> None:
        c = self.canvas
        keys = sorted(getattr(row["item"], f"{row['prop']}_keys"), key=lambda k: k.time)
        c.create_rectangle(LABEL_W, y, W, y + ROW_H, fill="#16161e", outline=GRID)
        ym = y + ROW_H / 2
        xs = [self._t2x(k.time) for k in keys]
        if len(xs) > 1:
            c.create_line(max(xs[0], LABEL_W), ym, min(xs[-1], W), ym, fill=KEY_COL, dash=(2, 3))
        for k, (key, x) in enumerate(zip(keys, xs)):
            if x < LABEL_W - KEY_R or x > W + KEY_R:
                continue
            orig_index = getattr(row["item"], f"{row['prop']}_keys").index(key)
            c.create_polygon(x, ym - KEY_R, x + KEY_R, ym, x, ym + KEY_R, x - KEY_R, ym, fill=KEY_COL, outline="#333", tags=(f"key_{row['idx']}_{row['prop']}_{orig_index}",))
            if len(xs) <= 12:
                c.create_text(x, y + 1, anchor="n", text=_fmt_value(row["prop"], key.value), fill="#bbb", font=("TkDefaultFont", 6))

    # ------------------------------------------------------------------ interacción
    def _hit(self, x: float, y: float) -> Optional[dict]:
        """Qué hay bajo el ratón: ruler/audio, límite de sección, borde o centro de una barra."""
        y = self.canvas.canvasy(y)
        if x < LABEL_W:
            row_i = int((y - RULER_H - AUDIO_H - SECTION_H) // ROW_H)
            if 0 <= row_i < len(self._rows):
                return {"what": "label", "row": self._rows[row_i]}
            return None
        if y < RULER_H + AUDIO_H:
            return {"what": "seek"}
        if y < RULER_H + AUDIO_H + SECTION_H:
            if self.project and self.project.sections != "auto":
                for i, sec in enumerate(self._sections):
                    if i > 0 and abs(self._t2x(sec.start) - x) <= EDGE_PX:
                        return {"what": "secbound", "idx": i}
            return {"what": "seek"}
        row_i = int((y - RULER_H - AUDIO_H - SECTION_H) // ROW_H)
        if 0 <= row_i < len(self._rows):
            row = self._rows[row_i]
            item = row["item"]
            if row["kind"] == "keys":
                keys = getattr(item, f"{row['prop']}_keys")
                for k, key in enumerate(keys):
                    if abs(self._t2x(key.time) - x) <= KEY_R + 2:
                        return {"what": "key", "row": row, "k": k}
                return {"what": "keyrow", "row": row}
            start = item.start if item.start is not None else 0.0
            end = item.end if item.end is not None else self.duration
            xa, xb = self._t2x(start), self._t2x(end)
            if abs(x - xa) <= EDGE_PX:
                return {"what": "edge_start", "row": row}
            if abs(x - xb) <= EDGE_PX:
                return {"what": "edge_end", "row": row}
            if xa <= x <= xb:
                return {"what": "move", "row": row, "grab": self._x2t(x) - start, "length": end - start}
            return {"what": "row", "row": row}
        return None

    def _on_motion(self, event) -> None:
        hit = self._hit(event.x, event.y)
        cursor = ""
        if hit:
            if hit["what"] in ("edge_start", "edge_end", "secbound", "key"):
                cursor = "sb_h_double_arrow"
            elif hit["what"] == "move":
                cursor = "fleur"
            elif hit["what"] in ("seek", "label", "row"):
                cursor = "hand2"
        self.canvas.configure(cursor=cursor)

    def _on_press(self, event) -> None:
        hit = self._hit(event.x, event.y)
        if not hit:
            return
        if hit["what"] == "seek":
            self._drag = {"what": "seek"}
            self._seek(event.x)
            return
        if hit["what"] in ("label", "row"):
            self.on_select(hit["row"]["kind"], hit["row"]["idx"])
            return
        if hit["what"] == "secbound":
            self._drag = {"what": "secbound", "idx": hit["idx"], "t": self._sections[hit["idx"]].start}
            return
        if hit["what"] == "keyrow":
            self.on_select("layers", hit["row"]["idx"])
            return
        if hit["what"] == "key":
            row = hit["row"]
            self.on_select("layers", row["idx"])
            key = getattr(row["item"], f"{row['prop']}_keys")[hit["k"]]
            self._drag = {"what": "key", "row": row, "k": hit["k"], "t": key.time, "moved": False}
            return
        row = hit["row"]
        item = row["item"]
        self.on_select(row["kind"], row["idx"])
        self._drag = {
            "what": hit["what"], "row": row, "grab": hit.get("grab", 0.0), "length": hit.get("length", 0.0),
            "start": item.start if item.start is not None else 0.0,
            "end": item.end if item.end is not None else self.duration,
            "orig_start": item.start, "orig_end": item.end, "moved": False,
        }

    def _on_drag(self, event) -> None:
        if not self._drag:
            return
        d = self._drag
        if d["what"] == "seek":
            self._seek(event.x)
            return
        t = self._x2t(event.x)
        if d["what"] == "key":
            d["t"] = t
            d["moved"] = True
            row = d["row"]
            y = RULER_H + AUDIO_H + SECTION_H + ROW_H * self._rows.index(row)
            ym = y + ROW_H / 2
            x = self._t2x(t)
            self.canvas.coords(f"key_{row['idx']}_{row['prop']}_{d['k']}", x, ym - KEY_R, x + KEY_R, ym, x, ym + KEY_R, x - KEY_R, ym)
            return
        if d["what"] == "secbound":
            i = d["idx"]
            lo = self._sections[i - 1].start + 0.1
            hi = self._sections[i].end - 0.1
            d["t"] = min(max(t, lo), hi)
            self._sections[i].start = d["t"]
            self._sections[i - 1].end = d["t"]
            self.redraw()
            return
        d["moved"] = True
        if d["what"] == "edge_start":
            d["start"] = min(max(t, 0.0), d["end"] - 0.05)
        elif d["what"] == "edge_end":
            d["end"] = max(min(t, self.duration), d["start"] + 0.05)
        elif d["what"] == "move":
            start = min(max(t - d["grab"], 0.0), self.duration - d["length"])
            d["start"], d["end"] = start, start + d["length"]
        self._preview_bar(d)

    def _preview_bar(self, d: dict) -> None:
        row = d["row"]
        y = RULER_H + AUDIO_H + SECTION_H + ROW_H * self._rows.index(row)
        tag = f"bar_{row['kind']}_{row['idx']}"
        self.canvas.coords(tag, max(self._t2x(d["start"]), LABEL_W), y + 3, min(self._t2x(d["end"]), self.canvas.winfo_width()), y + ROW_H - 3)

    def _on_release(self, event) -> None:
        d, self._drag = self._drag, None
        if not d:
            return
        if d["what"] == "secbound":
            self.on_section_change(d["idx"], round(d["t"], 2))
            return
        if d["what"] == "key":
            if d.get("moved") and self.on_key_move:
                self.on_key_move(d["row"]["idx"], d["row"]["prop"], d["k"], round(d["t"], 2))
            return
        if d["what"] in ("edge_start", "edge_end", "move") and d.get("moved"):
            start = round(d["start"], 2)
            end = round(d["end"], 2)
            new_start: Optional[float] = None if start <= 0.0 and d["orig_start"] is None and d["what"] != "edge_start" else start
            if start <= 0.0 and d["what"] in ("edge_start", "move"):
                new_start = None
            new_end: Optional[float] = None if end >= self.duration - 0.05 else end
            self.on_change(d["row"]["kind"], d["row"]["idx"], new_start, new_end)

    def _on_double(self, event) -> None:
        hit = self._hit(event.x, event.y)
        if not hit:
            return
        if hit["what"] == "key" and self.on_key_edit:
            self._drag = None
            self.on_key_edit(hit["row"]["idx"], hit["row"]["prop"], hit["k"])
        elif hit["what"] == "keyrow" and self.on_key_add:
            self.on_key_add(hit["row"]["idx"], hit["row"]["prop"], round(self._x2t(event.x), 2))

    def _on_right(self, event) -> None:
        hit = self._hit(event.x, event.y)
        if hit and hit["what"] == "key" and self.on_key_delete:
            self.on_key_delete(hit["row"]["idx"], hit["row"]["prop"], hit["k"])

    def _seek(self, x: float) -> None:
        t = self._x2t(x)
        self.set_time(t)
        self.on_seek(t)

    def _on_wheel(self, event, pan: bool = False) -> None:
        direction = -1 if (getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0) else 1
        span = self.view1 - self.view0
        if pan:
            shift = direction * span * 0.1
            self._set_view(self.view0 + shift, self.view1 + shift)
            return
        if event.x < LABEL_W:
            self.canvas.yview_scroll(direction, "units")
            return
        anchor = self._x2t(event.x)
        factor = 1.25 if direction > 0 else 0.8
        new_span = min(max(span * factor, 1.0), self.duration)
        frac = (anchor - self.view0) / max(span, 1e-6)
        v0 = anchor - frac * new_span
        self._set_view(v0, v0 + new_span)

    def _set_view(self, v0: float, v1: float) -> None:
        span = v1 - v0
        v0 = min(max(v0, 0.0), max(self.duration - span, 0.0))
        self.view0, self.view1 = v0, v0 + span
        self.redraw()

    def _hscroll(self, *args) -> None:
        span = self.view1 - self.view0
        if args[0] == "moveto":
            v0 = float(args[1]) * self.duration
        else:
            v0 = self.view0 + int(args[1]) * span * 0.1
        self._set_view(v0, v0 + span)

    def _update_hbar(self) -> None:
        if self.duration <= 0:
            return
        self.hbar.set(self.view0 / self.duration, self.view1 / self.duration)


def _fmt_value(prop: str, value) -> str:
    if prop == "position":
        return f"{value[0]:.2f}, {value[1]:.2f}"
    if prop == "rotation":
        return f"{value:.0f}°"
    return f"{value:.2f}"


def _nice_step(raw: float) -> float:
    for step in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
        if step >= raw:
            return step
    return 600.0


def _fmt(t: float, decimals: bool = False) -> str:
    t = max(t, 0.0)
    m, s = int(t // 60), t % 60
    return f"{m}:{s:04.1f}" if decimals else f"{m}:{int(s):02d}"
