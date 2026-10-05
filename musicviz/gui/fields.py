"""Introspección de los modelos pydantic para generar formularios automáticamente.

Es independiente de Tkinter para poder probarse sin pantalla.
"""
from __future__ import annotations

import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

LABELS: dict[str, str] = {
    "enabled": "Activo",
    "opacity": "Opacidad",
    "blend": "Fusión",
    "position": "Posición (x, y)",
    "glow": "Resplandor",
    "glow_radius": "Radio resplandor",
    "bands": "Bandas",
    "width": "Ancho",
    "height": "Alto",
    "gap": "Separación",
    "rounded": "Redondeado",
    "mirror": "Espejo",
    "symmetric": "Simétrico",
    "colors": "Colores",
    "color": "Color",
    "gradient": "Gradiente",
    "min_height": "Altura mínima",
    "style": "Estilo",
    "segment_height": "Alto segmento",
    "bass_boost": "Refuerzo graves",
    "baseline": "Línea base",
    "radius": "Radio",
    "length": "Longitud",
    "thickness": "Grosor",
    "inner": "Hacia dentro",
    "rotation": "Rotación",
    "rotation_speed": "Vel. rotación (°/s)",
    "pulse": "Pulso",
    "pulse_trigger": "Disparador pulso",
    "ring": "Anillo",
    "ring_thickness": "Grosor anillo",
    "ring_color": "Color anillo",
    "rings": "Nº anillos",
    "ring_spread": "Separación anillos",
    "amplitude": "Amplitud",
    "samples": "Muestras",
    "smooth": "Suavizado",
    "count": "Cantidad",
    "burst": "Ráfaga por beat",
    "burst_trigger": "Disparador ráfaga",
    "burst_threshold": "Umbral ráfaga",
    "size": "Tamaño",
    "size_variance": "Variación tamaño",
    "speed": "Velocidad",
    "burst_speed": "Vel. ráfaga",
    "energy_speed": "Vel. por energía",
    "direction": "Dirección",
    "emitter": "Emisor",
    "emitter_radius": "Radio emisor",
    "gravity": "Gravedad",
    "lifetime": "Vida (s)",
    "shape": "Forma",
    "react_size": "Tamaño reactivo",
    "twinkle": "Parpadeo",
    "seed": "Semilla",
    "file": "Archivo",
    "scale": "Escala",
    "circle_mask": "Máscara circular",
    "shake": "Vibración",
    "text": "Texto",
    "font": "Fuente",
    "letter_spacing": "Espaciado letras",
    "uppercase": "Mayúsculas",
    "bg_color": "Color fondo",
    "show_time": "Mostrar tiempo",
    "trigger": "Disparador",
    "threshold": "Umbral",
    "intensity": "Intensidad",
    "start": "Inicio (s)",
    "end": "Fin (s)",
    "block_shift": "Desplazamiento bloques",
    "rgb_split": "Separación RGB",
    "scanlines": "Scanlines",
    "noise": "Ruido",
    "blocks": "Nº bloques",
    "probability": "Probabilidad",
    "invert": "Inversión",
    "strength": "Fuerza",
    "amount": "Cantidad",
    "zoom": "Zoom",
    "softness": "Suavidad",
    "hue_speed": "Vel. tono (°/s)",
    "hue_react": "Tono reactivo",
    "saturation": "Saturación",
    "contrast": "Contraste",
    "brightness": "Brillo",
    "gamma": "Gamma",
    "posterize": "Posterizar",
    "segments": "Segmentos",
    "axis": "Eje",
    "spacing": "Espaciado",
    "darkness": "Oscuridad",
    "path": "Ruta de salida",
    "fps": "FPS",
    "codec": "Códec",
    "bitrate": "Bitrate",
    "preset": "Preset encoder",
    "audio_bitrate": "Bitrate audio",
    "workers": "Procesos",
    "duration": "Duración (s)",
    "fmin": "Frec. mínima (Hz)",
    "fmax": "Frec. máxima (Hz)",
    "fft_size": "Tamaño FFT",
    "smoothing": "Suavizado temporal",
    "attack": "Ataque",
    "release": "Caída",
    "spatial_smoothing": "Suavizado entre bandas",
    "gain": "Ganancia",
    "normalize": "Normalización",
    "tilt": "Compensación agudos",
    "dynamic_range": "Rango dinámico (dB)",
    "beat_sensitivity": "Sensibilidad beats",
    "min_beat_interval": "Intervalo mín. beats (s)",
    "type": "Tipo",
    "angle": "Ángulo",
    "image": "Imagen",
    "image_fit": "Ajuste imagen",
    "blur": "Desenfoque",
    "darken": "Oscurecer",
    "react": "Reacción",
    "react_trigger": "Disparador reacción",
    "name": "Nombre",
    "transparent": "Fondo transparente",
    "sections": "Sólo en secciones",
    "palette": "Paleta",
    "background_colors": "Colores del fondo",
    "transition": "Transición (s)",
    "pulse_trigger": "Disparador pulso",
    "shake_trigger": "Disparador vibración",
    "shake_rotation": "Giro vibración (°)",
    "min_length": "Duración mínima (s)",
    "palettes": "Paletas por tipo",
    "intensities": "Intensidades por tipo",
    "animate_in": "Animación entrada",
    "in_duration": "Duración entrada (s)",
    "animate_out": "Animación salida",
    "out_duration": "Duración salida (s)",
    "easing": "Suavizado animación",
    "slide_distance": "Distancia deslizamiento",
    "video": "Video de fondo",
    "video_start": "Inicio del video (s)",
    "video_loop": "Repetir video",
    "video_speed": "Velocidad del video",
    "aspect": "Proporción ancho/alto",
    "fit": "Encaje",
    "focus": "Foco del recorte (x, y)",
    "anchor": "Anclaje",
    "corner_radius": "Radio esquinas",
    "border": "Borde (px)",
    "border_color": "Color borde",
    "shadow": "Sombra",
    "shadow_blur": "Desenfoque sombra",
    "shadow_offset": "Desplaz. sombra (x, y)",
    "align": "Alineación",
    "valign": "Alineación vertical",
    "max_width": "Ancho máximo (ajuste)",
    "line_spacing": "Interlineado",
    "stroke_width": "Contorno (px)",
    "stroke_color": "Color contorno",
    "box_color": "Color de caja",
    "box_padding": "Relleno de caja",
    "box_radius": "Radio de caja",
}

COLOR_FIELDS = {"color", "bg_color", "ring_color", "border_color", "stroke_color", "box_color"}
FILE_FIELDS = {"file", "image", "path", "video"}


@dataclass
class FieldSpec:
    name: str
    kind: str  # bool | int | float | str | choice | color | colors | pair | file | font | model
    optional: bool = False
    choices: list[Any] = field(default_factory=list)
    model_cls: Optional[type] = None
    readonly: bool = False

    @property
    def label(self) -> str:
        return LABELS.get(self.name, self.name.replace("_", " ").capitalize())


def _strip_optional(ann: Any) -> tuple[Any, bool]:
    origin = get_origin(ann)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


def field_specs(model_cls: type[BaseModel], exclude: tuple[str, ...] = ()) -> list[FieldSpec]:
    specs: list[FieldSpec] = []
    for name, info in model_cls.model_fields.items():
        if name in exclude:
            continue
        ann, optional = _strip_optional(info.annotation)
        origin = get_origin(ann)
        if name == "type":
            specs.append(FieldSpec(name, "str", readonly=True))
        elif origin is Literal:
            specs.append(FieldSpec(name, "choice", optional, list(get_args(ann))))
        elif ann is bool:
            specs.append(FieldSpec(name, "bool", optional))
        elif ann is int:
            specs.append(FieldSpec(name, "int", optional))
        elif ann is float:
            specs.append(FieldSpec(name, "float", optional))
        elif ann is str:
            if name == "font":
                specs.append(FieldSpec(name, "font", optional))
            elif name in COLOR_FIELDS:
                specs.append(FieldSpec(name, "color", optional))
            elif name in FILE_FIELDS:
                specs.append(FieldSpec(name, "file", optional))
            else:
                specs.append(FieldSpec(name, "str", optional))
        elif origin in (list, typing.List) and get_args(ann) and get_args(ann)[0] is str:
            specs.append(FieldSpec(name, "strlist" if name in ("effects", "layers", "sections") else "colors", optional))
        elif origin in (tuple, typing.Tuple):
            specs.append(FieldSpec(name, "pair", optional))
        elif isinstance(ann, type) and issubclass(ann, BaseModel):
            specs.append(FieldSpec(name, "model", optional, model_cls=ann))
        elif origin in (dict, typing.Dict):
            continue  # diccionarios (paletas por tipo): se editan en el YAML
        else:  # pragma: no cover - tipos no previstos se editan como texto
            specs.append(FieldSpec(name, "str", optional))
    return specs


def format_value(value: Any, spec: FieldSpec) -> str:
    """Representación de texto para mostrar en un Entry."""
    if value is None:
        return ""
    if spec.kind in ("colors", "strlist"):
        return ", ".join(value)
    if spec.kind == "pair":
        return ", ".join(_fmt_num(v) for v in value)
    if spec.kind in ("float", "int"):
        return _fmt_num(value)
    return str(value)


def _fmt_num(v: Any) -> str:
    if isinstance(v, float):
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-0") else "0"
    return str(v)


def parse_value(raw: Any, spec: FieldSpec) -> Any:
    """Convierte lo escrito por el usuario al tipo del campo (None si está vacío y es opcional)."""
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            if spec.optional:
                return None
            if spec.kind in ("str", "file", "color"):
                return ""
            raise ValueError("Campo obligatorio")
    if spec.kind == "bool":
        if isinstance(raw, str):
            return raw.lower() in ("1", "true", "sí", "si", "yes", "on")
        return bool(raw)
    if spec.kind == "int":
        return int(float(raw))
    if spec.kind == "float":
        return float(raw)
    if spec.kind == "choice":
        for c in spec.choices:
            if str(c) == str(raw):
                return c
        raise ValueError(f"Valor no permitido: {raw}")
    if spec.kind in ("colors", "strlist"):
        parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
        return parts
    if spec.kind == "pair":
        parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("Se esperan dos números separados por coma")
        return (float(parts[0]), float(parts[1]))
    return raw


def apply_value(model: BaseModel, spec: FieldSpec, raw: Any) -> BaseModel:
    """Devuelve una copia validada del modelo con el campo cambiado. Lanza ValueError si no es válido."""
    value = parse_value(raw, spec)
    data = model.model_dump(mode="python")
    data[spec.name] = value
    try:
        return type(model).model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ValueError(first.get("msg", str(exc))) from exc


def replace_submodel(model: BaseModel, name: str, sub: BaseModel) -> BaseModel:
    data = model.model_dump(mode="python")
    data[name] = sub.model_dump(mode="python")
    return type(model).model_validate(data)
