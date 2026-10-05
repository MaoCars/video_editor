"""Esquema de configuración de un proyecto (archivo YAML) usando pydantic.

Todas las medidas en píxeles se expresan "a escala 1080p" y se escalan
automáticamente con la resolución de salida. Las posiciones y tamaños relativos
van de 0 a 1 respecto al ancho/alto del lienzo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .utils.color import parse_color

Color = str
Trigger = Literal["always", "beat", "kick", "bass", "energy", "drop", "treble"]
Anchor = Literal["center", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"]
ALPHA_CODECS = ("prores_4444", "qtrle", "vp9_alpha", "png_sequence")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_colors(values: List[str]) -> List[str]:
    if not values:
        raise ValueError("Se requiere al menos un color")
    for v in values:
        parse_color(v)
    return values


# --------------------------------------------------------------------------- salida / audio


class OutputConfig(StrictModel):
    path: str = "output.mp4"
    width: int = 1920
    height: int = 1080
    fps: int = 60
    codec: Literal["auto", "h264_nvenc", "hevc_nvenc", "libx264", "libx265", "prores_4444", "qtrle", "vp9_alpha", "png_sequence"] = "auto"
    """auto elige h264_nvenc/libx264 (o prores_4444 si transparent=true). Los últimos cuatro conservan el canal alfa."""
    transparent: bool = False
    """Fondo transparente: no se dibuja el fondo y el video conserva el canal alfa (.mov ProRes 4444 por defecto)."""
    bitrate: str = "16M"
    preset: Optional[str] = None
    """Preset del encoder (NVENC: p1..p7; x264: ultrafast..veryslow). None = por defecto."""
    audio_bitrate: str = "320k"
    workers: int = 0
    """Procesos de render en paralelo. 0 = automático según CPUs."""

    @field_validator("width", "height")
    @classmethod
    def _even(cls, v: int) -> int:
        if v < 16:
            raise ValueError("Resolución demasiado pequeña")
        return v - (v % 2)  # yuv420p requiere dimensiones pares


class SmoothingConfig(StrictModel):
    attack: float = Field(0.7, ge=0.01, le=1.0)
    release: float = Field(0.25, ge=0.01, le=1.0)


class AudioConfig(StrictModel):
    file: str
    start: float = 0.0
    duration: Optional[float] = None
    bands: int = Field(64, ge=4, le=512)
    fmin: float = 30.0
    fmax: float = 16000.0
    fft_size: int = 2048
    smoothing: SmoothingConfig = SmoothingConfig()
    spatial_smoothing: float = 0.8
    """Sigma (en bandas) del suavizado entre bandas vecinas. 0 = desactivado."""
    gain: float = 1.0
    gamma: float = 1.0
    """Curva de respuesta de las bandas (<1 realza valores bajos, >1 los comprime)."""
    normalize: Literal["hybrid", "per_band", "global"] = "hybrid"
    """hybrid: escala global en dB + compensación parcial por banda (conserva la forma del
    espectro pero hace visibles los agudos). per_band: cada banda usa todo el rango.
    global: dB puros (los agudos quedan muy bajos)."""
    tilt: float = Field(0.7, ge=0.0, le=1.0)
    """Sólo en hybrid: cuánto se compensan las bandas más silenciosas (0 = nada, 1 = per_band)."""
    dynamic_range: float = Field(50.0, gt=1.0)
    """Rango en dB entre el silencio visual (0) y el máximo (1)."""
    beat_sensitivity: float = 1.0
    min_beat_interval: float = 0.12


# --------------------------------------------------------------------------- fondo


class BackgroundConfig(StrictModel):
    type: Literal["solid", "gradient", "radial", "image", "video"] = "gradient"
    color: Color = "#0a0a14"
    colors: List[Color] = ["#0a0a14", "#1d0b3a"]
    angle: float = 90.0
    image: Optional[str] = None
    image_fit: Literal["cover", "contain", "stretch"] = "cover"
    video: Optional[str] = None
    """Archivo de video de fondo (mp4, mov, webm...). Su audio se ignora; se usa el de la canción."""
    video_start: float = 0.0
    """Segundo del video en el que empezar."""
    video_loop: bool = True
    """Si el video es más corto que la canción, repetirlo (si no, se congela el último frame)."""
    video_speed: float = Field(1.0, gt=0.0)
    blur: float = 0.0
    darken: float = 0.0
    pulse: float = 0.0
    """Zoom del fondo proporcional al kick (0.03 = 3%)."""
    react: float = 0.0
    """Brillo extra proporcional a la energía (0..1)."""
    react_trigger: Trigger = "bass"

    _vc = field_validator("colors")(_validate_colors)


# --------------------------------------------------------------------------- capas


Animation = Literal["none", "fade", "slide_left", "slide_right", "slide_up", "slide_down", "zoom_in", "zoom_out", "pop", "blur"]
Easing = Literal["linear", "ease_in", "ease_out", "ease_in_out", "back", "bounce"]


class LayerBase(StrictModel):
    """Campos comunes. `start`/`end` limitan cuándo existe la capa (segundos; None = toda la canción) y
    `animate_in`/`animate_out` definen cómo aparece y desaparece. Texto e imagen soportan todas las
    animaciones; el resto de capas usan fundido para cualquier animación distinta de `none`."""

    enabled: bool = True
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    blend: Literal["normal", "add", "screen"] = "normal"
    position: Tuple[float, float] = (0.5, 0.5)
    glow: float = 0.0
    glow_radius: float = 24.0
    start: Optional[float] = None
    end: Optional[float] = None
    animate_in: Animation = "none"
    in_duration: float = Field(0.6, ge=0.0)
    animate_out: Animation = "none"
    out_duration: float = Field(0.6, ge=0.0)
    easing: Easing = "ease_out"
    slide_distance: float = 0.15
    """Distancia de los deslizamientos, relativa al lado menor del lienzo."""


class BarsLayer(LayerBase):
    type: Literal["bars"]
    bands: Optional[int] = None
    width: float = 0.9
    height: float = 0.45
    gap: float = Field(0.35, ge=0.0, le=0.95)
    rounded: bool = True
    mirror: bool = True
    """Refleja las barras hacia abajo (simétricas respecto a la línea central)."""
    symmetric: bool = False
    """Espectro espejado izquierda/derecha (graves al centro)."""
    colors: List[Color] = ["#00f0ff", "#ff00e0"]
    gradient: Literal["index", "height"] = "index"
    min_height: float = 0.004
    style: Literal["solid", "outline", "dots", "segments"] = "solid"
    segment_height: float = 8.0
    bass_boost: float = 0.0
    baseline: bool = False

    _vc = field_validator("colors")(_validate_colors)


class CircleLayer(LayerBase):
    type: Literal["circle"]
    radius: float = 0.22
    bands: Optional[int] = None
    length: float = 0.22
    thickness: float = 4.0
    mirror: bool = True
    inner: bool = False
    colors: List[Color] = ["#ff2a6d", "#ff7a00"]
    gradient: Literal["angle", "value"] = "angle"
    rotation: float = -90.0
    rotation_speed: float = 0.0
    pulse: float = 0.12
    pulse_trigger: Trigger = "kick"
    style: Literal["bars", "line", "filled", "dots", "rays"] = "bars"
    ring: bool = True
    ring_thickness: float = 3.0
    ring_color: Optional[Color] = None
    rings: int = Field(1, ge=1, le=4)
    ring_spread: float = 0.06
    rounded: bool = True
    glow: float = 0.6

    _vc = field_validator("colors")(_validate_colors)


class WaveformLayer(LayerBase):
    type: Literal["waveform"]
    amplitude: float = 0.18
    thickness: float = 3.0
    colors: List[Color] = ["#ffffff"]
    mirror: bool = False
    style: Literal["line", "filled", "circular", "bars"] = "line"
    samples: int = 512
    width: float = 0.9
    radius: float = 0.25
    smooth: int = 3
    glow: float = 0.3

    _vc = field_validator("colors")(_validate_colors)


class ParticlesLayer(LayerBase):
    type: Literal["particles"]
    count: int = 200
    burst: int = 40
    burst_trigger: Literal["beat", "kick"] = "beat"
    burst_threshold: float = 0.0
    size: float = 3.0
    size_variance: float = 2.0
    colors: List[Color] = ["#ffffff", "#8ac4ff"]
    speed: float = 60.0
    burst_speed: float = 350.0
    energy_speed: float = 3.0
    direction: Literal["up", "down", "left", "right", "out", "in", "random"] = "up"
    emitter: Literal["screen", "center", "ring", "bottom", "top"] = "screen"
    emitter_radius: float = 0.25
    gravity: float = 0.0
    lifetime: float = 2.5
    shape: Literal["circle", "square", "streak"] = "circle"
    react_size: float = 1.0
    twinkle: float = 0.4
    seed: int = 1
    glow: float = 0.5
    blend: Literal["normal", "add", "screen"] = "add"

    _vc = field_validator("colors")(_validate_colors)


class ImageLayer(LayerBase):
    """Imagen (portada, foto del artista, logo).

    La imagen se encaja en una caja cuyo lado mayor mide `size` (relativo al lado menor del lienzo);
    `aspect` fija la proporción de la caja (None = la de la imagen), `fit` decide si se recorta (cover)
    o se encaja completa (contain) y `focus` qué zona conservar al recortar. `anchor` indica qué punto de
    la imagen se coloca en `position`.
    """

    type: Literal["image"]
    file: str
    size: float = 0.3
    aspect: Optional[float] = None
    shape: Literal["original", "square", "circle", "rounded"] = "original"
    fit: Literal["cover", "contain"] = "cover"
    focus: Tuple[float, float] = (0.5, 0.5)
    anchor: Anchor = "center"
    corner_radius: float = 40.0
    border: float = 0.0
    border_color: Color = "#ffffff"
    shadow: float = 0.0
    shadow_blur: float = 30.0
    shadow_offset: Tuple[float, float] = (0.0, 12.0)
    pulse: float = 0.08
    pulse_trigger: Trigger = "kick"
    rotation: float = 0.0
    rotation_speed: float = 0.0
    shake: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _compat(cls, data):
        # Compatibilidad con proyectos antiguos: scale -> size, circle_mask -> shape: circle
        if isinstance(data, dict):
            data = dict(data)
            if "scale" in data and "size" not in data:
                data["size"] = data.pop("scale")
            if data.pop("circle_mask", False):
                data.setdefault("shape", "circle")
        return data


class TextLayer(LayerBase):
    """Texto. `align` fija la alineación de las líneas y el punto horizontal que se coloca en `position`;
    `valign` el vertical. `max_width` (relativo al ancho) parte el texto en varias líneas. Usa \\n para
    saltos manuales."""

    type: Literal["text"]
    text: str
    font: Optional[str] = None
    size: float = 0.07
    color: Color = "#ffffff"
    align: Literal["left", "center", "right"] = "center"
    valign: Literal["top", "middle", "bottom"] = "middle"
    max_width: Optional[float] = None
    line_spacing: float = 1.15
    letter_spacing: float = 0.0
    uppercase: bool = False
    stroke_width: float = 0.0
    stroke_color: Color = "#000000"
    shadow: float = 0.0
    shadow_blur: float = 12.0
    shadow_offset: Tuple[float, float] = (3.0, 4.0)
    box_color: Optional[Color] = None
    box_padding: float = 16.0
    box_radius: float = 12.0
    pulse: float = 0.04
    pulse_trigger: Trigger = "kick"
    glow: float = 0.4


class ProgressLayer(LayerBase):
    type: Literal["progress"]
    thickness: float = 4.0
    color: Color = "#ffffff"
    bg_color: Color = "#ffffff30"
    width: float = 0.8
    position: Tuple[float, float] = (0.5, 0.95)
    show_time: bool = False
    font: Optional[str] = None


LayerConfig = Annotated[
    Union[BarsLayer, CircleLayer, WaveformLayer, ParticlesLayer, ImageLayer, TextLayer, ProgressLayer],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- efectos


class EffectBase(StrictModel):
    enabled: bool = True
    trigger: Trigger = "always"
    threshold: float = Field(0.5, ge=0.0, lt=1.0)
    intensity: float = 1.0
    start: Optional[float] = None
    end: Optional[float] = None


class GlitchEffect(EffectBase):
    type: Literal["glitch"]
    trigger: Trigger = "beat"
    block_shift: float = 1.0
    rgb_split: float = 1.0
    scanlines: float = 0.3
    noise: float = 0.15
    blocks: int = 10
    probability: float = 1.0
    invert: float = 0.0
    seed: int = 7


class BloomEffect(EffectBase):
    type: Literal["bloom"]
    radius: float = 30.0
    threshold: float = Field(0.55, ge=0.0, lt=1.0)
    strength: float = 0.9


class ChromaticEffect(EffectBase):
    type: Literal["chromatic"]
    amount: float = 4.0


class ShakeEffect(EffectBase):
    type: Literal["shake"]
    trigger: Trigger = "kick"
    amount: float = 14.0
    rotation: float = 0.0
    zoom: float = 0.03
    seed: int = 3


class VignetteEffect(EffectBase):
    type: Literal["vignette"]
    strength: float = 0.6
    softness: float = 0.65


class ColorEffect(EffectBase):
    type: Literal["color"]
    hue_speed: float = 0.0
    hue_react: float = 0.0
    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    gamma: float = 1.0
    posterize: int = 0


class PixelateEffect(EffectBase):
    type: Literal["pixelate"]
    trigger: Trigger = "beat"
    size: int = 12


class StrobeEffect(EffectBase):
    type: Literal["strobe"]
    trigger: Trigger = "kick"
    color: Color = "#ffffff"
    intensity: float = 0.35


class KaleidoEffect(EffectBase):
    type: Literal["kaleido"]
    segments: Literal[2, 4] = 2
    axis: Literal["horizontal", "vertical"] = "horizontal"


class RadialBlurEffect(EffectBase):
    type: Literal["radial_blur"]
    trigger: Trigger = "kick"
    amount: float = 0.08
    samples: int = 6


class ScanlinesEffect(EffectBase):
    type: Literal["scanlines"]
    spacing: int = 3
    darkness: float = 0.3


class FilmGrainEffect(EffectBase):
    type: Literal["grain"]
    amount: float = 0.05
    seed: int = 11


EffectConfig = Annotated[
    Union[
        GlitchEffect,
        BloomEffect,
        ChromaticEffect,
        ShakeEffect,
        VignetteEffect,
        ColorEffect,
        PixelateEffect,
        StrobeEffect,
        KaleidoEffect,
        RadialBlurEffect,
        ScanlinesEffect,
        FilmGrainEffect,
    ],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- proyecto


class ProjectConfig(StrictModel):
    name: str = "untitled"
    output: OutputConfig = OutputConfig()
    audio: AudioConfig
    background: BackgroundConfig = BackgroundConfig()
    layers: List[LayerConfig] = []
    effects: List[EffectConfig] = []

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectConfig":
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        project = cls.model_validate(data)
        return project.resolve_paths(path.parent)

    def resolve_paths(self, base: Path) -> "ProjectConfig":
        """Convierte rutas relativas (audio, imágenes) en absolutas respecto a `base`."""
        base = Path(base)

        def fix(p: Optional[str]) -> Optional[str]:
            if not p:
                return p
            pp = Path(p).expanduser()
            return str(pp if pp.is_absolute() else (base / pp))

        self.audio.file = fix(self.audio.file)  # type: ignore[assignment]
        self.background.image = fix(self.background.image)
        self.background.video = fix(self.background.video)
        for layer in self.layers:
            if isinstance(layer, ImageLayer):
                layer.file = fix(layer.file)  # type: ignore[assignment]
            if isinstance(layer, (TextLayer, ProgressLayer)) and layer.font and any(c in layer.font for c in "/\\"):
                layer.font = fix(layer.font)
        if not Path(self.output.path).is_absolute():
            self.output.path = str(base / self.output.path)
        return self

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")


def get_args_of_union(annotated_union) -> tuple[type, ...]:
    """Clases concretas de una unión discriminada (LayerConfig / EffectConfig)."""
    from typing import get_args

    inner = get_args(annotated_union)[0]  # Annotated[Union[...], Field] -> Union[...]
    return tuple(get_args(inner))
