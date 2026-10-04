# musicviz — generador de videos *music visualizer* en Python

Crea videos tipo **Trap Nation / Monstercat / NCS** para tus canciones de dubstep, EDM,
drum & bass, etc.: espectros de audio (circulares, barras, forma de onda), partículas
que reaccionan al ritmo, glitch, bloom, shake, aberración cromática y más. Todo se
configura con un archivo YAML por proyecto, así que puedes personalizar cada detalle
o partir de los presets incluidos.

Pensado para correr en tu laptop (MSI Cyborg 15: i7‑12650H, RTX 4060, 8 GB RAM, Windows 11):
el render usa varios núcleos de CPU en paralelo y la codificación del video se hace con
**NVENC** (la GPU) a través de ffmpeg.

---

## 1. Instalación (Windows 11)

1. **Python 3.10 o superior** → <https://www.python.org/downloads/> (marca *Add Python to PATH*).
2. **ffmpeg** (incluye soporte NVENC):
   ```powershell
   winget install Gyan.FFmpeg
   ```
   Cierra y vuelve a abrir la terminal, y comprueba con `ffmpeg -version`.
3. **musicviz** (desde la carpeta del repositorio):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -e .
   pip install sounddevice   # opcional: audio en la vista previa
   ```
4. Verifica el entorno:
   ```powershell
   musicviz check
   ```
   Debe aparecer `h264_nvenc → funciona`. Si no, actualiza los drivers NVIDIA; mientras tanto
   se usará `libx264` (CPU) automáticamente.

> En Linux/macOS funciona igual (instala ffmpeg con tu gestor de paquetes).

---

## 2. Interfaz gráfica

```powershell
musicviz gui                 # o: musicviz-gui, o: python -m musicviz.gui
musicviz gui mi_video.yaml   # abrir un proyecto existente
```

![Interfaz gráfica](docs/gui.png)

- **Arriba**: archivo de audio, video de salida y preset base (*Aplicar preset* reemplaza capas y efectos).
- **Proyecto**: resolución, FPS, códec, análisis de audio (bandas, suavizado, sensibilidad de beats) y fondo.
- **Capas / Efectos**: lista ordenable (añadir, duplicar, eliminar, subir/bajar) y un formulario con todas las
  opciones del elemento seleccionado; los campos inválidos se marcan en rojo y el motivo aparece en la barra de estado.
- **Vista previa**: se actualiza sola al cambiar cualquier valor (casilla *Auto*), con control de tiempo,
  calidad y botón *Reproducir* (con audio si instalaste `sounddevice`).
- **Renderizar video**: render completo o un fragmento (*Desde* / *Duración* / *Escala*) con barra de progreso;
  la ventana sigue usable mientras tanto.
- *Archivo → Guardar* escribe el YAML, que también puedes editar a mano o usar con la línea de comandos.

## 3. Uso rápido (línea de comandos)

```powershell
# 1) Crear un proyecto a partir de un preset
musicviz init mi_video.yaml --audio "C:\Musica\mi_cancion.mp3" --preset trap_nation

# 2) Ver un frame para ajustar el diseño (rápido)
musicviz snapshot mi_video.yaml --time 45 --output prueba.png

# 3) Vista previa en ventana (ESC salir, ESPACIO pausa, ←/→ saltar 5 s)
musicviz preview mi_video.yaml --scale 0.5

# 4) Render de prueba de 10 s a media resolución
musicviz render mi_video.yaml --start 40 --duration 10 --scale 0.5 -o prueba.mp4

# 5) Render final (1080p60 con NVENC)
musicviz render mi_video.yaml
```

Otros comandos:

| Comando | Qué hace |
|---|---|
| `musicviz presets` | Lista los presets y su descripción |
| `musicviz analyze cancion.mp3` | Duración, BPM estimado, beats, kicks y "drop" más fuerte |
| `musicviz check` | Comprueba ffmpeg, NVENC, OpenCV, CPUs |
| `musicviz gui [proyecto.yaml]` | Abre la interfaz gráfica |
| `musicviz render ... --workers 6` | Número de procesos de render en paralelo |

---

## 4. Presets incluidos

| Preset | Estilo |
|---|---|
| `trap_nation` | Círculo central con barras radiales simétricas que pulsan con el bass, partículas en cada golpe, bloom y shake |
| `monstercat` | Barras blancas limpias, título grande, barra de progreso con tiempo. Minimalista |
| `ncs` | Anillo con espectro relleno de gradiente neón, rotación lenta, partículas flotando, cambio de tono |
| `dnb_glitch` | Barras simétricas con espejo, glitch/pixelado en los kicks, strobe y desenfoque radial en los drops |
| `minimal` | Plantilla mínima para empezar desde cero |

`musicviz init` copia el preset (con comentarios) a tu YAML: edita, guarda y vuelve a
hacer `snapshot`/`render`. Mira `examples/proyecto_completo.yaml` para un ejemplo comentado
con todas las opciones.

---

## 5. Estructura del proyecto (YAML)

```yaml
name: mi_video
output:
  path: mi_video.mp4
  width: 1920
  height: 1080
  fps: 60
  codec: auto          # auto | h264_nvenc | hevc_nvenc | libx264 | libx265
  bitrate: 16M
  workers: 0           # 0 = automático
audio:
  file: cancion.mp3
  start: 0             # recorte opcional (segundos)
  duration: null
  bands: 64            # nº de bandas del espectro
  fmin: 30
  fmax: 16000
  smoothing: {attack: 0.7, release: 0.25}   # respuesta de las barras
  normalize: hybrid    # hybrid | per_band | global
  beat_sensitivity: 1.0
background: {...}
layers: [...]
effects: [...]
```

Las medidas en píxeles (grosores, radios de glow, cantidad de shake) se expresan "a 1080p"
y se escalan solas con la resolución. Posiciones y tamaños relativos van de 0 a 1.

### Disparadores (`trigger`)

Muchos parámetros (efectos, pulso del círculo, logo, texto, fondo) se controlan con un
disparador que toma un valor 0..1 por frame a partir del audio:

| trigger | Fuente |
|---|---|
| `always` | Siempre 1 |
| `beat` | Envolvente que salta en cada beat detectado y decae |
| `kick` | Igual pero sólo con golpes de graves (ideal para dubstep/dnb) |
| `bass` / `energy` / `treble` | Energía de graves / global / agudos por encima de `threshold` |
| `drop` | Secciones donde el volumen sube mucho respecto a los segundos previos |

Cada efecto admite `intensity`, `threshold`, `start` y `end` (ventana en segundos).

### Fondo (`background`)

`type: solid | gradient | radial | image`, `colors`, `angle`, `image`, `image_fit`,
`blur`, `darken`, `pulse` (zoom con el kick), `react` (brillo con la energía), `react_trigger`.

### Capas (`layers`)

Todas admiten `opacity`, `blend` (`normal | add | screen`), `position: [x, y]`, `glow`, `glow_radius`, `enabled`.

| type | Parámetros principales |
|---|---|
| `bars` | `bands`, `width`, `height`, `gap`, `rounded`, `mirror` (refleja hacia abajo), `symmetric` (graves al centro), `colors`, `gradient: index\|height`, `style: solid\|outline\|dots\|segments`, `bass_boost`, `baseline` |
| `circle` | `radius`, `length`, `thickness`, `mirror`, `inner`, `colors`, `gradient: angle\|value`, `rotation`, `rotation_speed`, `pulse`, `pulse_trigger`, `style: bars\|line\|filled\|dots\|rays`, `ring`, `ring_thickness`, `ring_color`, `rings`, `ring_spread` |
| `waveform` | `amplitude`, `thickness`, `colors`, `mirror`, `style: line\|filled\|circular\|bars`, `samples`, `width`, `radius`, `smooth` |
| `particles` | `count` (ambiente), `burst` (por beat), `burst_trigger: beat\|kick`, `burst_threshold`, `size`, `size_variance`, `colors`, `speed`, `burst_speed`, `energy_speed`, `direction: up\|down\|left\|right\|out\|in\|random`, `emitter: screen\|center\|ring\|bottom\|top`, `emitter_radius`, `gravity`, `lifetime`, `shape: circle\|square\|streak`, `react_size`, `twinkle`, `seed` |
| `image` | `file` (PNG con transparencia recomendado), `scale`, `pulse`, `pulse_trigger`, `rotation`, `rotation_speed`, `circle_mask`, `shake` |
| `text` | `text`, `font` (nombre o ruta .ttf), `size`, `color`, `pulse`, `pulse_trigger`, `letter_spacing`, `uppercase` |
| `progress` | `thickness`, `color`, `bg_color`, `width`, `show_time`, `font` |

### Efectos (`effects`) — se aplican en orden sobre el frame completo

| type | Parámetros |
|---|---|
| `glitch` | `block_shift`, `rgb_split`, `scanlines`, `noise`, `blocks`, `probability`, `invert`, `seed` |
| `bloom` | `radius`, `threshold`, `strength` |
| `chromatic` | `amount` (px) |
| `shake` | `amount` (px), `rotation` (grados), `zoom` |
| `vignette` | `strength`, `softness` |
| `color` | `hue_speed` (°/s), `hue_react`, `saturation`, `contrast`, `brightness`, `gamma`, `posterize` |
| `pixelate` | `size` |
| `strobe` | `color`, `intensity` |
| `kaleido` | `segments: 2\|4`, `axis` |
| `radial_blur` | `amount`, `samples` |
| `scanlines` | `spacing`, `darkness` |
| `grain` | `amount`, `seed` |

Ejemplo: glitch sólo durante el drop (de 1:02 a 1:30) disparado por los kicks:

```yaml
effects:
  - type: glitch
    trigger: kick
    start: 62
    end: 90
    intensity: 1.2
    probability: 0.8
```

---

## 6. Rendimiento y memoria (8 GB de RAM)

- El render es CPU (OpenCV + NumPy) en varios procesos; la codificación es GPU (NVENC).
  A 1080p60 cada proceso tarda ~80‑200 ms por frame según la cantidad de capas/efectos; con
  6‑8 procesos una canción de 3‑4 minutos tarda unos pocos minutos.
- Por defecto se usan `min(CPUs-2, 8)` procesos. Si notas que el sistema se queda sin memoria,
  baja con `--workers 4` (cada proceso usa ~150‑300 MB a 1080p).
- Ajusta primero con `snapshot` y renders cortos a `--scale 0.5`; el render final a 1080p sólo al terminar.
- `fps: 30` renderiza en la mitad de tiempo; 60 fps se ve más fluido en YouTube.
- Las capas que cubren toda la pantalla (partículas con muchas unidades, glow grande) son las más costosas.

---

## 7. Cómo funciona (para personalizar el código)

```
musicviz/
  audio/analysis.py   # STFT por frame, bandas log, bass/mid/treble, beats, kicks, drops, BPM
  config.py           # esquema YAML (pydantic) — añade aquí nuevos parámetros
  layers/             # background, bars, circle, waveform, particles, image, text, progress
  effects/            # glitch, bloom, chromatic, shake, vignette, color, pixelate, strobe, ...
  render/canvas.py    # lienzo float RGB + composición de capas RGBA con glow y blend modes
  render/engine.py    # Scene, render por frame, pool de procesos
  render/exporter.py  # ffmpeg (NVENC / libx264) + mezcla de audio
  render/preview.py   # ventana de vista previa (OpenCV)
  gui/app.py          # interfaz gráfica Tkinter; gui/fields.py genera los formularios desde los modelos
  presets/*.yaml
```

Para crear una capa nueva: define su modelo en `config.py` (añádelo a `LayerConfig`), implementa
una clase `Layer` con `prepare()` y `render()` en `layers/`, y regístrala en `layers/__init__.py`.
`render()` debe ser determinista respecto al número de frame (los frames se renderizan en
paralelo y fuera de orden), por eso las partículas se calculan analíticamente en función del tiempo.

Tests: `pip install -e .[dev]` y `pytest`.
