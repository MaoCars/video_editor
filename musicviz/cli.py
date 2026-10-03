"""Interfaz de línea de comandos de musicviz."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from . import __version__
from .config import ProjectConfig

app = typer.Typer(add_completion=False, help="Generador de videos music visualizer (espectros, partículas, glitch).", no_args_is_help=True)
console = Console()


def _load_project(path: Path) -> ProjectConfig:
    try:
        return ProjectConfig.load(path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error en el proyecto {path}:[/red]\n{exc}")
        raise typer.Exit(2)


def _analyze(project: ProjectConfig):
    from .render.engine import analyze_project

    with console.status("Analizando audio..."):
        t0 = time.perf_counter()
        features = analyze_project(project)
    console.print(f"Audio: {project.audio.file}  ·  {features.duration:.1f}s  ·  {features.n_frames} frames @ {features.fps:g} fps  ·  BPM≈{features.bpm:.0f}  ·  beats={int(features.beats.sum())}  ·  kicks={int(features.kick_onsets.sum())}  ({time.perf_counter() - t0:.1f}s)")
    return features


@app.callback()
def _main_callback(version: bool = typer.Option(False, "--version", help="Muestra la versión y sale.")):
    if version:
        console.print(f"musicviz {__version__}")
        raise typer.Exit()


@app.command()
def presets():
    """Lista los presets de estilo disponibles."""
    from .presets import preset_description, preset_names

    table = Table(title="Presets")
    table.add_column("Nombre", style="cyan")
    table.add_column("Descripción")
    for name in preset_names():
        table.add_row(name, preset_description(name))
    console.print(table)


@app.command()
def init(
    project: Path = typer.Argument(..., help="Ruta del archivo YAML de proyecto a crear."),
    audio: Path = typer.Option(..., "--audio", "-a", help="Archivo de audio (mp3, wav, flac, ogg, m4a...)."),
    preset: str = typer.Option("trap_nation", "--preset", "-p", help="Preset base (ver `musicviz presets`)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Ruta del video de salida."),
    force: bool = typer.Option(False, "--force", help="Sobrescribe si el proyecto ya existe."),
):
    """Crea un proyecto YAML a partir de un preset. Edítalo para personalizar colores, capas y efectos."""
    from .presets import load_preset, preset_yaml

    if project.exists() and not force:
        console.print(f"[red]{project} ya existe. Usa --force para sobrescribir.[/red]")
        raise typer.Exit(1)
    audio_str = str(audio)
    out_str = str(output) if output else f"{Path(audio).stem}_{preset}.mp4"
    try:
        load_preset(preset, audio_str, out_str)  # valida
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    project.write_text(preset_yaml(preset, audio_str, out_str), encoding="utf-8")
    console.print(f"[green]Proyecto creado:[/green] {project}\nEdítalo y luego ejecuta: [cyan]musicviz render {project}[/cyan]")


@app.command()
def analyze(audio: Path = typer.Argument(..., help="Archivo de audio."), fps: int = typer.Option(60, help="FPS de referencia.")):
    """Analiza un archivo de audio: duración, BPM estimado, beats y kicks detectados."""
    from .audio.analysis import analyze as analyze_audio
    from .config import AudioConfig

    cfg = AudioConfig(file=str(audio))
    with console.status("Analizando..."):
        f = analyze_audio(cfg, fps=float(fps))
    table = Table(title=str(audio))
    table.add_column("Métrica")
    table.add_column("Valor", justify="right")
    table.add_row("Duración", f"{f.duration:.2f} s")
    table.add_row("BPM estimado", f"{f.bpm:.1f}")
    table.add_row("Beats detectados", str(int(f.beats.sum())))
    table.add_row("Kicks (graves)", str(int(f.kick_onsets.sum())))
    table.add_row("Frames @ fps", f"{f.n_frames} @ {fps}")
    table.add_row("Pico de 'drop'", f"{float(f.drop.max()):.2f} en {float(f.drop.argmax() / fps):.1f} s")
    console.print(table)


@app.command()
def render(
    project: Path = typer.Argument(..., help="Archivo YAML del proyecto."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Sobrescribe la ruta de salida."),
    scale: float = typer.Option(1.0, "--scale", "-s", min=0.05, max=2.0, help="Factor de resolución (0.5 = vista rápida)."),
    start: float = typer.Option(0.0, "--start", help="Segundo inicial del clip a renderizar."),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Duración del clip (segundos). Por defecto toda la canción."),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Procesos en paralelo (por defecto automático)."),
):
    """Renderiza el video completo (o un fragmento) y lo codifica con ffmpeg."""
    from .render.exporter import ExportError, export_video

    cfg = _load_project(project)
    features = _analyze(cfg)
    try:
        with Progress(TextColumn("[bold blue]Render"), BarColumn(), TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as bar:
            task = bar.add_task("render", total=features.n_frames)

            def on_progress(done: int, total: int) -> None:
                bar.update(task, completed=done, total=total)

            result = export_video(cfg, features, output=output, scale=scale, start=start, duration=duration, workers=workers, progress=on_progress)
    except ExportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Listo:[/green] {result.path}  ·  {result.width}x{result.height}  ·  {result.codec}  ·  {result.frames} frames en {result.seconds:.1f}s ({result.fps_rendered:.1f} fps de render)")


@app.command()
def snapshot(
    project: Path = typer.Argument(..., help="Archivo YAML del proyecto."),
    time_s: float = typer.Option(30.0, "--time", "-t", help="Instante (segundos) a capturar."),
    output: Path = typer.Option(Path("snapshot.png"), "--output", "-o", help="Imagen PNG de salida."),
    scale: float = typer.Option(1.0, "--scale", "-s", min=0.05, max=2.0),
):
    """Guarda un solo frame como imagen para ajustar el diseño sin renderizar todo."""
    import cv2

    from .render.engine import render_frame_image

    cfg = _load_project(project)
    features = _analyze(cfg)
    img = render_frame_image(cfg, features, time_s, scale)
    cv2.imwrite(str(output), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    console.print(f"[green]Frame guardado:[/green] {output}")


@app.command()
def preview(
    project: Path = typer.Argument(..., help="Archivo YAML del proyecto."),
    scale: float = typer.Option(0.5, "--scale", "-s", min=0.1, max=1.0, help="Resolución de la vista previa."),
    start: float = typer.Option(0.0, "--start", help="Segundo inicial."),
    no_audio: bool = typer.Option(False, "--no-audio", help="No reproducir audio (requiere `pip install sounddevice`)."),
):
    """Abre una ventana con la vista previa en tiempo real (ESC salir, ESPACIO pausa, ←/→ saltar)."""
    from .render.preview import preview as run_preview

    cfg = _load_project(project)
    features = _analyze(cfg)
    try:
        run_preview(cfg, features, scale=scale, start=start, with_audio=not no_audio)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command()
def check():
    """Comprueba ffmpeg, encoders (NVENC) y CPUs disponibles."""
    import os

    from .render.exporter import ExportError, available_encoders, encoder_works, ffmpeg_path

    table = Table(title="Entorno")
    table.add_column("Componente")
    table.add_column("Estado")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("CPUs", str(os.cpu_count()))
    try:
        table.add_row("ffmpeg", ffmpeg_path())
        for enc in ("h264_nvenc", "hevc_nvenc", "libx264", "libx265"):
            if enc in available_encoders():
                ok = encoder_works(enc)
                table.add_row(enc, "[green]funciona[/green]" if ok else "[yellow]listado pero no funciona (¿sin GPU NVIDIA / drivers?)[/yellow]")
            else:
                table.add_row(enc, "[red]no disponible[/red]")
    except ExportError as exc:
        table.add_row("ffmpeg", f"[red]{exc}[/red]")
    try:
        import cv2

        table.add_row("OpenCV", f"{cv2.__version__} ({'con ventanas' if hasattr(cv2, 'imshow') else 'headless'})")
    except ImportError:
        table.add_row("OpenCV", "[red]no instalado[/red]")
    try:
        import sounddevice  # noqa: F401

        table.add_row("sounddevice (audio en preview)", "instalado")
    except ImportError:
        table.add_row("sounddevice (audio en preview)", "[yellow]opcional, no instalado[/yellow]")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
