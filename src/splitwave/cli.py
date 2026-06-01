"""Splitwave CLI (Track E) — a thin shell over the engine.

    splitwave song.mp3 --tier balanced --stems vocals,instrumental [--dereverb]
    splitwave env-info        # backend / acceleration diagnostics (design doc §6)
    splitwave models          # list the model catalog
    splitwave prefetch balanced   # pre-download checkpoints (design doc §6, A3)

``separate`` is the implicit command: a bare ``splitwave <file>`` is rewritten to
``splitwave separate <file>`` by :func:`app` so the file-first form in the design
doc works alongside the utility subcommands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Backend, EngineConfig
from .core import Splitwave
from .errors import SplitwaveError
from .registry import MODEL_CATALOG
from .tiers import resolve_tier
from .types import Stem, Tier

cli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="High-quality vocal / instrument separation (Splitwave).",
)
console = Console()
err_console = Console(stderr=True)

#: Subcommands that must NOT be rewritten into `separate <file>` by the shim.
_SUBCOMMANDS = {"separate", "env-info", "models", "prefetch"}


def _build_config(backend: Optional[str], fmt: Optional[str]) -> EngineConfig:
    cfg = EngineConfig.from_env()
    if backend:
        cfg = cfg.with_overrides(backend=Backend(backend.lower()))
    if fmt:
        cfg = cfg.with_overrides(output_format=fmt.lower())
    return cfg


@cli.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    """Global options."""
    if version:
        console.print(f"splitwave {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@cli.command()
def separate(
    input_file: Path = typer.Argument(..., help="Audio file to separate."),
    tier: str = typer.Option("balanced", "--tier", "-t", help="fast | balanced | best"),
    stems: str = typer.Option(
        "vocals,instrumental", "--stems", "-s", help="Comma-separated stem names."
    ),
    out: Path = typer.Option(Path("stems"), "--out", "-o", help="Output directory."),
    fmt: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (wav, flac, mp3)."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Force a backend."),
    dereverb: bool = typer.Option(False, "--dereverb", help="Also emit a dry (de-reverbed) vocal."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
) -> None:
    """Separate INPUT_FILE into stems."""
    progress = None if quiet else (lambda m: err_console.log(f"[dim]{m}[/dim]"))
    cfg = _build_config(backend, fmt)
    engine = Splitwave(cfg, progress=progress)
    try:
        result = engine.separate(
            input_file,
            tier=tier,
            stems=[s for s in stems.split(",") if s.strip()],
            dereverb=dereverb,
            out_dir=out,
        )
    except (SplitwaveError, FileNotFoundError, ValueError) as exc:
        err_console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    _print_result(result)


def _print_result(result) -> None:
    table = Table(title=f"Separated ({result.tier}) in {result.wall_seconds:.1f}s")
    table.add_column("stem", style="cyan")
    table.add_column("path")
    table.add_column("rate", justify="right")
    for sf in result.stems:
        label = f"{sf.stem.value}{' (wet)' if sf.wet else ''}"
        table.add_row(label, str(sf.path), f"{sf.sample_rate} Hz")
    console.print(table)
    rtf = result.realtime_factor
    suffix = f" · {rtf:.1f}x realtime" if rtf else ""
    console.print(f"[dim]models: {', '.join(result.models_used)}{suffix}[/dim]")


@cli.command("env-info")
def env_info() -> None:
    """Report Python, ffmpeg, acceleration, and backend availability (design doc §6)."""
    from .audio import ffmpeg_available
    from .backends import available_backends

    table = Table(title="Splitwave environment")
    table.add_column("component", style="cyan")
    table.add_column("status")
    table.add_row("python", sys.version.split()[0])
    table.add_row("ffmpeg", "available" if ffmpeg_available() else "[red]missing[/red]")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        coreml = "CoreMLExecutionProvider" in providers
        table.add_row("onnxruntime providers", ("[green]" if coreml else "") + ", ".join(providers))
    except Exception:  # noqa: BLE001
        table.add_row("onnxruntime", "[yellow]not installed[/yellow]")

    try:
        import torch

        table.add_row("torch", torch.__version__)
        table.add_row("torch MPS", "available" if torch.backends.mps.is_available() else "no")
    except Exception:  # noqa: BLE001
        table.add_row("torch", "[yellow]not installed[/yellow]")

    for name, ok in available_backends().items():
        table.add_row(
            f"backend: {name}",
            "[green]ready[/green]" if ok else "[yellow]unavailable[/yellow]",
        )
    console.print(table)


@cli.command("models")
def models() -> None:
    """List the model catalog (design doc §2/§9)."""
    table = Table(title="Model catalog")
    for col in ("id", "name", "backend", "kind", "vocal SDR", "verified"):
        table.add_column(col, style="cyan" if col == "id" else None)
    for spec in MODEL_CATALOG.values():
        table.add_row(
            spec.id,
            spec.display_name,
            spec.backend.value,
            spec.kind.value,
            f"{spec.approx_vocal_sdr:.1f}" if spec.approx_vocal_sdr else "-",
            "[green]yes[/green]" if spec.verified else "[yellow]TBD[/yellow]",
        )
    console.print(table)


@cli.command("prefetch")
def prefetch(
    tier: str = typer.Argument("balanced", help="Tier whose models to download."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
) -> None:
    """Pre-download the checkpoints a tier needs so the first run isn't slow."""
    from .backends import resolve_backend

    cfg = _build_config(backend, None)
    plan = resolve_tier(Tier(tier.lower()), (Stem.VOCALS, Stem.INSTRUMENTAL))
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)
    for model in plan.models:
        be = resolve_backend(model, cfg)
        if not be.is_available():
            err_console.print(f"[yellow]skip[/yellow] {model.id}: backend {be.name} unavailable")
            continue
        console.print(f"prefetching [cyan]{model.id}[/cyan] ({model.checkpoint}) …")
        try:
            _warm_model(model, cfg)
            console.print("  [green]ok[/green]")
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"  [red]failed[/red]: {exc}")


def _warm_model(model, cfg) -> None:
    """Trigger a checkpoint download without running a full separation."""
    if model.backend is Backend.DEMUCS:
        from demucs.pretrained import get_model as _get

        _get(model.checkpoint)
        return
    from audio_separator.separator import Separator

    sep = Separator(model_file_dir=str(cfg.model_cache_dir))
    sep.load_model(model_filename=model.checkpoint)


def app() -> None:
    """Console-script entry: rewrite a bare ``splitwave <file>`` to ``separate <file>``."""
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in _SUBCOMMANDS:
        sys.argv.insert(1, "separate")
    cli()


if __name__ == "__main__":  # pragma: no cover
    app()
