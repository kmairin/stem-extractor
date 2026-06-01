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
_SUBCOMMANDS = {"separate", "env-info", "models", "prefetch", "bench"}


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


def _discover_refs(refs_dir: Path, input_path: Path, stems: list[Stem]) -> dict:
    """Match ground-truth stem files in ``refs_dir`` to requested stems.

    Accepts ``<stem>.<ext>`` (shared refs) or ``<input-name>_<stem>.<ext>``
    (per-track refs); the per-track form wins. Used to enable real SI-SDR.
    """
    if not refs_dir.is_dir():
        return {}
    base = input_path.stem.lower()
    found: dict = {}
    candidates = [p for p in refs_dir.iterdir() if p.is_file()]
    for stem in stems:
        per_track = None
        shared = None
        for p in candidates:
            name = p.stem.lower()
            if name == f"{base}_{stem.value}":
                per_track = p
            elif name == stem.value:
                shared = p
        match = per_track or shared
        if match:
            found[stem] = match
    return found


@cli.command("bench")
def bench(
    inputs: list[Path] = typer.Argument(..., help="Audio file(s) to benchmark."),
    tiers: str = typer.Option("fast,balanced,best", "--tiers", help="Comma-separated tiers."),
    stems: str = typer.Option("vocals,instrumental", "--stems", "-s", help="Stem set to request."),
    refs: Optional[Path] = typer.Option(
        None, "--refs", help="Dir of ground-truth stems to score SI-SDR against."
    ),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write results JSON here."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Force a backend."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
) -> None:
    """Measure latency + quality across tiers, then lock defaults on evidence (B2)."""
    from .bench import BenchCase, run_case, write_json

    progress = None if quiet else (lambda m: err_console.log(f"[dim]{m}[/dim]"))
    cfg = _build_config(backend, None)
    engine = Splitwave(cfg, progress=progress)

    stem_list = [Stem.coerce(s) for s in stems.split(",") if s.strip()]
    tier_list = [Tier(t.strip().lower()) for t in tiers.split(",") if t.strip()]

    results = []
    for inp in inputs:
        ref_map = _discover_refs(refs, inp, stem_list) if refs else {}
        if refs and not ref_map:
            err_console.print(f"[yellow]no reference stems matched for {inp.name}[/yellow]")
        case = BenchCase(mixture=inp, references=ref_map)
        for tier in tier_list:
            if progress:
                progress(f"benchmarking {inp.name} @ {tier}")
            results.append(run_case(engine, case, tier, stem_list))

    scored = any(r.stem_si_sdr for r in results)
    _print_bench(results, scored=scored)
    if json_out:
        write_json(results, json_out)
        console.print(f"[dim]wrote {json_out}[/dim]")


def _print_bench(results, *, scored: bool) -> None:
    table = Table(title="Benchmark (latency + quality)")
    table.add_column("input", style="cyan", no_wrap=False)
    table.add_column("tier")
    table.add_column("models")
    table.add_column("wall s", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("infer s", justify="right")
    table.add_column("recon dB", justify="right")
    if scored:
        table.add_column("SI-SDR dB", justify="right")
    table.add_column("corr", justify="right")
    for r in results:
        if r.error:
            table.add_row(r.input_name, str(r.tier), f"[red]{r.error}[/red]", "", "", "", "", *([""] if scored else []), "")
            continue
        rtf = f"{r.realtime_factor:.2f}x" if r.realtime_factor else "-"
        recon = f"{r.reconstruction_db:.1f}" if r.reconstruction_db is not None else "-"
        corr = f"{r.max_stem_correlation:.3f}" if r.max_stem_correlation is not None else "-"
        row = [
            r.input_name,
            str(r.tier),
            ", ".join(r.models),
            f"{r.wall_seconds:.1f}",
            rtf,
            f"{r.inference_seconds:.1f}",
            recon,
        ]
        if scored:
            row.append(f"{r.mean_si_sdr:.1f}" if r.mean_si_sdr is not None else "-")
        row.append(corr)
        table.add_row(*row)
    console.print(table)
    if not scored:
        console.print(
            "[dim]no --refs given: SI-SDR (true accuracy) not measured; "
            "recon dB and corr are reference-free signals only.[/dim]"
        )


def app() -> None:
    """Console-script entry: rewrite a bare ``splitwave <file>`` to ``separate <file>``."""
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in _SUBCOMMANDS:
        sys.argv.insert(1, "separate")
    cli()


if __name__ == "__main__":  # pragma: no cover
    app()
