"""Benchmark harness (Track B2) — measure latency + separation quality.

This is the tool that turns "is HT-Demucs actually best for drums/bass?" into a
number. It runs the real engine over a matrix of (input × tier) and records:

* **latency** — wall-clock, real-time factor, and a decode/inference split read
  off :class:`~splitwave.types.StageTiming`; works on *any* input;
* **reconstruction fidelity** — how well a *complete* stem set sums back to the
  mixture (reference-free, design doc §3 P0.2); and
* **SI-SDR** — true scale-invariant separation accuracy, but only when caller
  supplies ground-truth reference stems (e.g. a MUSDB track or your own
  acapella/instrumental).

The metric functions are pure (numpy in, float out) so they unit-test without
any ML stack; :func:`run_benchmark` is the only part that touches a real engine.

Honesty note: without reference stems we **cannot** measure true accuracy — only
latency and reconstruction. SI-SDR columns are blank in that mode by design,
rather than reporting a misleading proxy as if it were quality.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .audio import AudioBuffer, load_audio
from .base import SeparationEngine
from .types import SeparationResult, Stem, Tier

__all__ = [
    "si_sdr",
    "reconstruction_db",
    "max_stem_correlation",
    "BenchCase",
    "RunMetric",
    "run_case",
    "run_benchmark",
    "results_to_dicts",
    "write_json",
    "COMPLETE_STEM_SETS",
]

#: Stem sets that should sum back to the mixture (used for reconstruction).
COMPLETE_STEM_SETS: tuple[frozenset[Stem], ...] = (
    frozenset({Stem.VOCALS, Stem.INSTRUMENTAL}),
    frozenset({Stem.VOCALS, Stem.DRUMS, Stem.BASS, Stem.OTHER}),
)

_EPS = 1e-8


# --- metric primitives -------------------------------------------------------
def _mono(x: np.ndarray) -> np.ndarray:
    """Collapse ``(n, channels)`` (or 1-D) to a mean-mono float64 1-D signal."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.reshape(-1)


def _align(*signals: np.ndarray) -> list[np.ndarray]:
    """Trim a set of 1-D signals to their common (shortest) length."""
    n = min(s.shape[0] for s in signals)
    return [s[:n] for s in signals]


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (Le Roux et al., 2019).

    Higher is better; the metric is invariant to a global gain on ``estimate``,
    which is what makes it the modern default over raw BSS-Eval SDR. Inputs may
    be mono or ``(n, channels)``; both are mixed to mono and length-aligned.
    """
    est, ref = _align(_mono(estimate), _mono(reference))
    ref = ref - ref.mean()
    est = est - est.mean()
    ref_energy = float(ref @ ref) + _EPS
    proj = (float(est @ ref) / ref_energy) * ref
    noise = est - proj
    ratio = (float(proj @ proj) + _EPS) / (float(noise @ noise) + _EPS)
    return 10.0 * float(np.log10(ratio))


def reconstruction_db(mixture: np.ndarray, stems: Sequence[np.ndarray]) -> float:
    """How well summed ``stems`` reconstruct ``mixture``, in dB (higher better).

    ``10·log10(‖mix‖² / ‖mix − Σstems‖²)``. For a complete decomposition a good
    model lands well above ~20 dB; a lossy or hallucinating model drops toward 0.
    Reference-free — it only needs the mixture the engine was given.
    """
    mono_stems = [_mono(s) for s in stems]
    mix = _mono(mixture)
    aligned = _align(mix, *mono_stems)
    mix = aligned[0]
    summed = np.sum(aligned[1:], axis=0)
    residual = mix - summed
    ratio = (float(mix @ mix) + _EPS) / (float(residual @ residual) + _EPS)
    return 10.0 * float(np.log10(ratio))


def max_stem_correlation(stems: Mapping[Stem, np.ndarray]) -> float | None:
    """Largest abs Pearson correlation between any two stems (leakage proxy).

    A *heuristic* only: high correlation often means one stem is bleeding into
    another, but uncorrelated stems can still be badly separated. Returned as a
    secondary signal, never as the headline quality number. ``None`` for <2 stems.
    """
    items = list(stems.items())
    if len(items) < 2:
        return None
    monos = {s: _mono(a) for s, a in items}
    worst = 0.0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = _align(monos[items[i][0]], monos[items[j][0]])
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.sqrt((a @ a) * (b @ b))) + _EPS
            worst = max(worst, abs(float(a @ b) / denom))
    return worst


# --- benchmark cases + results ----------------------------------------------
@dataclass(frozen=True, slots=True)
class BenchCase:
    """One input to benchmark, with optional ground-truth reference stems."""

    mixture: Path
    references: Mapping[Stem, Path] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.mixture).name


@dataclass(slots=True)
class RunMetric:
    """Latency + quality for one (input, tier) separation run."""

    input_name: str
    tier: Tier
    models: tuple[str, ...]
    source_seconds: float
    wall_seconds: float
    realtime_factor: float | None
    decode_seconds: float
    inference_seconds: float
    reconstruction_db: float | None = None
    stem_si_sdr: dict[str, float] = field(default_factory=dict)
    max_stem_correlation: float | None = None
    error: str | None = None

    @property
    def mean_si_sdr(self) -> float | None:
        vals = list(self.stem_si_sdr.values())
        return float(np.mean(vals)) if vals else None


def _timing_split(result: SeparationResult) -> tuple[float, float]:
    """Sum decode vs. inference seconds from the result's stage timings."""
    decode = sum(t.seconds for t in result.timings if t.name == "decode")
    infer = sum(t.seconds for t in result.timings if t.name == "inference")
    return decode, infer


def _is_complete(stems: Iterable[Stem]) -> bool:
    s = frozenset(stems)
    return any(req <= s for req in COMPLETE_STEM_SETS)


def run_case(
    engine: SeparationEngine,
    case: BenchCase,
    tier: Tier,
    stems: Sequence[Stem],
    *,
    out_dir: Path | None = None,
) -> RunMetric:
    """Separate one case at one tier and compute its metrics.

    Stems are written under ``out_dir`` (a throwaway temp dir if omitted) and
    read back for reconstruction/SI-SDR, so the harness measures the real files
    the engine produces — not just in-memory tensors.
    """
    tmp_ctx = tempfile.TemporaryDirectory() if out_dir is None else None
    target = Path(tmp_ctx.name) if tmp_ctx else Path(out_dir)  # type: ignore[union-attr]
    try:
        try:
            result = engine.separate(
                case.mixture, tier=tier, stems=tuple(stems), out_dir=target
            )
        except Exception as exc:  # noqa: BLE001 - record, don't abort the matrix
            return RunMetric(
                input_name=case.name,
                tier=Tier(tier),
                models=(),
                source_seconds=0.0,
                wall_seconds=0.0,
                realtime_factor=None,
                decode_seconds=0.0,
                inference_seconds=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        decode_s, infer_s = _timing_split(result)
        produced = {sf.stem: load_audio(sf.path) for sf in result.stems}

        recon: float | None = None
        if _is_complete(produced):
            mix = load_audio(case.mixture)
            recon = reconstruction_db(
                mix.samples, [produced[s].samples for s in produced]
            )

        si: dict[str, float] = {}
        for stem, ref_path in case.references.items():
            if stem in produced:
                ref = load_audio(ref_path)
                si[stem.value] = si_sdr(produced[stem].samples, ref.samples)

        corr = max_stem_correlation({s: b.samples for s, b in produced.items()})

        return RunMetric(
            input_name=case.name,
            tier=Tier(tier),
            models=tuple(result.models_used),
            source_seconds=result.source_seconds or 0.0,
            wall_seconds=result.wall_seconds,
            realtime_factor=result.realtime_factor,
            decode_seconds=decode_s,
            inference_seconds=infer_s,
            reconstruction_db=recon,
            stem_si_sdr=si,
            max_stem_correlation=corr,
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def run_benchmark(
    engine: SeparationEngine,
    cases: Sequence[BenchCase],
    tiers: Sequence[Tier],
    stems: Sequence[Stem],
    *,
    progress=None,
) -> list[RunMetric]:
    """Run the full (case × tier) matrix and return one :class:`RunMetric` each."""
    results: list[RunMetric] = []
    for case in cases:
        for tier in tiers:
            if progress:
                progress(f"benchmarking {case.name} @ {Tier(tier)}")
            results.append(run_case(engine, case, Tier(tier), stems))
    return results


def results_to_dicts(results: Sequence[RunMetric]) -> list[dict]:
    """JSON-serializable view of benchmark results (for --json / tracking)."""
    out: list[dict] = []
    for r in results:
        out.append(
            {
                "input": r.input_name,
                "tier": r.tier.value,
                "models": list(r.models),
                "source_seconds": round(r.source_seconds, 3),
                "wall_seconds": round(r.wall_seconds, 3),
                "realtime_factor": round(r.realtime_factor, 3) if r.realtime_factor else None,
                "decode_seconds": round(r.decode_seconds, 3),
                "inference_seconds": round(r.inference_seconds, 3),
                "reconstruction_db": round(r.reconstruction_db, 2)
                if r.reconstruction_db is not None
                else None,
                "stem_si_sdr": {k: round(v, 2) for k, v in r.stem_si_sdr.items()},
                "mean_si_sdr": round(r.mean_si_sdr, 2) if r.mean_si_sdr is not None else None,
                "max_stem_correlation": round(r.max_stem_correlation, 3)
                if r.max_stem_correlation is not None
                else None,
                "error": r.error,
            }
        )
    return out


def write_json(results: Sequence[RunMetric], path: Path) -> Path:
    """Write benchmark results to ``path`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results_to_dicts(results), indent=2))
    return path
