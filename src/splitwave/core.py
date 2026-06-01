"""The concrete ``Splitwave`` engine (Track A) — orchestration over the backends.

Pipeline per request (design doc §4):

1. decode the source once;
2. resolve the tier into a model plan (:mod:`splitwave.tiers`);
3. run the plan — RoFormer model(s) for vocals/instrumental (ensembled on the
   ``best`` tier), Demucs for drums/bass/other — resampling per model rate;
4. optional de-reverb post-chain producing the dry vocal (design doc §4.4);
5. restore the output sample rate and write each requested stem;
6. return a :class:`SeparationResult` with per-stage timings.

Backend inference is delegated; this module owns *policy* (which model feeds
which stem, ensembling, I/O), which is what we can verify with a fake backend.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio import AudioBuffer, load_audio, resample, save_audio
from .backends import ProgressFn, SeparationBackend, resolve_backend
from .base import BaseSeparationEngine
from .config import EngineConfig, SeparationRequest
from .errors import UnsupportedStemError
from .registry import DEFAULT_DEREVERB_MODEL, ModelSpec, get_model
from .tiers import TierPlan, resolve_tier
from .types import SeparationResult, StageTiming, Stem, StemFile

__all__ = ["Splitwave"]

_VOCAL_FAMILY = (Stem.VOCALS, Stem.INSTRUMENTAL)
_FOUR_STEM = (Stem.DRUMS, Stem.BASS, Stem.OTHER)


class Splitwave(BaseSeparationEngine):
    """Default engine implementing the frozen :class:`SeparationEngine` interface."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        progress: ProgressFn | None = None,
        backend_factory=resolve_backend,
    ) -> None:
        super().__init__(config)
        self._progress = progress
        # Injectable for tests (a fake backend) — defaults to real dispatch.
        self._backend_factory = backend_factory

    # -- helpers --------------------------------------------------------------
    def _emit(self, msg: str) -> None:
        if self._progress:
            self._progress(msg)

    def _backend(self, model: ModelSpec) -> SeparationBackend:
        return self._backend_factory(model, self.config)

    def _run_model(
        self,
        model: ModelSpec,
        source: AudioBuffer,
        rate_cache: dict[int, AudioBuffer],
        stems: set[Stem],
        timings: list[StageTiming],
    ) -> dict[Stem, AudioBuffer]:
        buf = rate_cache.get(model.native_sr)
        if buf is None:
            buf = source if source.sr == model.native_sr else resample(source, model.native_sr)
            rate_cache[model.native_sr] = buf
        backend = self._backend(model)
        t0 = time.perf_counter()
        out = backend.separate(
            buf,
            model,
            stems=tuple(stems),
            chunk=self.config.chunk,
            cache_dir=self.config.model_cache_dir,
            progress=self._progress,
        )
        timings.append(
            StageTiming(name="inference", seconds=time.perf_counter() - t0, model=model.id)
        )
        return out

    # -- main entry -----------------------------------------------------------
    def _run(self, request: SeparationRequest) -> SeparationResult:
        wall0 = time.perf_counter()
        plan = resolve_tier(request.tier, request.stems)
        for w in plan.warnings:
            self._emit(w)

        self._emit(f"decoding {request.input_path.name}")
        t0 = time.perf_counter()
        source = load_audio(request.input_path)
        timings: list[StageTiming] = [
            StageTiming(name="decode", seconds=time.perf_counter() - t0)
        ]
        rate_cache: dict[int, AudioBuffer] = {source.sr: source}

        separation_stems = [s for s in request.stems if s is not Stem.VOCALS_DRY]
        final: dict[Stem, AudioBuffer] = {}
        models_used: list[str] = []

        two_stem_models = [m for m in plan.models if Stem.DRUMS not in m.stems]
        four_stem_models = [m for m in plan.models if Stem.DRUMS in m.stems]

        # 1. Vocal-family stems via RoFormer (ensembled across models on `best`).
        vocal_targets = {s for s in separation_stems if s in _VOCAL_FAMILY}
        if two_stem_models and vocal_targets:
            ask = set(vocal_targets) | {Stem.VOCALS}  # vocals needed to derive instrumental
            per_model = []
            for m in two_stem_models:
                per_model.append(self._run_model(m, source, rate_cache, ask, timings))
                models_used.append(m.id)
            combined = _ensemble(per_model) if len(per_model) > 1 else per_model[0]
            for s in vocal_targets:
                final[s] = combined[s]

        # 2. Drums/bass/other (and vocals/instrumental on the fast tier) via Demucs.
        four_targets = {s for s in separation_stems if s in _FOUR_STEM}
        if not two_stem_models:
            four_targets |= vocal_targets  # fast tier: Demucs supplies everything
        if four_stem_models and four_targets:
            m = four_stem_models[0]
            out = self._run_model(m, source, rate_cache, four_targets, timings)
            for s in four_targets:
                final[s] = out[s]
            models_used.append(m.id)

        missing = set(separation_stems) - set(final)
        if missing:
            raise UnsupportedStemError(
                f"engine did not produce requested stems: "
                f"{sorted(s.value for s in missing)}"
            )

        # 3. Optional de-reverb post-chain -> dry vocal (design doc §4.4).
        if request.dereverb:
            final[Stem.VOCALS_DRY] = self._dereverb(final[Stem.VOCALS], rate_cache, timings)
            models_used.append(DEFAULT_DEREVERB_MODEL)

        # 4. Restore output rate + write.
        out_sr = self.config.output_sample_rate or source.sr
        result = SeparationResult(
            input_path=request.input_path,
            out_dir=request.out_dir,
            tier=request.tier,
            source_seconds=source.duration,
        )
        result.models_used = list(dict.fromkeys(models_used))
        for stem in request.effective_stems:
            buf = final.get(stem)
            if buf is None:
                continue
            written = self._write_stem(request, stem, buf, out_sr, timings)
            wet = stem is Stem.VOCALS and request.dereverb
            result.stems.append(
                StemFile(stem=stem, path=written, sample_rate=out_sr, wet=wet)
            )

        result.timings = timings
        result.wall_seconds = time.perf_counter() - wall0
        self._emit(f"done in {result.wall_seconds:.1f}s -> {request.out_dir}")
        return result

    def _dereverb(
        self,
        wet_vocals: AudioBuffer,
        rate_cache: dict[int, AudioBuffer],
        timings: list[StageTiming],
    ) -> AudioBuffer:
        model = get_model(DEFAULT_DEREVERB_MODEL)
        backend = self._backend(model)
        buf = (
            wet_vocals
            if wet_vocals.sr == model.native_sr
            else resample(wet_vocals, model.native_sr)
        )
        self._emit("de-reverbing vocals")
        t0 = time.perf_counter()
        out = backend.separate(
            buf,
            model,
            stems=(Stem.VOCALS_DRY,),
            chunk=self.config.chunk,
            cache_dir=self.config.model_cache_dir,
            progress=self._progress,
        )
        timings.append(
            StageTiming(name="dereverb", seconds=time.perf_counter() - t0, model=model.id)
        )
        return out[Stem.VOCALS_DRY]

    def _write_stem(
        self,
        request: SeparationRequest,
        stem: Stem,
        buf: AudioBuffer,
        out_sr: int,
        timings: list[StageTiming],
    ) -> Path:
        if buf.sr != out_sr:
            buf = resample(buf, out_sr)
        fmt = self.config.output_format
        name = f"{request.input_path.stem}_{stem.value}.{fmt}"
        path = request.out_dir / name
        t0 = time.perf_counter()
        save_audio(buf, path, fmt=fmt)
        timings.append(
            StageTiming(name=f"write:{stem.value}", seconds=time.perf_counter() - t0)
        )
        return path


def _ensemble(outputs: Sequence[dict[Stem, AudioBuffer]]) -> dict[Stem, AudioBuffer]:
    """Average per-stem buffers across models (best-tier ensembling, design doc §2)."""
    stems = set(outputs[0])
    for o in outputs[1:]:
        stems &= set(o)
    combined: dict[Stem, AudioBuffer] = {}
    for stem in stems:
        combined[stem] = _average([o[stem] for o in outputs])
    return combined


def _average(buffers: Sequence[AudioBuffer]) -> AudioBuffer:
    n = min(b.n_samples for b in buffers)
    ch = min(b.channels for b in buffers)
    acc = np.zeros((n, ch), dtype=np.float32)
    for b in buffers:
        acc += b.samples[:n, :ch]
    acc /= len(buffers)
    return AudioBuffer(acc, buffers[0].sr)
