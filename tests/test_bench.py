"""Benchmark harness tests (Track B2).

Metric functions are pure numpy and tested against known values; the
orchestration (`run_case`/`run_benchmark`) is exercised with the fake backend so
no ML stack or model downloads are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from splitwave.bench import (
    BenchCase,
    max_stem_correlation,
    reconstruction_db,
    results_to_dicts,
    run_benchmark,
    run_case,
    si_sdr,
    write_json,
)
from splitwave.config import EngineConfig
from splitwave.core import Splitwave
from splitwave.types import Stem, Tier


def _engine(fake_factory, **overrides):
    cfg = EngineConfig.default().with_overrides(**overrides)
    return Splitwave(cfg, backend_factory=fake_factory)


# --- metric primitives -------------------------------------------------------
@pytest.fixture
def ref_signal() -> np.ndarray:
    return np.random.default_rng(7).standard_normal(8000).astype(np.float32)


def test_si_sdr_identity_is_high(ref_signal):
    assert si_sdr(ref_signal, ref_signal) > 50


def test_si_sdr_is_scale_invariant(ref_signal):
    base = si_sdr(ref_signal, ref_signal)
    scaled = si_sdr(3.0 * ref_signal, ref_signal)
    # A pure gain must not change SI-SDR meaningfully.
    assert abs(scaled - base) < 1.0 or (scaled > 50 and base > 50)


def test_si_sdr_degrades_with_noise(ref_signal):
    rng = np.random.default_rng(1)
    noisy = ref_signal + 0.1 * rng.standard_normal(ref_signal.shape).astype(np.float32)
    # 1/0.1**2 power ratio ~ 20 dB.
    assert 15 < si_sdr(noisy, ref_signal) < 25
    assert si_sdr(rng.standard_normal(ref_signal.shape), ref_signal) < 5


def test_si_sdr_accepts_stereo():
    rng = np.random.default_rng(2)
    stereo = rng.standard_normal((4000, 2)).astype(np.float32)
    assert si_sdr(stereo, stereo) > 50


def test_reconstruction_exact_is_high():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2000, 2)).astype(np.float32)
    b = rng.standard_normal((2000, 2)).astype(np.float32)
    assert reconstruction_db(a + b, [a, b]) > 50


def test_reconstruction_lossy_is_finite_and_lower():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((2000, 2)).astype(np.float32)
    b = rng.standard_normal((2000, 2)).astype(np.float32)
    exact = reconstruction_db(a + b, [a, b])
    lossy = reconstruction_db(a + b, [a, 0.5 * b])
    assert lossy < exact
    assert np.isfinite(lossy)


def test_reconstruction_aligns_unequal_lengths():
    rng = np.random.default_rng(5)
    mix = rng.standard_normal((2000, 1)).astype(np.float32)
    # Shorter stem: function should trim to the common length, not crash.
    assert np.isfinite(reconstruction_db(mix, [mix[:1500]]))


def test_max_stem_correlation():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((3000, 2)).astype(np.float32)
    b = rng.standard_normal((3000, 2)).astype(np.float32)
    assert max_stem_correlation({Stem.VOCALS: a, Stem.DRUMS: a}) == pytest.approx(1.0, abs=1e-6)
    assert max_stem_correlation({Stem.VOCALS: a, Stem.DRUMS: b}) < 0.1
    assert max_stem_correlation({Stem.VOCALS: a}) is None


# --- orchestration (fake backend) -------------------------------------------
def test_run_case_records_latency_and_reconstruction(tone_wav: Path, fake_factory):
    case = BenchCase(mixture=tone_wav)
    m = run_case(_engine(fake_factory), case, Tier.BALANCED, [Stem.VOCALS, Stem.INSTRUMENTAL])
    assert m.error is None
    assert m.models == ("mel_band_roformer_kim",)
    assert m.source_seconds == pytest.approx(1.0, abs=0.05)
    assert m.wall_seconds >= 0
    # vocals+instrumental is a complete decomposition -> reconstruction computed.
    assert m.reconstruction_db is not None
    assert m.max_stem_correlation is not None
    assert m.stem_si_sdr == {}  # no references provided


def test_run_case_scores_si_sdr_when_refs_given(tone_wav: Path, fake_factory):
    # Fake stems are scalar multiples of the mix; SI-SDR is scale-invariant, so
    # scoring against the mix itself yields a high value for every stem.
    case = BenchCase(mixture=tone_wav, references={Stem.VOCALS: tone_wav})
    m = run_case(_engine(fake_factory), case, Tier.BALANCED, [Stem.VOCALS, Stem.INSTRUMENTAL])
    assert "vocals" in m.stem_si_sdr
    assert m.stem_si_sdr["vocals"] > 40
    assert m.mean_si_sdr is not None


def test_run_case_records_error_for_missing_file(tmp_path: Path, fake_factory):
    case = BenchCase(mixture=tmp_path / "nope.wav")
    m = run_case(_engine(fake_factory), case, Tier.FAST, [Stem.VOCALS])
    assert m.error is not None
    assert m.realtime_factor is None


def test_run_benchmark_matrix(tone_wav: Path, fake_factory):
    case = BenchCase(mixture=tone_wav)
    results = run_benchmark(
        _engine(fake_factory),
        [case],
        [Tier.FAST, Tier.BALANCED],
        [Stem.VOCALS, Stem.INSTRUMENTAL],
    )
    assert len(results) == 2
    assert {r.tier for r in results} == {Tier.FAST, Tier.BALANCED}
    assert all(r.error is None for r in results)


def test_results_to_dicts_and_json(tone_wav: Path, fake_factory, tmp_path: Path):
    case = BenchCase(mixture=tone_wav)
    results = run_benchmark(_engine(fake_factory), [case], [Tier.BALANCED], [Stem.VOCALS])
    dicts = results_to_dicts(results)
    assert dicts and dicts[0]["tier"] == "balanced"
    out = write_json(results, tmp_path / "bench.json")
    assert out.exists()
