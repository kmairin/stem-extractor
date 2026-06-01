"""Engine orchestration tests using a fake backend (no ML deps required).

These verify Splitwave's *policy* — model→stem routing, ensembling, dereverb
post-chain, output writing — which is the part independent of real inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splitwave import SeparationEngine, get_engine
from splitwave.config import EngineConfig
from splitwave.core import Splitwave
from splitwave.types import Stem, Tier


def _engine(fake_factory, **overrides):
    cfg = EngineConfig.default().with_overrides(**overrides)
    return Splitwave(cfg, backend_factory=fake_factory)


def test_splitwave_satisfies_protocol():
    assert isinstance(get_engine(), SeparationEngine)


def test_balanced_two_stem(tone_wav: Path, tmp_path: Path, fake_factory):
    out = tmp_path / "stems"
    res = _engine(fake_factory).separate(
        tone_wav, tier="balanced", stems=["vocals", "instrumental"], out_dir=out
    )
    assert {sf.stem for sf in res.stems} == {Stem.VOCALS, Stem.INSTRUMENTAL}
    for sf in res.stems:
        assert sf.path.exists()
    assert res.models_used == ["mel_band_roformer_kim"]
    assert res.source_seconds == pytest.approx(1.0, abs=0.05)


def test_fast_tier_runs_without_torch(tone_wav: Path, tmp_path: Path, fake_factory):
    # htdemucs would need torch, but the injected fake backend stands in.
    res = _engine(fake_factory).separate(
        tone_wav, tier="fast", stems=["vocals", "instrumental"], out_dir=tmp_path
    )
    assert res.models_used == ["htdemucs_ft"]
    assert res.get(Stem.VOCALS).path.exists()


def test_four_stem_balanced(tone_wav: Path, tmp_path: Path, fake_factory):
    res = _engine(fake_factory).separate(
        tone_wav,
        tier="balanced",
        stems=["vocals", "drums", "bass", "other"],
        out_dir=tmp_path,
    )
    produced = {sf.stem for sf in res.stems}
    assert {Stem.VOCALS, Stem.DRUMS, Stem.BASS, Stem.OTHER} <= produced
    assert set(res.models_used) == {"mel_band_roformer_kim", "htdemucs_ft"}


def test_dereverb_emits_wet_and_dry(tone_wav: Path, tmp_path: Path, fake_factory):
    res = _engine(fake_factory).separate(
        tone_wav, tier="balanced", stems=["vocals"], dereverb=True, out_dir=tmp_path
    )
    wet = res.get(Stem.VOCALS)
    dry = res.get(Stem.VOCALS_DRY)
    assert wet is not None and wet.wet is True
    assert dry is not None and dry.path.exists()
    assert "uvr_deecho_dereverb" in res.models_used


def test_best_tier_ensembles(tone_wav: Path, tmp_path: Path, fake_factory):
    res = _engine(fake_factory).separate(
        tone_wav, tier="best", stems=["vocals", "instrumental"], out_dir=tmp_path
    )
    assert set(res.models_used) == {"mel_band_roformer_kim", "bs_roformer_1297"}
    assert all(sf.path.exists() for sf in res.stems)


def test_output_format_override(tone_wav: Path, tmp_path: Path, fake_factory):
    res = _engine(fake_factory, output_format="flac").separate(
        tone_wav, tier="balanced", stems=["vocals"], out_dir=tmp_path
    )
    assert res.get(Stem.VOCALS).path.suffix == ".flac"


def test_timings_recorded(tone_wav: Path, tmp_path: Path, fake_factory):
    res = _engine(fake_factory).separate(
        tone_wav, tier="balanced", stems=["vocals"], out_dir=tmp_path
    )
    names = {t.name for t in res.timings}
    assert "decode" in names
    assert any(t.name == "inference" for t in res.timings)
    assert res.wall_seconds >= 0
