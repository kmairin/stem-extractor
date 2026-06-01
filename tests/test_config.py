from __future__ import annotations

from pathlib import Path

import pytest

from splitwave.config import ChunkConfig, EngineConfig, SeparationRequest
from splitwave.types import Stem, Tier


def test_chunkconfig_validation():
    with pytest.raises(ValueError):
        ChunkConfig(segment_seconds=0)
    with pytest.raises(ValueError):
        ChunkConfig(overlap=1.0)
    ChunkConfig(segment_seconds=8, overlap=0.25)  # ok


def test_request_build_coerces_and_dedupes(tone_wav: Path, tmp_path: Path):
    req = SeparationRequest.build(
        tone_wav, tmp_path, tier="best", stems="vocal, instrumental ,vocals"
    )
    assert req.tier is Tier.BEST
    # 'vocal' and 'vocals' collapse to one Stem.VOCALS, order preserved
    assert req.stems == (Stem.VOCALS, Stem.INSTRUMENTAL)


def test_request_build_default_stems(tone_wav: Path, tmp_path: Path):
    req = SeparationRequest.build(tone_wav, tmp_path)
    assert req.stems == (Stem.VOCALS, Stem.INSTRUMENTAL)


def test_request_validate_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SeparationRequest.build(tmp_path / "nope.wav", tmp_path)


def test_request_dereverb_requires_vocals(tone_wav: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        SeparationRequest.build(tone_wav, tmp_path, stems="instrumental", dereverb=True)


def test_effective_stems_adds_dry_vocal(tone_wav: Path, tmp_path: Path):
    req = SeparationRequest.build(tone_wav, tmp_path, stems="vocals", dereverb=True)
    assert Stem.VOCALS_DRY in req.effective_stems
    assert Stem.VOCALS_DRY not in req.stems


def test_engineconfig_from_env(monkeypatch):
    monkeypatch.setenv("SPLITWAVE_BACKEND", "demucs")
    monkeypatch.setenv("SPLITWAVE_OUTPUT_FORMAT", "FLAC")
    cfg = EngineConfig.from_env()
    assert cfg.backend.value == "demucs"
    assert cfg.output_format == "flac"


def test_engineconfig_cache_dir_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SPLITWAVE_CACHE_DIR", str(tmp_path / "mc"))
    # default_factory reads env at construction time
    cfg = EngineConfig()
    assert cfg.model_cache_dir == tmp_path / "mc"
