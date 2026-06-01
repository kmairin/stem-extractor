from __future__ import annotations

from pathlib import Path

import pytest

from splitwave.types import SeparationResult, StageTiming, Stem, StemFile, Tier


def test_stem_coerce_aliases():
    assert Stem.coerce("vocal") is Stem.VOCALS
    assert Stem.coerce("No_Vocals") is Stem.INSTRUMENTAL
    assert Stem.coerce("DRY") is Stem.VOCALS_DRY
    assert Stem.coerce(Stem.BASS) is Stem.BASS


def test_stem_coerce_unknown_raises():
    with pytest.raises(ValueError):
        Stem.coerce("kazoo")


def test_tier_str_is_value():
    assert str(Tier.BALANCED) == "balanced"
    assert Tier("fast") is Tier.FAST


def test_stemfile_coerces_path():
    sf = StemFile(stem=Stem.VOCALS, path="x/y.wav", sample_rate=44100)
    assert isinstance(sf.path, Path)


def test_result_lookup_and_paths():
    res = SeparationResult(
        input_path="in.wav",
        out_dir="out",
        tier=Tier.BALANCED,
        stems=[
            StemFile(Stem.VOCALS, "out/in_vocals.wav", 44100, wet=True),
            StemFile(Stem.VOCALS_DRY, "out/in_vocals_dry.wav", 44100),
        ],
        source_seconds=120.0,
        wall_seconds=60.0,
    )
    assert res.get(Stem.VOCALS, wet=True) is not None
    assert res.get("vocals_dry") is not None
    assert "vocals" in res.stem_paths
    assert res.realtime_factor == pytest.approx(2.0)


def test_realtime_factor_none_without_timing():
    res = SeparationResult(input_path="in.wav", out_dir="out", tier=Tier.FAST)
    assert res.realtime_factor is None
