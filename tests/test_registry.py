from __future__ import annotations

import pytest

from splitwave.errors import ModelNotAvailableError
from splitwave.registry import (
    MODEL_CATALOG,
    ModelKind,
    get_model,
    models_by_kind,
    separators_for_stem,
)
from splitwave.types import Stem


def test_get_model_unknown_raises():
    with pytest.raises(ModelNotAvailableError):
        get_model("does_not_exist")


def test_catalog_has_expected_models():
    for mid in ("mel_band_roformer_kim", "bs_roformer_1297", "htdemucs_ft", "uvr_deecho_dereverb"):
        assert mid in MODEL_CATALOG


def test_separators_for_vocals_sorted_by_sdr():
    seps = separators_for_stem(Stem.VOCALS)
    sdrs = [m.approx_vocal_sdr or 0 for m in seps]
    assert sdrs == sorted(sdrs, reverse=True)
    assert all(m.kind is ModelKind.SEPARATOR for m in seps)


def test_dereverb_model_is_dereverb_kind():
    derev = models_by_kind(ModelKind.DEREVERB)
    assert derev and all(m.kind is ModelKind.DEREVERB for m in derev)
    assert derev[0].can_produce(Stem.VOCALS_DRY)


def test_demucs_model_produces_four_stems():
    m = get_model("htdemucs_ft")
    for s in (Stem.DRUMS, Stem.BASS, Stem.OTHER, Stem.VOCALS):
        assert m.can_produce(s)
