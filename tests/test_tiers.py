from __future__ import annotations

import pytest

from splitwave.config import Backend
from splitwave.errors import UnsupportedStemError
from splitwave.tiers import resolve_tier
from splitwave.types import Stem, Tier


def test_fast_tier_uses_demucs_only():
    plan = resolve_tier(Tier.FAST, (Stem.VOCALS, Stem.INSTRUMENTAL))
    assert plan.model_ids == ("htdemucs_ft",)
    assert plan.backends == {Backend.DEMUCS}
    assert plan.ensemble is False


def test_balanced_tier_default_roformer():
    plan = resolve_tier(Tier.BALANCED, (Stem.VOCALS, Stem.INSTRUMENTAL))
    assert plan.model_ids == ("mel_band_roformer_kim",)
    assert plan.ensemble is False
    assert plan.warnings == ()


def test_balanced_tier_adds_demucs_for_four_stem():
    plan = resolve_tier(Tier.BALANCED, (Stem.VOCALS, Stem.DRUMS, Stem.BASS, Stem.OTHER))
    assert "mel_band_roformer_kim" in plan.model_ids
    assert "htdemucs_ft" in plan.model_ids
    assert any("4-stem" in w for w in plan.warnings)


def test_best_tier_is_ensemble_with_budget_warning():
    plan = resolve_tier(Tier.BEST, (Stem.VOCALS, Stem.INSTRUMENTAL))
    assert plan.ensemble is True
    assert set(plan.model_ids) == {"mel_band_roformer_kim", "bs_roformer_1297"}
    assert any("budget" in w for w in plan.warnings)


def test_dry_vocal_does_not_break_coverage():
    # VOCALS_DRY is satisfied by the post-chain, not the tier plan
    plan = resolve_tier(Tier.BALANCED, (Stem.VOCALS, Stem.VOCALS_DRY))
    assert Stem.VOCALS in plan.produced_stems


def test_produced_stems_includes_instrumental_from_vocals():
    plan = resolve_tier(Tier.BALANCED, (Stem.VOCALS,))
    assert Stem.INSTRUMENTAL in plan.produced_stems
