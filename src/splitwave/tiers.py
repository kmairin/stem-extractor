"""Tier resolver (Track A) — maps a quality tier + requested stems to a model plan.

Implements the tier table in design doc §4.2:

* ``fast``     → HT-Demucs FT (one 4-stem pass; vocals/instrumental derived).
* ``balanced`` → Mel-Band RoFormer (Kim); adds HT-Demucs only if 4-stem asked.
* ``best``     → ensemble of Mel-Band + BS-RoFormer; adds HT-Demucs for 4-stem.

The resolver does not run anything — it produces a :class:`TierPlan` the core
engine executes. Keeping this pure makes tier policy unit-testable without models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import Backend
from .errors import UnsupportedStemError
from .registry import ModelSpec, get_model
from .types import Stem, Tier

__all__ = ["TierPlan", "resolve_tier", "FOUR_STEM_ONLY"]

#: Stems that require a 4-stem (Demucs-class) decomposition.
FOUR_STEM_ONLY: frozenset[Stem] = frozenset({Stem.DRUMS, Stem.BASS, Stem.OTHER})

# Budget guardrail wording reused in CLI/server warnings (design doc §4.2).
_BEST_BUDGET_WARNING = (
    "'best' tier runs a 2-3 model ensemble and may exceed the ~2-min latency "
    "budget by design (design doc §4.2) — it is gated behind explicit selection."
)


@dataclass(frozen=True, slots=True)
class TierPlan:
    """An ordered set of models to run for a request, plus advisory warnings."""

    tier: Tier
    models: tuple[ModelSpec, ...]
    #: True when ``models`` 2-stem outputs are blended (best tier).
    ensemble: bool
    warnings: tuple[str, ...] = ()

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.models)

    @property
    def backends(self) -> set[Backend]:
        return {m.backend for m in self.models}

    @property
    def produced_stems(self) -> set[Stem]:
        """Separation stems this plan can yield (excludes the dereverb dry stem)."""
        out: set[Stem] = set()
        for m in self.models:
            out.update(m.stems)
            if Stem.VOCALS in m.stems:
                out.add(Stem.INSTRUMENTAL)
        return out


def _needs_four_stem(stems: Iterable[Stem]) -> bool:
    return any(s in FOUR_STEM_ONLY for s in stems)


def resolve_tier(tier: Tier, stems: Iterable[Stem]) -> TierPlan:
    """Resolve ``tier`` + requested ``stems`` into an executable :class:`TierPlan`.

    Raises :class:`UnsupportedStemError` if the resolved plan cannot cover a
    requested separation stem (the dereverb ``vocals_dry`` stem is satisfied by
    the post-chain, not here).
    """
    stem_tuple = tuple(stems)
    four = _needs_four_stem(stem_tuple)
    warnings: list[str] = []

    if tier is Tier.FAST:
        # One Demucs pass covers all stems (vocals/instrumental by recombination).
        models = [get_model("htdemucs_ft")]
        ensemble = False
    elif tier is Tier.BALANCED:
        models = [get_model("mel_band_roformer_kim")]
        ensemble = False
        if four:
            models.append(get_model("htdemucs_ft"))
            warnings.append(
                "4-stem output requested at 'balanced'; adding HT-Demucs for "
                "drums/bass/other while keeping the RoFormer vocal/instrumental."
            )
    elif tier is Tier.BEST:
        models = [get_model("mel_band_roformer_kim"), get_model("bs_roformer_1297")]
        ensemble = True
        if four:
            models.append(get_model("htdemucs_ft"))
        warnings.append(_BEST_BUDGET_WARNING)
    else:  # pragma: no cover - exhaustive over the enum
        raise UnsupportedStemError(f"unknown tier: {tier!r}")

    plan = TierPlan(tier=tier, models=tuple(models), ensemble=ensemble, warnings=tuple(warnings))

    # Coverage check at the planning boundary — fail loud, not mid-inference.
    produced = plan.produced_stems
    for stem in stem_tuple:
        if stem is Stem.VOCALS_DRY:
            continue  # produced by the dereverb post-chain
        if stem not in produced:
            raise UnsupportedStemError(
                f"tier {tier} cannot produce stem '{stem}'; produced: "
                f"{sorted(s.value for s in produced)}"
            )
    return plan
