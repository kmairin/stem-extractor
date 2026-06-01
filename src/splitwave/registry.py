"""Model registry — catalog of candidate checkpoints (Track B seed).

Seeded from design doc §2 (landscape) and §9 (references). Each :class:`ModelSpec`
carries the metadata the tier resolver (Track A) and benchmark harness (Track B)
need: which backend runs it, the checkpoint identifier, the stems it yields, and
rough quality/size figures.

NOTE ON CHECKPOINT IDS: ``checkpoint`` holds the backend-specific identifier
(audio-separator ``--model_filename`` or Demucs model name). All catalog
identifiers were confirmed against ``Separator.list_supported_model_files()``
on 2026-05-31 (task B2), so every spec is ``verified=True``. The backend still
fails loudly if an identifier cannot be resolved, so a future drift in the
upstream model list surfaces as a clear error rather than wrong output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import Backend
from .errors import ModelNotAvailableError
from .types import Stem

__all__ = [
    "ModelKind",
    "ModelSpec",
    "MODEL_CATALOG",
    "get_model",
    "models_by_kind",
    "separators_for_stem",
    "DEFAULT_DEREVERB_MODEL",
]

#: 2-stem outputs from a vocal-isolation model.
_TWO_STEM: tuple[Stem, ...] = (Stem.VOCALS, Stem.INSTRUMENTAL)
#: 4-stem outputs from a Demucs-class model.
_FOUR_STEM: tuple[Stem, ...] = (Stem.VOCALS, Stem.DRUMS, Stem.BASS, Stem.OTHER)


class ModelKind(str, Enum):
    SEPARATOR = "separator"
    DEREVERB = "dereverb"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static metadata for one model checkpoint."""

    id: str
    display_name: str
    backend: Backend
    checkpoint: str
    family: str
    kind: ModelKind
    stems: tuple[Stem, ...]
    native_sr: int = 44100
    #: Reported vocal SDR (dB) on MUSDB18-HQ-class sets; informational only.
    approx_vocal_sdr: float | None = None
    approx_size_mb: int | None = None
    #: True once Track B confirms the checkpoint identifier resolves.
    verified: bool = False
    source: str | None = None
    notes: str = ""

    def can_produce(self, stem: Stem) -> bool:
        if stem is Stem.VOCALS_DRY:
            return self.kind is ModelKind.DEREVERB
        if stem is Stem.INSTRUMENTAL:
            # Any model that yields vocals can yield instrumental by residual.
            return Stem.INSTRUMENTAL in self.stems or Stem.VOCALS in self.stems
        return stem in self.stems


# --- Catalog -----------------------------------------------------------------
# Quality figures are the doc's reported numbers (§2); treat as ballpark until
# Track B's harness measures them on our eval set.

_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="mel_band_roformer_kim",
        display_name="Mel-Band RoFormer (Kim)",
        backend=Backend.AUDIO_SEPARATOR,
        checkpoint="vocals_mel_band_roformer.ckpt",
        family="mel_band_roformer",
        kind=ModelKind.SEPARATOR,
        stems=_TWO_STEM,
        approx_vocal_sdr=12.9,
        approx_size_mb=650,
        verified=True,  # confirmed against audio-separator model list (B2)
        source="https://arxiv.org/abs/2310.01809",
        notes="Community default for vocals; balanced-tier default. (§2)",
    ),
    ModelSpec(
        id="bs_roformer_1297",
        display_name="BS-RoFormer (Viperx 1297)",
        backend=Backend.AUDIO_SEPARATOR,
        checkpoint="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        family="bs_roformer",
        kind=ModelKind.SEPARATOR,
        stems=_TWO_STEM,
        approx_vocal_sdr=12.97,
        approx_size_mb=650,
        verified=True,  # confirmed against audio-separator model list (B2)
        source="https://arxiv.org/abs/2309.02612",
        notes="SDX'23 winner family; strongest instrumental. (§2)",
    ),
    ModelSpec(
        id="bs_roformer_1296",
        display_name="BS-RoFormer (Viperx 1296)",
        backend=Backend.AUDIO_SEPARATOR,
        checkpoint="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        family="bs_roformer",
        kind=ModelKind.SEPARATOR,
        stems=_TWO_STEM,
        approx_vocal_sdr=12.96,
        approx_size_mb=650,
        verified=True,  # confirmed against audio-separator model list (B2)
        source="https://arxiv.org/abs/2309.02612",
        notes="Alternate BS-RoFormer checkpoint for ensembling. (§2)",
    ),
    ModelSpec(
        id="htdemucs_ft",
        display_name="HT-Demucs v4 (fine-tuned)",
        backend=Backend.DEMUCS,
        checkpoint="htdemucs_ft",
        family="htdemucs",
        kind=ModelKind.SEPARATOR,
        stems=_FOUR_STEM,
        approx_vocal_sdr=9.2,
        approx_size_mb=320,
        verified=True,  # stable Demucs model name
        source="https://github.com/facebookresearch/demucs",
        notes="Fast 4-stem tier; clean drums/bass. (§2)",
    ),
    ModelSpec(
        id="uvr_deecho_dereverb",
        display_name="UVR-DeEcho-DeReverb",
        backend=Backend.AUDIO_SEPARATOR,
        checkpoint="UVR-DeEcho-DeReverb.pth",
        family="uvr_dereverb",
        kind=ModelKind.DEREVERB,
        stems=(Stem.VOCALS_DRY,),
        approx_size_mb=60,
        verified=True,  # confirmed against audio-separator model list (B2)
        source="https://github.com/openmirlab/melband-roformer-infer",
        notes="Post-stage to dry out vocals; off by default. (§4.4)",
    ),
)

MODEL_CATALOG: dict[str, ModelSpec] = {spec.id: spec for spec in _SPECS}

#: Default de-reverb post-stage model (design doc §4.4).
DEFAULT_DEREVERB_MODEL = "uvr_deecho_dereverb"


def get_model(model_id: str) -> ModelSpec:
    """Look up a model by id, raising :class:`ModelNotAvailableError` if unknown."""
    try:
        return MODEL_CATALOG[model_id]
    except KeyError:
        known = ", ".join(sorted(MODEL_CATALOG))
        raise ModelNotAvailableError(
            f"unknown model id {model_id!r}; known models: {known}"
        ) from None


def models_by_kind(kind: ModelKind) -> list[ModelSpec]:
    return [m for m in MODEL_CATALOG.values() if m.kind is kind]


def separators_for_stem(stem: Stem) -> list[ModelSpec]:
    """Separator models that can produce ``stem``, best vocal SDR first."""
    matches = [
        m
        for m in MODEL_CATALOG.values()
        if m.kind is ModelKind.SEPARATOR and m.can_produce(stem)
    ]
    return sorted(matches, key=lambda m: m.approx_vocal_sdr or 0.0, reverse=True)
