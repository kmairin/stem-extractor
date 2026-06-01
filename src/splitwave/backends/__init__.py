"""Backend dispatch (Track A, design doc §4.1).

Maps a :class:`ModelSpec` to the backend that runs it and answers availability
queries for the CLI ``env-info`` command. Backend instances are stateless
wrappers (model weights load inside ``separate``), so they're cheap to build.
"""

from __future__ import annotations

from ..config import Backend, EngineConfig
from ..errors import BackendUnavailableError
from ..registry import ModelSpec
from .audio_separator import AudioSeparatorBackend
from .base import ProgressFn, SeparationBackend
from .demucs import DemucsBackend

__all__ = [
    "SeparationBackend",
    "ProgressFn",
    "AudioSeparatorBackend",
    "DemucsBackend",
    "resolve_backend",
    "available_backends",
]


def resolve_backend(model: ModelSpec, config: EngineConfig) -> SeparationBackend:
    """Return the backend instance that should run ``model`` under ``config``.

    ``model.backend`` decides the family; ``config.backend`` only narrows the
    audio-separator family toward MLX when explicitly requested. MLX is not yet
    implemented (design doc M4), so selecting it fails loudly rather than silently
    downgrading.
    """
    if model.backend is Backend.DEMUCS:
        return DemucsBackend(device=config.demucs_device)

    if model.backend is Backend.AUDIO_SEPARATOR:
        if config.backend is Backend.MLX:
            raise BackendUnavailableError(
                "MLX backend is not implemented yet (design doc M4); use the "
                "default audio-separator backend for RoFormer models."
            )
        return AudioSeparatorBackend(onnx_providers=config.onnx_providers)

    raise BackendUnavailableError(f"no backend for model backend {model.backend!r}")


def available_backends(config: EngineConfig | None = None) -> dict[str, bool]:
    """Map ``backend-name -> importable?`` for diagnostics (CLI ``env-info``)."""
    cfg = config or EngineConfig.default()
    return {
        Backend.AUDIO_SEPARATOR.value: AudioSeparatorBackend(
            onnx_providers=cfg.onnx_providers
        ).is_available(),
        Backend.DEMUCS.value: DemucsBackend(device=cfg.demucs_device).is_available(),
        Backend.MLX.value: False,
    }
