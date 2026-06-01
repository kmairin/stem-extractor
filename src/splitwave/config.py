"""Configuration schema + request contract — part of the Milestone-0 frozen interface.

``EngineConfig`` is *how* the engine runs (backend, devices, caching, chunking);
``SeparationRequest`` is *what* a single job asks for (input, tier, stems,
dereverb). Both are plain dataclasses so they serialize trivially for the server
(Track C) and CLI (Track E).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from .types import Stem, Tier

__all__ = [
    "Backend",
    "ChunkConfig",
    "EngineConfig",
    "SeparationRequest",
    "DEFAULT_STEMS",
]

#: Default 2-stem split (design doc §3 P0.1).
DEFAULT_STEMS: tuple[Stem, ...] = (Stem.VOCALS, Stem.INSTRUMENTAL)


class Backend(str, Enum):
    """Inference backend selector (design doc §4.1).

    ``AUTO`` lets the engine pick the best available backend for the host
    (MLX > audio-separator/CoreML on Apple Silicon, CPU otherwise).
    """

    AUTO = "auto"
    AUDIO_SEPARATOR = "audio_separator"  # UVR-derived, ONNX + CoreML EP (default)
    MLX = "mlx"                          # Apple-Silicon-native (opt-in)
    DEMUCS = "demucs"                    # PyTorch + MPS, 4-stem fast tier

    def __str__(self) -> str:
        return self.value


def _default_cache_dir() -> Path:
    """Model cache root, overridable via ``SPLITWAVE_CACHE_DIR``."""
    env = os.environ.get("SPLITWAVE_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "splitwave" / "models"


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Overlap-add chunking for long audio (design doc §4.3).

    Bounds peak memory and avoids edge artifacts. ``segment_seconds`` is the
    window length; ``overlap`` is the fraction (0–1) shared between adjacent
    windows and cross-faded on recombine.
    """

    enabled: bool = True
    segment_seconds: float = 10.0
    overlap: float = 0.25

    def __post_init__(self) -> None:
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be > 0")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """How the engine runs. Construct via :meth:`default` or :meth:`from_env`."""

    backend: Backend = Backend.AUTO
    model_cache_dir: Path = field(default_factory=_default_cache_dir)
    #: Output container format (passed to soundfile/ffmpeg).
    output_format: str = "wav"
    #: Output sample rate; ``None`` preserves the source rate.
    output_sample_rate: int | None = None
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    #: Torch device for the Demucs backend (``mps`` on Apple Silicon).
    demucs_device: str = "mps"
    #: ONNX Runtime execution providers, in priority order (design doc §4.1).
    onnx_providers: tuple[str, ...] = (
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_cache_dir", Path(self.model_cache_dir))

    @classmethod
    def default(cls) -> "EngineConfig":
        return cls()

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Build a config from ``SPLITWAVE_*`` environment variables.

        Recognized: ``SPLITWAVE_BACKEND``, ``SPLITWAVE_CACHE_DIR``,
        ``SPLITWAVE_OUTPUT_FORMAT``, ``SPLITWAVE_DEMUCS_DEVICE``,
        ``SPLITWAVE_LOG_LEVEL``. Unset variables fall back to the defaults.
        """
        cfg = cls()
        backend = os.environ.get("SPLITWAVE_BACKEND")
        if backend:
            cfg = replace(cfg, backend=Backend(backend.lower()))
        fmt = os.environ.get("SPLITWAVE_OUTPUT_FORMAT")
        if fmt:
            cfg = replace(cfg, output_format=fmt.lower())
        device = os.environ.get("SPLITWAVE_DEMUCS_DEVICE")
        if device:
            cfg = replace(cfg, demucs_device=device.lower())
        log_level = os.environ.get("SPLITWAVE_LOG_LEVEL")
        if log_level:
            cfg = replace(cfg, log_level=log_level.upper())
        return cfg

    def with_overrides(self, **changes: object) -> "EngineConfig":
        """Return a copy with the given fields replaced (immutable update)."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class SeparationRequest:
    """A single separation job — *what* to produce.

    Mirrors the frozen ``SeparationEngine.separate`` signature (design doc §7,
    Milestone 0). Use :meth:`build` to coerce loose CLI/HTTP input into a
    validated request.
    """

    input_path: Path
    out_dir: Path
    tier: Tier = Tier.BALANCED
    stems: tuple[Stem, ...] = DEFAULT_STEMS
    dereverb: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "out_dir", Path(self.out_dir))

    @classmethod
    def build(
        cls,
        input_path: "str | Path",
        out_dir: "str | Path",
        *,
        tier: "str | Tier" = Tier.BALANCED,
        stems: "str | list[str] | tuple[str, ...] | None" = None,
        dereverb: bool = False,
    ) -> "SeparationRequest":
        """Construct and validate a request from loose input.

        ``stems`` accepts a comma-separated string or a sequence of names; each
        is normalized via :meth:`Stem.coerce`. ``tier`` accepts a name or enum.
        """
        tier_val = tier if isinstance(tier, Tier) else Tier(str(tier).lower())
        if stems is None:
            stem_vals: tuple[Stem, ...] = DEFAULT_STEMS
        else:
            if isinstance(stems, str):
                raw = [s for s in stems.split(",") if s.strip()]
            else:
                raw = list(stems)
            # de-dupe while preserving order
            seen: dict[Stem, None] = {}
            for s in raw:
                seen.setdefault(Stem.coerce(s), None)
            stem_vals = tuple(seen)
        req = cls(
            input_path=Path(input_path),
            out_dir=Path(out_dir),
            tier=tier_val,
            stems=stem_vals,
            dereverb=dereverb,
        )
        req.validate()
        return req

    def validate(self) -> None:
        """Raise ``ValueError``/``FileNotFoundError`` if the request is malformed.

        Boundary validation only (design doc principle: validate at edges). Does
        not check model availability — that surfaces from the backend.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"input file not found: {self.input_path}")
        if not self.input_path.is_file():
            raise ValueError(f"input path is not a file: {self.input_path}")
        if not self.stems:
            raise ValueError("at least one stem must be requested")
        if self.dereverb and Stem.VOCALS not in self.stems:
            raise ValueError("dereverb requires 'vocals' to be among the requested stems")

    @property
    def effective_stems(self) -> tuple[Stem, ...]:
        """Requested stems plus the dry vocal when ``dereverb`` is on."""
        if self.dereverb and Stem.VOCALS_DRY not in self.stems:
            return (*self.stems, Stem.VOCALS_DRY)
        return self.stems
