"""Splitwave — high-quality vocal / instrument separation engine.

Public API (the Milestone-0 frozen interface). The concrete engine and backends
import heavy ML deps lazily, so importing this package only needs the light core
dependencies.
"""

from __future__ import annotations

from .base import BaseSeparationEngine, SeparationEngine
from .config import (
    DEFAULT_STEMS,
    Backend,
    ChunkConfig,
    EngineConfig,
    SeparationRequest,
)
from .errors import (
    AudioIOError,
    BackendUnavailableError,
    ModelNotAvailableError,
    SplitwaveError,
    UnsupportedStemError,
)
from .types import (
    SeparationResult,
    StageTiming,
    Stem,
    StemFile,
    Tier,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # interface
    "SeparationEngine",
    "BaseSeparationEngine",
    # config
    "Backend",
    "ChunkConfig",
    "EngineConfig",
    "SeparationRequest",
    "DEFAULT_STEMS",
    # types
    "Tier",
    "Stem",
    "StemFile",
    "StageTiming",
    "SeparationResult",
    # errors
    "SplitwaveError",
    "BackendUnavailableError",
    "ModelNotAvailableError",
    "UnsupportedStemError",
    "AudioIOError",
]


def get_engine(config: EngineConfig | None = None) -> SeparationEngine:
    """Construct the default concrete engine.

    Imported lazily so ``import splitwave`` stays light; the engine pulls in the
    audio/backends layer only when actually instantiated.
    """
    from .core import Splitwave

    return Splitwave(config)
