"""Backend interface (Track A, design doc §4.1).

A backend runs *one* model on an in-memory :class:`AudioBuffer` and returns a
``{Stem: AudioBuffer}`` mapping. The core engine owns tier resolution, ensembling
and writing; backends only do inference. This keeps adding a backend (e.g. MLX)
to implementing one method.

``self_segments`` records whether the underlying library does its own overlap-add
chunking (audio-separator, Demucs both do). When True the engine hands it the
segment/overlap params and makes a single call; when False the engine drives
:mod:`splitwave.chunker` itself (design doc §4.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Sequence

from ..audio import AudioBuffer
from ..config import Backend, ChunkConfig
from ..errors import BackendUnavailableError
from ..registry import ModelSpec
from ..types import Stem

__all__ = ["SeparationBackend", "ProgressFn"]

#: Optional status sink for UI/logging: ``progress(message)``.
ProgressFn = Callable[[str], None]


class SeparationBackend(ABC):
    """Runs a single model; subclasses implement availability + inference."""

    #: Stable id this backend implements (see :class:`splitwave.config.Backend`).
    backend_id: Backend
    #: Whether the library performs its own internal chunking.
    self_segments: bool = True

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def is_available(self) -> bool:
        """True if the backend's dependencies import and the device is usable."""

    def ensure_available(self) -> None:
        """Raise :class:`BackendUnavailableError` if the backend can't run."""
        if not self.is_available():
            raise BackendUnavailableError(
                f"{self.name} is unavailable — install the required extra "
                f"(e.g. `pip install 'splitwave[ml]'`) or pick another backend."
            )

    @abstractmethod
    def separate(
        self,
        audio: AudioBuffer,
        model: ModelSpec,
        *,
        stems: Sequence[Stem],
        chunk: ChunkConfig,
        cache_dir: Path,
        progress: ProgressFn | None = None,
    ) -> dict[Stem, AudioBuffer]:
        """Separate ``audio`` with ``model`` and return the requested ``stems``."""
