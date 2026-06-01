"""Exception hierarchy shared across tracks.

All engine-raised failures derive from :class:`SplitwaveError` so callers (CLI,
server) can catch one base type and render a friendly message.
"""

from __future__ import annotations

__all__ = [
    "SplitwaveError",
    "BackendUnavailableError",
    "ModelNotAvailableError",
    "UnsupportedStemError",
    "AudioIOError",
]


class SplitwaveError(Exception):
    """Base class for all Splitwave errors."""


class BackendUnavailableError(SplitwaveError):
    """A required inference backend is not installed or not usable on this host.

    Typically means the ``[ml]`` extra (audio-separator/onnxruntime/demucs) or
    ``[mlx]`` extra is missing. Raised loudly per design doc §6 ("fail loud").
    """


class ModelNotAvailableError(SplitwaveError):
    """A requested model checkpoint could not be resolved or downloaded."""


class UnsupportedStemError(SplitwaveError):
    """The selected tier/model cannot produce one of the requested stems."""


class AudioIOError(SplitwaveError):
    """Decoding or encoding audio failed."""
