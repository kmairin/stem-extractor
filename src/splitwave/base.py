"""The ``SeparationEngine`` interface — the keystone of the Milestone-0 freeze.

Two surfaces:

* :class:`SeparationEngine` — a ``Protocol`` for *consumers* (CLI, server, eval).
  Code should type against this, never against a concrete engine.
* :class:`BaseSeparationEngine` — an ABC for *implementers*. It owns request
  construction/validation and the public ``separate``/``separate_request``
  methods, leaving subclasses to implement only :meth:`_run`.

The ``separate`` signature is frozen exactly as written in design doc §7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from .config import DEFAULT_STEMS, EngineConfig, SeparationRequest
from .types import SeparationResult, Tier

__all__ = ["SeparationEngine", "BaseSeparationEngine"]

_DEFAULT_STEM_NAMES: tuple[str, ...] = tuple(s.value for s in DEFAULT_STEMS)


@runtime_checkable
class SeparationEngine(Protocol):
    """Structural contract every engine satisfies (design doc §7, Milestone 0)."""

    def separate(
        self,
        input_path: "str | Path",
        *,
        tier: "str | Tier" = Tier.BALANCED,
        stems: Sequence[str] = _DEFAULT_STEM_NAMES,
        dereverb: bool = False,
        out_dir: "str | Path",
    ) -> SeparationResult:
        """Separate ``input_path`` into ``stems`` at the given ``tier``."""
        ...


class BaseSeparationEngine(ABC):
    """Base class handling validation + dispatch; subclasses implement :meth:`_run`."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig.default()

    @property
    def name(self) -> str:
        return type(self).__name__

    def separate(
        self,
        input_path: "str | Path",
        *,
        tier: "str | Tier" = Tier.BALANCED,
        stems: Sequence[str] = _DEFAULT_STEM_NAMES,
        dereverb: bool = False,
        out_dir: "str | Path",
    ) -> SeparationResult:
        """Build a validated :class:`SeparationRequest`, then run it.

        Loose-typed entry point matching the frozen interface — coerces ``tier``
        and ``stems`` strings into canonical enums via :meth:`SeparationRequest.build`.
        """
        request = SeparationRequest.build(
            input_path,
            out_dir,
            tier=tier,
            stems=list(stems),
            dereverb=dereverb,
        )
        return self.separate_request(request)

    def separate_request(self, request: SeparationRequest) -> SeparationResult:
        """Validate and execute a fully-formed request."""
        request.validate()
        request.out_dir.mkdir(parents=True, exist_ok=True)
        return self._run(request)

    @abstractmethod
    def _run(self, request: SeparationRequest) -> SeparationResult:
        """Perform the separation. Subclass responsibility."""
        raise NotImplementedError
