"""I/O contracts for the separation engine — part of the Milestone-0 frozen interface.

These types are the stable boundary every track depends on (Tracks A–F). Treat
changes here as interface changes: they ripple into the CLI, the server, and the
eval harness, so keep them additive where possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "Tier",
    "Stem",
    "StemFile",
    "StageTiming",
    "SeparationResult",
]


class Tier(str, Enum):
    """Quality/speed tier. See design doc §4.2.

    ``str`` subclassing means a ``Tier`` serializes to its plain value (e.g.
    ``"balanced"``) in JSON and CLI output without extra handling.
    """

    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"

    def __str__(self) -> str:  # nicer CLI / log rendering
        return self.value


class Stem(str, Enum):
    """Canonical stem names produced by the engine.

    ``VOCALS_DRY`` is the de-reverbed vocal (design doc §4.4); ``VOCALS`` is the
    raw ("wet") vocal. The 4-stem set (drums/bass/other) comes from Demucs-class
    models; 2-stem (vocals/instrumental) from the RoFormer family.
    """

    VOCALS = "vocals"
    INSTRUMENTAL = "instrumental"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    VOCALS_DRY = "vocals_dry"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: "str | Stem") -> "Stem":
        """Normalize a user-supplied stem name to a :class:`Stem`.

        Accepts case-insensitive names and a few common aliases. Raises
        ``ValueError`` with the list of valid names on an unknown stem.
        """
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        aliases = {
            "vocal": cls.VOCALS,
            "voice": cls.VOCALS,
            "acapella": cls.VOCALS,
            "inst": cls.INSTRUMENTAL,
            "instrumentals": cls.INSTRUMENTAL,
            "accompaniment": cls.INSTRUMENTAL,
            "no_vocals": cls.INSTRUMENTAL,
            "drum": cls.DRUMS,
            "dry": cls.VOCALS_DRY,
            "vocals_dereverb": cls.VOCALS_DRY,
        }
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"unknown stem {value!r}; valid stems: {valid}") from None


@dataclass(frozen=True, slots=True)
class StemFile:
    """A single written stem output."""

    stem: Stem
    path: Path
    sample_rate: int
    #: True when this is the pre-de-reverb ("wet") vocal kept alongside the dry one.
    wet: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Wall-clock spent in one pipeline stage (decode/inference/dereverb/write).

    Powers the latency profiler (Track B) and the <2-min budget gate (design doc
    §3 P0.3). ``model`` is set for inference stages so per-model latency is
    attributable.
    """

    name: str
    seconds: float
    model: str | None = None


@dataclass(slots=True)
class SeparationResult:
    """Outcome of one separation request."""

    input_path: Path
    out_dir: Path
    tier: Tier
    stems: list[StemFile] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    timings: list[StageTiming] = field(default_factory=list)
    #: Duration of the source audio in seconds (None until decoded).
    source_seconds: float | None = None
    #: Total wall-clock for the whole request.
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.out_dir = Path(self.out_dir)

    def get(self, stem: "str | Stem", *, wet: bool | None = None) -> StemFile | None:
        """Return the written file for ``stem``, or ``None`` if it was not produced."""
        want = Stem.coerce(stem)
        for sf in self.stems:
            if sf.stem == want and (wet is None or sf.wet == wet):
                return sf
        return None

    @property
    def stem_paths(self) -> dict[str, Path]:
        """Map of ``stem-name -> path`` for produced stems (dry variants win ties)."""
        out: dict[str, Path] = {}
        for sf in self.stems:
            if sf.stem.value not in out or not sf.wet:
                out[sf.stem.value] = sf.path
        return out

    @property
    def realtime_factor(self) -> float | None:
        """``source_seconds / wall_seconds`` — >1 means faster than real time."""
        if not self.source_seconds or self.wall_seconds <= 0:
            return None
        return self.source_seconds / self.wall_seconds
