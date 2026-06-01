"""Audio decode/encode + resampling (Track A, design doc §4.3).

Fast path uses libsndfile (via ``soundfile``) for WAV/FLAC/OGG. Anything else, or
any resample, is routed through ffmpeg (soxr resampler) which the design assumes
is present for format coverage. All audio is carried as float32 ``(n, channels)``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import AudioIOError

__all__ = ["AudioBuffer", "load_audio", "save_audio", "resample", "ffmpeg_available"]

# Formats libsndfile reads/writes directly; others go through ffmpeg.
_SNDFILE_FORMATS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".w64"}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass(slots=True)
class AudioBuffer:
    """PCM audio as float32 ``(n_samples, channels)`` plus its sample rate."""

    samples: np.ndarray
    sr: int

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            self.samples = self.samples[:, None]
        if self.samples.dtype != np.float32:
            self.samples = self.samples.astype(np.float32)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return self.n_samples / self.sr if self.sr else 0.0


def _run_ffmpeg(args: list[str]) -> None:
    if not ffmpeg_available():
        raise AudioIOError("ffmpeg not found on PATH; required for this format/resample")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AudioIOError(f"ffmpeg failed: {proc.stderr.strip() or proc.returncode}")


def _decode_via_ffmpeg(path: Path, target_sr: int | None) -> AudioBuffer:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        args = ["-i", str(path)]
        if target_sr:
            args += ["-af", "aresample=resampler=soxr", "-ar", str(target_sr)]
        args += ["-c:a", "pcm_f32le", str(wav)]
        _run_ffmpeg(args)
        data, sr = sf.read(wav, always_2d=True, dtype="float32")
    return AudioBuffer(data, sr)


def _resample(buffer: AudioBuffer, target_sr: int) -> AudioBuffer:
    """Resample via ffmpeg/soxr (high quality), falling back to linear interp."""
    if buffer.sr == target_sr:
        return buffer
    if ffmpeg_available():
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.wav"
            sf.write(src, buffer.samples, buffer.sr)
            return _decode_via_ffmpeg(src, target_sr)
    # Fallback: linear interpolation per channel (lower quality but dependency-free).
    n_out = int(round(buffer.n_samples * target_sr / buffer.sr))
    x_old = np.linspace(0.0, 1.0, buffer.n_samples, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.stack(
        [np.interp(x_new, x_old, buffer.samples[:, c]) for c in range(buffer.channels)],
        axis=1,
    )
    return AudioBuffer(out.astype(np.float32), target_sr)


def resample(buffer: AudioBuffer, target_sr: int) -> AudioBuffer:
    """Public wrapper around the internal resampler (ffmpeg/soxr, linear fallback)."""
    return _resample(buffer, target_sr)


def load_audio(path: "str | Path", target_sr: int | None = None) -> AudioBuffer:
    """Decode ``path`` to a float32 :class:`AudioBuffer`, optionally resampled."""
    path = Path(path)
    if not path.exists():
        raise AudioIOError(f"audio file not found: {path}")

    ext = path.suffix.lower()
    if ext in _SNDFILE_FORMATS:
        try:
            data, sr = sf.read(path, always_2d=True, dtype="float32")
            buf = AudioBuffer(data, sr)
            return _resample(buf, target_sr) if target_sr and sr != target_sr else buf
        except AudioIOError:
            raise
        except Exception:  # noqa: BLE001 - fall back to ffmpeg for odd encodings
            pass
    return _decode_via_ffmpeg(path, target_sr)


def save_audio(
    buffer: AudioBuffer,
    path: "str | Path",
    *,
    fmt: str | None = None,
    subtype: str | None = None,
) -> Path:
    """Encode ``buffer`` to ``path``. WAV/FLAC/OGG via libsndfile, else ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or path.suffix.lstrip(".")).lower()

    if f".{fmt}" in _SNDFILE_FORMATS:
        sf.write(path, buffer.samples, buffer.sr, subtype=subtype)
        return path

    # Lossy/other containers: write a temp WAV then transcode with ffmpeg.
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "stem.wav"
        sf.write(wav, buffer.samples, buffer.sr)
        _run_ffmpeg(["-i", str(wav), str(path)])
    return path
