"""Overlap-add chunking (Track A, design doc §4.3).

Long tracks are split into fixed-length windows with a configurable overlap,
separated independently, then recombined with linear cross-fades across the
overlap regions. This bounds peak memory and removes the edge artifacts that
appear at hard chunk boundaries.

:func:`plan_chunks` is pure integer math (no numpy needed) so the boundary logic
is cheap to unit-test; :func:`overlap_add` does the weighted recombination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Chunk", "plan_chunks", "overlap_add"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One window as half-open sample indices ``[start, end)`` plus fade lengths.

    ``fade_in``/``fade_out`` are the sample counts shared with the previous/next
    chunk — the cross-fade ramp lengths used by :func:`overlap_add`. They are 0
    at the track's true edges (no neighbor to blend with).
    """

    index: int
    start: int
    end: int
    fade_in: int = 0
    fade_out: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


def plan_chunks(
    n_samples: int,
    sr: int,
    segment_seconds: float,
    overlap: float,
    *,
    enabled: bool = True,
) -> list[Chunk]:
    """Plan overlap-add windows over ``n_samples`` at sample rate ``sr``.

    A single full-length chunk is returned when chunking is disabled, the audio
    is empty, or it already fits in one window. ``overlap`` is the fraction (in
    [0, 1)) of each window shared with its neighbour.
    """
    if n_samples <= 0:
        return [Chunk(index=0, start=0, end=0)]

    seg = int(round(segment_seconds * sr))
    seg = max(seg, 1)
    if not enabled or n_samples <= seg:
        return [Chunk(index=0, start=0, end=n_samples)]

    ov = int(round(seg * overlap))
    ov = min(max(ov, 0), seg - 1)  # keep hop >= 1
    hop = seg - ov

    starts: list[int] = []
    start = 0
    while start < n_samples:
        starts.append(start)
        if start + seg >= n_samples:
            break
        start += hop

    chunks: list[Chunk] = []
    n = len(starts)
    for i, s in enumerate(starts):
        e = min(s + seg, n_samples)
        chunks.append(Chunk(index=i, start=s, end=e))

    # Second pass: fade lengths from actual neighbour positions (handles the
    # possibly-short final chunk correctly).
    resolved: list[Chunk] = []
    for i, c in enumerate(chunks):
        fade_in = 0
        fade_out = 0
        if i > 0:
            fade_in = max(0, chunks[i - 1].end - c.start)
        if i < n - 1:
            fade_out = max(0, c.end - chunks[i + 1].start)
        # A fade cannot exceed the chunk it lives in.
        fade_in = min(fade_in, c.length)
        fade_out = min(fade_out, c.length - fade_in if c.length - fade_in > 0 else c.length)
        resolved.append(
            Chunk(index=i, start=c.start, end=c.end, fade_in=fade_in, fade_out=fade_out)
        )
    return resolved


def _window(length: int, fade_in: int, fade_out: int) -> np.ndarray:
    """Trapezoidal weight window: ramp up over ``fade_in``, flat, ramp down."""
    w = np.ones(length, dtype=np.float64)
    if fade_in > 0:
        # linspace excluding 0 so the very first sample isn't fully zeroed
        w[:fade_in] = np.linspace(0.0, 1.0, fade_in, endpoint=False) + (1.0 / (2 * fade_in))
        w[:fade_in] = np.clip(w[:fade_in], 0.0, 1.0)
    if fade_out > 0:
        w[length - fade_out :] = w[length - fade_out :] * np.linspace(
            1.0, 0.0, fade_out, endpoint=False
        )
    return w


def overlap_add(pieces: list[np.ndarray], plan: list[Chunk], total_length: int) -> np.ndarray:
    """Recombine per-chunk arrays into one signal via weighted overlap-add.

    ``pieces[i]`` must correspond to ``plan[i]`` and have length ``plan[i].length``
    along axis 0 (mono ``(n,)`` or multichannel ``(n, c)``). Output has shape
    ``(total_length,)`` or ``(total_length, c)``. Normalising by accumulated
    weights makes the result correct regardless of the exact ramp shape.
    """
    if len(pieces) != len(plan):
        raise ValueError(f"pieces ({len(pieces)}) and plan ({len(plan)}) length mismatch")
    if not pieces:
        return np.zeros((total_length,), dtype=np.float32)

    sample = pieces[0]
    channels = 1 if sample.ndim == 1 else sample.shape[1]
    out_shape = (total_length,) if channels == 1 else (total_length, channels)
    acc = np.zeros(out_shape, dtype=np.float64)
    wsum = np.zeros((total_length,), dtype=np.float64)

    for piece, chunk in zip(pieces, plan):
        if chunk.length == 0:
            continue
        if piece.shape[0] != chunk.length:
            raise ValueError(
                f"chunk {chunk.index}: piece length {piece.shape[0]} != planned {chunk.length}"
            )
        w = _window(chunk.length, chunk.fade_in, chunk.fade_out)
        seg = piece.astype(np.float64)
        if channels == 1:
            acc[chunk.start : chunk.end] += seg * w
        else:
            acc[chunk.start : chunk.end] += seg * w[:, None]
        wsum[chunk.start : chunk.end] += w

    nonzero = wsum > 1e-8
    if channels == 1:
        acc[nonzero] /= wsum[nonzero]
    else:
        acc[nonzero] /= wsum[nonzero, None]
    return acc.astype(np.float32)
