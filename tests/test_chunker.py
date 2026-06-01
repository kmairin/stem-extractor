from __future__ import annotations

import numpy as np
import pytest

from splitwave.chunker import overlap_add, plan_chunks


def test_single_chunk_when_short():
    chunks = plan_chunks(1000, 44100, segment_seconds=10, overlap=0.25)
    assert len(chunks) == 1
    assert chunks[0].start == 0 and chunks[0].end == 1000
    assert chunks[0].fade_in == 0 and chunks[0].fade_out == 0


def test_single_chunk_when_disabled():
    chunks = plan_chunks(10_000_000, 44100, segment_seconds=10, overlap=0.25, enabled=False)
    assert len(chunks) == 1


def test_multi_chunk_covers_signal_and_overlaps():
    n = 44100 * 25  # 25 s
    chunks = plan_chunks(n, 44100, segment_seconds=10, overlap=0.25)
    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].end == n
    # interior chunks carry symmetric fades of the overlap length
    ov = int(round(44100 * 10 * 0.25))
    assert chunks[1].fade_in == ov
    assert chunks[0].fade_out == ov
    # last chunk has no trailing fade
    assert chunks[-1].fade_out == 0


def test_zero_overlap_tiles_without_fades():
    n = 44100 * 25
    chunks = plan_chunks(n, 44100, segment_seconds=10, overlap=0.0)
    assert all(c.fade_in == 0 and c.fade_out == 0 for c in chunks)
    # contiguous, non-overlapping coverage
    for a, b in zip(chunks, chunks[1:]):
        assert b.start == a.end


def test_overlap_add_reconstructs_signal():
    n = 44100 * 25
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((n, 2)).astype(np.float32)
    plan = plan_chunks(n, 44100, segment_seconds=10, overlap=0.25)
    pieces = [signal[c.start : c.end] for c in plan]
    recon = overlap_add(pieces, plan, n)
    assert recon.shape == signal.shape
    assert np.allclose(recon, signal, atol=1e-4)


def test_overlap_add_length_mismatch_raises():
    plan = plan_chunks(1000, 44100, segment_seconds=10, overlap=0.25)
    with pytest.raises(ValueError):
        overlap_add([], plan, 1000)
