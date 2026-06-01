from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from splitwave.audio import AudioBuffer
from splitwave.backends.base import SeparationBackend
from splitwave.config import Backend


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    """A 1-second 44.1k stereo sine written to disk for engine/decode tests."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = 0.2 * np.sin(2 * np.pi * 220 * t)
    right = 0.2 * np.sin(2 * np.pi * 440 * t)
    data = np.stack([left, right], axis=1).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, data, sr)
    return path


class FakeBackend(SeparationBackend):
    """Deterministic backend that fabricates the requested stems (no ML)."""

    backend_id = Backend.AUDIO_SEPARATOR

    def is_available(self) -> bool:
        return True

    def separate(self, audio, model, *, stems, chunk, cache_dir, progress=None):
        out = {}
        for i, s in enumerate(stems):
            # distinct content per stem so averaging/writes are observable
            out[s] = AudioBuffer(audio.samples * (0.1 * (i + 1)), audio.sr)
        return out


@pytest.fixture
def fake_factory():
    backend = FakeBackend()
    return lambda model, config: backend
