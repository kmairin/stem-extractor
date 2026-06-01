"""Demucs backend (Track A) — PyTorch + MPS, fast 4-stem tier (design doc §4.1).

Runs HT-Demucs on Apple Silicon's MPS device when available (CPU fallback).
Demucs does its own split/overlap, so we pass ``chunk.overlap`` straight to
``apply_model``. torch/demucs are imported lazily (``[ml]`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..audio import AudioBuffer
from ..config import Backend, ChunkConfig
from ..errors import BackendUnavailableError, ModelNotAvailableError
from ..registry import ModelSpec
from ..types import Stem
from .base import ProgressFn, SeparationBackend

__all__ = ["DemucsBackend"]


class DemucsBackend(SeparationBackend):
    backend_id = Backend.DEMUCS
    self_segments = True

    def __init__(self, device: str = "mps") -> None:
        self.requested_device = device

    def is_available(self) -> bool:
        try:
            import demucs.apply  # noqa: F401
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def _resolve_device(self) -> str:
        import torch

        dev = self.requested_device
        if dev == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return dev

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
        self.ensure_available()
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        want = set(stems)
        if progress:
            progress(f"loading {model.display_name}")
        try:
            net = get_model(model.checkpoint)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailableError(
                f"could not load Demucs model {model.checkpoint!r}: {exc}"
            ) from exc

        device = self._resolve_device()
        net.to(device)
        net.eval()

        # Demucs wants stereo float32 at the model's native rate (engine resamples).
        samples = audio.samples
        if audio.channels == 1:
            samples = np.repeat(samples, 2, axis=1)
        wav = torch.from_numpy(samples.T.copy()).float()  # (channels, samples)

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        if progress:
            progress(f"separating with {model.display_name} on {device}")
        with torch.no_grad():
            out = apply_model(
                net,
                wav[None],
                device=device,
                split=True,
                overlap=chunk.overlap,
                progress=False,
            )[0]
        out = out * ref.std() + ref.mean()

        sources = list(net.sources)  # e.g. ['drums','bass','other','vocals']
        produced: dict[Stem, AudioBuffer] = {}
        for name, tensor in zip(sources, out):
            try:
                stem = Stem.coerce(name)
            except ValueError:
                continue
            arr = tensor.cpu().numpy().T.astype(np.float32)  # (samples, channels)
            produced[stem] = AudioBuffer(arr, audio.sr)

        if Stem.INSTRUMENTAL in want and Stem.INSTRUMENTAL not in produced:
            produced[Stem.INSTRUMENTAL] = _sum_non_vocal(produced, audio.sr)

        missing = want - set(produced)
        if missing:
            raise BackendUnavailableError(
                f"Demucs did not produce stems: {sorted(s.value for s in missing)}"
            )
        return {s: produced[s] for s in want if s in produced}


def _sum_non_vocal(produced: dict[Stem, AudioBuffer], sr: int) -> AudioBuffer:
    """instrumental = drums + bass + other."""
    parts = [produced[s].samples for s in (Stem.DRUMS, Stem.BASS, Stem.OTHER) if s in produced]
    if not parts:
        raise BackendUnavailableError("cannot build instrumental: no non-vocal stems")
    n = min(p.shape[0] for p in parts)
    ch = min(p.shape[1] for p in parts)
    acc = np.zeros((n, ch), dtype=np.float32)
    for p in parts:
        acc += p[:n, :ch]
    return AudioBuffer(acc, sr)
