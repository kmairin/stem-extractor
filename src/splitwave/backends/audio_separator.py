"""audio-separator backend (Track A, default) — UVR-derived, ONNX + CoreML EP.

Wraps ``python-audio-separator`` (design doc §0/§4.1). The library is file-based
and does its own segmentation, so this backend writes the in-memory buffer to a
temp WAV, runs the model, then reads the produced stem files back. The library is
imported lazily so the package stays light when the ``[ml]`` extra isn't present.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from ..audio import AudioBuffer, load_audio, save_audio
from ..config import Backend, ChunkConfig
from ..errors import BackendUnavailableError, ModelNotAvailableError
from ..registry import ModelKind, ModelSpec
from ..types import Stem
from .base import ProgressFn, SeparationBackend

__all__ = ["AudioSeparatorBackend"]

# Substrings audio-separator embeds in output filenames -> our canonical stem.
_NAME_TO_STEM: tuple[tuple[str, Stem], ...] = (
    ("no reverb", Stem.VOCALS_DRY),
    ("noreverb", Stem.VOCALS_DRY),
    ("no_reverb", Stem.VOCALS_DRY),
    ("dry", Stem.VOCALS_DRY),
    ("instrumental", Stem.INSTRUMENTAL),
    ("vocals", Stem.VOCALS),
    ("vocal", Stem.VOCALS),
    ("drums", Stem.DRUMS),
    ("bass", Stem.BASS),
    ("other", Stem.OTHER),
)


def _classify(filename: str) -> Stem | None:
    low = filename.lower()
    # "no reverb" must win over "vocals" when both could match the parenthetical.
    for needle, stem in _NAME_TO_STEM:
        if needle in low:
            return stem
    return None


class AudioSeparatorBackend(SeparationBackend):
    backend_id = Backend.AUDIO_SEPARATOR
    self_segments = True

    def __init__(self, onnx_providers: Sequence[str] | None = None) -> None:
        self.onnx_providers = tuple(onnx_providers or ())

    def is_available(self) -> bool:
        try:
            import audio_separator.separator  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def _make_separator(self, output_dir: Path, cache_dir: Path):
        from audio_separator.separator import Separator

        kwargs: dict = {
            "output_dir": str(output_dir),
            "model_file_dir": str(cache_dir),
            "output_format": "WAV",
        }
        # Pass execution providers when the installed version supports it.
        if self.onnx_providers:
            try:
                return Separator(onnx_execution_provider=list(self.onnx_providers), **kwargs)
            except TypeError:
                pass
        return Separator(**kwargs)

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
        cache_dir.mkdir(parents=True, exist_ok=True)
        want = set(stems)

        def emit(msg: str) -> None:
            if progress:
                progress(msg)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            in_wav = tmpdir / "mixture.wav"
            out_dir = tmpdir / "out"
            out_dir.mkdir()
            save_audio(audio, in_wav)

            separator = self._make_separator(out_dir, cache_dir)
            emit(f"loading {model.display_name}")
            try:
                separator.load_model(model_filename=model.checkpoint)
            except Exception as exc:  # noqa: BLE001
                raise ModelNotAvailableError(
                    f"could not load checkpoint {model.checkpoint!r} for "
                    f"{model.id}: {exc}"
                ) from exc

            emit(f"separating with {model.display_name}")
            produced = separator.separate(str(in_wav))

            results: dict[Stem, AudioBuffer] = {}
            for fname in produced:
                fpath = out_dir / Path(fname).name
                if not fpath.exists():
                    fpath = Path(fname)
                stem = _classify(fpath.name)
                if stem is None:
                    continue
                if model.kind is ModelKind.DEREVERB and stem is not Stem.VOCALS_DRY:
                    continue
                results[stem] = load_audio(fpath)

        if Stem.VOCALS in results and Stem.INSTRUMENTAL in want and Stem.INSTRUMENTAL not in results:
            results[Stem.INSTRUMENTAL] = _residual_instrumental(audio, results[Stem.VOCALS])

        missing = want - set(results)
        if missing:
            raise BackendUnavailableError(
                f"{model.display_name} did not produce requested stems: "
                f"{sorted(s.value for s in missing)} (got {sorted(s.value for s in results)})"
            )
        return {s: results[s] for s in want if s in results}


def _residual_instrumental(mix: AudioBuffer, vocals: AudioBuffer) -> AudioBuffer:
    """instrumental = mix - vocals, length/channel aligned."""
    import numpy as np

    n = min(mix.n_samples, vocals.n_samples)
    ch = min(mix.channels, vocals.channels)
    inst = mix.samples[:n, :ch] - vocals.samples[:n, :ch]
    return AudioBuffer(np.ascontiguousarray(inst), mix.sr)
