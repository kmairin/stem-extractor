"""Per-model MUSDB18 eval — settles design-doc open question Q2 (§8).

The tier system never runs a single RoFormer checkpoint in isolation: ``best``
*ensembles* Mel-Band + BS-RoFormer, so the 10-track tier eval (``eval_musdb.py``)
can't say whether **BS-RoFormer alone** beats **Mel-Band alone** on instrumental.
This probe drives each model standalone via ``resolve_backend`` +
``backend.separate`` and scores vocals + instrumental against MUSDB ground truth
with both SI-SDR and the standard museval BSS-Eval SDR.

Usage::

    python scripts/eval_models.py --n-tracks 10 \
        --models mel_band_roformer_kim,bs_roformer_1297 \
        --json /tmp/musdb_models.json

The MUSDB 7-s previews download on first run (free, no registration) and cache
under ``~/MUSDB18``; the audio is gitignored.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import musdb
import museval
import numpy as np
import soundfile as sf

from splitwave.audio import load_audio
from splitwave.backends import resolve_backend
from splitwave.bench import si_sdr
from splitwave.config import EngineConfig
from splitwave.registry import get_model
from splitwave.types import Stem

#: stems every 2-stem RoFormer emits -> the MUSDB target that is ground truth.
TARGETS: dict[Stem, str] = {Stem.VOCALS: "vocals", Stem.INSTRUMENTAL: "accompaniment"}


def _as_stereo(x: np.ndarray, nch: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == nch:
        return x
    if x.shape[1] == 1:
        return np.repeat(x, nch, axis=1)
    return np.repeat(x.mean(axis=1, keepdims=True), nch, axis=1)


def _summary(vals: list[float]) -> dict:
    clean = [v for v in vals if np.isfinite(v)]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": round(float(statistics.fmean(clean)), 2),
        "std": round(float(statistics.pstdev(clean)), 2) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tracks", type=int, default=10)
    ap.add_argument("--subset", default="test", choices=["train", "test"])
    ap.add_argument("--models", default="mel_band_roformer_kim,bs_roformer_1297")
    ap.add_argument("--json", type=Path, default=Path("/tmp/musdb_models.json"))
    args = ap.parse_args()

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    models = [get_model(m) for m in model_ids]  # raises on unknown id
    for m in models:
        if set(m.stems) != set(TARGETS):
            raise SystemExit(
                f"{m.id} emits {[s.value for s in m.stems]}; this probe needs a "
                "2-stem vocals/instrumental model"
            )

    cfg = EngineConfig.default()
    mus = musdb.DB(download=True, subsets=[args.subset])
    tracks = list(mus)[: args.n_tracks]
    if not tracks:
        raise SystemExit("no MUSDB tracks available")

    # (model_id, stem) -> {"si_sdr": [...], "bss_sdr": [...]}
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"si_sdr": [], "bss_sdr": []}
    )
    per_track: list[dict] = []

    for ti, track in enumerate(tracks):
        rate = track.rate
        print(f"[{ti + 1}/{len(tracks)}] {track.name}", flush=True)
        with tempfile.TemporaryDirectory() as td:
            mix_path = Path(td) / "mixture.wav"
            sf.write(mix_path, track.audio, rate)
            buf = load_audio(mix_path)

            for model in models:
                backend = resolve_backend(model, cfg)
                try:
                    backend.ensure_available()
                    produced = backend.separate(
                        buf,
                        model,
                        stems=(Stem.VOCALS, Stem.INSTRUMENTAL),
                        chunk=cfg.chunk,
                        cache_dir=cfg.model_cache_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"    {model.id}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue

                names = list(TARGETS)
                refs, ests = [], []
                for stem in names:
                    est = produced[stem].samples
                    ref = track.targets[TARGETS[stem]].audio
                    n = min(len(ref), len(est))
                    r = _as_stereo(ref[:n], 2)
                    e = _as_stereo(est[:n], 2)
                    refs.append(r)
                    ests.append(e)
                    si = si_sdr(e, r)
                    acc[(model.id, stem.value)]["si_sdr"].append(si)
                    per_track.append(
                        {"track": track.name, "model": model.id,
                         "stem": stem.value, "si_sdr": round(si, 2)}
                    )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sdr, _isr, _sir, _sar = museval.evaluate(
                        np.stack(refs), np.stack(ests)
                    )
                line = []
                for i, stem in enumerate(names):
                    bss = float(np.nanmedian(sdr[i]))
                    acc[(model.id, stem.value)]["bss_sdr"].append(bss)
                    si_last = acc[(model.id, stem.value)]["si_sdr"][-1]
                    line.append(f"{stem.value} SI={si_last:.1f}/BSS={bss:.1f}")
                print(f"    {model.id}: " + "  ".join(line), flush=True)

    # ---- aggregate ----
    report = {"subset": args.subset, "n_tracks": len(tracks), "models": {}}
    print("\n" + "=" * 64)
    print(f"MUSDB18 per-model accuracy — {len(tracks)} {args.subset} tracks (mean over tracks)")
    print("=" * 64)
    for model in models:
        report["models"][model.id] = {}
        print(f"\n{model.id}:")
        print(f"  {'stem':12s} {'SI-SDR dB':>16s} {'BSS-SDR dB':>16s}")
        for stem in TARGETS:
            s_si = _summary(acc[(model.id, stem.value)]["si_sdr"])
            s_bss = _summary(acc[(model.id, stem.value)]["bss_sdr"])
            report["models"][model.id][stem.value] = {"si_sdr": s_si, "bss_sdr": s_bss}
            si_txt = f"{s_si['mean']:>7}±{s_si['std']:<5}" if s_si["mean"] is not None else "      -"
            bss_txt = f"{s_bss['mean']:>7}±{s_bss['std']:<5}" if s_bss["mean"] is not None else "      -"
            print(f"  {stem.value:12s} {si_txt:>16s} {bss_txt:>16s}")

    # ---- head-to-head on instrumental (the Q2 question) ----
    if len(models) == 2:
        a, b = models[0].id, models[1].id
        ia = report["models"][a]["instrumental"]["bss_sdr"]["mean"]
        ib = report["models"][b]["instrumental"]["bss_sdr"]["mean"]
        if ia is not None and ib is not None:
            winner, delta = (a, ia - ib) if ia >= ib else (b, ib - ia)
            print("\n" + "-" * 64)
            print(f"Q2 instrumental head-to-head (BSS-SDR): "
                  f"{a}={ia:.2f}  vs  {b}={ib:.2f}")
            print(f"  -> {winner} wins by {delta:.2f} dB")

    args.json.write_text(__import__("json").dumps(
        {"summary": report, "per_track": per_track}, indent=2
    ))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
