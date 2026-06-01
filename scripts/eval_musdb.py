"""Multi-track MUSDB18 accuracy eval — the verdict-grade quality measurement.

Runs the real Splitwave engine over N MUSDB18 test tracks and scores each
produced stem two ways:

* **SI-SDR** — our scale-invariant metric (``splitwave.bench.si_sdr``);
* **BSS-Eval SDR** — the standard MUSDB leaderboard metric via ``museval``
  (median over 1-s windows, the usual reporting convention).

Averaging across tracks turns the single-clip probe into something that can
actually *lock* tier defaults (design doc §7 Track B2 / open question Q2).

Usage::

    python scripts/eval_musdb.py --n-tracks 10 --tiers fast,balanced,best \
        --json /tmp/musdb_eval.json

The MUSDB 7-s previews are downloaded on first run (free, no registration)
and cached under ``~/MUSDB18``; the audio is gitignored.
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

from splitwave import get_engine
from splitwave.audio import load_audio
from splitwave.bench import si_sdr
from splitwave.types import Stem

#: tier -> (requested stems, {our stem name: MUSDB target name})
CONFIGS: dict[str, tuple[list[str], dict[str, str]]] = {
    "fast": (
        ["vocals", "drums", "bass", "other"],
        {"vocals": "vocals", "drums": "drums", "bass": "bass", "other": "other"},
    ),
    "balanced": (
        ["vocals", "instrumental"],
        {"vocals": "vocals", "instrumental": "accompaniment"},
    ),
    "best": (
        ["vocals", "instrumental"],
        {"vocals": "vocals", "instrumental": "accompaniment"},
    ),
}


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
    ap.add_argument("--tiers", default="fast,balanced,best")
    ap.add_argument("--json", type=Path, default=Path("/tmp/musdb_eval.json"))
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    for t in tiers:
        if t not in CONFIGS:
            raise SystemExit(f"unknown tier {t!r}; choose from {list(CONFIGS)}")

    mus = musdb.DB(download=True, subsets=[args.subset])
    tracks = list(mus)[: args.n_tracks]
    if not tracks:
        raise SystemExit("no MUSDB tracks available")
    engine = get_engine()

    # (tier, stem) -> {"si_sdr": [...], "bss_sdr": [...]}
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

            for tier in tiers:
                stems, target_map = CONFIGS[tier]
                with tempfile.TemporaryDirectory() as out:
                    try:
                        result = engine.separate(
                            mix_path, tier=tier, stems=stems, out_dir=out
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"    {tier}: ERROR {type(exc).__name__}: {exc}", flush=True)
                        continue
                    produced = {sf_.stem: load_audio(sf_.path).samples for sf_ in result.stems}

                    names = list(target_map)
                    refs, ests = [], []
                    for our in names:
                        est = produced[Stem.coerce(our)]
                        ref = track.targets[target_map[our]].audio
                        n = min(len(ref), len(est))
                        r = _as_stereo(ref[:n], 2)
                        e = _as_stereo(est[:n], 2)
                        refs.append(r)
                        ests.append(e)
                        si = si_sdr(e, r)
                        acc[(tier, our)]["si_sdr"].append(si)
                        per_track.append(
                            {"track": track.name, "tier": tier, "stem": our, "si_sdr": round(si, 2)}
                        )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sdr, _isr, _sir, _sar = museval.evaluate(
                            np.stack(refs), np.stack(ests)
                        )
                    line = []
                    for i, our in enumerate(names):
                        bss = float(np.nanmedian(sdr[i]))
                        acc[(tier, our)]["bss_sdr"].append(bss)
                        line.append(f"{our} SI={acc[(tier, our)]['si_sdr'][-1]:.1f}/BSS={bss:.1f}")
                    print(f"    {tier}: " + "  ".join(line), flush=True)

    # ---- aggregate ----
    report = {"subset": args.subset, "n_tracks": len(tracks), "tiers": {}}
    print("\n" + "=" * 64)
    print(f"MUSDB18 accuracy — {len(tracks)} {args.subset} tracks (mean over tracks)")
    print("=" * 64)
    for tier in tiers:
        stems = list(CONFIGS[tier][1])
        report["tiers"][tier] = {}
        print(f"\n{tier}:")
        print(f"  {'stem':12s} {'SI-SDR dB':>16s} {'BSS-SDR dB':>16s}")
        si_means, bss_means = [], []
        for stem in stems:
            s_si = _summary(acc[(tier, stem)]["si_sdr"])
            s_bss = _summary(acc[(tier, stem)]["bss_sdr"])
            report["tiers"][tier][stem] = {"si_sdr": s_si, "bss_sdr": s_bss}
            si_txt = f"{s_si['mean']:>7}±{s_si['std']:<5}" if s_si["mean"] is not None else "      -"
            bss_txt = f"{s_bss['mean']:>7}±{s_bss['std']:<5}" if s_bss["mean"] is not None else "      -"
            print(f"  {stem:12s} {si_txt:>16s} {bss_txt:>16s}")
            if s_si["mean"] is not None:
                si_means.append(s_si["mean"])
            if s_bss["mean"] is not None:
                bss_means.append(s_bss["mean"])
        if si_means:
            mean_si = round(float(statistics.fmean(si_means)), 2)
            mean_bss = round(float(statistics.fmean(bss_means)), 2)
            report["tiers"][tier]["mean"] = {"si_sdr": mean_si, "bss_sdr": mean_bss}
            print(f"  {'MEAN':12s} {mean_si:>13}    {mean_bss:>13}")

    args.json.write_text(__import__("json").dumps(
        {"summary": report, "per_track": per_track}, indent=2
    ))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
