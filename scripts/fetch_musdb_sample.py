"""Fetch one MUSDB18 preview track and write its ground-truth stems as WAVs.

MUSDB18 is the standard music-source-separation benchmark. Its 7-second
*preview* clips are freely downloadable (no SISEC registration) and ship with
true ``vocals/drums/bass/other`` stems plus ``accompaniment`` — exactly the
ground truth we need to score real SI-SDR with ``splitwave bench --refs``
(design doc §3/§7, Track B2). Reference-free reconstruction can't rank true
quality; this gives the harness honest references.

Usage::

    python scripts/fetch_musdb_sample.py --index 0 \
        --mix-dir musdb_sample --refs-dir musdb_refs

then, e.g.::

    splitwave bench "musdb_sample/<track>.wav" --tiers fast \
        --stems vocals,drums,bass,other --refs musdb_refs/ --json /tmp/si.json

The downloaded preview audio is gitignored (``*.wav`` etc.), so only this
script is version-controlled — the data regenerates on demand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import musdb
import soundfile as sf

#: Our stem name -> MUSDB target name. ``instrumental`` is MUSDB's
#: ``accompaniment`` (drums + bass + other), matching Stem.INSTRUMENTAL.
STEM_TARGETS = {
    "vocals": "vocals",
    "drums": "drums",
    "bass": "bass",
    "other": "other",
    "instrumental": "accompaniment",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=0, help="Track index within the subset.")
    ap.add_argument("--subset", default="test", choices=["train", "test"])
    ap.add_argument("--mix-dir", type=Path, default=Path("musdb_sample"))
    ap.add_argument("--refs-dir", type=Path, default=Path("musdb_refs"))
    ap.add_argument("--root", type=Path, default=None, help="MUSDB cache root.")
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="Reuse an already-downloaded MUSDB set under --root instead of fetching.",
    )
    args = ap.parse_args()

    mus = musdb.DB(
        root=str(args.root) if args.root else None,
        download=not args.no_download,
        subsets=[args.subset],
    )
    if not len(mus):
        raise SystemExit("no MUSDB tracks found (download failed or empty subset)")
    if not 0 <= args.index < len(mus):
        raise SystemExit(f"--index {args.index} out of range (0..{len(mus) - 1})")

    track = mus[args.index]
    rate = track.rate
    slug = track.name.replace("/", "_")

    args.mix_dir.mkdir(parents=True, exist_ok=True)
    args.refs_dir.mkdir(parents=True, exist_ok=True)

    mix_path = args.mix_dir / f"{slug}.wav"
    sf.write(mix_path, track.audio, rate)

    written = []
    for stem_name, target in STEM_TARGETS.items():
        if target not in track.targets:
            print(f"  (skip {stem_name}: MUSDB target '{target}' unavailable)")
            continue
        sf.write(args.refs_dir / f"{stem_name}.wav", track.targets[target].audio, rate)
        written.append(stem_name)

    dur = track.audio.shape[0] / rate
    print(f"track : {track.name}  ({dur:.1f}s @ {rate} Hz, {track.audio.shape[1]}ch)")
    print(f"mixture -> {mix_path}")
    print(f"refs    -> {args.refs_dir}/  [{', '.join(written)}]")
    print()
    print("Score real SI-SDR with, e.g.:")
    print(
        f'  splitwave bench "{mix_path}" --tiers fast '
        f"--stems vocals,drums,bass,other --refs {args.refs_dir}/"
    )


if __name__ == "__main__":
    main()
