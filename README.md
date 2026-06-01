# Splitwave

High-quality vocal / instrument separation engine for Apple Silicon (CLI + server).
Inference + orchestration over open SOTA checkpoints (Mel-Band / BS-RoFormer,
HT-Demucs) — see [`stem-separation-design-doc.md`](stem-separation-design-doc.md)
for the full design and the live status tracker (§7).

> **Status: v0.1 working.** The frozen engine interface (Milestone 0), the core
> orchestration (Track A: tier resolver, chunker, backend dispatch, engine), and the
> CLI (Track E) are in place and unit-tested against a fake backend. With the `[ml]`
> extra installed, real separation is **verified end-to-end on Apple Silicon** across
> all tiers (fast/balanced/best + dereverb); every registry checkpoint is confirmed
> against the audio-separator model list. The **Track B2 benchmark** (`splitwave bench`)
> measures latency + reconstruction on any song, and true SI-SDR/SDR against ground
> truth; a 10-track **MUSDB18 eval locks the tier defaults on evidence** (see
> *Which tier?* below). Remaining: the server (Track C) and CI eval gates (Track F).
> See the design doc §7 log.

## Install

```bash
uv venv && uv pip install -e ".[dev]"      # light core + tests (no ML stack)
uv pip install -e ".[ml]"                  # add audio-separator + Demucs for real inference
# optional: ".[server]" (FastAPI), ".[mlx]" (Apple-Silicon MLX), ".[eval]" (museval)
```

ffmpeg must be on `PATH` (used for non-WAV decode and resampling).

## Usage

```bash
# Separate (the file-first form is the implicit `separate` command):
splitwave song.mp3 --tier balanced --stems vocals,instrumental --out stems/

# Quality/speed tiers (design doc §4.2):
splitwave song.wav --tier fast        # HT-Demucs, ~4-stem, fastest
splitwave song.wav --tier balanced    # Mel-Band RoFormer (default)
splitwave song.wav --tier best        # RoFormer ensemble (may exceed the 2-min budget)

# 4-stem + dry vocal:
splitwave song.wav --stems vocals,drums,bass,other
splitwave song.wav --stems vocals --dereverb        # emits wet + dry vocals

# Utilities:
splitwave env-info            # ffmpeg / CoreML / MPS / backend availability
splitwave models              # the model catalog
splitwave prefetch balanced   # pre-download a tier's checkpoints

# Benchmark latency + quality across tiers (Track B2):
splitwave bench song.wav --tiers fast,balanced,best         # reference-free: latency + reconstruction
splitwave bench song.wav --refs truth_stems/ --json b.json  # add true SI-SDR when ground-truth stems exist
```

Configuration via env vars: `SPLITWAVE_BACKEND`, `SPLITWAVE_CACHE_DIR`,
`SPLITWAVE_OUTPUT_FORMAT`, `SPLITWAVE_DEMUCS_DEVICE`, `SPLITWAVE_LOG_LEVEL`.

## Which tier should I use?

Locked on a 10-track **MUSDB18** eval (museval BSS-Eval SDR — the standard metric):

| Goal | Use | Measured (SDR dB, higher = better) |
|---|---|---|
| **Vocals / instrumental** | `balanced` *(default)* | vocals **11.4**, instrumental **15.8** — beats `fast` on vocals by **+2.6 dB** |
| **Drums / bass / other** (4-stem) | `fast` | bass **9.7**, drums **7.7**, other **5.1** |
| **Max quality, time no object** | `best` | only **+0.6 dB** over `balanced` for ~2.3× the runtime |

**Rule of thumb:** `balanced` for vocals, `fast` for drums/bass/other, `best` only when
wait time doesn't matter. "Other" (synths/guitars/fx) is the hardest stem for every
model. Reproduce the numbers with `scripts/eval_musdb.py` (needs the `[eval]` extra).

## Library API

```python
from splitwave import get_engine

engine = get_engine()
result = engine.separate("song.wav", tier="balanced",
                         stems=["vocals", "instrumental"], out_dir="stems")
print(result.stem_paths, result.realtime_factor)
```

`get_engine()` returns something satisfying the `SeparationEngine` protocol; the CLI
and (forthcoming) server are thin shells over it.

## Layout

```
src/splitwave/
  types.py      config.py    base.py      # M0: frozen interface (contracts, config, ABC)
  registry.py   bench.py                   # B1: model catalog · B2: latency+quality harness
  tiers.py      chunker.py                 # A2: tier policy + overlap-add
  audio.py      backends/                  # A1: I/O + backend dispatch (audio-separator, Demucs)
  core.py                                  # A:  the Splitwave engine
  cli.py                                   # E1: CLI
```

## Develop

```bash
.venv/bin/python -m pytest        # 52 tests, pure-logic + engine orchestration + bench metrics (no ML deps)
```

The engine takes an injectable `backend_factory`, so orchestration is tested with a
fake backend — no model downloads required.

Quality is measured against **MUSDB18** ground truth (`uv pip install -e '.[eval]'`):
`scripts/fetch_musdb_sample.py` grabs one reference track for `splitwave bench --refs`,
and `scripts/eval_musdb.py` runs the multi-track SI-SDR + BSS-Eval SDR matrix that
locks the tier defaults above.
