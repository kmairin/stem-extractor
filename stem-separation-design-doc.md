# Design Doc: High-Quality Vocal / Instrument Separation Engine

**Codename:** `Splitwave`
**Status:** Draft v1.0 — ready for implementation
**Targets:** macOS (Apple Silicon) first → server-deployable second
**North star:** Clean stems, quality > speed, but **never >~2 min for a 4-min song**

---

## 0. TL;DR (read this if nothing else)

- **Engine of record:** [`python-audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) (UVR-derived, mature, CoreML support, 70+ models, built-in ensembling).
- **Default quality model:** **Mel-Band RoFormer** (Kim) or **BS-RoFormer** (Viperx-1297). These are the current SOTA for vocals (~12.4–12.9 dB vocal SDR, ~17 dB instrumental SDR).
- **Speed tier:** **HT-Demucs FT** (`htdemucs_ft`) for fast 4-stem (vocals/drums/bass/other).
- **Max-quality tier:** **ensemble** of 2–3 RoFormer/Demucs models (slower, optional).
- **Apple Silicon acceleration:** CoreML execution provider via ONNX Runtime; optional [`mlx-audio-separator`](https://github.com/ssmall256/mlx-audio-separator) MLX backend for native M-series speed.
- **De-reverb (P1, not P0):** chained post-stage using a dedicated dereverb RoFormer / UVR-DeEcho-DeReverb model.
- **Server path:** FastAPI + job queue wrapping the same core engine. Build the core so the CLI and server share one `SeparationEngine` interface.

---

## 1. The Council (who designed this)

A virtual advisory council of field archetypes. Each owns a concern and signs off on the relevant section. (Personas representing real bodies of work — citations are to the actual research, not to individuals' statements.)

| Seat | Persona | Concern | Anchored in |
|---|---|---|---|
| 🎛️ **MSS Architect** | Source-separation modeling lead | Model selection, quality ceiling | BS-RoFormer (SDX'23 winner), Mel-Band RoFormer |
| ⚡ **Perf Engineer** | Apple Silicon / inference optimization | Latency, MPS/CoreML/MLX, chunking | mlx-audio-separator, ONNX CoreML EP |
| 🧪 **Eval Scientist** | Metrics & QA | SDR/SI-SDR harness, regression gates | MUSDB18-HQ, Sound Demixing Challenge |
| 🧰 **Platform Eng** | API / server / packaging | FastAPI service, job queue, Mac app | — |
| 🎚️ **Audio DSP** | Pre/post-processing, de-reverb | Resampling, chunk overlap, dereverb chain | MSR Challenge 2025 sequential RoFormer pipeline |
| 📋 **Program Manager** | Parallelization & tracking | Work tracks, dependencies, acceptance | §7 |

**Council ruling:** Do **not** train a model. The open SOTA checkpoints already beat anything we'd train on a reasonable budget. Splitwave is an **inference + orchestration** product, not a research project.

---

## 2. Background & landscape (state of the art, 2026)

Music source separation (MSS) splits a mix into stems (vocals, drums, bass, other). Two architecture families:

- **Spectrogram / mask models** — best on harmonic sources (vocals). The RoFormer family dominates here.
- **Waveform models** — better on percussion/bass. Demucs is the reference.

**Current quality leaders:**
- **BS-RoFormer** — band-split + hierarchical transformer with Rotary Position Embedding. Won SDX'23 music track by a large SDR margin. ~12.9 dB vocal SDR.
- **Mel-Band RoFormer** — mel-scale band projection (perceptually weighted, overlapping subbands). Edges out BS-RoFormer on vocals/drums; the "Kim" vocal checkpoint is the community default.
- **HT-Demucs (v4) / `htdemucs_ft`** — hybrid spectrogram+waveform with cross-domain transformer. ~9.0–9.2 dB SDR, but **fast** and gives a clean 4-stem split.

**Ensembling** (combine RoFormer + Demucs outputs) yields the highest measured quality (reported ~13.6 dB vocal SNR/SDR) at the cost of running multiple models.

**De-reverb / de-echo:** handled as a *separate* model stage. The MSR Challenge 2025 winning system chained sequential RoFormers: separate → denoise → dereverb. We mirror that as an optional post-chain.

**Implication for us:** quality is a *model-choice + ensemble* dial; speed is a *backend + chunking* dial. Architect both as configurable tiers.

---

## 3. Requirements

### P0 (must ship)
1. Separate any input song into **vocals** and **instrumental** (and ideally 4-stem) with SOTA quality.
2. Run on macOS Apple Silicon with hardware acceleration.
3. A 4-minute song completes in **< ~2 min wall-clock** in the default quality tier.
4. Robust across genres (pop, rock, EDM, hip-hop, acoustic, classical) — no genre-specific failure.
5. Clean CLI + a single reusable `SeparationEngine` Python API.

### P1 (next)
6. **De-reverb / de-echo** option to produce dry vocals.
7. Quality tiers exposed to the user: `fast` / `balanced` / `best`.
8. Server deployment (FastAPI + queue) reusing the core engine unchanged.

### P2 (later)
9. Batch processing UI / drag-and-drop Mac app.
10. Stem caching, resumable jobs, multi-GPU server scaling.

### Non-goals
- Training new models. Real-time/streaming separation. Mobile/iOS.

---

## 4. Architecture

```
                ┌────────────────────────────────────────────┐
                │              Clients                         │
                │  CLI  •  FastAPI HTTP  •  (later) Mac app     │
                └───────────────────┬──────────────────────────┘
                                    │  same interface
                ┌───────────────────▼──────────────────────────┐
                │            SeparationEngine (core)            │
                │  - tier resolver (fast/balanced/best)         │
                │  - model loader + cache                       │
                │  - chunker (overlap-add)                      │
                │  - backend dispatch                           │
                │  - optional post-chain (dereverb/denoise)     │
                └───────────────────┬──────────────────────────┘
            ┌───────────────────────┼───────────────────────────┐
            ▼                       ▼                           ▼
   ┌─────────────────┐   ┌────────────────────┐    ┌───────────────────┐
   │ audio-separator │   │  mlx-audio-separator│    │  Demucs (htdemucs) │
   │ (ONNX+CoreML)   │   │  (Apple Silicon)    │    │  4-stem fallback   │
   └─────────────────┘   └────────────────────┘    └───────────────────┘
```

**Core principle:** the CLI and server are thin shells over `SeparationEngine`. Build the engine once; wrap it twice.

### 4.1 Backend selection (Perf Engineer)
- **Default backend:** `python-audio-separator` with ONNX Runtime **CoreML execution provider** on Apple Silicon. Verify with `audio-separator --env_info` (should log `CoreMLExecutionProvider available`).
- **Speed-max backend:** `mlx-audio-separator` (MLX-native, no PyTorch/ONNX at inference — lowest latency on M-series). Treat as an optional, swappable backend behind the same interface.
- **Demucs:** run via PyTorch with **MPS** device for the 4-stem fast tier.

### 4.2 Quality tiers (MSS Architect)
| Tier | Models | ~Speed (4-min song, M-series) | Use |
|---|---|---|---|
| `fast` | HT-Demucs FT (MPS) | ~20–45 s | quick previews, batch |
| `balanced` *(default)* | Mel-Band RoFormer (Kim) **or** BS-RoFormer Viperx-1297 | ~45–110 s | best quality/speed trade |
| `best` | Ensemble: MelBand RoFormer + BS-RoFormer (+ Demucs for drums/bass) | ~2–4 min | mastering / archival |

> Council guardrail: `best` may exceed the 2-min budget by design — gate it behind an explicit `--tier best` flag and warn the user. Default is always `balanced`.

### 4.3 Chunking & memory (Audio DSP)
- Long audio → segment with **overlap-add** (e.g. 8–12 s windows, 25% overlap) to bound memory and avoid edge artifacts. (`python-audio-separator` exposes segment/overlap params; reuse them rather than rolling our own.)
- Always resample to model's native rate (typically 44.1 kHz), separate, then restore original rate/format on output.

### 4.4 De-reverb chain (P1)
Post-process the separated vocal through a dedicated dereverb model (UVR-DeEcho-DeReverb or a dereverb RoFormer checkpoint). Sequential, optional, off by default:
```
mix → [separation model] → vocal_wet → [dereverb model] → vocal_dry
```
Expose `--dereverb` flag; emit both wet and dry stems.

---

## 5. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem; all SOTA tooling is Python |
| Core engine | `python-audio-separator` | Mature, CoreML, 70+ models, ensembling, auto model download |
| Apple accel | ONNX Runtime CoreML EP; optional MLX backend | Native M-series speed |
| 4-stem fast | `demucs` (PyTorch + MPS) | Fast, clean drums/bass |
| Server | FastAPI + Uvicorn | Async, simple, reuses core |
| Queue | Redis + RQ (or Celery) | Background jobs, no request-blocking |
| Audio I/O | ffmpeg, soundfile/torchaudio | Format coverage |
| Packaging | `uv` for deps; PyInstaller (later, Mac app) | Reproducible env |
| Eval | `museval` / SI-SDR on MUSDB18-HQ subset | Regression gate |

---

## 6. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| CoreML EP not installing / silently CPU | `--env_info` check in CI + startup assertion; fail loud |
| `best` tier blows latency budget | Gate behind explicit flag; default to `balanced` |
| RoFormer memory spikes on long tracks | Mandatory chunking + overlap-add; cap segment length |
| Model checkpoints large / slow first run | Pre-download + cache in known dir; ship a `prefetch` command |
| MLX backend parity drift vs ONNX | Parity smoke test in eval harness; MLX is opt-in, not default |
| Genre-specific failures | Eval set spans genres; per-genre SDR gate before release |
| Dereverb model degrades clean inputs | Off by default; only on `--dereverb`; A/B in eval |

---

## 7. Parallelization plan (Program Manager)

Six tracks. **A** is on the critical path; the rest fan out once the `SeparationEngine` interface is frozen (Milestone 0). Each track is sized for **one agent** working independently.

### Milestone 0 — Interface freeze (blocks everything, do first, ~½ day)
Define and commit the `SeparationEngine` ABC + config schema + I/O contracts. Once merged, **B–F unblock and run in parallel.**

```python
class SeparationEngine(Protocol):
    def separate(self, input_path: str, *, tier: Tier,
                 stems: list[str], dereverb: bool=False,
                 out_dir: str) -> SeparationResult: ...
```

### Track table

| Track | Title | Owner (agent) | Depends on | Parallel-safe? |
|---|---|---|---|---|
| **A** | Core engine + backend dispatch | Agent-1 | M0 | Critical path |
| **B** | Model registry + benchmark harness | Agent-2 | M0 | ✅ |
| **C** | API server + job queue | Agent-3 | M0 (mock engine ok) | ✅ |
| **D** | De-reverb / post-processing chain | Agent-4 | M0 | ✅ |
| **E** | CLI + packaging (Mac) | Agent-5 | M0 (mock engine ok) | ✅ |
| **F** | Eval / QA harness + regression gates | Agent-6 | M0 | ✅ |

### Per-track scope & acceptance

**Track A — Core engine** *(Agent-1)*
- Implement `SeparationEngine` over `python-audio-separator`; CoreML EP wired; MPS for Demucs.
- Tier resolver (`fast`/`balanced`/`best`), chunker with overlap-add, model cache.
- ✅ Done when: `balanced` separates a 4-min song < 2 min on M-series, returns vocal+instrumental.

**Track B — Model registry + benchmark** *(Agent-2)*
- Catalog candidate checkpoints (MelBand RoFormer Kim, BS-RoFormer 1296/1297, htdemucs_ft, dereverb models) with metadata (SDR, size, native rate).
- Latency profiler recording decode/preprocess/inference/postprocess/write medians + p95.
- ✅ Done when: one command outputs a quality×latency table per model; feeds tier defaults in A.

**Track C — API server + queue** *(Agent-3)*
- FastAPI: `POST /jobs` (upload + tier), `GET /jobs/{id}`, `GET /jobs/{id}/stems`. Redis/RQ worker calls the engine.
- Build against a mock engine until A lands; swap in real engine via interface.
- ✅ Done when: submit → poll → download stems works end-to-end against the real engine.

**Track D — De-reverb chain** *(Agent-4)*
- Implement optional post-stage; emit wet + dry vocals; `--dereverb` flag.
- ✅ Done when: dereverb stage runs on a vocal stem and improves dry-vocal SDR on reverb test clips without harming clean inputs (verified via F).

**Track E — CLI + Mac packaging** *(Agent-5)*
- `splitwave <file> --tier balanced --stems vocals,instrumental [--dereverb]`; progress bar; `prefetch` command.
- `uv`-based reproducible env; PyInstaller spike for a double-click Mac binary (P2).
- ✅ Done when: clean-machine install + single command produces stems.

**Track F — Eval / QA** *(Agent-6)*
- `museval`/SI-SDR harness over a MUSDB18-HQ subset + a custom multi-genre + reverb test set.
- Per-genre SDR gate; MLX-vs-ONNX parity smoke test; wire into CI.
- ✅ Done when: `make eval` prints per-model, per-genre SDR and fails the build on regression.

### Integration milestones
- **M1:** A + B → tier defaults locked, latency budget verified.
- **M2:** C + E → engine usable via CLI and HTTP.
- **M3:** D + F → dereverb shipped behind gate, regression gates green.
- **M4:** `best` ensemble tier + server scaling (P2).

### Status tracker (update in PRs)

Legend: ✅ done · 🟡 in progress (code landed, real-model/hardware verification pending) · ☐ todo

| ID | Task | Track | Status | Owner | Blocks |
|---|---|---|---|---|---|
| M0 | Freeze `SeparationEngine` interface | — | ✅ done | kmairin | A,B,C,D,E,F |
| A1 | Backend dispatch + CoreML/MPS wiring | A | 🟡 in progress | kmairin | A2,A3 |
| A2 | Tier resolver + chunker | A | ✅ done | kmairin | M1 |
| A3 | Model cache + prefetch | A | 🟡 in progress | kmairin | E |
| B1 | Model registry metadata | B | ✅ done | kmairin | A2 |
| B2 | Latency + quality benchmark harness | B | ☐ todo | — | M1 |
| C1 | FastAPI endpoints (mock engine) | C | ☐ todo | — | C2 |
| C2 | Redis/RQ worker + real engine swap | C | ☐ todo | — | M2 |
| D1 | Dereverb post-stage + flags | D | 🟡 in progress | kmairin | D2 |
| D2 | Wet/dry output contract | D | ✅ done | kmairin | M3 |
| E1 | CLI + progress + prefetch | E | ✅ done | kmairin | M2 |
| E2 | uv env + PyInstaller spike | E | 🟡 in progress | kmairin | M4 |
| F1 | SDR/SI-SDR eval harness | F | ☐ todo | — | F2 |
| F2 | Per-genre + parity gates in CI | F | ☐ todo | — | M3 |

### Implementation log

**2026-05-31 — v0.1 foundation (M0 + Track A critical path + CLI).**
Landed the frozen `SeparationEngine` interface and the orchestration spine. The
engine runs end-to-end against a fake backend (39 unit tests green); real
separation needs the `[ml]` extra installed.

- **M0 (done):** `splitwave.types` (Tier/Stem/StemFile/SeparationResult/StageTiming),
  `splitwave.config` (Backend/ChunkConfig/EngineConfig/SeparationRequest),
  `splitwave.base` (`SeparationEngine` Protocol + `BaseSeparationEngine` ABC).
- **A2 (done):** `tiers.resolve_tier` (fast/balanced/best → model plan, §4.2 incl.
  the `best` budget guardrail) and `chunker` (overlap-add, §4.3) — both unit-tested.
- **A1/A3 (in progress):** `backends/` dispatch with lazy audio-separator (ONNX/CoreML
  providers) + Demucs (MPS) wrappers; `prefetch` command + cache dir. *Pending:*
  end-to-end run on real checkpoints/hardware; in-process model-load cache.
- **B1 (done):** `registry.MODEL_CATALOG` seeded from §2/§9. ⚠️ RoFormer checkpoint
  filenames are unverified placeholders (`verified=False`) — **B2 must confirm them
  via `audio-separator --list_models` before tier defaults are locked.**
- **D1/D2:** de-reverb post-chain + wet/dry output contract wired (§4.4); real
  dereverb-model run unverified.
- **E1 (done):** `splitwave` CLI — `separate` (implicit), `env-info`, `models`,
  `prefetch`. **E2:** uv env done; PyInstaller spike todo.
- **Next (unblocked, parallel-safe):** B2 benchmark, C server, F eval harness.

See `README.md` for quickstart. Run `pip install -e '.[ml]'` then
`splitwave env-info` to confirm CoreML/MPS before first separation.

---

## 8. Open questions for the team
1. Ship MLX backend in v1 or keep ONNX/CoreML only and add MLX in M4?
2. Default to Mel-Band RoFormer (best vocals) or BS-RoFormer (best instrumental)? → resolve via Track B benchmark.
3. Server: stateless stem download vs. persisted stem store (S3) — depends on hosting plan.

---

## 9. Appendix — references
- BS-RoFormer (Lu et al., 2023), SDX'23 winner — arxiv 2309.02612
- Mel-Band RoFormer (Wang, Lu, Won, 2023) — arxiv 2310.01809
- Hybrid Transformer Demucs v4 — github.com/facebookresearch/demucs
- `python-audio-separator` — github.com/nomadkaraoke/python-audio-separator
- `mlx-audio-separator` — github.com/ssmall256/mlx-audio-separator
- `melband-roformer-infer` (incl. dereverb/denoise models) — github.com/openmirlab/melband-roformer-infer
- MSR Challenge 2025 (sequential separate→denoise→dereverb RoFormers) — github.com/ModistAndrew/xlance-msr
