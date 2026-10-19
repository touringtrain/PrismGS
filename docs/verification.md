# Verification

For the fresh 2026-10-19 calibration reproduction (26 configurations, 81 views
each), see [calibration reproduction](calibration.md) and its
machine-readable verification (local file: `calibration_verification.json`).
The subsequent [decoding test](decoding.md) freshly decoded 96 configurations
over 20 frames and verified initial/dynamic-frame renders; see
decoding verification (local file: `decoding_verification.json`).
The [GPU decoding adaptation and test](gpu_decoding.md) verifies CUDA batched
reconstruction against CPU outputs, with separate measured timing and limitations.

On 2026-10-18 the one-command pipeline completed a real GPU smoke test using
the existing `videogs` conda environment, HiFi4G `4k_4_dancing` frames 100–101,
resolution divisor 8 and training camera index 23. Every training stage ran
2 iterations; videos used QPs 0 and 40, with a 0.1 KB DP bin.

All eight stages completed in about 128 seconds: training, packing, encoding,
calibration, frontier generation, actual decode/render, walking simulation,
and driving simulation. Four predicted Pareto points and six measured RD
points (including uniform controls) were produced. See
smoke_test_summary.json (local file: `smoke_test_summary.json`) for timings and measured values.
Full outputs remain local under `runs/e2e_smoke/`, excluded from Git/export.

The result verifies connected execution with actual CUDA training, H.264
encoding/decoding, scene-specific measured calibration and actual render metrics.
It does not establish convergence, all-view/all-scene correctness, a clean CUDA
extension build, publication throughput/latency numbers, or camera-ready quality.
The single training view is explicitly a diagnostic; full evaluation must use
the intended held-out split and sufficient training.

CPU verification includes a real FFmpeg round trip with partial groups,
constant attributes and changing point counts, a small DP comparison to
exhaustive enumeration, and throughput unit/selection tests. Run:

```bash
python -m unittest discover -s tests -v
```

`training/verified_environment.json` records observed package versions.
The host's NVML management library reports a driver/library mismatch, but CUDA
allocation, training, rendering and checkpoint generation worked. No driver or
system configuration changes were made.

## Independent source archive

The source archive was extracted to a temporary directory outside this repository.
A second full training-to-streaming run completed there with the same short-test
options. Import paths for `prismgs`, `scene` and `gaussian_renderer` were checked
and all resolved inside that extracted source. The run used user-supplied prepared
data and the installed `videogs` dependencies; it did not load implementation
modules from the original VideoGS/AceGS workspaces.

See independent_source_summary.json (local file: `independent_source_summary.json`) for the
completed stages. These are short stochastic training runs, so the number of
retained Pareto configurations can differ between runs.

## Greeting: all 20 frames

On 2026-10-19, local `datasets/4k_1_greeting` frames 0–19 completed training,
packing, encoding, scene calibration, frontier generation, parallel decode,
actual rendering and both walking/driving throughput simulations in `videogs`.
Each training stage ran 2 iterations at resolution divisor 8. Camera 23 of the
training split was evaluated for every frame. Two Pareto configurations plus
two uniform-QP controls yielded 80 actual frame/view measurements.

The first run saved all 20 trained checkpoints under `runs/greeting_20_smoke/`.
It was paused during evaluation to fix repeated image loading. The maintained
renderer now caches selected cameras with an image-count bound rather than
retaining every view in a two-frame-only cache. The follow-up run explicitly
reused those checkpoints and completed all subsequent stages under
`runs/greeting_20_verified/`. All reconstructed frame files, metrics counts and
predicted/measured payload sizes were checked; 7 regression tests passed.
See greeting_20_verification.json (local file: `greeting_20_verification.json`) for settings,
stage timings and measured values. This remains a diagnostic execution test,
not held-out paper-quality evaluation or converged training.

## Greeting: full default training and compression measurement

On 2026-10-19, `runs/greeting_20_full` completed frames 0–19 with 12,000
keyframe iterations, 4,000 pruning iterations, and 800 motion + 2,000 finetune
iterations per dynamic frame, at resolution divisor 2. Training took 3,035 s.
All 81 training views were measured. This is an in-sample end-to-end test,
not held-out evaluation or a claim of reproducing paper results.

Six uniform QP controls (0/10/20/30/40/50) and all 92 predicted frontier
configurations were decoded and rendered across every frame: 158,760 view
records. The measurement loop was interrupted after calibration to replace
configuration-first image loading with frame-first camera reuse; trained
checkpoints, encoded videos and calibration were retained. Per-frame metrics
are persisted for interrupted measurement recovery. All configuration outputs
remain local, including decoded PLYs and per-view metrics.

| Representation | PSNR (dB) | SSIM | Video payload (KB/frame) |
| --- | ---: | ---: | ---: |
| Uncompressed SH0 PLY | 35.5465 | 0.99213 | — |
| Uniform QP 0 | 35.4372 | 0.99197 | 924.1148 |
| Uniform QP 10 | 35.4175 | 0.99175 | 749.0403 |
| Uniform QP 20 | 35.1242 | 0.98998 | 434.4487 |
| Uniform QP 30 | 32.1059 | 0.97760 | 257.2749 |
| Uniform QP 40 | 25.4313 | 0.94938 | 161.7768 |
| Uniform QP 50 | 23.4685 | 0.93496 | 82.2742 |

KB is decimal; payload excludes the required reconstruction metadata.
Walking/driving throughput simulations also completed. Their mean FPS was
28.4063 / 28.6697, respectively; these simulations exclude decode cost, RTT
and buffering and must not be described as measured network streaming FPS.

See greeting_20_full_verification.json (local file: `greeting_20_full_verification.json`)
for configuration, timings, point counts, integrity checks and summary values.

## Initial-anchor measurement protocol

The paper measures quality using only the initial frame of the 20-frame sequence.
The maintained pipeline now evaluates frame `--start` for every compression
configuration. Training and video encoding still cover all 20 frames, and rate
is the actual video payload averaged over the full encoded sequence. Calibration
also uses the initial frame. All selected camera views of that frame are retained.

The all-frame measurements above are additional diagnostics, not the paper's
anchor protocol. In `runs/greeting_20_full`, `measured_rd.json` and the streaming
summaries now use frame 0 quality. Previous aggregate outputs are preserved with
an `all_frames_` prefix. Existing per-frame measurements were reused without
retraining or rerendering. See `docs/greeting_20_anchor_verification.json`.
