# Decoding verification

The maintained decoder reconstructs SH0 Gaussian PLY checkpoints from encoded
attribute videos and `metadata.json`. Each configuration selects QPs independently
for position low bytes, color, opacity, scale and rotation. Position high bytes
remain at QP=0. Gaussian scale and opacity are restored as stored raw parameters;
the renderer applies their activations.

## Reproduce the test after calibration

From the repository root in the `videogs` CUDA environment:

```bash
PYTHONPATH=. python scripts/verify_decode.py \
  --videos runs/greeting_20_full/feature_video \
  --dataset datasets/4k_1_greeting \
  --profile runs/greeting_20_calibration/profile.json \
  --frontier runs/greeting_20_calibration/frontier.json \
  --original-checkpoints runs/greeting_20_full/checkpoint \
  --output runs/greeting_20_decode --workers 2
```

Use a fresh output directory. The script deduplicates the frontier and uniform
QP controls by configuration, preserving their names in `aliases`. It decodes
every unique configuration for every encoded frame, then checks:

- PLY point counts match metadata; all 17 SH0 parameters are finite, within their
  recorded ranges, and unused normals are zero.
- QP=0 reconstruction differs from original checkpoints by at most the packing
  quantization step (16-bit position, 8-bit other attributes), plus float tolerance.
- Position high-byte videos are identical across all supplied QPs.
- Actual selected MP4 payload per frame matches the frontier's reported size.
- Single-worker and parallel decoding produce byte-identical PLYs for low,
  middle and high rate configurations, across the entire sequence.
- Every configuration renders the initial keyframe with the calibration's
  camera selection, split, resolution and background. A middle-rate mixed-QP
  configuration additionally renders the middle and last dynamic frames.

`verification.json` records checks and measurements; `measured_rd.json` contains
the freshly rendered initial-frame RD points. Each configuration retains its
decoded `checkpoint/`, `ply_sha256.json`, and per-view `anchor/metrics.json`.
`dynamic/metrics.json` contains the additional temporal checks.

The CPU regression test also exercises partial final groups, changing point
counts, constant attributes, mixed-QP channel selection, serial/parallel equality
and rejection of truncated videos before any PLY is written for that group:

```bash
python -m unittest discover -s tests -v
```

## Implementation boundary

`prismgs/codec.py` supports CPU reconstruction (the default) and CUDA batched
reconstruction (`--device cuda` for `decode`, `--decode-device cuda` for `calibrate`
and `run`). Both paths read video channels concurrently with CPU OpenCV and
prefetch the next group in the background. CUDA reconstruction processes all
frames/points in a group using batched tensors, then transfers the reconstructed
batch back for parallel PLY writing. See [GPU decoding](gpu_decoding.md).

This adapts the frame batching and I/O prefetch approach in paper Section 3.3
and the historical GPU script. H.264 entropy decoding remains on CPU, as in that
script; this is not NVDEC hardware video decoding. Explicit CUDA selection raises
an error if CUDA is unavailable. Decode statistics distinguish total wall time
from CUDA upload/reconstruction events. Neither is a paper throughput claim.

Quality uses the initial-frame anchor protocol. The two extra dynamic frames
are diagnostic checks, not a full-sequence quality aggregate or held-out evaluation.

## CPU local result: 2026-10-19

The test passed on `4k_1_greeting` frames 0–19, using the new calibration profile
and frontier, resolution divisor 2, white background, all 81 training views.
See machine-readable verification (local file: `decoding_verification.json`).

- 92 frontier points plus six uniform controls reduced to 96 unique configurations.
  All 1,920 decoded checkpoints passed integrity and range checks.
- 7,776 fresh initial-frame renders reproduced prior PSNR/SSIM exactly; selected
  video sizes also matched exactly. QP=0 baseline PSNR was 35.8223 dB.
- Low/middle/high rate configurations passed serial/parallel byte equality on
  all 20 frames (60 additional checkpoints). Both uniform and mixed-QP CLI
  invocations matched the verification runner (40 additional checkpoints).
- The mixed configuration `{position_low:40, color:30, opacity:20, scale:20,
  rotation:20}` rendered frames 10 and 19 in all views: 162 records, mean
  PSNR 33.5437 dB and SSIM 0.98075. All per-view PSNR/SSIM matched prior renders.
- All nine regression tests passed, including the extended codec test.

Measured CPU decoding/reconstruction/PLY writing for the 96 main configurations
took 73.695 seconds in total. This is a single local correctness run, with cache
effects and rendering between decodes; it is not a controlled throughput benchmark.
Full outputs and logs remain under `runs/greeting_20_decode/`.
