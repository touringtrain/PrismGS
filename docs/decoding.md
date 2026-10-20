# Decoding verification

The maintained decoder reconstructs SH0 Gaussian PLY checkpoints from encoded
attribute videos and `metadata.json`. Each configuration selects QPs independently
for position low bytes, color, opacity, scale and rotation. Position high bytes
remain at QP=0. Gaussian scale and opacity are restored as stored raw parameters;
the renderer applies their activations.

## Reproduce the test after calibration

From the repository root in an environment with CUDA enabled:

```bash
PYTHONPATH=. python scripts/verify_decode.py \
  --videos RUN/feature_video \
  --dataset datasets/4k_1_greeting \
  --profile RUN/calibration/profile.json \
  --frontier RUN/frontier.json \
  --original-checkpoints RUN/checkpoint \
  --output RUN/decode --workers 2
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
