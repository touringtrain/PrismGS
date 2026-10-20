# GPU parallel decoding

The CUDA backend in `prismgs/codec.py` adapts the group batching and background
prefetch architecture of `training/prismgs_backend/compress/decompress_video_2_ckpt.py`
to the maintained format. It supports `metadata.json`, independently selected
attribute QPs, partial groups, changing point counts and actual PLY output.
It preserves raw opacity/log-scale values and zero unused normals.

The stages are:

1. CPU thread pool reads independent MP4 attribute channels with OpenCV.
2. A background worker prefetches the next group while the current group is processed.
3. Packed uint8 planes transfer to CUDA. Batched tensor operations combine
   position bytes and dequantize all frames/points of each attribute on GPU.
4. One reconstructed group transfers back to CPU; threads write per-frame PLYs.

This follows paper Section 3.3's frame-wise parallel attribute reconstruction.
Video entropy decoding remains on CPU, matching the repository's legacy GPU
script. No NVDEC support is claimed. The adaptation eliminates its per-frame
GPU tensor filling/min-max loops and restores the previously disabled PLY writes.

## Commands

From the repository root in an environment with CUDA enabled:

```bash
python -m prismgs decode \
  --videos RUN/feature_video \
  --output RUN/selected/checkpoint --qp 0 --workers 2 --device cuda
```

For a mixed-QP configuration replace `--qp 0` with `--config CONFIG.json`.
`--device cuda:1` selects GPU 1. Without `--device`, decoding uses CPU for
compatibility. Explicit CUDA requests fail if CUDA is unavailable; they never
silently switch to CPU. `calibrate` and `run` accept `--decode-device cuda`.
In `run`, `--cuda` controls visible GPUs, so use `cuda:0` for the selected GPU.

Full verification against the previously tested CPU outputs:

```bash
PYTHONPATH=. python scripts/verify_decode.py \
  --videos RUN/feature_video \
  --dataset datasets/4k_1_greeting \
  --profile RUN/calibration/profile.json \
  --frontier RUN/frontier.json \
  --original-checkpoints RUN/checkpoint \
  --output RUN/gpu_decode --workers 2 --device cuda \
  --cpu-reference RUN/decode
```

Use fresh verification output directories. Every decoded frame is compared to
the corresponding CPU parameters using a float32 tolerance scaled to its recorded
ranges. This allows rounding differences between NumPy division and CUDA kernels.
The test also checks metadata ranges, QP=0 quantization error, worker-count
independence, payload sizes and actual initial/dynamic-frame renders.

`tests/test_gpu_decode.py` tests real CUDA reconstruction on multiple groups,
including a partial group, varying point counts, constant opacity and mixed QPs.
It checks that background read failures propagate and no PLY is written for a
truncated group. CUDA tests are explicitly skipped on hosts without CUDA.

## Timing

Each decode writes `decode_stats.json`:

- `wall_seconds`: initialization, video reads, reconstruction, transfers and PLY
  writing, including normal I/O overlap. It excludes writing the statistics file.
- `cuda_upload_and_reconstruct_ms`: CUDA events around H2D and batched
  reconstruction; excludes CPU packing, D2H and PLY writing. First-call kernel
  initialization and scheduling can affect this value.

For a CPU/CUDA comparison using identical PLY output work:

```bash
PYTHONPATH=. python scripts/benchmark_decode.py \
  --videos RUN/feature_video \
  --frontier RUN/frontier.json \
  --output RUN/gpu_benchmark --workers 2 --repeats 3
```

The benchmark performs one warmup per device/configuration and alternates execution
order across repeats. It reports medians and retains raw samples. Run it without
concurrent GPU verification/rendering. Disk/video decoding and PLY output can
dominate this 20-frame workload, so GPU reconstruction does not imply a faster
end-to-end decoder. These measurements do not establish the paper's throughput.
