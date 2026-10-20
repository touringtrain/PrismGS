# Pipeline stages

The root README gives the one-command `python -m prismgs run` workflow.
The same stages can be run separately. All paths below are user inputs/outputs;
all required runtime code is included in this repository.

```bash
python -m prismgs train --dataset DATA --output RUN --start 100 --end 105 --group-size 5
python -m prismgs pack --checkpoints RUN/checkpoint --output RUN/feature_image --start 100 --end 105 --group-size 5
python -m prismgs encode --images RUN/feature_image --output RUN/feature_video --qps 0 10 20 30 40 50
python -m prismgs calibrate --videos RUN/feature_video --dataset DATA --output RUN/calibration --qps 0 10 20 30 40 50
python -m prismgs frontier --profile RUN/calibration/profile.json --output RUN/frontier.json --step-kb 1
```

`frontier.json` contains a `points` array. Extract a selected point's `config`
object to `CONFIG.json`, then:

```bash
python -m prismgs decode --videos RUN/feature_video --output RUN/selected/checkpoint --config CONFIG.json --workers 4
python -m prismgs render --checkpoints RUN/selected/checkpoint --dataset DATA --output RUN/selected --start 100 --end 105
```

The `run` command automates extraction, decoding and rendering of every selected
configuration, includes uniform-QP controls, measures payload sizes and writes
`measured_rd.json`. Its `points` contain actual `psnr`, `ssim`, and `size_kb`.
The [decoding verification guide](decoding.md) gives a standalone test of every
frontier configuration, PLY integrity, serial/parallel equality and actual renders.
Then:

```bash
python -m prismgs evaluate --trace evaluation/Lumos5G-v1.0.csv --results RUN/measured_rd.json --output RUN/streaming.json --mobility walking
```

## Measurement conventions

- Frame ranges are integer, contiguous and end-exclusive; partial final groups
  are supported. All frames in a group are padded to a common plane size.
- The SH0 codec preserves the stored (pre-activation) opacity and scale values.
  It does not mix the legacy codec's activated-value normalization with raw PLY
  reconstruction. Constant attributes and unused normals are handled explicitly.
- Position high bytes are encoded at QP=0. The remaining five attribute groups
  choose only QPs that were actually encoded and measured.
- Metadata is versioned separately from the legacy min/max format. Repack old
  checkpoints before using the maintained CLI.
- Calibration renders the first keyframe and normalizes each attribute's PSNR
  to the all-QP=0 PSNR. DP maximizes the product of those normalized factors.
  Interaction errors are possible: predictions are not empirical metrics.
  See [calibration reproduction](calibration.md) for the exact sweep and outputs.
  Calibration requires a fresh output directory to prevent mixed measurements.
- Profile sizes and measured sizes count MP4 files in decimal KB per frame
  over all groups. Metadata and raw scene transfer are excluded.
- DP stores a best candidate in each discretized bin. It is approximate for
  arbitrary fractional costs; final output sizes are recomputed payload sizes.
- By default rendering evaluates every training camera. `--cameras` explicitly
  restricts indices within the selected split. `--split test` in `run` or `train`
  excludes held-out views during training, using the backend loader's split.
  With reused checkpoints, the caller must ensure they used the correct split.
- Render time measures CUDA render calls only, not data loading or video decode;
  it has not been calibrated as a publication benchmark.
- Streaming uses measured quality, a target FPS and independent throughput samples.
  It reports a throughput-only upper bound, not a client playback benchmark.

## Preparation boundary

The executable chain begins with calibrated multiview frames and initialization
points. Dataset downloads, camera calibration and optional NeuS2 point generation
remain dataset-dependent preparation tasks. Preprocessing helpers are included
under `training/prismgs_backend/preprocess/`; see `training/README.md`.
