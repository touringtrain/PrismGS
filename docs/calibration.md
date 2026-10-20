# Calibration reproduction

This module implements the offline attribute quality calibration in the
camera-ready paper, Section 3.2, Eq. (8). It consumes encoded Gaussian attribute
videos and calibrated reference images; it does not estimate camera poses.

For the first keyframe of each input sequence:

1. Decode all attributes at QP=0 and render the selected camera views to obtain
   `Q_base`, the arithmetic mean of per-view PSNR in dB.
2. Sweep one attribute's QP while keeping the other four at QP=0. The five
   groups are position low bytes, color (SH0), opacity, scale and rotation.
   Position high bytes remain lossless at QP=0.
3. Record the measured `Q_i(q)` and normalized factor `f_i(q) = Q_i(q)/Q_base`.
   SSIM is also recorded, but the quality model uses PSNR.
4. Count each attribute's MP4 bytes over **all encoded frames**, divided by
   frame count and 1000. Count position high-byte payload once as `fixed_kb`.
5. Predict a mixed configuration using
   `Q(qp) = Q_base * product(f_i(qp_i))` and pass the profile to the DP frontier.

For six QPs and five attributes, there are 26 unique decode/render jobs:
one baseline and 25 nonzero single-attribute configurations. The five QP=0
rows reuse the same baseline, yielding 30 attribute rows. Factors are empirical;
they are not clipped, smoothed or fitted to historical measurements. Only
measured QPs are supported, as in the maintained README pipeline.

## Commands

Activate the CUDA environment described in the root README. After training,
packing and encoding `RUN`, reproduce calibration with:

```bash
python -m prismgs calibrate \
  --videos RUN/feature_video \
  --dataset datasets/4k_1_greeting \
  --output RUN/calibration \
  --qps 0 10 20 30 40 50 --workers 2 --resolution 2 --split train

python -m prismgs frontier \
  --profile RUN/calibration/profile.json \
  --output RUN/calibration/frontier.json --step-kb 1

PYTHONPATH=. python scripts/report_calibration.py \
  --profile RUN/calibration/profile.json \
  --output RUN/calibration/report
```

Use a fresh calibration directory. Invalid/duplicate QPs, a missing QP=0,
empty input sequences and mixed measurement protocols are rejected.
Omitting `--cameras` measures every camera in the selected split. `--split train`
is an in-sample diagnostic. Held-out evaluation requires checkpoints trained
with the matching split. Background and resolution must also match the intended
experiment. QP=0 is the encoded baseline, not the original floating-point PLY;
packing itself quantizes Gaussian values.

## Outputs

| File | Meaning |
| --- | --- |
| `calibration_config.json` | Input paths, QPs, view/frame protocol and job count |
| `uniform_qp0/metrics.json` | Shared baseline and all per-view metrics |
| `ATTRIBUTE_qpN/config.json` | Exact five-attribute configuration |
| `ATTRIBUTE_qpN/metrics.json` | Measured anchor PSNR/SSIM and per-view records |
| `profile.json` | `Q_base`, `f_i`, attribute payloads, units and evaluation protocol |
| `report/anchors.csv` | All 30 attribute rows |
| `report/calibration.png`, `.pdf` | Normalized quality and attribute payload curves |
| `report/summary.json` | Machine-readable experiment summary |

The report optionally accepts `--measured-rd PATH` to compare Eq. (8) predictions
against existing mixed-configuration renders. Use results from the same encoded
source, background and evaluation settings. It checks frame/split/camera/resolution
agreement and counts duplicate configurations only once. This diagnostic does
not refit calibration, and is not the multi-scene experiment in paper Table 1.

The implementation is in `prismgs/pipeline.py` (`calibrate`) and
`prismgs/quality.py` (`collect_profile`, `predict_quality`). Regression tests in
`tests/test_calibration.py` verify isolated attribute sweeps, baseline reuse,
full-sequence rate normalization, Eq. (8), and rejection of mixed camera views.
