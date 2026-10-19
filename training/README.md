# Training implementation

The maintained entry points are `python -m prismgs train` and `python -m prismgs run`.
Both launch `training/prismgs_backend/train.py`, `prune_gaussian.py`, and
`train_dynamic.py` using the same Python environment as the caller. Paths resolve
inside this checkout, not a neighboring project.

The backend was recovered from `/home/old_disk/sjtu/ljh/videogs/VideoGS` and then
consolidated with the author-provided `AceGS` source. `SOURCE.json` records the
initial old-disk import; `ACEGS_SOURCE.json` records the later consolidated
import. Later release changes are recorded in `RELEASE_CHANGES.md`.

## Defaults

- Keyframe: 12000 iterations; scale regularization 0.001, anisotropy weight 0.001,
  anisotropy threshold 5.
- Opacity-based pruning followed by 4000 finetuning iterations.
- Residual motion network: 800 iterations; dynamic finetuning: 2000 iterations.
- SH degree 0; white background; resolution divisor 2.
- Group-local keyframe, with frame interval 1.

All these training counts and regularization weights are configurable through
`python -m prismgs train --help`. Set a regularization weight to zero for ablations.
Short iteration counts only test execution; they do not produce converged models.

## Input preparation

The loader supports calibrated COLMAP scenes (`FRAME/images/` plus
`FRAME/sparse/0/`) or `FRAME/transforms.json` with initialization `points3d.ply`.
Preprocessing scripts are included in `prismgs_backend/preprocess/`. Their CLI
options and the camera JSON schema are documented in the retained upstream
[README](../third_party/videogs/README.md).

Optional preprocessing dependencies are in `../requirements-preprocess.txt`.
HiFi4G/N3DV download, calibration, background processing and NeuS2 initial-point
construction depend on the chosen dataset and are not performed by `prismgs run`.
Supply calibrated data and initialization points before training. Backend utilities
may create converted point-cloud files and missing per-frame calibration folders
inside the supplied dataset; use a prepared working copy if source data must stay
read-only.

## Environment

Use the root README installation instructions. CUDA rasterizer and KNN source
are bundled in `../submodules/`. Tiny CUDA NN, PyTorch and FFmpeg are installation
dependencies. `verified_environment.json` records the existing `videogs`
environment actually used for GPU smoke verification; it is not a clean-build lockfile.

`train_sequence.py` is a compatibility wrapper for the unified training entry.
The original `train_PrismGS.py` and `train_sequence_PrismGS.py` are retained as
provenance references; the maintained launchers use the consolidated scripts.
`third_party/videogs/` retains an unmodified upstream snapshot for comparison.
