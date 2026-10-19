# Preparing your own test data

Dataset images are not distributed in Git or source exports. Download HiFi4G
from its official source and follow the root [README](../README.md#download-and-prepare-hifi4g)
and [preparation guide](../docs/data_preparation.md). Use
`scripts/prepare_hifi4g.py` to arrange frames 0–19 into `datasets/4k_1_greeting/`.

# Local development copy

`4k_1_greeting/` contains HiFi4G frames **0–19**, copied without image resizing
or recompression from the author's prepared VideoGS data. Each frame contains
81 multiview images, camera metadata and initialization/calibration files.
Nested experimental image directories and generated training outputs are omitted.
Where later frames lacked `sparse/`, frame 0 fixed-camera calibration is copied
explicitly, matching the original sequence preprocessing convention.
`manifest.json` records every copied file's size and SHA-256.

This is prepared input, so no NeuS2 or camera calibration run is required to
start the maintained training pipeline. It is a local dataset copy, excluded
from Git and the source archive. Confirm HiFi4G redistribution terms before
publishing a dataset archive; source-only users must obtain the data separately.

From the repository root in the `videogs` conda environment, test all 20 frames:

```bash
python -m prismgs run --dataset datasets/4k_1_greeting \
  --output runs/greeting_20_smoke --start 0 --end 20 --group-size 20 \
  --resolution 8 --keyframe-iterations 2 --prune-iterations 2 \
  --motion-iterations 2 --finetune-iterations 2 \
  --qps 0 40 --cameras 23 --workers 2 --step-kb 0.1
```

For a faster initial check, use `--end 2 --group-size 2` and a different output
folder. For converged training omit the four short iteration flags, choose the
intended resolution, and omit `--cameras 23` to evaluate all views. Use
`--split test` to exclude held-out views from training/evaluate that split.
Full 20-frame training is more expensive than a two-frame smoke test.

For Neural 3D Video (N3DV), see the [N3DV preparation guide](../docs/data_preparation.md#neural-3d-video-n3dv). Extract synchronized camera videos first, then prepare per-frame calibration and initialization points; raw `poses_bounds.npy` is not loaded directly.
