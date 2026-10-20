# Preparing your own test data

Dataset images are not distributed in Git or source exports. Download HiFi4G
from its official source and follow the root [README](../README.md#download-and-prepare-hifi4g)
and [preparation guide](../docs/data_preparation.md). Use
`scripts/prepare_hifi4g.py` to arrange frames 0–19 into `datasets/4k_1_greeting/`.

## Preparation and testing

After preparing a dataset, run the maintained pipeline from the repository root:

```bash
python -m prismgs run --dataset /path/to/prepared/scene \
  --output runs/scene --start 0 --end 20 --group-size 20 \
  --qps 0 10 20 30 40 50
```

Use shorter frame ranges and fewer iterations for an initial smoke test. Omit
those reductions for the full experiment. Dataset downloads, camera calibration,
initialization point generation and redistribution permissions remain the
responsibility of the dataset provider and user.
