# Source export boundary

`scripts/export_source.py` exports an explicit set of maintained sources:
`prismgs/`, the training backend and preprocessing scripts, CUDA source and
licenses, tests, installation files, and pipeline documentation.
No implementation is loaded from adjacent VideoGS or AceGS folders.

The export omits historical plotting scripts/assets, raw datasets, trained
checkpoints, generated results and verification reports, media, throughput traces, compiled extensions and Python caches.
`runs/` remains local and is ignored by Git. PyTorch, Tiny CUDA NN, Python
dependencies and FFmpeg are installed separately, and data is supplied by users.
An editable source installation is the supported installation mode.
