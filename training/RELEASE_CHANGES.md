# Training integration

- The maintained CLI uses the training backend included in this repository.
- The unified `run` command orchestrates training, compression, calibration,
  frontier extraction, rendering and throughput simulation. It saves run options,
  stage timings and measured rate-distortion points.
- Calibration reuses the QP=0 baseline. Rendering caches selected camera images
  with a bounded cache to reduce repeated loading.
- Training supports held-out views, explicit background selection and CUDA device
  selection through the maintained CLI.
- CPU and CUDA decoding support independent attribute QPs and per-frame PLY output.
- CUDA extension source and original license notices are included. Source exports
  exclude datasets, compiled binaries and generated experiment outputs.

See [training instructions](README.md) and the root [installation guide](../README.md#installation).
