# Release integration changes

1. Initial old-disk VideoGS files were imported with original hashes in
   `prismgs_backend/SOURCE.json`.
2. The author's consolidated AceGS tree supplied maintained codec, profile/DP,
   rendering, streaming, training modules, adjusted backend source, CUDA source
   and original license texts. Import hashes are in `ACEGS_SOURCE.json`.
3. Modules were connected to the in-repository `training/prismgs_backend/` path.
   The unified `run` command now orchestrates every stage and records run options,
   stage timings and measured RD points rather than only predicted quality.
4. Calibration avoids rendering identical QP=0 anchors repeatedly. Camera images
   are cached for at most two frame/dataset/split configurations during rendering.
5. Training propagates the held-out split and explicit black-background choice.
   Backend scripts accept the new black-background override without changing
   their original default. CUDA selection is applied to the whole pipeline.
6. The previous standalone sequence entry delegates to the maintained CLI; the
   old fetch-only extension helper now prints build commands for bundled sources.
7. Source export explicitly excludes local datasets, compiled binaries and
   generated run artifacts. Original imports remain recorded separately from
   edited release files.
