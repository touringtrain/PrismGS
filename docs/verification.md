# Verification

The repository provides CPU regression tests for the codec, calibration model,
frontier extraction, streaming selection and data validation. Run them from an
installed environment:

```bash
python -m unittest discover -s tests -v
```

The test suite exercises a real FFmpeg round trip, partial frame groups, changing
point counts, constant attributes, mixed-QP channel selection, serial/parallel
consistency, calibration protocol validation and GPU decoding when CUDA is
available. GPU-specific tests are skipped automatically on machines without CUDA.

For a complete experiment, prepare calibrated multiview data, run the pipeline
with `python -m prismgs run`, and inspect the generated `stages.json`,
`calibration/profile.json`, `frontier.json`, `measured_rd.json` and streaming
summaries. These files are generated per run and are not distributed with the
source repository.

The implementation supports CPU and CUDA reconstruction. Use
`--decode-device cuda` with `calibrate` or `run`, and `--device cuda` with
`decode` when a compatible CUDA installation is available. Performance depends
on GPU, CPU, storage, video length and the selected evaluation settings; the
provided tests establish functional correctness rather than a hardware benchmark.
