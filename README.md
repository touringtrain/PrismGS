# [ACM MM 2026] PrismGS

PyTorch-based research code release for **PrismGS: Coordinated Training and Compression for Pareto-Optimal 4D Gaussian Streaming**.

Paper: [PrismGS_0402_1856.pdf](./PrismGS_0402_1856.pdf)

PrismGS studies bandwidth-adaptive streaming for dynamic 4D Gaussian scenes. This repository is a cleaned and self-contained open-source release reorganized from our internal research workspace, with bundled example assets, evaluation scripts, and plotting code.

## Abstract

Dynamic 4D Gaussian scene streaming requires balancing visual quality, storage cost, and transmission bandwidth under rapidly changing network conditions. PrismGS addresses this problem with a coordinated training-and-compression pipeline that models the rate-distortion behavior of different Gaussian attribute groups and selects Pareto-optimal compression configurations for streaming. Instead of relying on a single uniform compression level, PrismGS performs attribute-aware allocation across geometry- and appearance-related components, enabling more effective bitrate use under bandwidth constraints. This repository includes the core quality modeling and Pareto frontier generation code, bandwidth-trace evaluation on dynamic scenes, and a bundled example scene for reproducing the main workflow.

## Overview

PrismGS is organized around the following pipeline:

1. Convert reconstructed 4D Gaussian checkpoints into feature images.
2. Compress feature images into videos under different QP settings.
3. Reconstruct anchor checkpoints from compressed feature videos.
4. Measure attribute-wise sensitivity and fit a quality model.
5. Generate Pareto-optimal compression configurations.
6. Evaluate streaming quality and smoothness under real bandwidth traces.

## Repository Structure

```text
PrismGS/
├── core/                  # Core compression, decompression, quality modeling, and testing scripts
├── evaluation/            # Bandwidth-trace evaluation scripts and bundled Lumos-5G trace
├── plot/                  # Paper plotting scripts and generated figures
├── quality_testset/       # Pareto-frontier checkpoints and comparison utilities
├── assets/
│   ├── example_scene/     # Self-contained bundled example scene assets
│   ├── visualization/     # Selected figures and qualitative outputs
│   └── original_readme.md # Original internal project notes for reference
├── video_scene_process/   # Visualization preprocessing helpers
├── video_scene_process_new/
├── requirements.txt
└── PrismGS_0402_1856.pdf
```

## Installation

1. Create and activate a fresh conda environment:

   ```bash
   conda create -n prismgs python=3.10 -y
   conda activate prismgs
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `ffmpeg` is available:

   ```bash
   ffmpeg -version
   ```

## Dependency Notes

This repository has two levels of dependencies.

### 1. Core analysis environment

The packaged dependencies in `requirements.txt` are sufficient for:

- Pareto frontier generation
- bandwidth-trace evaluation
- most plotting scripts
- reading and analyzing the bundled example assets

### 2. Full rendering / reconstruction environment

Some scripts in `core/` depend on external 3DGS / VideoGS components used in the original project environment. In particular, `core/render_test.py` expects modules such as:

- `torch`
- `scene`
- `gaussian_renderer`
- `utils.*`
- `arguments`

These modules are not fully vendored in this repository. If you want to run rendering-based evaluation end-to-end, please additionally prepare the original 3DGS / VideoGS environment used by your project.

## Dataset

This release does not include the full raw datasets used in the paper.

- For 4D Gaussian reconstruction, the project uses dynamic multi-view scenes such as HiFi4G and N3DV-style sequences.
- For network evaluation, the repository includes the Lumos-5G throughput trace in [evaluation/Lumos5G-v1.0.csv](/home/new_disk/ljh/PrismGS/evaluation/Lumos5G-v1.0.csv).

To keep the repository self-contained, we bundle one example scene under `assets/example_scene/`, including:

- `feature_video/`
- `pareto_frontier_dp_4k_4.json`
- `test_results.json`
- lightweight metadata files (`min_max.json`, `viewer_min_max.json`, `group_info.json`)

To make the repository suitable for GitHub release, large local experimental caches and heavyweight intermediate results were removed from the public repo tree during cleanup.

## Quick Start

Generate a Pareto frontier from the bundled calibration setup:

```bash
cd core
python quality_model.py
```

Run bandwidth evaluation on the bundled example result:

```bash
cd evaluation
python overall_performance.py --results ../assets/example_scene/test_results.json
```

## Main Components

### Core scripts

- `core/compress_image_2_video_sensitivity.py`: compress feature images into attribute videos under multiple QP settings
- `core/decompress_video_2_ckpt_sensitivity.py`: reconstruct sensitivity checkpoints from compressed feature videos
- `core/decompress_video_2_ckpt_testset.py`: reconstruct checkpoints for Pareto-frontier testing
- `core/quality_model.py`: fit the quality model and generate Pareto frontiers
- `core/render_test.py`: rendering-based evaluation entry that depends on the upstream 3DGS / VideoGS environment

### Evaluation

- `evaluation/overall_performance.py`: evaluate streaming quality and smoothness under Lumos-5G bandwidth traces
- `plot/`: scripts used to generate figures in the paper

## Notes

- This repository is a cleaned release from a much larger research workspace.
- The bundled example scene is intended as the main entry point for understanding the codebase.
- `quality_testset/` is preserved because it contains useful checked results, but it is still one of the larger folders in the repository.
- Several plotting scripts still reference additional scene-specific experimental outputs that were not bundled into this public release.
- Large local-only results were moved out of the repository during cleanup to keep the public release lightweight.

## Citation

If you find this repository useful, please cite our paper.

```bibtex
@inproceedings{prismgs2026,
  title={PrismGS: Coordinated Training and Compression for Pareto-Optimal 4D Gaussian Streaming},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```

## Acknowledgement

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [VideoGS / V^3](https://github.com/AuthorityWang/VideoGS)
