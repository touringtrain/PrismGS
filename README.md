# PrismGS

### Coordinated Training and Compression for Pareto-Optimal 4D Gaussian Streaming

**ACM MM 2026** · Jinghang Li, Zhe Wang, Yifei Zhu · Shanghai Jiao Tong University

[Project Page](https://touringtrain.github.io/PrismGS/)

PrismGS combines compression-aware training, attribute-wise video compression,
and rate allocation for 4D Gaussian streaming.

## Installation

Requires Linux, an NVIDIA GPU, CUDA 11.8, a C++ compiler, Conda, and FFmpeg
with `libx264` support. Use a CUDA toolkit matching your PyTorch installation.

```bash
git clone https://github.com/touringtrain/PrismGS.git
cd PrismGS

conda create -n prismgs python=3.10 -y
conda activate prismgs
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch'
python -m pip install --no-build-isolation -e .
```

On Ubuntu, install system dependencies with
`sudo apt-get install build-essential git ffmpeg`.

## Data Preparation

Download [HiFi4G](https://github.com/moqiyinlun/HiFi4G_Dataset) or
[N3DV](https://neural-3d-video.github.io/) and prepare synchronized, undistorted
multiview images with matching cameras and initialization points:

```text
datasets/scene/
├── 0/
│   ├── images/
│   └── sparse/0/       # COLMAP cameras, poses, and points3D
├── 1/
│   ├── images/
│   └── sparse/0/
└── ...
```

For HiFi4G with a fixed camera rig:

```bash
python scripts/prepare_hifi4g.py \
  --images /path/to/undistorted_images \
  --calibration /path/to/colmap/sparse/0 \
  --output datasets/scene --start 0 --end 20
```

The helper arranges prepared data; it does not perform camera calibration or
reconstruct points. See [data preparation](docs/data_preparation.md) for both datasets.

## Run

Run training, compression, calibration, rate allocation, GPU reconstruction,
and evaluation with one command:

```bash
python -m prismgs run \
  --dataset datasets/scene --output runs/scene \
  --start 0 --end 20 --group-size 20 \
  --qps 0 10 20 30 40 50 --decode-device cuda
```

`--end` is exclusive. Use a fresh output directory. Add `--split test` for
held-out evaluation, or `--black-background` for images with a black background.
GPU reconstruction uses batched CUDA operations; video reading uses CPU OpenCV.

Results are saved under `runs/scene`: checkpoints, attribute videos,
`calibration/profile.json`, `frontier.json`, `measured_rd.json`, and streaming
summaries. Quality is measured on the first frame; payload is averaged over all
encoded frames. Streaming evaluation simulates throughput from a bandwidth trace.

## Training

To run training separately:

```bash
python -m prismgs train \
  --dataset datasets/scene --output runs/scene \
  --start 0 --end 20 --group-size 20
```

## Compression and Decoding

```bash
python -m prismgs pack --checkpoints runs/scene/checkpoint \
  --output runs/scene/feature_image --start 0 --end 20 --group-size 20
python -m prismgs encode --images runs/scene/feature_image \
  --output runs/scene/feature_video --qps 0 10 20 30 40 50
python -m prismgs decode --videos runs/scene/feature_video \
  --output runs/scene/decoded --qp 20 --device cuda
```

The codec supports SH degree 0. For attribute-specific QPs, replace `--qp` with
`--config CONFIG.json`. See [pipeline commands](docs/pipeline.md) for calibration,
frontier selection, rendering and evaluation. Each command accepts `--help`.

## Citation

```bibtex
@inproceedings{prismgs2026,
  author={Li, Jinghang and Wang, Zhe and Zhu, Yifei},
  title={PrismGS: Coordinated Training and Compression for Pareto-Optimal 4D Gaussian Streaming},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```

## Acknowledgements

Built on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and
[V³ / VideoGS](https://github.com/AuthorityWang/VideoGS).
See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution and licenses.
