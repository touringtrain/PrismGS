# [ACM MM 2026] PrismGS

**PrismGS: Coordinated Training and Compression for Pareto-Optimal 4D Gaussian Streaming**
Jinghang Li, Zhe Wang, Yifei Zhu · Shanghai Jiao Tong University

[Project page and demo](https://touringtrain.github.io/PrismGS/)

PrismGS combines regularized keyframe reconstruction, V³ residual dynamic training,
attribute video compression, measured quality modeling, and rate allocation.
The maintained `python -m prismgs` interface contains the full pipeline from
**prepared, calibrated multiview data and initialization points** to measured
streaming results. Runtime code and rasterizer/KNN source are included here;
a neighboring VideoGS or AceGS checkout is not required.

Raw datasets and initialization points must be obtained/prepared separately.
Full default-length training and camera-ready metrics have not yet been reproduced
with this release. See [verification](docs/verification.md) for the exact tests.

## Installation

Requirements: Linux, an NVIDIA GPU with a CUDA-compatible driver, a CUDA toolkit
with `nvcc`, a C++ compiler, and Conda. FFmpeg must include the `libx264` encoder.
The steps below use Python 3.10 and PyTorch 2.1.2 with CUDA 11.8; install a
matching CUDA 11.8 toolkit before building the extensions.

Clone the repository and create a dedicated environment:

```bash
git clone https://github.com/touringtrain/PrismGS.git
cd PrismGS
conda create -n prismgs python=3.10 -y
conda activate prismgs
python -m pip install --upgrade pip setuptools wheel
```

Install FFmpeg and build tools using your system package manager. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ffmpeg
```

Install PyTorch, Python dependencies, and the CUDA extensions:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch'
python -m pip install --no-build-isolation -e .
```

Keep the checkout available: the editable installation loads the training backend
and bundled extension source from this repository. If the CUDA toolkit is installed
outside the default search path, set `CUDA_HOME` to its installation directory and
add its `bin` directory to `PATH` before building.

Check the installation:

```bash
nvcc --version
ffmpeg -hide_banner -h encoder=libx264
python -c "import torch; assert torch.cuda.is_available(); import simple_knn._C, diff_gaussian_rasterization, tinycudann; print(torch.cuda.get_device_name(0))"
python -m prismgs --help
```

Optional raw-data preparation helpers require additional dependencies:

```bash
python -m pip install -r requirements-preprocess.txt
```

## Download and prepare HiFi4G

Obtain HiFi4G from the [official dataset repository](https://github.com/moqiyinlun/HiFi4G_Dataset)
following its access instructions and terms. Dataset images and point clouds are
not included in our Git repository or source archive. The local example copy is
only for development; see [datasets/README.md](datasets/README.md).

For `4k_1_greeting`, select frames **0–19**. Frame numbers refer to time; each
frame folder must contain all calibrated camera views, not a single video image.
Use the dataset's **undistorted, white-background** images and their matching
COLMAP calibration. Keep original camera filenames and resolution. Do not mix
calibration for distorted images with undistorted images.

The preparation command expects the following input layout (the actual download
folder names can differ; pass their real paths with the two arguments):

```text
/path/to/download/4k_1_greeting/
  image_undistortion_white/       # some releases call this image_white_undistortion
    0/0.png ... 80.png           # camera IDs; use actual supplied names
    1/0.png ... 80.png
    ...
    19/0.png ... 80.png
  colmap/sparse/0/
    cameras.bin                 # PINHOLE/SIMPLE_PINHOLE intrinsics
    images.bin                  # camera poses and image names
    points3D.bin                # initialization points (or points3D.ply/.txt)
```

`cameras.txt`/`images.txt` are also accepted instead of the binary camera files.
The calibration may live inside the image folder in some releases. Pass the
actual model directory; do not rename or regenerate calibration unnecessarily.

From the repository root, arrange your downloaded data without moving or changing
the originals:

```bash
python scripts/prepare_hifi4g.py \
  --images /path/to/download/4k_1_greeting/image_undistortion_white \
  --calibration /path/to/download/4k_1_greeting/colmap/sparse/0 \
  --output datasets/4k_1_greeting --start 0 --end 20
```

Use an empty output directory. The script checks calibrated image filenames,
rejects distorted camera models, copies images without resizing/recompression,
and places the fixed-camera calibration and initialization points in every frame.
It also accepts input frame folders already containing `images/`. The result is:

```text
datasets/4k_1_greeting/
  preparation.json
  0/
    images/                     # all calibrated camera images
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin               # or supplied points3D.ply/.txt
  1/
    images/
    sparse/0/                   # shared fixed-camera calibration
  ...
  19/
    images/
    sparse/0/
```

This **COLMAP layout is directly trainable**: `transforms.json` and a separate
NeuS2 run are not required when valid COLMAP initialization points are supplied.
The loader converts `points3D.bin/.txt` to `points3D.ply` on first use if necessary.
Sharing the model assumes a fixed camera rig; moving cameras need per-frame poses.

If your download contains only distorted images, undistort them with COLMAP first;
if it lacks initialization points, reconstruct/provide them before training.
[Detailed preparation instructions](docs/data_preparation.md) cover these cases
and the alternative `transforms.json` layout. The arrangement script performs
no undistortion, background removal, camera calibration or point reconstruction.

## Download and prepare N3DV

The paper also uses **Neural 3D Video (N3DV / DyNeRF)**. Obtain it through the
[official project page](https://neural-3d-video.github.io/). A typical scene
contains per-camera `camXX.mp4` videos and `poses_bounds.npy`.

1. Extract identical frame indices from every camera video without temporal
   resampling, and arrange them as `FRAME/images/CAMERA.png`.
2. Prepare matching undistorted COLMAP cameras and initialization points under
   `FRAME/sparse/0/`, or per-frame `transforms.json` plus `points3d.ply`.
   The loader does not directly accept videos or `poses_bounds.npy`.
3. Run the same `python -m prismgs run` pipeline, pointing `--dataset` at the
   prepared scene. Select the appropriate background and held-out views.

```text
datasets/coffee_martini/
  0/images/cam00.png ... camXX.png
  0/sparse/0/                  # cameras, poses and initialization points
  1/images/cam00.png ... camXX.png
  1/sparse/0/
  ...
```

The [N3DV preparation guide](docs/data_preparation.md#neural-3d-video-n3dv)
includes extraction commands, calibration/conversion requirements, arrangement,
and a short test command. Raw N3DV conversion and paper-specific evaluation
splits are not automatically inferred; the N3DV recipe remains unverified end
to end in this release. Dataset files are not shipped in Git or source exports.

## One-command pipeline

From the activated `prismgs` environment:

```bash
python -m prismgs run --dataset /path/to/prepared/scene --output runs/scene \
  --start 100 --end 105 --group-size 5 --qps 0 10 20 30 40 50
```

`--end` is exclusive. Use a fresh output directory. The command performs:

1. Regularized keyframe training, pruning and residual dynamic training.
2. SH0 checkpoint packing and per-attribute H.264 encoding.
3. Keyframe sensitivity measurements with other attributes fixed at QP=0.
4. A measured quality/size profile and discretized DP frontier.
5. Parallel decoding and actual rendering of every selected configuration and uniform-QP controls.
6. Walking/driving throughput simulation using the bundled Lumos5G trace.

A short execution test (not a converged model or paper-quality experiment):

```bash
python -m prismgs run --dataset /path/to/4k_4_dancing --output runs/smoke \
  --start 100 --end 102 --group-size 2 --resolution 8 \
  --keyframe-iterations 2 --prune-iterations 2 \
  --motion-iterations 2 --finetune-iterations 2 \
  --qps 0 40 --cameras 23 --workers 2 --step-kb 0.1
```

Omit `--cameras` to evaluate every view in the selected split. The default
`--split train` is an in-sample diagnostic. Use `--split test` for the loader's
held-out split; the launcher then excludes those views during training too.
Use `--black-background` consistently for scenes requiring a black background.
Use `--checkpoints /path/to/checkpoint` to explicitly reuse existing training
outputs. QP=0 must be encoded; optimization only selects measured QP levels.

Outputs under `--output` include:

| Output | Contents |
|---|---|
| `checkpoint/` | Trained Gaussian checkpoints |
| `feature_image/`, `feature_video/` | Attribute planes, videos and versioned metadata |
| `calibration/profile.json` | Scene-specific measured anchors and sizes |
| `frontier.json` | Predicted Pareto configurations |
| `measured_rd.json` | Actually rendered PSNR/SSIM and payload sizes |
| `streaming_walking.json`, `streaming_driving.json` | Throughput simulation results |
| `run_config.json`, `stages.json` | Run options and completed-stage timings |

## Individual stages

`python -m prismgs --help` lists `train`, `pack`, `encode`, `decode`, `render`,
`calibrate`, `frontier`, `evaluate`, and `run`. Each accepts `--help`.
See [pipeline details](docs/pipeline.md) for commands and measurement conventions.
The [calibration reproduction guide](docs/calibration.md) gives the paper's
single-attribute measurement protocol, commands, normalized quality curves and
local verification results.
The [decoding verification guide](docs/decoding.md) tests every frontier
configuration, reconstructed PLY integrity and serial/parallel consistency.
For the paper's batched GPU attribute reconstruction, use `decode --device cuda`
or `calibrate`/`run --decode-device cuda`; see [GPU decoding](docs/gpu_decoding.md).

## Scope and limitations

- The maintained codec supports **SH degree 0**. It stores raw Gaussian parameters,
  16-bit positions and 8-bit appearance/opacity/scale/rotation planes. No extra
  activation is applied during reconstruction. High position bytes use QP=0.
- The codec uses its own versioned `metadata.json`. Legacy feature videos under
  `assets/example_scene/` use another format; repack checkpoints for the new CLI.
- Calibration renders the first keyframe; rate measurements cover all supplied
  frames. Model predictions are verified on all requested frames/views afterwards.
- Sizes are decimal **KB/frame of video payload**, including MP4 overhead and
  excluding metadata, checkpoints and dataset transfer. DP is discretized;
  smaller `--step-kb` retains more candidates, with more computation.
- The bandwidth evaluation is a throughput-only model, without buffering, RTT,
  packet loss or decode cost. It is not a network streaming client.
- Legacy `core/`, `evaluation/`, `plot/` and `quality_testset/` remain for historical
  comparison. Their embedded measurements and figures are not refreshed paper results.
- All configurations on the generated frontier are decoded/rendered. Full-scene
  calibration and evaluation can take substantial time and disk space.

## Verification and source export

```bash
python -m unittest discover -s tests -v
python scripts/export_source.py --output dist/PrismGS-source.tar.gz
```

The export contains maintained runtime source, CUDA extension source and licenses,
preprocessing helpers and tests. It excludes datasets, throughput traces,
trained outputs, generated reports, historical media and compiled binaries.
For streaming evaluation from a source export, provide a trace with --trace. See
[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
A license for original contributions remains to be selected by the authors.

## Citation

```bibtex
@inproceedings{prismgs2026,
  author={Li, Jinghang and Wang, Zhe and Zhu, Yifei},
  title={PrismGS: Coordinated Training and Compression for Pareto-Optimal 4D Gaussian Streaming},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```

Training and rendering build on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
and [V³ / VideoGS](https://github.com/AuthorityWang/VideoGS).

### Anchor measurement protocol

Compression quality is evaluated on the initial frame (`--start`) of the
20-frame sequence, matching the paper's anchor test. Training and encoding cover
the entire sequence; bitrate uses the average encoded video payload per frame.
Camera selection applies to the anchor frame.
