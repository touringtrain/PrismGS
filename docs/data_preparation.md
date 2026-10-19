# Dataset preparation details

Download HiFi4G through https://github.com/moqiyinlun/HiFi4G_Dataset and follow
its access/license terms. We do not redistribute its images or initialization
points. Directory names below are examples, not a guarantee of archive layout.

## Already-undistorted data (recommended)

Follow the root README: pass the directory of numbered frame folders and the
matching undistorted COLMAP model to `scripts/prepare_hifi4g.py`. The script
supports `FRAME/CAMERA.png` and `FRAME/images/CAMERA.png` inputs. Numeric frame
folders must match the requested frame numbers; `--end` is exclusive.

It copies only camera-referenced image basenames and model files, checks for
missing images before copying, and rejects nonempty outputs. A PINHOLE or
SIMPLE_PINHOLE camera model is required by the training loader. All camera
basenames must be unique. Camera names are not assumed to be consecutive.

Keep images at the calibration resolution. Use the runtime `--resolution` option
for training/rendering downsampling; do not manually resize without updating
intrinsics. Image masks must already be composited against the intended background.

## Distorted images

Install COLMAP first. For each frame, use the model that matches the original
camera images and run the standard undistorter, for example:

```bash
colmap image_undistorter \
  --image_path /path/to/raw/0/images \
  --input_path /path/to/original/calibration \
  --output_path /path/to/undistorted/0 --output_type COLMAP
```

Repeat for every selected frame. COLMAP writes `images/` and a new undistorted
model, commonly under `sparse/` (inspect the output version's actual layout).
Use the **new** model with the undistorted images, never the old distorted one.
For a fixed rig, collect the output frames as `UNDISTORTED/FRAME/images/` and run:

```bash
python scripts/prepare_hifi4g.py --images /path/to/undistorted \
  --calibration /path/to/undistorted/0/sparse \
  --output datasets/4k_1_greeting --start 0 --end 20
```

If the new model is under `sparse/0`, adjust that argument. Confirm camera IDs,
image dimensions and poses are consistent across frames before sharing calibration.
Background segmentation/compositing is dataset-specific and is not performed by
this command. Changing backgrounds does not change camera intrinsics.

## Missing initialization points

Camera calibration alone is insufficient. The COLMAP route expects
`points3D.ply`, `points3D.bin` or `points3D.txt` in each model folder.
Use the dataset's initialization points or reconstruct a compatible point cloud
with COLMAP/another reconstruction tool. The cloud must be in the same coordinate
system as the cameras. Do not substitute a trained Gaussian checkpoint as an
RGB initialization cloud.

V³ also uses NeuS2_K for keyframe point generation. Its optional external source
and build instructions are in `third_party/videogs/README.md`; it is not bundled
or automatically run. The legacy sequence scripts illustrate sampling a NeuS2
mesh into `points3d.ply`, but the maintained pipeline requires that input to be
prepared first. No verified one-command raw-images-to-NeuS2 recipe is claimed.

## Alternative transforms layout

If using the recovered VideoGS transforms loader rather than COLMAP, each frame
must contain:

```text
FRAME/
  images/...
  transforms.json
  points3d.ply
```

`transforms.json` has a top-level `frames` array; each entry needs `file_path`
(relative to that frame folder), a 4x4 camera-to-world `transform_matrix`, and
`fl_x`, `fl_y`, `cx`, `cy`. The transforms follow the original V³/NGP convention:
the loader applies a Y/Z camera-axis flip before converting to world-to-camera.
Do not directly insert COLMAP world-to-camera matrices.

The retained `training/prismgs_backend/preprocess/colmap2k.py` is an upstream
conversion helper, with separate prerequisites and original third-party notices;
the recommended COLMAP route does not need to run it. If both `sparse/` and
`transforms.json` exist, the loader prioritizes **COLMAP sparse/**. For the
transforms route provide initialization `points3d.ply` in every selected frame.

## First test

After preparing frames 0–19, start with two frames:

```bash
python -m prismgs run --dataset datasets/4k_1_greeting \
  --output runs/greeting_smoke --start 0 --end 2 --group-size 2 \
  --resolution 8 --keyframe-iterations 2 --prune-iterations 2 \
  --motion-iterations 2 --finetune-iterations 2 \
  --qps 0 40 --cameras 23 --workers 2 --step-kb 0.1
```

Camera index 23 is a diagnostic example for this 81-view sequence. Use a valid
index for other datasets, or omit `--cameras` to render all selected-split views.
Change `--end 20 --group-size 20` for all twenty frames. Short training counts
only verify execution. Remove them for default training and select the intended
held-out evaluation protocol with `--split test`.

## Neural 3D Video (N3DV)

PrismGS also evaluates on **Neural 3D Video**, introduced with DyNeRF. Obtain the
videos and camera metadata through the [official project page](https://neural-3d-video.github.io/)
and its linked download instructions, subject to the dataset's terms. Do not
redistribute the videos through this repository. Scene names commonly include
`coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`,
`flame_steak`, and `sear_steak`; use the names in your downloaded release.

### 1. Extract synchronized frames

A typical original scene contains one video per fixed camera and LLFF-style
camera metadata:

```text
/path/to/N3DV/coffee_martini/
  cam00.mp4
  cam01.mp4
  ...
  poses_bounds.npy
```

Unlike HiFi4G, these are **per-camera videos**, not frame folders. Decode the
same frame indices from every camera without temporal resampling. For example,
extract frames 0–19 of one camera:

```bash
mkdir -p /path/to/extracted/coffee_martini/cam00
ffmpeg -i /path/to/N3DV/coffee_martini/cam00.mp4 \
  -frames:v 20 -vsync 0 -start_number 0 \
  /path/to/extracted/coffee_martini/cam00/%06d.png
```

Repeat for every camera, then arrange the extracted images by **time**, keeping
a stable camera filename in every frame, for example:

```text
/path/to/frame_images/coffee_martini/
  0/cam00.png
  0/cam01.png
  ...
  1/cam00.png
  1/cam01.png
  ...
  19/cam00.png
  19/cam01.png
```

`0/` contains the first image from every camera, not the first camera's entire
video. Preserve timestamps/frame alignment and original resolution. The runtime
`--fps` must match your chosen temporal sampling (commonly 30 FPS; inspect your
source videos). Renaming cameras is allowed only if calibration image names are
updated to match.

### 2. Convert cameras and reconstruct initialization points

The PrismGS loader **does not read `poses_bounds.npy` directly**. Prepare one of
the supported camera formats before invoking the main pipeline:

- **COLMAP route:** use a matching supplied/reconstructed COLMAP model with
  PINHOLE/SIMPLE_PINHOLE intrinsics and a nonempty initialization point cloud.
  Camera poses and point coordinates must share the same coordinate system.
  If reconstructing with COLMAP, calibrate/triangulate synchronized multiview
  images of the first frame (or every group keyframe as needed), then undistort
  if necessary using the procedure above. Verify view alignment. Sharing the
  first-frame model assumes a fixed camera rig.
- **Provided N3DV/LLFF poses:** convert each row's 3x5 pose/HWF block to the
  V³/NGP `transforms.json` convention described above, maintaining the exact
  camera-to-row ordering. Intrinsics must be scaled if extracted image dimensions
  differ from the metadata. Triangulate/reconstruct a compatible RGB point cloud
  using the same poses. `poses_bounds.npy` near/far bounds are not point coordinates
  and do not replace an initialization point cloud.

No verified automatic raw-N3DV pose-conversion/triangulation command is included
in the maintained interface. The original local `n3v2blender.py` mixes extraction,
pose changes, COLMAP and dense reconstruction with scene-specific assumptions;
it is intentionally not presented as a working general converter. A generic
DyNeRF transforms JSON is not necessarily compatible: this loader expects
per-view intrinsics inside each frame entry, not just top-level intrinsics.

### 3. Arrange the prepared scene

With frame-organized, undistorted images and a fixed-camera COLMAP model, reuse
the arrangement helper (despite its HiFi4G name, this step is dataset-independent):

```bash
python scripts/prepare_hifi4g.py \
  --images /path/to/frame_images/coffee_martini \
  --calibration /path/to/coffee_martini/undistorted_model \
  --output datasets/coffee_martini --start 0 --end 20
```

The output is `datasets/coffee_martini/FRAME/images/` and
`datasets/coffee_martini/FRAME/sparse/0/`, exactly as for HiFi4G. Alternatively,
prepare per-frame `images/`, `transforms.json` and `points3d.ply` without `sparse/`.
If using per-frame COLMAP reconstruction, place each frame's own model under
its `sparse/0/` rather than copying the first frame's model.

### 4. Run the same pipeline

```bash
python -m prismgs run --dataset datasets/coffee_martini \
  --output runs/coffee_martini_smoke --start 0 --end 2 --group-size 2 \
  --resolution 8 --keyframe-iterations 2 --prune-iterations 2 \
  --motion-iterations 2 --finetune-iterations 2 \
  --qps 0 40 --workers 2 --step-kb 0.1 --split test --black-background
```

Use the background choice matching your prepared images; the command explicitly
overrides the HiFi4G white-background default. No hardcoded 81-view count or
camera index is used here. `--split test` uses the bundled loader's held-out
split (COLMAP: every eighth view; transforms: seeded 30% split). This is **not**
a claim of matching the official N3DV or paper split: explicitly establish and
record the required view split before reporting comparable benchmark numbers.
This N3DV preparation/run recipe has not yet been validated end to end in this
release; the recorded GPU smoke runs used HiFi4G only.
