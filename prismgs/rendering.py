"""Evaluate explicit frames/views; reuse camera images within a process."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

_CAMERA_CACHE = OrderedDict()


def evaluate(checkpoints, dataset, output, frames, camera_indices=None, resolution=2,
             white_background=True, split='train'):
    import sys
    backend = str(Path(__file__).resolve().parents[1] / 'training/prismgs_backend')
    if backend not in sys.path:
        sys.path.insert(0, backend)
    import torch
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from .codec import write_json, latest_ply
    if not torch.cuda.is_available():
        raise RuntimeError('Rendering requires CUDA.')
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1., 1., 1.] if white_background else [0., 0., 0.], device='cuda')
    records = []
    with torch.no_grad():
        for frame in frames:
            key = (str(Path(dataset).resolve()), frame, resolution, white_background, split,
                   None if camera_indices is None else tuple(camera_indices))
            gaussians = GaussianModel(0)
            if key not in _CAMERA_CACHE:
                args = SimpleNamespace(model_path=str(Path(checkpoints)/str(frame)),
                    source_path=str(Path(dataset)/str(frame)), sh_degree=0, images='images',
                    eval=split == 'test', white_background=white_background, resolution=resolution, data_device='cuda')
                scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)
                cameras = scene.getTestCameras() if split == 'test' else scene.getTrainCameras()
                selected = list(range(len(cameras))) if camera_indices is None else camera_indices
                if not selected or any(i < 0 or i >= len(cameras) for i in selected):
                    raise ValueError(f'Invalid/empty camera selection for frame {frame}: {selected}')
                cameras = {i: cameras[i] for i in selected}
                _CAMERA_CACHE[key] = cameras
                # Bound cached images, not frames; a single-view diagnostic can reuse
                # a whole short sequence without retaining every training camera.
                while sum(len(c) for c in _CAMERA_CACHE.values()) > 162 and len(_CAMERA_CACHE) > 1:
                    _CAMERA_CACHE.popitem(last=False)
            else:
                cameras = _CAMERA_CACHE[key]
                _CAMERA_CACHE.move_to_end(key)
                gaussians.load_ply(str(latest_ply(checkpoints, frame)))
            indices = list(cameras) if camera_indices is None else camera_indices
            if not indices or any(i not in cameras for i in indices):
                raise ValueError(f'Invalid/empty camera selection for frame {frame}: {indices}')
            for index in indices:
                cam = cameras[index]
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                image = render(cam, gaussians, pipeline, background)['render'].clamp(0, 1)
                end.record()
                torch.cuda.synchronize()
                gt = cam.original_image.cuda()
                records.append({'frame': frame, 'camera': index, 'psnr': psnr(image, gt).mean().item(),
                                'ssim': ssim(image, gt).mean().item(), 'render_ms': start.elapsed_time(end)})
    if not records:
        raise ValueError('No frames evaluated')
    result = {key: sum(r[key] for r in records)/len(records) for key in ['psnr','ssim','render_ms']}
    result.update({'split': split, 'resolution': resolution, 'views': records})
    write_json(Path(output)/'metrics.json', result)
    (Path(output)/'eval.txt').write_text(f"PSNR: {result['psnr']:.6f} dB\nSSIM: {result['ssim']:.6f}\n")
    return result
