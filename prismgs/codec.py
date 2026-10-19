"""Attribute image packing and independent, parallel video decoding (SH degree 0)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil
import subprocess
import time

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

ATTRIBUTES = {'position_low': [0, 2, 4], 'color': [9, 10, 11],
              'opacity': [12], 'scale': [13, 14, 15], 'rotation': [16, 17, 18, 19]}
HIGH = [1, 3, 5]
CHANNELS = HIGH + [c for channels in ATTRIBUTES.values() for c in channels]
PROPERTIES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
              'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def latest_ply(checkpoints, frame):
    folder = Path(checkpoints) / str(frame) / 'point_cloud'
    iterations = [p for p in folder.glob('iteration_*') if (p / 'point_cloud.ply').is_file()]
    if not iterations:
        raise FileNotFoundError(f'No Gaussian PLY checkpoint in {folder}')
    return max(iterations, key=lambda p: int(p.name.split('_')[-1])) / 'point_cloud.ply'


def read_ply(path):
    data = PlyData.read(path)['vertex'].data
    extra = [n for n in data.dtype.names if n.startswith('f_rest_')]
    if extra:
        raise ValueError('The released 20-channel codec requires SH degree 0; train with --sh_degree 0.')
    matrix = np.column_stack([data[n] for n in PROPERTIES])
    if not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError(f'Empty or non-finite point cloud: {path}')
    return matrix


def write_ply(path, matrix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(len(matrix), dtype=[(n, '<f4') for n in PROPERTIES])
    for i, name in enumerate(PROPERTIES):
        vertices[name] = matrix[:, i]
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False, byte_order='<').write(str(path))


def pack(checkpoints, output, start, end, group_size):
    """Preserve stored Gaussian parameters; positions use low/high 8-bit planes."""
    if start >= end or group_size <= 0:
        raise ValueError('Require start < end and group_size > 0 (end is exclusive).')
    output = Path(output)
    metadata = {'format_version': 1, 'sh_degree': 0, 'groups': []}
    for group, first in enumerate(range(start, end, group_size)):
        frames = list(range(first, min(first + group_size, end)))
        clouds = [read_ply(latest_ply(checkpoints, f)) for f in frames]
        # All frames of a video must have the same dimensions, even after densification.
        side = max(8, int(np.ceil(np.sqrt(max(map(len, clouds))) / 8)) * 8)
        folder = output / f'group{group}'
        folder.mkdir(parents=True, exist_ok=True)
        info = {'index': group, 'frames': frames, 'side': side, 'metadata': {}}
        for frame, cloud in zip(frames, clouds):
            ranges = []
            for i in range(17):
                lo, hi = float(cloud[:, i].min()), float(cloud[:, i].max())
                ranges.append([lo, hi])
                if i in (3, 4, 5):
                    continue  # Normals are unused by Gaussian splatting.
                limit = 65535 if i < 3 else 255
                values = np.zeros(len(cloud)) if hi == lo else (cloud[:, i] - lo) / (hi - lo) * limit
                quantized = np.clip(values, 0, limit).astype(np.uint16 if i < 3 else np.uint8)
                planes = [(2*i, quantized & 255), (2*i+1, quantized >> 8)] if i < 3 else [(i+3, quantized)]
                for channel, plane in planes:
                    image = np.zeros(side * side, dtype=np.uint8)
                    image[:len(cloud)] = plane
                    if not cv2.imwrite(str(folder / f'{frame}_{channel}.png'), image.reshape(side, side)):
                        raise OSError(f'Failed to write image for frame {frame}, channel {channel}')
            info['metadata'][str(frame)] = {'count': len(cloud), 'ranges': ranges}
        metadata['groups'].append(info)
    write_json(output / 'metadata.json', metadata)


def encode(images, videos, qps, fps=30):
    if not shutil.which('ffmpeg'):
        raise RuntimeError('Install FFmpeg with libx264 support before encoding.')
    if not qps or any(q < 0 or q > 51 for q in qps) or fps <= 0:
        raise ValueError('QP must be in [0, 51] and FPS must be positive.')
    images, videos = Path(images), Path(videos)
    metadata = json.loads((images / 'metadata.json').read_text())
    for qp in qps:
        for group in metadata['groups']:
            folder = videos / f'png_all_{qp}' / f"group{group['index']}"
            folder.mkdir(parents=True, exist_ok=True)
            for channel in CHANNELS:
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-framerate', str(fps), '-start_number', str(group['frames'][0]),
                    '-i', str(images / f"group{group['index']}" / f'%d_{channel}.png'),
                    '-frames:v', str(len(group['frames'])), '-c:v', 'libx264',
                    '-qp', str(0 if channel in HIGH else qp), '-pix_fmt', 'yuvj444p',
                    str(folder / f'{channel}.mp4')], check=True)
    write_json(videos / 'metadata.json', metadata)


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise OSError(f'Cannot open video: {path}')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    return frames


def reconstruct(info, frame, offset, planes):
    spec = info['metadata'][str(frame)]
    cloud = np.zeros((spec['count'], 17), dtype=np.float32)
    for i in range(17):
        if i in (3, 4, 5):
            continue
        if i < 3:
            pixels = planes[2*i][offset].astype(np.uint16) + (planes[2*i+1][offset].astype(np.uint16) << 8)
            limit = 65535
        else:
            pixels, limit = planes[i+3][offset], 255
        lo, hi = spec['ranges'][i]
        cloud[:, i] = pixels.reshape(-1)[:spec['count']].astype(np.float32) / limit * (hi-lo) + lo
    return cloud


def reconstruct_group_cuda(info, planes, device):
    """Batch all frames/points on CUDA, following the legacy GPU decoder.

    Video entropy decoding stays on CPU. Upload packed uint8 planes once per
    group, reconstruct raw parameters on GPU, then transfer one PLY batch back.
    The returned CUDA event time includes H2D and reconstruction, not D2H.
    """
    import torch
    frames = info['frames']
    counts = [info['metadata'][str(f)]['count'] for f in frames]
    max_points = max(counts)
    # [channel, frame, pixel]; trim padding before transferring to the GPU.
    packed = np.stack([np.stack(planes[c]).reshape(len(frames), -1)[:, :max_points]
                       for c in CHANNELS])
    # Match CPU scalar conversion: subtract ranges in Python float precision
    # before casting spans to float32, rather than subtracting rounded bounds.
    ranges = [info['metadata'][str(f)]['ranges'] for f in frames]
    lows = np.asarray([[lo for lo, hi in r] for r in ranges], dtype=np.float32)
    spans = np.asarray([[hi - lo for lo, hi in r] for r in ranges], dtype=np.float32)
    with torch.cuda.device(device), torch.no_grad():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        data = torch.from_numpy(packed).to(device)
        low = torch.from_numpy(lows).to(device)
        span = torch.from_numpy(spans).to(device)
        channels = {c: data[i] for i, c in enumerate(CHANNELS)}
        clouds = torch.zeros((len(frames), max_points, len(PROPERTIES)), dtype=torch.float32, device=device)
        for i in range(len(PROPERTIES)):
            if i in (3, 4, 5):
                continue
            if i < 3:
                pixels = channels[2*i].to(torch.int32) + (channels[2*i+1].to(torch.int32) << 8)
                limit = 65535
            else:
                pixels, limit = channels[i+3], 255
            clouds[:, :, i] = pixels.float() / limit * span[:, i, None] + low[:, i, None]
        end.record()
        host = clouds.cpu().numpy()  # Synchronizes before CPU writers see data.
        elapsed_ms = start.elapsed_time(end)
    return host, elapsed_ms


def decode(videos, output, config, workers=4, device='cpu'):
    if set(config) != set(ATTRIBUTES) or any(not isinstance(q, int) or not 0 <= q <= 51 for q in config.values()):
        raise ValueError(f'Configuration must specify integer QPs [0, 51] for {list(ATTRIBUTES)}')
    if workers < 1:
        raise ValueError('workers must be positive')
    started = time.perf_counter()
    if device != 'cpu':
        import torch
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Decode device must be cpu, cuda or cuda:N.')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA decoding requested but CUDA is unavailable.')
        # Initialize the selected device before starting any background I/O.
        torch.empty(0, device=device)
        device = str(device)
    videos, output = Path(videos), Path(output)
    metadata = json.loads((videos / 'metadata.json').read_text())
    if metadata.get('format_version') != 1 or metadata.get('sh_degree') != 0:
        raise ValueError('Decoder requires format_version=1 and SH degree 0 metadata.')
    if not metadata['groups'] or any(not g['frames'] for g in metadata['groups']):
        raise ValueError('Decoder requires nonempty frame groups.')
    mapping = {c: config[attr] for attr, channels in ATTRIBUTES.items() for c in channels}
    # High position bytes are identical in every QP folder.
    mapping.update({c: config['position_low'] for c in HIGH})
    def load_group(group):
        paths = [videos / f'png_all_{mapping[c]}' / f"group{group['index']}" / f'{c}.mp4' for c in CHANNELS]
        with ThreadPoolExecutor(max_workers=workers) as readers:
            planes = dict(zip(CHANNELS, readers.map(read_video, paths)))
        for channel, frames in planes.items():
            if len(frames) != len(group['frames']) or any(f.shape != (group['side'], group['side']) for f in frames):
                raise ValueError(f'Video frame count or shape mismatch: group {group["index"]}, channel {channel}')
        return planes

    # Keep at most the current and next group in memory. Background channel I/O
    # overlaps independent frame reconstruction and PLY output for the current group.
    groups = iter(metadata['groups'])
    gpu_ms = 0.0
    with ThreadPoolExecutor(max_workers=1) as prefetch, ThreadPoolExecutor(max_workers=workers) as pool:
        current = next(groups, None)
        pending = prefetch.submit(load_group, current) if current is not None else None
        while current is not None:
            planes = pending.result()
            following = next(groups, None)
            pending = prefetch.submit(load_group, following) if following is not None else None
            batch = None
            if device != 'cpu':
                batch, elapsed_ms = reconstruct_group_cuda(current, planes, device)
                gpu_ms += elapsed_ms
            def restore(item):
                offset, frame = item
                cloud = (reconstruct(current, frame, offset, planes) if batch is None else
                         batch[offset, :current['metadata'][str(frame)]['count']])
                write_ply(output / str(frame) / 'point_cloud/iteration_1/point_cloud.ply', cloud)
            list(pool.map(restore, enumerate(current['frames'])))
            current = following
    write_json(output / 'compression.json', config)
    stats = {'device': device, 'workers': workers,
             'frames': sum(len(g['frames']) for g in metadata['groups']),
             'groups': len(metadata['groups']), 'wall_seconds': time.perf_counter() - started,
             'cuda_upload_and_reconstruct_ms': gpu_ms if device != 'cpu' else None,
             'video_decode': 'CPU OpenCV',
             'scope': 'Wall time includes initialization, video I/O, reconstruction, transfers and PLY writes; CUDA events include H2D and reconstruction only.'}
    write_json(output / 'decode_stats.json', stats)
    return stats
