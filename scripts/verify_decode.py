"""Decode every frontier configuration and uniform control, then verify/render.

Uses the calibration profile's evaluation settings. Outputs must be fresh.
Run from the repository root with PYTHONPATH=. in the CUDA environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from prismgs.codec import ATTRIBUTES, HIGH, PROPERTIES, decode, latest_ply, read_ply, write_json
from prismgs.rendering import evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(args):
    videos, output = Path(args.videos), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh verification output directory.')
    profile = json.loads(Path(args.profile).read_text())
    frontier = json.loads(Path(args.frontier).read_text())['points']
    metadata = json.loads((videos/'metadata.json').read_text())
    frames = [f for g in metadata['groups'] for f in g['frames']]
    specs = {int(f): spec for g in metadata['groups'] for f, spec in g['metadata'].items()}
    if not frontier or not frames or len(set(frames)) != len(frames):
        raise ValueError('Require a nonempty frontier and unique encoded frames.')
    protocol = profile['evaluation']
    if protocol['frames'] != [frames[0]]:
        raise ValueError('Profile must measure the first encoded frame.')
    # Decoder selects high-byte files from the position-low QP folder. Verify
    # that these duplicated files really are the same lossless payload.
    for group in metadata['groups']:
        for c in HIGH:
            paths = [videos/f'png_all_{q}/group{group["index"]}/{c}.mp4' for q in profile['qps']]
            if len({digest(path) for path in paths}) != 1:
                raise ValueError(f'Position high-byte payload differs across QPs: {paths}')
    candidates = [(f'uniform_qp{q}', {a: q for a in ATTRIBUTES}, None) for q in profile['qps']]
    candidates += [(f'pareto_{i}', p['config'], p['size_kb']) for i, p in enumerate(frontier)]
    unique = {}
    for name, config, expected_size in candidates:
        key = tuple(config[a] for a in ATTRIBUTES)
        entry = unique.setdefault(key, {'name': name, 'config': config, 'aliases': [], 'expected_sizes': []})
        entry['aliases'].append(name)
        if expected_size is not None:
            entry['expected_sizes'].append(expected_size)
    write_json(output/'verification_config.json', vars(args))
    results = []
    zero_error = np.zeros(len(PROPERTIES))
    for index, entry in enumerate(unique.values()):
        name, config = entry['name'], entry['config']
        print(f'Decode {index + 1}/{len(unique)}: {name}', flush=True)
        folder = output/name
        start = time.perf_counter()
        stats = decode(videos, folder/'checkpoint', config, args.workers, device=args.device)
        elapsed = time.perf_counter() - start
        hashes = {}
        cpu_max_error = 0.0
        for frame in frames:
            path = latest_ply(folder/'checkpoint', frame)
            cloud = read_ply(path)  # Rejects empty, nonfinite or non-SH0 PLYs.
            if cloud.shape != (specs[frame]['count'], len(PROPERTIES)):
                raise ValueError(f'Point count mismatch: {path}')
            ranges = np.asarray(specs[frame]['ranges'])
            lo, hi = ranges[:, 0], ranges[:, 1]
            active = [i for i in range(len(PROPERTIES)) if i not in (3, 4, 5)]
            tolerance = 8 * np.finfo(np.float32).eps * np.maximum(1, np.maximum(abs(lo), abs(hi)))
            if (np.any(cloud[:, active] < lo[active] - tolerance[active]) or
                    np.any(cloud[:, active] > hi[active] + tolerance[active])):
                raise ValueError(f'Decoded values outside recorded ranges: {path}')
            if np.any(cloud[:, 3:6] != 0):
                raise ValueError(f'Unused normals must be zero: {path}')
            hashes[str(frame)] = digest(path)
            if args.cpu_reference:
                reference_folder = Path(args.cpu_reference)/name/'checkpoint'
                reference_config = json.loads((reference_folder/'compression.json').read_text())
                if reference_config != config:
                    raise ValueError(f'CPU reference configuration mismatch: {name}')
                reference = read_ply(latest_ply(reference_folder, frame))
                if reference.shape != cloud.shape:
                    raise ValueError(f'CPU/GPU point count mismatch: {name} frame {frame}')
                error = np.max(abs(reference - cloud), axis=0)
                if np.any(error > tolerance):
                    raise ValueError(f'CPU/GPU parameters differ beyond float tolerance: {name} frame {frame}')
                cpu_max_error = max(cpu_max_error, float(error.max()))
            if args.original_checkpoints and all(q == 0 for q in config.values()):
                original = read_ply(latest_ply(args.original_checkpoints, frame))
                if original.shape != cloud.shape:
                    raise ValueError('Original point count differs from decoded baseline.')
                error = np.max(abs(original - cloud), axis=0)
                limits = np.array([65535 if i < 3 else 255 for i in range(len(PROPERTIES))])
                bound = (hi - lo) / limits + tolerance
                if np.any(error[active] > bound[active]):
                    raise ValueError(f'QP=0 exceeds packing quantization error at frame {frame}')
                zero_error = np.maximum(zero_error, error)
        write_json(folder/'ply_sha256.json', hashes)
        mapping = {c: config[a] for a, channels in ATTRIBUTES.items() for c in channels}
        mapping.update({c: config['position_low'] for c in HIGH})
        size = sum((videos/f'png_all_{q}/group{g["index"]}/{c}.mp4').stat().st_size
                   for g in metadata['groups'] for c, q in mapping.items()) / len(frames) / 1000
        if any(not np.isclose(size, expected, rtol=0, atol=1e-9) for expected in entry['expected_sizes']):
            raise ValueError(f'Frontier payload mismatch: {name}')
        metrics = evaluate(folder/'checkpoint', args.dataset, folder/'anchor', [frames[0]],
                           protocol['cameras'], protocol['resolution'], protocol['white_background'], protocol['split'])
        result = {'name': name, 'aliases': entry['aliases'], 'config': config,
                  'size_kb': size, 'psnr': metrics['psnr'], 'ssim': metrics['ssim'],
                  'decode_seconds': elapsed, 'decoded_frames': len(frames),
                  'decode_stats': stats,
                  'cpu_max_absolute_error': cpu_max_error if args.cpu_reference else None,
                  'anchor_view_count': len(metrics['views'])}
        results.append(result)
        write_json(output/'progress.json', results)
    # Check scheduling independence at low, middle and high rate.
    representatives = [frontier[i]['config'] for i in [0, len(frontier)//2, len(frontier)-1]]
    serial_checks = []
    for config in representatives:
        entry = unique[tuple(config[a] for a in ATTRIBUTES)]
        destination = output/'serial'/entry['name']
        decode(videos, destination, config, workers=1, device=args.device)
        hashes = json.loads((output/entry['name']/'ply_sha256.json').read_text())
        if any(digest(latest_ply(destination, f)) != hashes[str(f)] for f in frames):
            raise ValueError(f'Serial/parallel mismatch: {entry["name"]}')
        serial_checks.append(entry['name'])
    # Render middle and final dynamic frames of a mixed-QP configuration too.
    middle = unique[tuple(representatives[1][a] for a in ATTRIBUTES)]
    dynamic_frames = sorted({frames[len(frames)//2], frames[-1]} - {frames[0]})
    dynamic = None
    if dynamic_frames:
        print(f'Render dynamic frames {dynamic_frames}: {middle["name"]}', flush=True)
        dynamic = evaluate(output/middle['name']/'checkpoint', args.dataset, output/'dynamic', dynamic_frames,
                           protocol['cameras'], protocol['resolution'], protocol['white_background'], protocol['split'])
    summary = {
        'status': 'passed', 'inputs': vars(args), 'evaluation': protocol, 'frames': frames,
        'requested_configurations': len(candidates), 'unique_configurations': len(results),
        'decoded_checkpoints': len(results)*len(frames),
        'anchor_view_records': sum(r['anchor_view_count'] for r in results),
        'serial_parallel_identical': serial_checks,
        'serial_checkpoints': len(serial_checks)*len(frames),
        'high_bytes_identical_across_qps': True,
        'qp0_max_absolute_error_by_property': dict(zip(PROPERTIES, zero_error.tolist())) if args.original_checkpoints else None,
        'decode_seconds_total': sum(r['decode_seconds'] for r in results),
        'timing_scope': 'CPU OpenCV video decode + selected-device reconstruction + transfers + PLY writes; excludes verification and rendering. Not a controlled throughput benchmark.',
        'dynamic_render': {'config': middle['config'], 'frames': dynamic_frames,
                           'view_records': len(dynamic['views']), 'psnr': dynamic['psnr'],
                           'ssim': dynamic['ssim']} if dynamic else None,
        'points': results}
    write_json(output/'verification.json', summary)
    write_json(output/'measured_rd.json', {'size_unit': profile['size_unit'],
                                         'evaluation': protocol, 'points': results})
    print(f'PASS: {len(results)} configurations, {len(frames)} frames each; {output}/verification.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['videos', 'dataset', 'profile', 'frontier', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--original-checkpoints')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cpu', help='cpu, cuda or cuda:N')
    parser.add_argument('--cpu-reference', help='Previous verify_decode output to compare every PLY')
    verify(parser.parse_args())
