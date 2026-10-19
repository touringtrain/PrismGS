"""Compare CPU/CUDA decode with warmup, alternating order and the same PLY I/O."""
import argparse
import json
from pathlib import Path
import statistics
import time

from prismgs.codec import decode, write_json


def benchmark(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh benchmark output directory.')
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    points = json.loads(Path(args.frontier).read_text())['points']
    choices = [('low', points[0]['config']), ('middle', points[len(points)//2]['config']),
               ('high', points[-1]['config'])]
    results = []
    for name, config in choices:
        samples = {'cpu': [], 'cuda': []}
        # One unmeasured full decode per device to initialize CUDA/kernels and
        # warm file caches; both devices persist the same number of PLY files.
        for device in samples:
            decode(args.videos, output/name/device, config, args.workers, device=device)
        for repeat in range(args.repeats):
            for device in (['cpu', 'cuda'] if repeat % 2 == 0 else ['cuda', 'cpu']):
                started = time.perf_counter()
                stats = decode(args.videos, output/name/device, config, args.workers, device=device)
                samples[device].append({'wall_seconds': time.perf_counter() - started,
                                        'cuda_upload_and_reconstruct_ms': stats['cuda_upload_and_reconstruct_ms']})
        cpu = statistics.median(s['wall_seconds'] for s in samples['cpu'])
        gpu = statistics.median(s['wall_seconds'] for s in samples['cuda'])
        results.append({'name': name, 'config': config, 'frames': stats['frames'],
                        'cpu_median_seconds': cpu, 'cuda_median_seconds': gpu,
                        'cpu_over_cuda_wall_speedup': cpu / gpu, 'samples': samples})
        print(f'{name}: CPU {cpu:.4f}s, CUDA {gpu:.4f}s, ratio {cpu/gpu:.3f}', flush=True)
    import torch
    report = {'inputs': vars(args), 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'warmups_per_device_per_config': 1,
              'scope': 'Warm-cache local benchmark including CPU video decode, reconstruction, transfers and PLY writes. No rendering. Not paper GPU-only or network throughput.',
              'points': results}
    write_json(output/'benchmark.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['videos', 'frontier', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    benchmark(parser.parse_args())
