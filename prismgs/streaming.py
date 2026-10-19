"""Throughput-only simulation; reports its assumptions alongside its output."""
import csv
import json
import math
from pathlib import Path
from .codec import write_json


def evaluate_trace(trace, results, output, mobility='walking', fps=30):
    if fps <= 0:
        raise ValueError('FPS must be positive')
    points = json.loads(Path(results).read_text())['points']
    if not points or any('psnr' not in p or p['size_kb'] <= 0 for p in points):
        raise ValueError('Supply measured points with positive size_kb and actual psnr (not predicted_psnr).')
    samples = []
    with open(trace, newline='') as handle:
        for row in csv.DictReader(handle):
            if row['mobility_mode'] != mobility:
                continue
            # Lumos Throughput is Mbps; payload sizes are decimal KB/frame.
            throughput = float(row['Throughput'])
            if not math.isfinite(throughput) or throughput < 0:
                raise ValueError('Throughput must be finite and nonnegative')
            candidates = [(min(fps, throughput * 1000 / 8 / p['size_kb']), p) for p in points]
            realtime = [(f,p) for f,p in candidates if f >= fps]
            actual_fps, point = max(realtime, key=lambda pair: pair[1]['psnr']) if realtime else max(candidates, key=lambda pair: (pair[0], pair[1]['psnr']))
            samples.append({'throughput_mbps':throughput,'fps':actual_fps,'psnr':point['psnr'],'size_kb':point['size_kb']})
    if not samples:
        raise ValueError(f'No {mobility} samples found in {trace}')
    write_json(output,{'assumptions':'Independent 1-second samples; throughput-only upper bound; no buffering, RTT or decode cost.',
        'mobility':mobility,'mean_fps':sum(p['fps'] for p in samples)/len(samples),
        'mean_psnr':sum(p['psnr'] for p in samples)/len(samples),'samples':samples})
