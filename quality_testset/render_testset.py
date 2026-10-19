"""Evaluate decoded Pareto checkpoints using an external VideoGS renderer."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pareto', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('test_results.json'))
    args = parser.parse_args()
    for path in (args.pareto, args.checkpoints, args.dataset, args.renderer):
        if not path.exists():
            parser.error(f'Path does not exist: {path}')
    points = json.loads(args.pareto.read_text())
    results = []
    keys = ('position_low', 'color', 'opacity', 'scale', 'rotation')
    for point in points:
        config = {key: point['config'][key] for key in keys}
        checkpoint = args.checkpoints / ('test_' + '_'.join(str(config[k]) for k in keys))
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        run = subprocess.run(
            [sys.executable, str(args.renderer.resolve()),
             '--model_path', str(checkpoint.resolve()),
             '--source_path', str(args.dataset.resolve()),
             '--output_dir', str(checkpoint.resolve())],
            capture_output=True, text=True, check=True,
        )
        (checkpoint / 'eval.txt').write_text(run.stdout)
        metrics = {}
        for metric in ('PSNR', 'SSIM'):
            values = re.findall(rf'{metric}:\s*([0-9.eE+\-]+)', run.stdout)
            if len(values) != 1:
                raise ValueError(f'Expected one aggregate {metric}, got {len(values)}; '
                                 'use a renderer that reports aggregate evaluation metrics.')
            metrics[metric.lower()] = float(values[0])
        results.append({**config, 'bitrate_kb': point['bitrate_kb'], **metrics})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
