"""Export measured calibration curves and optional Eq. 8 prediction diagnostics.

Run from the repository root after installing prismgs (or set PYTHONPATH=.).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prismgs.codec import ATTRIBUTES, write_json
from prismgs.quality import predict_quality


def report(profile_path, output, measured_rd=None):
    profile = json.loads(Path(profile_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for attr in ATTRIBUTES:
        for anchor in profile['attributes'][attr]:
            rows.append({'attribute': attr, 'qp': anchor['qp'],
                         'psnr_db': anchor['psnr'], 'ssim': anchor['ssim'],
                         'quality_factor': anchor['psnr'] / profile['base_psnr'],
                         'size_kb_per_frame': anchor['size_kb']})
    with (output/'anchors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'profile': str(Path(profile_path).resolve()),
               'base_psnr': profile['base_psnr'], 'fixed_kb': profile['fixed_kb'],
               'evaluation': profile['evaluation'],
               'rate_frame_count': profile['rate_frame_count'],
               'qps': profile['qps'], 'anchor_rows': len(rows),
               'unique_configurations': 1 + len(ATTRIBUTES) * (len(profile['qps']) - 1)}
    if measured_rd:
        measured = json.loads(Path(measured_rd).read_text())
        for key in ['frames', 'split', 'cameras', 'resolution']:
            if measured['evaluation'][key] != profile['evaluation'][key]:
                raise ValueError(f'Measured RD protocol differs from calibration: {key}')
        # This comparison diagnoses attribute interactions; it is not refitting.
        comparisons = []
        seen = set()
        for point in measured['points']:
            key = tuple(point['config'][a] for a in ATTRIBUTES)
            if key in seen:
                continue
            seen.add(key)
            predicted = predict_quality(profile, point['config'])
            comparisons.append({'name': point['name'], 'config': point['config'],
                                'measured_psnr': point['psnr'], 'predicted_psnr': predicted,
                                'error_db': predicted - point['psnr']})
        if not comparisons:
            raise ValueError('Measured RD contains no configurations.')
        predicted = np.array([p['predicted_psnr'] for p in comparisons])
        actual = np.array([p['measured_psnr'] for p in comparisons])
        correlation = (float(np.corrcoef(predicted, actual)[0, 1])
                       if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0 else None)
        summary['prediction_diagnostic'] = {
            'source': str(Path(measured_rd).resolve()),
            'note': 'Existing measurements; unique configurations, same anchor protocol. Not paper Table 1.',
            'configurations': len(comparisons),
            'mae_db': float(np.mean(np.abs(predicted - actual))),
            'max_abs_error_db': float(np.max(np.abs(predicted - actual))),
            'pearson_r': correlation}
        write_json(output/'prediction_comparison.json', comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for attr in ATTRIBUTES:
        anchors = sorted(profile['attributes'][attr], key=lambda a: a['qp'])
        qps = [a['qp'] for a in anchors]
        axes[0].plot(qps, [a['psnr']/profile['base_psnr'] for a in anchors], 'o-', label=attr)
        axes[1].plot(qps, [a['size_kb'] for a in anchors], 'o-', label=attr)
    axes[0].set_ylabel('Normalized PSNR: Q_i / Q_base')
    axes[1].set_ylabel('Attribute video payload (KB/frame)')
    for ax in axes:
        ax.set_xlabel('QP')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Initial-keyframe calibration; full-sequence average payload')
    fig.savefig(output/'calibration.png', dpi=180)
    fig.savefig(output/'calibration.pdf')
    plt.close(fig)
    write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--measured-rd', help='Optional existing RD measurements from the same source and protocol')
    args = parser.parse_args()
    print(json.dumps(report(args.profile, args.output, args.measured_rd), indent=2))
