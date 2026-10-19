"""Measured attribute quality model and discretized DP (paper Eq. 8 / Algorithm 1)."""
import json
import math
from pathlib import Path
from .codec import ATTRIBUTES, HIGH, write_json


def validate_qps(qps):
    if (not qps or any(type(q) is not int or not 0 <= q <= 51 for q in qps)
            or 0 not in qps or len(set(qps)) != len(qps)):
        raise ValueError('Calibration requires distinct integer QPs in [0, 51], including 0.')


def predict_quality(profile, config):
    """Paper Eq. 8, using only measured anchors (no extrapolation)."""
    if set(config) != set(ATTRIBUTES):
        raise ValueError(f'Configuration must specify {list(ATTRIBUTES)}')
    base = float(profile['base_psnr'])
    quality = base
    for attr, qp in config.items():
        choices = {c['qp']: c for c in profile['attributes'][attr]}
        if qp not in choices:
            raise ValueError(f'Unmeasured QP for {attr}: {qp}')
        quality *= float(choices[qp]['psnr']) / base
    return quality


def frontier(profile, step_kb=1.0, max_kb=None):
    if not math.isfinite(step_kb) or step_kb <= 0:
        raise ValueError('step_kb must be finite and positive')
    base = float(profile['base_psnr'])
    fixed = float(profile['fixed_kb'])
    if not math.isfinite(base) or base <= 0 or not math.isfinite(fixed) or fixed < 0:
        raise ValueError('Invalid base PSNR or fixed size')
    if max_kb is not None and (not math.isfinite(max_kb) or max_kb <= 0):
        raise ValueError('max_kb must be finite and positive')
    states = {math.ceil(fixed / step_kb): (1.0, fixed, {})}
    for attr in ATTRIBUTES:
        choices = profile['attributes'][attr]
        if not choices:
            raise ValueError(f'No measurements for {attr}')
        next_states = {}
        for quality, size, config in states.values():
            for choice in choices:
                rate, factor = float(choice['size_kb']), float(choice['psnr']) / base
                if not math.isfinite(rate) or rate < 0 or not math.isfinite(factor) or factor <= 0:
                    raise ValueError(f'Invalid measurement for {attr}')
                new_size = size + rate
                if max_kb is not None and new_size > max_kb:
                    continue
                index = math.ceil(new_size / step_kb)
                candidate = (quality * factor, new_size, {**config, attr: int(choice['qp'])})
                previous = next_states.get(index)
                if previous is None or (candidate[0], -candidate[1]) > (previous[0], -previous[1]):
                    next_states[index] = candidate
        states = next_states
    result, best = [], -math.inf
    for quality, size, config in sorted(states.values(), key=lambda x: (x[1], -x[0])):
        if quality > best:
            result.append({'config': config, 'predicted_psnr': base * quality, 'size_kb': size})
            best = quality
    return {'size_unit': 'decimal KB per frame (video payload)', 'step_kb': step_kb,
            'note': 'Discretized DP; smaller bins retain more candidate configurations.', 'points': result}


def collect_profile(videos, measurements, qps, output):
    """Collect keyframe quality anchors and full-sequence video payload rates."""
    validate_qps(qps)
    videos, measurements = Path(videos), Path(measurements)
    metadata = json.loads((videos / 'metadata.json').read_text())
    frames = sum(len(g['frames']) for g in metadata['groups'])
    if frames == 0:
        raise ValueError('Calibration requires a nonempty encoded sequence.')
    def size(channels, qp):
        return sum((videos / f'png_all_{qp}' / f"group{g['index']}" / f'{c}.mp4').stat().st_size
                   for g in metadata['groups'] for c in channels) / frames / 1000
    baseline = json.loads((measurements / 'uniform_qp0/metrics.json').read_text())
    base = float(baseline['psnr'])
    if not math.isfinite(base) or base <= 0:
        raise ValueError('Baseline PSNR must be finite and positive.')
    anchor = metadata['groups'][0]['frames'][0]
    def view_ids(metrics):
        views = metrics['views']
        ids = [(v['frame'], v['camera']) for v in views]
        if not ids or len(set(ids)) != len(ids) or any(f != anchor for f, _ in ids):
            raise ValueError('Calibration must measure unique views of the initial keyframe.')
        return set(ids)
    baseline_views = view_ids(baseline)
    profile = {'format_version': 1, 'base_psnr': base,
               'size_unit': 'decimal KB per frame (video payload)',
               'rate_frame_count': frames, 'qps': list(qps),
               'quality_model': 'Q = base_psnr * product(quality_factor)',
               'fixed_kb': size(HIGH, 0), 'attributes': {}}
    for attr, channels in ATTRIBUTES.items():
        profile['attributes'][attr] = []
        for qp in qps:
            metrics = json.loads((measurements / f'{attr}_qp{qp}/metrics.json').read_text())
            if (view_ids(metrics) != baseline_views or
                    any(metrics[k] != baseline[k] for k in ['split', 'resolution'])):
                raise ValueError(f'Inconsistent evaluation protocol for {attr} QP={qp}')
            psnr, ssim = float(metrics['psnr']), float(metrics['ssim'])
            if not math.isfinite(psnr) or psnr <= 0 or not math.isfinite(ssim):
                raise ValueError(f'Invalid quality measurement for {attr} QP={qp}')
            if qp == 0 and (psnr != base or ssim != baseline['ssim']):
                raise ValueError(f'QP=0 anchor must equal the shared baseline: {attr}')
            profile['attributes'][attr].append(
                {'qp': qp, 'size_kb': size(channels, qp), 'psnr': psnr,
                 'ssim': ssim, 'quality_factor': psnr / base})
    write_json(output, profile)
    return profile
