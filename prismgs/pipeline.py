"""Prepared calibrated data to measured streaming results in one invocation."""
import json
import os
from pathlib import Path
import time
from .codec import ATTRIBUTES, pack, encode, decode, write_json
from .quality import collect_profile, frontier, validate_qps
from .rendering import evaluate
from .streaming import evaluate_trace
from .training import train


def calibrate(videos, dataset, output, qps, workers=4, resolution=2,
              cameras=None, white_background=True, split='train', decode_device='cpu'):
    validate_qps(qps)
    if workers < 1:
        raise ValueError('workers must be positive')
    output = Path(output)
    metadata = json.loads((Path(videos)/'metadata.json').read_text())
    if not metadata['groups'] or any(not g['frames'] for g in metadata['groups']):
        raise ValueError('Calibration requires a nonempty encoded sequence.')
    frames = [metadata['groups'][0]['frames'][0]]
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh calibration output directory to avoid mixing measurements.')
    protocol = {'frames': frames, 'split': split, 'cameras': cameras,
                'resolution': resolution, 'white_background': white_background}
    write_json(output/'calibration_config.json', {
        'videos': str(Path(videos).resolve()), 'dataset': str(Path(dataset).resolve()),
        'qps': list(qps), 'workers': workers, 'decode_device': decode_device, 'evaluation': protocol,
        'protocol': 'initial-keyframe single-attribute sweep; all other QPs fixed at 0',
        'unique_configurations': 1 + len(ATTRIBUTES) * (len(qps) - 1)})
    config = {a: 0 for a in ATTRIBUTES}
    folder = output/'uniform_qp0'
    write_json(folder/'config.json', config)
    decode(videos, folder/'checkpoint', config, workers, device=decode_device)
    base = evaluate(folder/'checkpoint', dataset, folder, frames, cameras, resolution, white_background, split)
    for attr in ATTRIBUTES:
        for qp in qps:
            folder = output/f'{attr}_qp{qp}'
            config = {a: (qp if a == attr else 0) for a in ATTRIBUTES}
            write_json(folder/'config.json', config)
            if qp == 0:
                write_json(folder/'metrics.json', base)
                continue
            print(f'Calibrating {attr} QP={qp}', flush=True)
            decode(videos, folder/'checkpoint', config, workers, device=decode_device)
            evaluate(folder/'checkpoint', dataset, folder, frames, cameras, resolution, white_background, split)
    profile = collect_profile(videos, output, qps, output/'profile.json')
    profile['evaluation'] = protocol
    write_json(output/'profile.json', profile)
    return profile


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    if args.start >= args.end or args.group_size < 1 or args.workers < 1:
        raise ValueError('Require start < end, positive group-size and workers.')
    if any(q < 0 or q > 51 for q in args.qps) or args.fps <= 0:
        raise ValueError('Require QPs in [0, 51] and positive FPS.')
    if not Path(args.trace).is_file():
        raise FileNotFoundError(args.trace)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory to avoid mixing runs.')
    output.mkdir(parents=True, exist_ok=True)
    if 0 not in args.qps or len(set(args.qps)) != len(args.qps):
        raise ValueError('QP list must contain 0 and have no duplicates.')
    stages = []
    def stage(name, operation):
        print(f'=== {name} ===', flush=True)
        start = time.monotonic()
        value = operation()
        stages.append({'stage': name, 'seconds': time.monotonic() - start})
        write_json(output/'stages.json', stages)
        return value
    options = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    write_json(output/'run_config.json', options)
    checkpoints = Path(args.checkpoints).resolve() if args.checkpoints else output/'checkpoint'
    if not args.checkpoints:
        stage('training', lambda: train(args))
    stage('pack', lambda: pack(checkpoints, output/'feature_image', args.start, args.end, args.group_size))
    stage('encode', lambda: encode(output/'feature_image', output/'feature_video', args.qps, args.fps))
    profile = stage('calibration', lambda: calibrate(output/'feature_video', args.dataset,
        output/'calibration', args.qps, args.workers, args.resolution, args.cameras,
        not args.black_background, args.split, getattr(args, 'decode_device', 'cpu')))
    predicted = stage('frontier', lambda: frontier(profile, args.step_kb, args.max_kb))
    write_json(output/'frontier.json', predicted)
    if not predicted['points']:
        raise ValueError('No feasible frontier point; increase max-kb or reduce step-kb.')
    stage('decode_render', lambda: measure_configurations(args, predicted))
    for mobility in ['walking','driving']:
        stage('streaming_'+mobility, lambda m=mobility: evaluate_trace(args.trace,
              output/'measured_rd.json', output/f'streaming_{m}.json', m, args.fps))
    print(f'Complete: {output}', flush=True)


def measure_configurations(args, predicted):
    """Measure the initial anchor frame; rate covers the complete encoded sequence.

    Persist each frame separately to permit an interrupted measurement to resume.
    Call only with the run's original configuration and unchanged checkpoints.
    """
    output = Path(args.output).resolve()
    configs = [('uniform_qp'+str(q), {a:q for a in ATTRIBUTES}) for q in args.qps]
    configs += [('pareto_'+str(i), p['config']) for i,p in enumerate(predicted['points'])]
    metadata = json.loads((output/'feature_video/metadata.json').read_text())
    total_frames = sum(len(g['frames']) for g in metadata['groups'])
    from .codec import HIGH, latest_ply
    for name, config in configs:
        folder = output/'measured'/name
        try:
            for frame in range(args.start, args.end):
                latest_ply(folder/'checkpoint', frame)
        except FileNotFoundError:
            print(f'Decoding {name}', flush=True)
            decode(output/'feature_video', folder/'checkpoint', config, args.workers,
                   device=getattr(args, 'decode_device', 'cpu'))
    for frame in [args.start]:
        print(f'Measuring frame {frame}: {len(configs)} configurations', flush=True)
        for name, config in configs:
            folder = output/'measured'/name
            frame_output = folder/'frames'/str(frame)
            if not (frame_output/'metrics.json').exists():
                evaluate(folder/'checkpoint', args.dataset, frame_output, [frame],
                    args.cameras, args.resolution, not args.black_background, args.split)
        print(f'Measured frame {frame} complete', flush=True)
    points = []
    for name, config in configs:
        folder = output/'measured'/name
        records = []
        for frame in [args.start]:
            records.extend(json.loads((folder/'frames'/str(frame)/'metrics.json').read_text())['views'])
        metrics = {k: sum(r[k] for r in records)/len(records) for k in ['psnr','ssim','render_ms']}
        metrics.update({'split':args.split, 'resolution':args.resolution, 'views':records})
        write_json(folder/'metrics.json', metrics)
        (folder/'eval.txt').write_text(f"PSNR: {metrics['psnr']:.6f} dB\nSSIM: {metrics['ssim']:.6f}\n")
        mapping = {c:config[a] for a,channels in ATTRIBUTES.items() for c in channels}
        mapping.update({c:config['position_low'] for c in HIGH})
        size = sum((output/'feature_video'/f'png_all_{q}'/f"group{g['index']}"/f'{c}.mp4').stat().st_size
                   for g in metadata['groups'] for c,q in mapping.items()) / total_frames / 1000
        points.append({'name':name, 'config':config, 'size_kb':size,
                       'psnr':metrics['psnr'], 'ssim':metrics['ssim']})
    value = {'size_unit':'decimal KB/frame, video payload only', 'points':points,
             'evaluation':{'split':args.split, 'cameras':args.cameras, 'resolution':args.resolution,
                           'frames':[args.start], 'rate_start':args.start, 'rate_end':args.end,
                           'protocol':'initial-frame anchor quality; full-sequence average payload'}}
    write_json(output/'measured_rd.json',value)
    return value
