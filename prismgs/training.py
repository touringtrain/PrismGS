"""Launch bundled keyframe training, pruning, and residual training safely."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1] / 'training' / 'prismgs_backend'


def train(args):
    dataset, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    if args.start >= args.end or args.group_size < 1:
        raise ValueError('Require start < end and group_size > 0')
    for frame in range(args.start, args.end):
        if not (dataset / str(frame)).is_dir():
            raise FileNotFoundError(dataset / str(frame))
    if min(args.keyframe_iterations, args.motion_iterations, args.finetune_iterations, args.resolution) < 1 or args.prune_iterations < 0:
        raise ValueError('Training iteration counts/resolution must be positive; pruning may be zero.')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.cuda)}
    def run(script, *options):
        subprocess.run([sys.executable, str(ROOT/script), *map(str, options)], cwd=ROOT, env=env, check=True)
    for start in range(args.start, args.end, args.group_size):
        stop = min(start+args.group_size, args.end)
        model = output/'checkpoint'/str(start)
        common = ['-s', dataset/str(start), '-m', model, '--sh_degree', 0,
                  '-r', args.resolution]
        common += ['--black_background'] if args.black_background else ['--white_background']
        if getattr(args, 'split', 'train') == 'test':
            common += ['--eval']
        run('train.py', *common, '--iterations', args.keyframe_iterations,
            '--save_iterations', args.keyframe_iterations,
            '--scale_reg_weight', args.scale_reg_weight,
            '--anisotropy_reg_weight', args.anisotropy_reg_weight,
            '--max_anisotropy', args.max_anisotropy)
        if args.prune_iterations > 0:
            run('prune_gaussian.py', *common, '--iterations', args.prune_iterations)
        if stop > start+1:
            dynamic = ['-s', dataset, '-m', output/'checkpoint', '--sh_degree', 0,
                       '-r', args.resolution, '--st', start, '--ed', stop-1, '--interval', 1,
                       '--motion_iterations', args.motion_iterations, '--finetune_iterations', args.finetune_iterations]
            dynamic += ['--black_background'] if args.black_background else ['--white_background']
            if getattr(args, 'split', 'train') == 'test':
                dynamic += ['--eval']
            run('train_dynamic.py', *dynamic)
