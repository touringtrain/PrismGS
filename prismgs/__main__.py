"""Run with python -m prismgs; all user paths are relative to the caller's cwd."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='prismgs', description='PrismGS training and attribute-level compression')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train', help='Train regularized keyframes and residual frames')
    p.add_argument('--dataset', required=True); p.add_argument('--output', required=True)
    p.add_argument('--start', type=int, required=True); p.add_argument('--end', type=int, required=True)
    p.add_argument('--group-size', type=int, default=200); p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--keyframe-iterations', type=int, default=12000)
    p.add_argument('--prune-iterations', type=int, default=4000)
    p.add_argument('--motion-iterations', type=int, default=800)
    p.add_argument('--finetune-iterations', type=int, default=2000)
    p.add_argument('--scale-reg-weight', type=float, default=0.001)
    p.add_argument('--anisotropy-reg-weight', type=float, default=0.001)
    p.add_argument('--max-anisotropy', type=float, default=5)
    p.add_argument('--black-background', action='store_true')
    p = commands.add_parser('pack', help='Pack SH0 checkpoints into attribute image planes')
    p.add_argument('--checkpoints', required=True); p.add_argument('--output', required=True)
    p.add_argument('--start', type=int, required=True); p.add_argument('--end', type=int, required=True)
    p.add_argument('--group-size', type=int, default=20)
    p = commands.add_parser('encode', help='Encode attribute videos at specified QPs')
    p.add_argument('--images', required=True); p.add_argument('--output', required=True)
    p.add_argument('--qps', nargs='+', type=int, default=[0,10,20,30,40,50]); p.add_argument('--fps', type=float, default=30)
    p = commands.add_parser('decode', help='Decode a uniform QP or attribute configuration')
    p.add_argument('--videos', required=True); p.add_argument('--output', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--qp', type=int); g.add_argument('--config', help='JSON file with the five attribute QPs')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cpu', help='Gaussian reconstruction device: cpu, cuda or cuda:N')
    p = commands.add_parser('render', help='Measure PSNR/SSIM with the bundled CUDA renderer')
    p.add_argument('--checkpoints', required=True); p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True); p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True); p.add_argument('--cameras', nargs='+', type=int)
    p.add_argument('--resolution', type=int, default=2); p.add_argument('--split', choices=['train','test'], default='train')
    p.add_argument('--black-background', action='store_true')
    p = commands.add_parser('calibrate', help='Decode/render single-attribute anchors and collect measurements')
    p.add_argument('--videos', required=True); p.add_argument('--dataset', required=True); p.add_argument('--output', required=True)
    p.add_argument('--qps', nargs='+', type=int, default=[0,10,20,30,40,50]); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resolution', type=int, default=2); p.add_argument('--cameras', nargs='+', type=int)
    p.add_argument('--split', choices=['train','test'], default='train'); p.add_argument('--black-background', action='store_true')
    p.add_argument('--decode-device', default='cpu', help='Gaussian reconstruction device: cpu, cuda or cuda:N')
    p = commands.add_parser('frontier', help='Generate the measured quality model and Pareto frontier')
    p.add_argument('--profile', required=True); p.add_argument('--output', required=True)
    p.add_argument('--step-kb', type=float, default=1); p.add_argument('--max-kb', type=float)
    p = commands.add_parser('evaluate', help='Simulate throughput-limited rendering from measured RD points')
    p.add_argument('--trace', required=True); p.add_argument('--results', required=True); p.add_argument('--output', required=True)
    p.add_argument('--mobility', choices=['walking','driving'], default='walking'); p.add_argument('--fps', type=float, default=30)
    p = commands.add_parser('run', help='Prepared data to measured streaming results')
    p.add_argument('--dataset', required=True); p.add_argument('--output', required=True)
    p.add_argument('--start', type=int, required=True); p.add_argument('--end', type=int, required=True)
    p.add_argument('--group-size', type=int, default=20); p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--keyframe-iterations', type=int, default=12000)
    p.add_argument('--prune-iterations', type=int, default=4000)
    p.add_argument('--motion-iterations', type=int, default=800)
    p.add_argument('--finetune-iterations', type=int, default=2000)
    p.add_argument('--scale-reg-weight', type=float, default=0.001)
    p.add_argument('--anisotropy-reg-weight', type=float, default=0.001)
    p.add_argument('--max-anisotropy', type=float, default=5)
    p.add_argument('--black-background', action='store_true')
    p.add_argument('--qps', nargs='+', type=int, default=[0,10,20,30,40,50])
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--decode-device', default='cpu', help='Gaussian reconstruction device: cpu, cuda or cuda:N')
    p.add_argument('--cameras', nargs='+', type=int)
    p.add_argument('--split', choices=['train','test'], default='train')
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--step-kb', type=float, default=1)
    p.add_argument('--max-kb', type=float)
    p.add_argument('--trace', default=str(Path(__file__).resolve().parents[1]/'evaluation/Lumos5G-v1.0.csv'))
    p.add_argument('--checkpoints', help='Explicitly reuse existing checkpoints instead of training')
    args = parser.parse_args()
    if args.command == 'run':
        from .pipeline import run
        run(args)
    elif args.command == 'train':
        from .training import train
        train(args)
    elif args.command == 'pack':
        from .codec import pack
        pack(args.checkpoints,args.output,args.start,args.end,args.group_size)
    elif args.command == 'encode':
        from .codec import encode
        encode(args.images,args.output,args.qps,args.fps)
    elif args.command == 'decode':
        from .codec import decode, ATTRIBUTES
        config = json.loads(Path(args.config).read_text()) if args.config else {a: args.qp for a in ATTRIBUTES}
        decode(args.videos,args.output,config,args.workers,args.device)
    elif args.command == 'render':
        from .rendering import evaluate
        evaluate(args.checkpoints,args.dataset,args.output,range(args.start,args.end),args.cameras,
                 args.resolution,not args.black_background,args.split)
    elif args.command == 'calibrate':
        from .pipeline import calibrate
        calibrate(args.videos, args.dataset, args.output, args.qps, args.workers,
                  args.resolution, args.cameras, not args.black_background, args.split, args.decode_device)
    elif args.command == 'frontier':
        from .quality import frontier
        from .codec import write_json
        write_json(args.output,frontier(json.loads(Path(args.profile).read_text()),args.step_kb,args.max_kb))
    elif args.command == 'evaluate':
        from .streaming import evaluate_trace
        evaluate_trace(args.trace,args.results,args.output,args.mobility,args.fps)

if __name__ == '__main__':
    main()
