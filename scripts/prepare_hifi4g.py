"""Arrange already-undistorted HiFi4G images and COLMAP calibration for PrismGS."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Folder containing numeric frame folders of camera images')
    parser.add_argument('--calibration', type=Path, required=True, help='COLMAP model directory containing cameras/images and points3D')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, required=True, help='Exclusive')
    args = parser.parse_args()
    images, calibration, output = (p.resolve() for p in (args.images,args.calibration,args.output))
    if args.start < 0 or args.end <= args.start:
        parser.error('Require 0 <= start < end')
    if output == images or images in output.parents or output == calibration or calibration in output.parents:
        parser.error('Output must be separate from input directories')
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be empty; existing data will not be overwritten')
    spec = importlib.util.spec_from_file_location('colmap_loader', ROOT/'training/prismgs_backend/scene/colmap_loader.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    binary = (calibration/'cameras.bin').is_file() and (calibration/'images.bin').is_file()
    if binary:
        cameras = loader.read_intrinsics_binary(str(calibration/'cameras.bin'))
        views = loader.read_extrinsics_binary(str(calibration/'images.bin'))
    else:
        cameras = loader.read_intrinsics_text(str(calibration/'cameras.txt'))
        views = loader.read_extrinsics_text(str(calibration/'images.txt'))
    if not views or any(c.model not in ('PINHOLE','SIMPLE_PINHOLE') for c in cameras.values()):
        parser.error('Supply an undistorted PINHOLE/SIMPLE_PINHOLE COLMAP model')
    if not any((calibration/name).is_file() for name in ['points3D.ply','points3D.bin','points3D.txt']):
        parser.error('Initialization point cloud missing from calibration model')
    names = [Path(v.name).name for v in views.values()]
    if len(names) != len(set(names)):
        parser.error('COLMAP image basenames must be unique')
    plan = []
    for frame in range(args.start,args.end):
        folder = images/str(frame)
        if (folder/'images').is_dir():
            folder = folder/'images'
        for name in names:
            source = folder/name
            if not source.is_file():
                parser.error(f'Missing calibrated image: {source}')
            plan.append((source,output/str(frame)/'images'/name))
    for source,target in plan:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for frame in range(args.start,args.end):
        target = output/str(frame)/'sparse/0'
        target.mkdir(parents=True,exist_ok=True)
        for name in ['cameras.bin','images.bin','points3D.bin','cameras.txt','images.txt','points3D.txt','points3D.ply']:
            if (calibration/name).is_file():
                shutil.copy2(calibration/name,target/name)
    report = {'start':args.start,'end':args.end,'views_per_frame':len(views),
              'images_copied':len(plan),'calibration':'Shared fixed-camera undistorted COLMAP model',
              'note':'No undistortion, resizing, masking or point reconstruction was performed.'}
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
