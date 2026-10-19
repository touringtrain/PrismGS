"""Export maintained runtime source, without scene data or generated binaries."""
import argparse
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'THIRD_PARTY_NOTICES.md', 'SOURCE_MANIFEST.md',
         'requirements.txt', 'requirements-preprocess.txt', 'pyproject.toml', '.gitignore',
         'docs/calibration.md', 'docs/decoding.md', 'docs/gpu_decoding.md',
         'docs/data_preparation.md', 'docs/pipeline.md', 'docs/verification.md',
         'datasets/README.md']
DIRECTORIES = ['prismgs', 'training', 'submodules', 'scripts', 'licenses', 'third_party/videogs']
EXCLUDED = {'.git', '__pycache__', 'build', '.pytest_cache', 'local_reference'}


def source_files():
    files = [ROOT/name for name in FILES]
    for directory in DIRECTORIES:
        for path in sorted((ROOT/directory).rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED or part.endswith('.egg-info') for part in relative.parts):
                continue
            if (path.name == '.gitkeep' or path.suffix.lower() in {'.py', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.inl', '.md', '.txt', '.json', '.sh', '.cmake', '.in', '.yml', '.yaml'}) and path.name != 'verified_environment.json':
                files.append(path)
    files += [ROOT/'tests/test_pipeline.py', ROOT/'tests/test_streaming.py',
              ROOT/'tests/test_calibration.py', ROOT/'tests/test_gpu_decode.py']
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = source_files()
    for path in files:
        if not path.is_file():
            parser.error(f'Missing manifest entry: {path}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=Path('PrismGS')/path.relative_to(ROOT), recursive=False)
    print(f'Exported {len(files)} files to {args.output}')


if __name__ == '__main__':
    main()
