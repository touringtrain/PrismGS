"""Print build commands for bundled sources; no downloads or system changes."""
from pathlib import Path
import shlex
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    for name in ['simple-knn', 'diff-gaussian-rasterization']:
        path = root / 'submodules' / name
        if not (path / 'setup.py').is_file():
            raise FileNotFoundError(path)
        print(shlex.join([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', str(path)]))
