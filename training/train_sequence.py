"""Compatibility entry: translate old sequence flags to the maintained CLI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismgs.__main__ import main

if __name__ == '__main__':
    aliases = {'--data':'--dataset', '--gtp-iterations':'--motion-iterations',
               '--sh-degree':None}
    arguments = []
    values = iter(sys.argv[1:])
    for value in values:
        if value == '--sh-degree':
            if next(values) != '0':
                raise SystemExit('Only SH degree 0 is supported.')
        else:
            arguments.append(aliases.get(value, value))
    if '--dry-run' in arguments:
        raise SystemExit('Use python -m prismgs train --help; --dry-run is no longer supported. No training was started.')
    sys.argv = [sys.argv[0], 'train', *arguments]
    main()
