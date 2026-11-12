"""Check pinned source availability and public configuration; never use GPUs."""
import ast
import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('act', 'osvi', 'wam', 'openpi'),
                        help='Also check installed package versions in the selected model environment')
    args = parser.parse_args()
    lock = json.loads((root / 'sources.lock.json').read_text(encoding='utf-8'))
    for item in lock['components']:
        path = root / item['path']
        if not (path / '.git').exists():
            raise RuntimeError(f'Missing {item["path"]}: run git submodule update --init')
        sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, encoding='utf-8').strip()
        assert sha == item['commit'], f'Changed pin: {item["path"]}'
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True, encoding='utf-8').strip()
        if dirty:
            raise RuntimeError(f'Tracked source changed in {item["path"]}; commit/review it and update the source lock before training')
        for entry in item.get('entrypoints', []):
            assert (path / entry).is_file(), (item['path'], entry)
            if entry.endswith('.py'):
                ast.parse((path / entry).read_text(encoding='utf-8-sig'), filename=entry)
    for path in (root / 'configs').glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
    for path in (root / 'tools').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    print(f'PASS: {len(lock["components"])} pinned components, entrypoints, JSON and helper syntax')
    if args.profile:
        requirements = (root / 'models/openpi/examples/segmented_demo/requirements-linux-py312.txt'
                        if args.profile == 'openpi' else root / f'environments/{args.profile}-py312.txt')
        failures = []
        count = 0
        for line in requirements.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            name, expected = line.strip().split('==', 1)
            count += 1
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                failures.append(f'{name}: missing (expected {expected})')
                continue
            # PEP 440: an unqualified ==2.x also accepts the matching +cuXXX wheel.
            comparable = actual if '+' in expected else actual.split('+', 1)[0]
            if comparable != expected:
                failures.append(f'{name}: {actual} (expected {expected})')
        if failures:
            raise SystemExit('Environment mismatch:\n' + '\n'.join(failures))
        print(f'PASS: {args.profile} environment, {count} package version constraints')
    print('This checks source/config and requested package versions; data, GPU forward/backward and convergence are not validated.')

if __name__ == '__main__':
    main()
