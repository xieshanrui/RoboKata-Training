"""Check pinned source availability and public configuration; never use GPUs."""
import ast
import json
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'sources.lock.json').read_text(encoding='utf-8'))
    for item in lock['components']:
        path = root / item['path']
        if not (path / '.git').exists():
            raise RuntimeError(f'Missing {item["path"]}: run git submodule update --init')
        sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, encoding='utf-8').strip()
        assert sha == item['commit'], f'Changed pin: {item["path"]}'
        for entry in item.get('entrypoints', []):
            assert (path / entry).is_file(), (item['path'], entry)
            if entry.endswith('.py'):
                ast.parse((path / entry).read_text(encoding='utf-8-sig'), filename=entry)
    for path in (root / 'configs').glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
    for path in (root / 'tools').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    print(f'PASS: {len(lock["components"])} pinned components, entrypoints, JSON and helper syntax')
    print('This is a source/config check, not a data, dependency, CUDA or convergence test.')

if __name__ == '__main__':
    main()
