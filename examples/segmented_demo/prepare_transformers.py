"""Check/apply the upstream OpenPI Transformers replacement in a dedicated venv."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if importlib.metadata.version('transformers') != '4.53.2':
        raise RuntimeError('This replacement requires Transformers 4.53.2')
    target = Path(importlib.util.find_spec('transformers').origin).parent.resolve()
    source = Path(__file__).resolve().parents[2] / 'src/openpi/models_pytorch/transformers_replace'
    if args.apply and (sys.prefix == sys.base_prefix or not target.is_relative_to(Path(sys.prefix).resolve())):
        raise RuntimeError('Apply only in a dedicated venv containing its own Transformers installation')
    results = {}
    for path in sorted(source.rglob('*.py')):
        relative = path.relative_to(source)
        destination = target / relative
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest() if destination.exists() else None
        results[relative.as_posix()] = dict(expected=expected, actual=actual, matches=expected == actual)
    passed = bool(results) and all(row['matches'] for row in results.values())
    print(json.dumps(dict(passed=passed, applied=args.apply, files=results), indent=2))
    if not passed:
        raise SystemExit('Replacement differs; prepare an isolated venv with --apply')


if __name__ == '__main__':
    main()
