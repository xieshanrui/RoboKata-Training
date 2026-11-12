"""Fit train-only action statistics and select the C15 human descriptions."""
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'statistics', 'selection'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.statistics.exists() or args.selection.exists():
        raise FileExistsError('Preserve existing statistics/selection; use new output paths')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    assert len(manifest['sim_episodes']) == 3000 and len(manifest['human_episodes']) == 750
    human_ids = sorted({r['human_repo_id'] for r in manifest['sim_episodes']})
    assert len(human_ids) == 15
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environments/imitator_game'))
    from examples.baselines.zero_wam.zero_wam.action_space import fit_statistics
    fit_statistics(manifest, 'relative_episode', args.statistics)
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(json.dumps({'human_ids': human_ids}, indent=2), encoding='utf-8')
    print('Prepared C15 training statistics and 15 human text identities')

if __name__ == '__main__':
    main()
