"""Download the pinned HumanGen release, including WAM latents and statistics."""
import argparse
import json
import sys
from pathlib import Path

def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models/act'))
    from act_robotwin.protocol import DATASET, REVISION, training_rows
    snapshot_download(DATASET, repo_type='dataset', revision=REVISION, local_dir=a.output)
    rows = training_rows(a.output)
    assert (a.output / 'robotwin_data/meta/action_stats.json').is_file()
    (a.output / 'download_receipt.json').write_text(json.dumps(dict(repo=DATASET, revision=REVISION, train_pairs=len(rows), download='complete pinned HF snapshot; train/test membership remains in original manifests'), indent=2), encoding='utf-8')
    print('Downloaded pinned release; training selects only train.jsonl')

if __name__ == '__main__':
    main()
