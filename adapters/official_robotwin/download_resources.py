"""Download pinned official weights and seven HumanGen latents; no evaluation."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--human', type=Path, required=True)
    p.add_argument('--asset-archives', type=Path, help='Optional official asset ZIP download directory')
    a = p.parse_args()
    from huggingface_hub import snapshot_download
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from evaluation.robotwin.robotwin_icl_human_videos import ROBOTWIN_ICL_HUMAN_VIDEOS
    provenance = json.loads(Path(__file__).with_name('provenance.json').read_text(encoding='utf-8'))
    snapshot_download('robbyant-research/zero-wam-posttrain-robotwin', revision=provenance['model_revision'], local_dir=a.model)
    patterns = []
    for task in provenance['protocol']['tasks']:
        sample = Path(ROBOTWIN_ICL_HUMAN_VIDEOS[task][0]).parent.name
        patterns.append('human_latents/robotwin/run_robotwin_20260728_013720_robotwin_n1000/samples/' + sample + '/generated_video_kling-v3.pth')
    snapshot_download('Robbyant-Research/HumanGen', repo_type='dataset', revision=provenance['human_revision'], local_dir=a.human, allow_patterns=patterns + ['README.md', 'LICENSE*'])
    for relative in patterns:
        if not (a.human / relative).is_file():
            raise FileNotFoundError(a.human / relative)
    if a.asset_archives:
        snapshot_download('TianxingChen/RoboTwin2.0', repo_type='dataset', revision=provenance['asset_revision'], local_dir=a.asset_archives, allow_patterns=['embodiments.zip', 'objects.zip', 'background_texture.zip', 'README.md', 'LICENSE*'])


if __name__ == '__main__':
    main()
