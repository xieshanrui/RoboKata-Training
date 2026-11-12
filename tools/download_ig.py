"""Download and verify the frozen C15 files; does not build GPU caches."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(path, algorithm, prefix=b''):
    value = hashlib.new(algorithm)
    value.update(prefix)
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(part)
    return value.hexdigest()

def main():
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/ig_c15_download_manifest.json').read_text(encoding='utf-8'))
    for i, item in enumerate(manifest['files'], 1):
        target = (root / item['path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Unsafe dataset path')
        hf_hub_download(manifest['repo'], item['path'], repo_type='dataset', revision=manifest['revision'], local_dir=root)
        assert target.stat().st_size == item['size'], item['path']
        if item.get('lfs'):
            assert digest(target, 'sha256') == item['lfs']['oid'], item['path']
        else:
            prefix = ('blob ' + str(item['size']) + '\0').encode()
            assert digest(target, 'sha1', prefix) == item['oid'], item['path']
        print(f'{i}/{len(manifest["files"])} verified {item["path"]}', flush=True)
    (root / 'c15_download_receipt.json').write_text(json.dumps(dict(repo=manifest['repo'], revision=manifest['revision'], files=len(manifest['files']), verified=True), indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
