"""Independent-start route pairing and timed RGB-only Demo sampling."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

PROMPT = 'Follow the demonstration and complete the demonstrated task.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dense_demo_indices(recorded_times, duration, sample_hz):
    """Targets in [0,duration); select the latest actual frame at each target."""
    times = np.asarray(recorded_times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('Demo requires a finite nonempty recorded time axis')
    if times[0] != 0 or np.any(np.diff(times) <= 0) or duration <= times[-1] or sample_hz <= 0:
        raise ValueError('Invalid Demo duration, sampling rate or recorded clock')
    targets = np.arange(int(np.ceil(duration * sample_hz)), dtype=np.float64) / sample_hz
    targets = targets[targets < duration - 1e-8]
    # Resolve floating-point representations of the same recorded instant.
    indices = np.searchsorted(times, targets + 1e-8, side='right') - 1
    if (indices < 0).any() or len(np.unique(indices)) != len(indices):
        raise ValueError('Dense sampling requires distinct available recorded frames')
    return indices, targets


def normalize(value, stats):
    value = np.asarray(value, dtype=np.float32).copy()
    value[..., :3] = 2 * (value[..., :3] - np.asarray(stats['low_xyz'])) / (
        np.asarray(stats['high_xyz']) - np.asarray(stats['low_xyz'])) - 1
    return value


class Dataset:
    def __init__(self, data, *, frames, sample_hz=5., split='train', seed=20261017, cache=None):
        self.data, self.frames, self.sample_hz = Path(data), frames, sample_hz
        self.manifest, self.stats = read(self.data/'manifest.json'), read(self.data/'normalization.json')
        self.views = self.manifest.get('demo_views', 2)
        self.cache = None if cache is None else Path(cache)
        self._load_cache(frames, sample_hz)
        self._load_pairs(split, seed)

    def _load_cache(self, frames, sample_hz):
        if self.cache is not None:
            self.cache_manifest = read(self.cache/'cache_manifest.json')
            require(self.cache_manifest['passed'] and self.cache_manifest['source_manifest_sha256'] == sha(self.data/'manifest.json'),
                    'Raw patch cache does not match the data manifest')
            require(self.cache_manifest['frames'] == frames and self.cache_manifest['sample_hz'] == sample_hz,
                    'Raw patch cache has a different Demo sampling contract')
            self.cache_verification = read(self.cache/'verification_receipt.json')
            require(self.cache_verification['passed'] and self.cache_verification['all_file_sha256_verified'],
                    'Raw patch cache requires a successful complete SHA256 verification receipt')
            require(self.cache_verification['cache_manifest_sha256'] == sha(self.cache/'cache_manifest.json'),
                    'Cache verification receipt refers to a different manifest')
            require(self.cache_verification['fingerprint'] == self.cache_manifest['fingerprint'],
                    'Cache verification fingerprint differs')
            require(self.cache_verification['records'].keys() == self.cache_manifest['records'].keys(),
                    'Cache verification record inventory differs')
    def _load_pairs(self, split, seed):
        self.rows = [row for row in self.manifest['rows'] if row['group'] == split]
        self.routes = sorted({row['route'] for row in self.rows})
        self.by_route = {route: [r for r in self.rows if r['route'] == route] for route in self.routes}
        self.pairs = None
        if self.manifest.get('pairing_file'):
            path = self.data / self.manifest['pairing_file']
            require(sha(path) == self.manifest['pairing_sha256'], 'Frozen pairing manifest differs')
            by_id = {r['id']: r for r in self.rows}
            self.pairs = []
            for pair in read(path)['pairs']:
                if pair['demo'] not in by_id or pair['query'] not in by_id:
                    continue
                demo, query = by_id[pair['demo']], by_id[pair['query']]
                require(demo['route'] == query['route'] == pair['task_id'], 'Pair task differs')
                require(pair['self_pair'] == (demo['id'] == query['id']), 'Self-pair flag differs')
                require(demo['id'] == query['id'] or demo['state'] != query['state'], 'Cross-record pair shares a physical family')
                self.pairs.append((demo, query))
            self.pair_ids = {(d['id'], q['id']) for d, q in self.pairs}
            require(self.pairs and len(self.pair_ids) == len(self.pairs), 'Empty or duplicate pair manifest')
        self.rng = np.random.default_rng(seed)
        if not self.rows or any(len({r['state'] for r in group}) < 2 for group in self.by_route.values()):
            raise ValueError('Every split/route needs independently recorded physical starts')

    @lru_cache(maxsize=64)
    def labels(self, identity):
        with np.load(self.data/'records'/identity/'supervision.npz', allow_pickle=False) as item:
            return {key: item[key] for key in item.files}

    @lru_cache(maxsize=64)
    def rgb(self, identity):
        return np.load(self.data/'records'/identity/'rgb.npy', mmap_mode='r')

    @lru_cache(maxsize=128)
    def demo(self, identity, duration):
        labels = self.labels(identity)
        indices, times = dense_demo_indices(labels['timestamp_s'], duration, self.sample_hz)
        times = labels['timestamp_s'][indices]
        if len(times) > self.frames:
            raise ValueError(f'Demo {identity} needs {len(times)} frames; cap {self.frames} cannot truncate it')
        if self.cache is not None:
            import safetensors.torch
            entry = self.cache_manifest['records'][identity]
            require(entry['input_indices'] == indices.tolist(), 'Cached Demo input indices differ')
            require(Path(entry['file']).name == entry['file'], 'Cache entry must be a local basename')
            path = self.cache/entry['file']
            verified = self.cache_verification['records'][identity]
            stat = path.stat()
            require(stat.st_size == entry['bytes'] == verified['bytes'] and stat.st_mtime_ns == verified['mtime_ns'],
                    'Verified Demo patch cache was modified')
            saved = safetensors.torch.load_file(str(path))
            patch_count, dtype = 256, torch.bfloat16
            require(saved['patches'].dtype == dtype and saved['patches'].shape == (len(times), self.views, patch_count, 1152),
                    'Cached visual features have an incompatible shape or precision')
            patches = torch.zeros((self.frames, self.views, patch_count, 1152), dtype=dtype)
            patches[:len(times)] = saved['patches']
            valid = np.zeros((self.frames, self.views), dtype=bool)
            valid[:len(times)] = True
            timestamps = np.zeros(self.frames, dtype=np.float32)
            timestamps[:len(times)] = times
            return patches, valid, timestamps
        rgb = np.zeros((self.frames, self.views, 224, 224, 3), dtype=np.uint8)
        valid = np.zeros((self.frames, self.views), dtype=bool)
        timestamps = np.zeros(self.frames, dtype=np.float32)
        rgb[:len(times)] = self.rgb(identity)[indices]
        valid[:len(times)] = True
        timestamps[:len(times)] = times
        return rgb, valid, timestamps

    def window(self, query, index, demo):
        require(query['route'] == demo['route'] and query['group'] == demo['group'], 'Demo/Query task or split differs')
        require((demo['id'], query['id']) in self.pair_ids if self.pairs is not None else query['state'] != demo['state'],
                'Demo/Query pair is not allowed by its frozen manifest')
        require(0 <= index < query['frames'], 'Query observation index is outside the recording')
        labels = self.labels(query['id'])
        state = normalize(labels['state250'][int(labels['observation_step'][index])], self.stats)
        state = np.pad(state, (0, 22))
        steps = np.arange(index, index + 50)
        available = steps < query['frames']
        actions = normalize(labels['action'][np.minimum(steps, query['frames'] - 1)], self.stats)
        actions = np.pad(actions, ((0, 0), (0, 22)))
        action_valid = np.zeros((50, 32), dtype=bool)
        action_valid[available, :10] = True
        payload, demo_valid, times = self.demo(demo['id'], demo['seconds'])
        return dict(image=np.array(self.rgb(query['id'])[index], copy=True), state=state,
                    actions=actions, action_valid=action_valid,
                    **({'demo_rgb': payload} if self.cache is None else {'demo_patches': payload}),
                    demo_valid=demo_valid, demo_timestamps=times, demo_id=demo['id'])


def collate(samples, device, tokens, token_mask, image_keys=('base_0_rgb', 'left_wrist_0_rgb')):
    from openpi.models.model import Observation
    batch = len(samples)
    def tensor(key):
        return torch.from_numpy(np.stack([sample[key] for sample in samples])).to(device)
    images = tensor('image').float().permute(0, 1, 4, 2, 3) / 127.5 - 1
    cached = 'demo_patches' in samples[0]
    require(all(('demo_patches' in sample) == cached for sample in samples), 'Mixed raw/cached Demo batch')
    demo = None if cached else tensor('demo_rgb').float().permute(0, 1, 2, 5, 3, 4) / 127.5 - 1
    patches = torch.stack([sample['demo_patches'] for sample in samples]).to(device) if cached else None
    keys = tuple(image_keys)
    require(images.shape[1] == len(keys), 'Query RGB count differs from ordered camera contract')
    observation = Observation(
        images={key: images[:, i] for i, key in enumerate(keys)},
        image_masks={key: torch.ones(batch, dtype=torch.bool, device=device) for key in keys},
        state=tensor('state'), tokenized_prompt=tokens[None].expand(batch, -1),
        tokenized_prompt_mask=token_mask[None].expand(batch, -1),
        source_id=torch.zeros(batch, dtype=torch.long, device=device),
        demo_rgb=demo, demo_patches=patches, demo_valid=tensor('demo_valid'), demo_timestamps=tensor('demo_timestamps'))
    return observation, tensor('actions').float(), tensor('action_valid')


def prepare_data(data):
    data = Path(data)
    manifest = read(data/'manifest.json')
    count, maximum = 0, 0
    for row in manifest['rows']:
        folder = data/'records'/row['id']
        require(sha(folder/'policy_recording.npz') == row['recording_sha256'], 'Recording SHA256 differs')
        require(sha(folder/'supervision.npz') == row['labels_sha256'], 'Supervision SHA256 differs')
        with np.load(folder/'policy_recording.npz', allow_pickle=False) as item:
            require(item['rgb'].shape == (row['frames'], 2, 224, 224, 3), 'Recording RGB shape differs')
            require(np.array_equal(item['timestamp_s'], np.arange(row['frames']) * .02), 'Recording clock differs')
            indices, targets = dense_demo_indices(item['timestamp_s'], row['seconds'], 5.)
            require(np.all(targets - item['timestamp_s'][indices] < .020001) and
                    np.all(targets - item['timestamp_s'][indices] >= -1e-8), 'Demo sampling timing differs')
            maximum = max(maximum, len(targets))
            target = folder/'rgb.npy'
            if not target.exists():
                np.save(target, item['rgb'], allow_pickle=False)
            require(np.array_equal(np.load(target, mmap_mode='r'), item['rgb']), 'RGB mmap differs from recording')
        count += 1
        if count % 100 == 0:
            print(json.dumps(dict(event='records_verified', records=count, total=len(manifest['rows']))), flush=True)
    result = dict(passed=True, records=count, maximum_demo_samples_5hz=maximum,
                  manifest_sha256=sha(data/'manifest.json'), normalization_sha256=sha(data/'normalization.json'),
                  original_record_and_label_sha_verified=True, rgb_memmaps_equal_canonical=True)
    write(data/'preparation_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-data', type=Path, required=True)
    prepare_data(parser.parse_args().prepare_data)
