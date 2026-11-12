"""Native RoboTwin/IG recordings; human conditioning remains RGB-only."""
from pathlib import Path

import numpy as np

from data import Dataset, read, require, sha


class TransferRecords(Dataset):
    def __init__(self, data, *, frames, sample_hz=5., split='train', seed=0, cache=None):
        self.data, self.frames, self.sample_hz = Path(data), frames, sample_hz
        self.manifest = read(self.data/'manifest.json')
        require(self.manifest['schema'] == 'segmented_transfer_v1' and split == 'train', 'Unknown transfer split')
        self.stats = read(self.data/'normalization.json')
        self.views = self.manifest['demo_views']
        self.cache = None if cache is None else Path(cache)
        self._load_cache(frames, sample_hz)
        self.rows = self.manifest['rows']
        by_query = {r['id']: r for r in self.rows}
        by_demo = {r['id']: r for r in self.manifest['demos']}
        path = self.data/self.manifest['pairing_file']
        require(sha(path) == self.manifest['pairing_sha256'], 'Pairing manifest changed')
        self.pairs = [(by_demo[d], by_query[q]) for d, q in read(path)['pairs']]
        self.pair_ids = {(d['id'], q['id']) for d, q in self.pairs}
        require(len(self.pair_ids) == len(self.pairs) > 0, 'Duplicate or empty transfer pairs')
        require(all(d['task'] == q['task'] for d, q in self.pairs), 'Cross-task Demo pair')
        require({q['id'] for _, q in self.pairs} == set(by_query), 'Unpaired Query')
        for row in [*self.rows, *by_demo.values()]:
            folder = self.data/'records'/row['id']
            require(sha(folder/'supervision.npz') == row['labels_sha256'], 'Recorded labels changed')
            stat = (folder/'rgb.npy').stat()
            require(stat.st_size == row['rgb_bytes'] and stat.st_mtime_ns == row['rgb_mtime_ns'], 'Verified RGB changed')

    def window(self, query, index, demo):
        require((demo['id'], query['id']) in self.pair_ids, 'Pair not allowed')
        require(0 <= index < query['frames'], 'Invalid Query index')
        labels = self.labels(query['id'])
        state = self.normalize(labels['state'][index], 'state')
        steps = np.arange(index, index+50)
        available = steps < query['frames']
        actions = self.normalize(labels['action'][np.minimum(steps, query['frames']-1)], 'action')
        state = np.pad(state, (0, 32-state.shape[-1]))
        actions = np.pad(actions, ((0, 0), (0, 32-actions.shape[-1])))
        valid = np.zeros((50, 32), dtype=bool)
        valid[available, :16] = True
        payload, demo_valid, times = self.demo(demo['id'], demo['seconds'])
        return dict(image=np.array(self.rgb(query['id'])[index], copy=True), state=state,
                    actions=actions, action_valid=valid, demo_valid=demo_valid, demo_timestamps=times,
                    **({'demo_rgb': payload} if self.cache is None else {'demo_patches': payload}), demo_id=demo['id'])

    def normalize(self, value, kind):
        s = self.stats[kind]
        return ((value-np.asarray(s['mean'], np.float32))/np.asarray(s['std'], np.float32)).astype(np.float32)

    def decode_actions(self, value):
        """Unpad/denormalize; preserve each benchmark's native control contract."""
        s = self.stats['action']
        result = np.asarray(value, np.float32)[..., :16]*np.asarray(s['std'], np.float32)+np.asarray(s['mean'], np.float32)
        require(np.isfinite(result).all(), 'Nonfinite predicted control target')
        if self.manifest['dataset'] == 'robotwin':
            for start in (3, 11):
                quaternion = result[..., start:start+4]
                norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
                require((norm > 1e-6).all(), 'Degenerate predicted EEF quaternion')
                result[..., start:start+4] = quaternion/norm
        return result
