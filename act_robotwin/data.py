"""Exact source-frame RGB plus absolute 16-D EEF targets (not joint angles)."""
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .protocol import read, video_paths, sha

@lru_cache(maxsize=6)
def decoder(path):
    from torchcodec.decoders import VideoDecoder
    return VideoDecoder(str(path), device='cpu', num_ffmpeg_threads=1)

def rgb_at(path, indices):
    d = decoder(str(path))
    frames = d.get_frames_at(indices=[int(i) for i in indices]).data.permute(0,2,3,1).numpy()
    return np.stack([cv2.resize(f, (224,224), interpolation=cv2.INTER_AREA) for f in frames])

def demo_rgb(path):
    d = decoder(str(path))
    indices = np.linspace(0, len(d)-1, min(10,len(d))).astype(np.int64)
    # Preserve native image resolution before the official DINO processor.
    return d.get_frames_at(indices=indices.tolist()).data.permute(0,2,3,1)

def episode(root, row):
    import pyarrow.parquet as pq
    cols = pq.read_table(Path(root)/row['robot_data'], columns=['frame_index','episode_index','timestamp','action','observation.state']).to_pydict()
    n = len(cols['frame_index'])
    assert cols['frame_index'] == list(range(n))
    assert set(cols['episode_index']) == {int(row['episode_id'].split('_')[-1])}
    assert np.allclose(np.diff(cols['timestamp']), .02, atol=1e-4)
    action, state = [np.asarray(cols[k],dtype=np.float32) for k in ['action','observation.state']]
    assert action.shape == state.shape == (n,16) and np.isfinite(action).all() and np.isfinite(state).all()
    for v in [action,state]:
        for start in [3,11]:
            assert np.allclose(np.linalg.norm(v[:,start:start+4],axis=1),1,atol=.02)
    return action,state

class RobotwinDataset(Dataset):
    def __init__(self, root, prepared, feature_cache, horizon=24):
        self.root=Path(root); self.prepared=read(prepared); self.rows=self.prepared['rows']
        self.stats=self.prepared['stats']; self.horizon=horizon; self.cache=Path(feature_cache)
        self.episodes=[episode(root,r) for r in self.rows]
        self.ends=np.cumsum([len(a) for a,s in self.episodes]).tolist()
        self.features={}
        for r in self.rows:
            name=r['human_video']
            if name not in self.features:
                key=self.prepared['files'][name]
                features=torch.load(self.cache/(key+'.pt'),map_location='cpu',weights_only=True)
                assert features['encoder_revision']==self.prepared['dino_revision'] and features['demo_sha256']==key
                self.features[name]=features['raw_cls'].float().reshape(1024)

    def __len__(self): return self.ends[-1]

    def __getitem__(self,index):
        i=bisect_right(self.ends,index); t=index-(self.ends[i-1] if i else 0)
        row=self.rows[i]; action,state=self.episodes[i]
        # Official IG ACT has a fixed-size posterior: repeat terminal targets.
        ids=np.minimum(np.arange(t,t+self.horizon),len(action)-1)
        rgb=np.stack([rgb_at(self.root/p,[t])[0] for p in video_paths(row)])
        def norm(value,key):
            return (value-np.asarray(self.stats[key]['mean'],np.float32))/np.asarray(self.stats[key]['std'],np.float32)
        return {'rgb':torch.from_numpy(rgb).permute(0,3,1,2).float()/255,
                'state':torch.from_numpy(norm(state[t],'state')),
                'actions':torch.from_numpy(norm(action[ids],'action')),
                'demo_cls':self.features[row['human_video']]}
