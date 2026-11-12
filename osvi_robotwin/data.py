"""RGB-only human Demo, six Query images, native EEF action supervision."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from act_robotwin.data import episode, rgb_at, decoder
from act_robotwin.protocol import read, training_rows, sha
from .model import encode_pose

MEAN = torch.tensor([.485,.456,.406])[None,:,None,None]
STD = torch.tensor([.229,.224,.225])[None,:,None,None]

def normalize(rgb):
    return (torch.from_numpy(np.asarray(rgb)).permute(0,3,1,2).float()/255-MEAN)/STD

class PairedDataset(Dataset):
    def __init__(self,root,prepared,photometric=False):
        self.root=Path(root);self.prepared=read(prepared);self.rows=self.prepared['rows']
        allowed={r['pair_id']:r for r in training_rows(root)}
        assert all(allowed.get(r['pair_id'])==r for r in self.rows)
        self.smoke_only=self.prepared['smoke_only']
        for row in self.rows:
            for key in ('robot_data','robot_video','human_video'):
                name=row[key]
                assert sha(self.root/name)==self.prepared['files'][name],name
        self.cached={}
        self.photometric=photometric
        if photometric:
            from torchvision.transforms import ColorJitter
            self.jitter=ColorJitter(.4,.4,.4)

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        if i not in self.cached:
            row=self.rows[i];action,state=episode(self.root,row)
            qi=np.linspace(0,len(state)-1,6).astype(int)
            video=decoder(str(self.root/row['human_video']))
            di=np.linspace(0,len(video)-1,10).astype(int)
            assert len(decoder(str(self.root/row['robot_video'])))==len(state)
            ti=np.linspace(0,len(state)-1,50).astype(int)
            self.cached[i]=dict(images=normalize(rgb_at(self.root/row['robot_video'],qi)),
                context=normalize(rgb_at(self.root/row['human_video'],di)),
                trajectory=torch.from_numpy(encode_pose(action)[ti]),
                initial_pose=torch.from_numpy(state[0].copy()))
        item={k:v.clone() for k,v in self.cached[i].items()}
        if self.photometric:
            with torch.random.fork_rng(devices=[]):
                if hasattr(self,'epoch_seed'):
                    torch.random.default_generator.manual_seed(self.epoch_seed*1000003+i)
                for key in ('images','context'):
                    rgb=(item[key]*STD+MEAN).clamp(0,1)
                    item[key]=(self.jitter(rgb)-MEAN)/STD
        return item
