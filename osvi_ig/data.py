"""Random same-task human pairing over the frozen C15 recording split."""
import numpy as np
import torch
from act_robotwin.protocol import read,sha
from osvi_robotwin.data import normalize,MEAN,STD

class PairedDataset:
    def __init__(self,root,prepared):
        self.prepared=read(prepared);self.smoke_only=self.prepared['smoke_only'];self.photometric=False
        assert self.prepared['schema']=='osvi_ig_c15_native_joint_v1'
        self.rows=self.prepared['sim'];self.human=self.prepared['human'];self.groups={};self.loaded={}
        if not self.smoke_only:assert len(self.rows)==3000 and len(self.human)==750
        for i,row in enumerate(self.human):self.groups.setdefault(row['human_repo_id'],[]).append(i)
        for row in self.rows+self.human:assert sha(row['path'])==row['sha256'],row['path']
        self.adaptation=dict(dataset='ig10k_c15',pairing='random same-task human per epoch/query',
            geometry='native absolute joint action targets; no image flips/crops',
            action_representation='native_joint',normalization=self.prepared['normalization'])

    def __len__(self):return len(self.rows)

    def load(self,row):
        path=row['path']
        if path not in self.loaded:self.loaded[path]=torch.load(path,map_location='cpu',weights_only=True)
        return self.loaded[path]

    def __getitem__(self,i):
        row=self.rows[i];seed=getattr(self,'epoch_seed',42)*1000003+i
        candidates=self.groups[row['human_repo_id']];di=candidates[int(np.random.default_rng(seed).integers(len(candidates)))]
        query=self.load(row);demo=self.load(self.human[di])
        item=dict(images=normalize(query['rgb'].numpy()),context=normalize(demo['rgb'].numpy()),
            trajectory=query['trajectory'].clone(),initial_action=query['initial_action'].clone())
        if self.photometric:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                for key in ('images','context'):item[key]=(self.jitter((item[key]*STD+MEAN).clamp(0,1))-MEAN)/STD
        return item
