"""Native text-conditioned pi0.5 on the verified RoboTwin43 recordings.

Only data/contract adaptation lives here; optimization and checkpointing use
scripts/train_pytorch.py. No human video or segmented Demo modules are inputs.
"""
import argparse
import dataclasses
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'examples/segmented_demo'))
from data import read, write, sha
from frame_data import FrameLoader, RemainingBatches
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training.config import TrainConfig, DataConfig
from openpi.training.optimizer import AdamW, CosineDecaySchedule


class NativePi05(PI0Pytorch):
    def forward(self, observation, actions, action_valid=None, noise=None, time=None):
        loss = super().forward(observation, actions, noise=noise, time=time)
        if action_valid is None:
            return loss
        return (loss*action_valid).sum((1, 2))/action_valid.sum((1, 2)).clamp_min(1)


@dataclasses.dataclass(frozen=True)
class NativeConfig(Pi0Config):
    def create_pytorch(self):
        return NativePi05(self)


class Frames(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.root = Path(cfg['data'])
        self.manifest = read(self.root/'manifest.json')
        assert self.manifest['dataset'] == 'robotwin' and not self.manifest['smoke_only']
        self.rows = self.manifest['rows']
        self.ends = np.cumsum([r['frames'] for r in self.rows])
        self.assets = read(cfg['assets'])
        assert self.assets['manifest_sha256'] == sha(self.root/'manifest.json')
        self.stats = self.assets['normalization']
        self.tokenizer = PaligemmaTokenizer(200, tokenizer_path=Path(cfg['tokenizer']))
        for row in self.rows:
            folder = self.root/'records'/row['id']
            assert sha(folder/'supervision.npz') == row['labels_sha256']
            st = (folder/'rgb.npy').stat()
            assert st.st_size == row['rgb_bytes'] and st.st_mtime_ns == row['rgb_mtime_ns']

    def __len__(self):
        return int(self.ends[-1])

    @functools.lru_cache(maxsize=64)
    def labels(self, identity):
        with np.load(self.root/'records'/identity/'supervision.npz') as d:
            return {k:d[k] for k in ('state','action')}

    @functools.lru_cache(maxsize=64)
    def rgb(self, identity):
        return np.load(self.root/'records'/identity/'rgb.npy', mmap_mode='r')

    def normalize(self, x, key):
        st = self.stats[key]
        return ((x-np.asarray(st['q01'],np.float32))/(np.asarray(st['q99'],np.float32)-np.asarray(st['q01'],np.float32)+1e-6)*2-1).astype(np.float32)

    def __getitem__(self, index):
        ri = int(np.searchsorted(self.ends,index,side='right'))
        row = self.rows[ri]; frame = int(index-(self.ends[ri-1] if ri else 0))
        labels = self.labels(row['id'])
        state = np.pad(self.normalize(labels['state'][frame],'state'),(0,16))
        ids = np.arange(frame,frame+50)
        actions = np.pad(self.normalize(labels['action'][ids.clip(max=row['frames']-1)],'action'),((0,0),(0,16)))
        valid = np.zeros((50,32),bool);valid[ids<row['frames'],:16] = True
        tokens,mask = self.tokenizer.tokenize(self.assets['prompts'][row['id']],state)
        assert mask.sum() < 200, 'Prompt/state truncation must be resolved before training'
        return dict(rgb=np.array(self.rgb(row['id'])[frame],copy=True),state=state,actions=actions,
                    valid=valid,tokens=tokens,mask=mask)


class Loader(FrameLoader):
    dynamic_graph = False

    def __init__(self,cfg,rank,world):
        self.dataset = Frames(cfg)
        self.rank,self.world,self.global_batch = rank,world,cfg['global_batch']
        self.gradient_receipt = None
        assert world*cfg['per_gpu_batch'] == self.global_batch
        self.sampler = DistributedSampler(self.dataset,num_replicas=world,rank=rank,shuffle=True,seed=cfg['seed'],drop_last=True)
        self.batches = RemainingBatches(self.sampler,cfg['per_gpu_batch'],drop_last=True)
        self.epoch,self.cursor = 0,0
        self.loader = DataLoader(self.dataset,batch_sampler=self.batches,num_workers=0,collate_fn=self.collate,
                                 generator=torch.Generator().manual_seed(cfg['seed']+rank))

    def collate(self,samples):
        def stack(k):return torch.from_numpy(np.stack([s[k] for s in samples]))
        rgb = stack('rgb').float().permute(0,1,4,2,3)/127.5-1
        keys = self.dataset.manifest['query_image_keys']
        obs = Observation(images={k:rgb[:,i] for i,k in enumerate(keys)},
            image_masks={k:torch.ones(len(samples),dtype=torch.bool) for k in keys},state=stack('state'),
            tokenized_prompt=stack('tokens'),tokenized_prompt_mask=stack('mask'))
        return obs,stack('actions'),stack('valid')

    def verify_gradients(self,model):
        if self.gradient_receipt is not None:return
        raw = model.module if hasattr(model,'module') else model
        params = dict(raw.named_parameters())
        names = ['action_out_proj.weight',
            'paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight',
            'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight',
            'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight']
        values = {n:float(params[n].grad.float().norm()) if params[n].grad is not None else None for n in names}
        assert all(v is not None and np.isfinite(v) and v>0 for v in values.values()),values
        self.gradient_receipt = values


def prepare(data,output):
    root = Path(data);manifest = read(root/'manifest.json')
    upstream = read(manifest['input_manifest'])
    assert sha(manifest['input_manifest']) == manifest['input_sha256']
    rows = manifest['rows'];assert len(rows)==2143 and len({r['task'] for r in rows})==43
    assert not {r['task'] for r in rows}.intersection(upstream['eval_tasks']+[upstream['excluded_eval_task']])
    lookup = {r['pair_id']:r for r in upstream['rows']}
    prompts = {r['id']:lookup[r['source_id']]['robot_instruction'].strip() for r in rows}
    assert all(prompts.values())
    labels = {'state':[],'action':[]}
    for row in rows:
        p=root/'records'/row['id']/'supervision.npz';assert sha(p)==row['labels_sha256']
        with np.load(p) as d:
            for k in labels:
                x=d[k];assert x.shape==(row['frames'],16) and np.isfinite(x).all();labels[k].append(x)
    stats={}
    for k,parts in labels.items():
        x=np.concatenate(parts);q=np.quantile(x,[.01,.99],axis=0)
        stats[k]=dict(q01=q[0].tolist(),q99=q[1].tolist(),mean=x.mean(0).tolist(),std=x.std(0).tolist())
    write(output,dict(manifest_sha256=sha(root/'manifest.json'),source_manifest_sha256=manifest['input_sha256'],
        prompts=prompts,normalization=stats,normalization_method='OpenPI quantiles, no clipping, training split only',
        training_episodes=len(rows),training_frames=sum(r['frames'] for r in rows),held_out_tasks=upstream['eval_tasks'],
        prompt_source='Released robot_instruction per Query episode; no human video input'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-data',type=Path);p.add_argument('--assets',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);p.add_argument('--steps',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.prepare_data:prepare(a.prepare_data,a.assets);return
    cfg=read(a.config);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    assert world==cfg['world_size'];torch.set_num_threads(cfg.get('cpu_threads',4))
    model_cfg=NativeConfig(pi05=True,pytorch_compile_mode=None)
    loader=Loader(cfg,rank,world);output=a.output.resolve()
    paths=[ROOT/'scripts/train_pytorch.py',Path(__file__),ROOT/'examples/segmented_demo/frame_data.py',ROOT/'examples/segmented_demo/data.py',
           ROOT/'src/openpi/models/model.py',ROOT/'src/openpi/models/pi0_config.py',ROOT/'src/openpi/models/tokenizer.py',
           *list((ROOT/'src/openpi/models_pytorch').rglob('*.py'))]
    loader.contract=dict(model='native_pi05_full_finetune',config=cfg,model_config=dataclasses.asdict(model_cfg),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in paths},
        assets_sha256=sha(cfg['assets']),base_sha256=sha(cfg['base_checkpoint']),tokenizer_sha256=sha(cfg['tokenizer']))
    loader.contract=json.loads(json.dumps(loader.contract))
    native=TrainConfig(name=output.parent.name,exp_name=output.name,checkpoint_base_dir=str(output.parent.parent),model=model_cfg,
        batch_size=cfg['global_batch'],num_train_steps=a.steps or cfg['steps'],seed=cfg['seed'],num_workers=0,
        save_interval=5000,log_interval=10,wandb_enabled=False,resume=a.resume,ema_decay=None,
        pytorch_weight_path=None if a.resume else str(Path(cfg['base_checkpoint']).parent),optimizer=AdamW(),
        lr_schedule=CosineDecaySchedule(warmup_steps=cfg['warmup'],peak_lr=cfg['lr_peak'],decay_steps=cfg['steps'],decay_lr=cfg['lr_end']))
    assert native.checkpoint_dir==output
    spec=importlib.util.spec_from_file_location('openpi_native_trainer',ROOT/'scripts/train_pytorch.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer);trainer.init_logging()
    if rank==0:
        output.mkdir(parents=True,exist_ok=True);write(output/'contract.json',loader.contract)
    trainer.train_loop(native,data_loader=(loader,DataConfig()))


if __name__=='__main__':main()
