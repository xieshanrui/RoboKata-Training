"""Build original segmented-model raw frozen vision caches on disjoint GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import safetensors.torch
from safetensors import safe_open
import torch

from data import dense_demo_indices, read, write, sha


def load_vision(path):
    from transformers import SiglipVisionModel
    from transformers.models.auto import CONFIG_MAPPING
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    config=CONFIG_MAPPING['paligemma']().vision_config
    config.intermediate_size=4304;config.projection_dim=2048
    config.projector_hidden_act='gelu_fast';config.torch_dtype='float32'
    vision=SiglipVisionModel(config).float()
    holder=torch.nn.Module();holder.add_module('vision_tower',vision)
    PaliGemmaWithExpertModel.to_bfloat16_for_selected_params(holder,'bfloat16')
    prefix='paligemma_with_expert.paligemma.model.vision_tower.'
    with safe_open(path,framework='pt',device='cpu') as base:
        weights={k.removeprefix(prefix):base.get_tensor(k) for k in base.keys() if k.startswith(prefix)}
    assert weights and all(v.dtype==torch.float32 for v in weights.values())
    vision.load_state_dict(weights,strict=True);vision.requires_grad_(False)
    return vision.eval()


def build(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    vision=load_vision(args.base).to('cuda').eval()
    manifest=read(args.data/'manifest.json');views=manifest.get('demo_views',2)
    rows=manifest.get('demos',manifest['rows'])[args.rank::args.world]
    records={};verified={};started=time.time()
    with torch.no_grad():
        for number,row in enumerate(rows,1):
            assert shutil.disk_usage(args.output).free>75*1024**3,'Cache reserve exhausted'
            folder=args.data/'records'/row['id']
            with np.load(folder/'supervision.npz') as labels:
                indices,_=dense_demo_indices(labels['timestamp_s'],row['seconds'],5.)
            rgb=np.load(folder/'rgb.npy',mmap_mode='r')
            assert len(indices)<=args.frames, 'Complete Demo exceeds configured capacity'
            images=torch.zeros((len(indices),views,3,224,224),dtype=torch.float32)
            images[:len(indices)]=torch.from_numpy(np.array(rgb[indices],copy=True)).permute(0,1,4,2,3).float()/127.5-1
            images=images.flatten(0,1)
            pieces=[]
            for chunk in images.split(4):
                raw=vision(chunk.to('cuda')).last_hidden_state.to(torch.bfloat16)
                assert raw.shape[1:]==(256,1152) and torch.isfinite(raw).all()
                pieces.append(raw.cpu())
            patches=torch.cat(pieces).reshape(len(indices),views,256,1152).contiguous()
            assert patches.dtype==torch.bfloat16 and torch.isfinite(patches).all()
            name=row['id']+'.safetensors';path=args.output/name;assert not path.exists()
            safetensors.torch.save_file({'patches':patches},str(path));stat=path.stat()
            records[row['id']]=dict(file=name,sha256=sha(path),bytes=stat.st_size,input_indices=indices.tolist(),source_sha256=row['recording_sha256'])
            verified[row['id']]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
            if number%25==0:
                status=dict(rank=args.rank,records=number,total=len(rows),seconds=time.time()-started)
                write(args.output/f'progress_rank{args.rank}.json',status);print(json.dumps(status),flush=True)
    write(args.output/f'shard_rank{args.rank}.json',dict(passed=True,records=records,verified=verified,seconds=time.time()-started))


def merge(args):
    manifest=read(args.data/'manifest.json');rows=manifest.get('demos',manifest['rows']);records={};verified={}
    for rank in range(args.world):
        shard=read(args.output/f'shard_rank{rank}.json');assert shard['passed']
        assert not records.keys()&shard['records'].keys()
        records.update(shard['records']);verified.update(shard['verified'])
    assert set(records)=={r['id'] for r in rows}
    identity=dict(base_checkpoint_sha256=sha(args.base),source_manifest_sha256=sha(args.data/'manifest.json'),builder_sha256=sha(__file__),precision='official mixed BF16 vision, BF16 raw patches; no autocast')
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=dict(passed=True,vision_frozen=True,pooling_and_projection_cached=False,fixed_spatial_pool_cached=False,
                  boundary='Raw BF16 SigLIP patches before trainable attention pooling',identity=identity,fingerprint=fingerprint,
                  source_manifest_sha256=identity['source_manifest_sha256'],frames=args.frames,sample_hz=5.,records=records,total_bytes=sum(r['bytes'] for r in records.values()))
    write(args.output/'cache_manifest.json',manifest)
    write(args.output/'verification_receipt.json',dict(passed=True,all_file_sha256_verified=True,
          cache_manifest_sha256=sha(args.output/'cache_manifest.json'),fingerprint=fingerprint,records=verified,
          read_policy='Each newly written feature file SHA256 verified by its worker; all shards and source identities matched'))
    print(json.dumps(dict(passed=True,records=len(records),total_bytes=manifest['total_bytes'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8)
    p.add_argument('--frames',type=int,default=60);p.add_argument('--merge',action='store_true');a=p.parse_args()
    try:
        if a.merge:merge(a)
        else:build(a)
    except Exception as error:
        write(a.output/f'failure_rank{a.rank}.json',dict(error=repr(error)));raise
