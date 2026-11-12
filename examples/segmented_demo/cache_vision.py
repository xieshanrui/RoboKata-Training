"""Frozen shared SigLIP raw patches; pooling/projector remain outside the cache."""
from pathlib import Path
import argparse,hashlib,inspect,json,shutil,time
import numpy as np
import safetensors
import safetensors.torch
import torch
from data import dense_demo_indices,read,sha,write


@torch.no_grad()
def encode_patches(vision,rgb,valid,chunk_size=4):
    assert not vision.training and not any(p.requires_grad for p in vision.parameters())
    assert rgb.shape[:3]==valid.shape and rgb.shape[-3:]==(3,224,224)
    rgb=torch.where(valid[...,None,None,None],rgb,0)
    patches=torch.cat([vision(chunk).last_hidden_state.to(torch.bfloat16)
        for chunk in rgb.flatten(0,2).split(chunk_size)])
    return patches.reshape(*valid.shape,256,1152)


def main(args):
    from openpi.models.pi0_frame_demo_config import RoboICLConfig
    cfg=read(args.config);assert cfg['vision_frozen'] and cfg['demo_cache']
    torch.set_num_threads(4);torch.cuda.set_device(0)
    manifest=read(Path(cfg['data'])/'manifest.json')
    estimate=len(manifest['rows'])*cfg['demo_frames']*2*256*1152*2
    args.output.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.output).free>estimate+200_000_000_000,'Insufficient cache/checkpoint reserve'
    model=RoboICLConfig(action_dim=32,action_encoding='world_tcp10',demo_frames=cfg['demo_frames']).create_pytorch()
    model.load_base_checkpoint(cfg['base_checkpoint']);model.set_phase('A')
    vision=model.paligemma_with_expert.paligemma.model.vision_tower
    del model
    digest=hashlib.sha256()
    for name,value in sorted(vision.state_dict().items()):
        digest.update(name.encode());digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    identity=dict(base_checkpoint_sha256=sha(cfg['base_checkpoint']),vision_state_sha256=digest.hexdigest(),
        vision_implementation_sha256=sha(Path(inspect.getfile(type(vision)))),torch=str(torch.__version__),
        builder_sha256=sha(Path(__file__)),sampling_function_sha256=hashlib.sha256(inspect.getsource(dense_demo_indices).encode()).hexdigest(),
        frames=cfg['demo_frames'],sample_hz=cfg['demo_sample_hz'],encode_batch=4,
        boundary='Raw SigLIP BF16 patches, before trainable pooling and visual projection',
        preprocessing='Recorded224x224 RGB/127.5-1; invalid normalized pixels zero; no Demo augmentation')
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    marker=args.output/'encoder.json'
    if marker.exists():assert read(marker)['fingerprint']==fingerprint
    else:write(marker,dict(fingerprint=fingerprint,identity=identity))
    vision.to('cuda').eval();records={};started=time.perf_counter()
    for index,row in enumerate(manifest['rows']):
        folder=Path(cfg['data'])/'records'/row['id'];path=args.output/(row['id']+'.safetensors')
        with np.load(folder/'supervision.npz',allow_pickle=False) as labels:
            indices,times=dense_demo_indices(labels['timestamp_s'],row['seconds'],cfg['demo_sample_hz'])
        assert len(times)<=cfg['demo_frames']
        if path.exists():
            with safetensors.safe_open(str(path),framework='pt',device='cpu') as saved:
                meta=saved.metadata();assert meta['fingerprint']==fingerprint and meta['source_sha256']==row['recording_sha256']
                assert json.loads(meta['input_indices'])==indices.tolist()
        else:
            rgb=np.zeros((cfg['demo_frames'],2,224,224,3),np.uint8)
            rgb[:len(times)]=np.load(folder/'rgb.npy',mmap_mode='r')[indices]
            valid=torch.zeros((1,cfg['demo_frames'],2),dtype=torch.bool,device='cuda');valid[:,:len(times)]=True
            images=torch.from_numpy(rgb).to('cuda').float().permute(0,1,4,2,3)[None]/127.5-1
            patches=encode_patches(vision,images,valid,4)[0,:len(times)].cpu().contiguous()
            assert torch.isfinite(patches).all()
            temporary=path.with_suffix('.tmp')
            safetensors.torch.save_file(dict(patches=patches),str(temporary),metadata=dict(
                fingerprint=fingerprint,source_sha256=row['recording_sha256'],input_indices=json.dumps(indices.tolist())))
            temporary.replace(path)
        loaded=safetensors.torch.load_file(str(path))['patches']
        assert loaded.shape==(len(times),2,256,1152) and loaded.dtype==torch.bfloat16 and torch.isfinite(loaded).all()
        records[row['id']]=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,
            input_indices=indices.tolist(),source_sha256=row['recording_sha256'])
        if (index+1)%100==0:print(json.dumps(dict(event='frozen_demo_cache',records=index+1,total=len(manifest['rows']))),flush=True)
    result=dict(passed=True,fingerprint=fingerprint,identity=identity,frames=cfg['demo_frames'],sample_hz=cfg['demo_sample_hz'],
        source_manifest_sha256=sha(Path(cfg['data'])/'manifest.json'),records=records,
        total_bytes=sum(row['bytes'] for row in records.values()),seconds=time.perf_counter()-started,
        demo_rgb_only=True,pooling_and_projection_cached=False,vision_frozen=True)
    write(args.output/'cache_manifest.json',result)
    print(json.dumps(dict(event='frozen_demo_cache_complete',records=len(records),bytes=result['total_bytes'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
