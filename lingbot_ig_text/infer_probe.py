"""Decode one native text-conditioned action chunk; never execute a robot."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

import torch

from data import IGTextDataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args()
    settings=json.loads(args.config.read_text(encoding='utf-8'))
    root=Path(settings['run_root'])
    checkpoint=root/'train/checkpoints'/f"checkpoint_step_{settings['steps']}"/'transformer'
    sys.path.insert(0,str(Path(settings['official_source'])/'wan_va'))
    from modules.utils import load_transformer
    from configs.va_robotwin_cfg import va_robotwin_cfg
    from wan_va_server import VA_Server
    from utils import FlowMatchScheduler
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(20261110)
    started=time.perf_counter()
    dataset=IGTextDataset(settings['manifest'],settings['statistics'],settings['text_cache'],
                          settings['action_space_source'],cfg_prob=0)
    index=settings['probe_indices'][0][2]
    sample=dataset[index]
    record=dataset.records[index]
    payload=torch.load(record['cache_path'],map_location='cpu',weights_only=False,mmap=True)
    origin=torch.as_tensor(payload['states'][0]).float()
    server=VA_Server.__new__(VA_Server)
    server.device=torch.device('cuda')
    server.dtype=torch.bfloat16
    server.cache_name='text_probe'
    server.save_root=str(root/'inference')
    config=copy.deepcopy(va_robotwin_cfg)
    config.update(height=sample['latents'].shape[-2]*16,
                  width=sample['latents'].shape[-1]*16,env_type='ig_single_zed2i',
                  obs_cam_keys=['observation.images.zed2i'],used_action_channel_ids=dataset.channels)
    server.job_config=config
    server.env_type=config.env_type
    server.transformer=load_transformer(str(checkpoint),torch.bfloat16,'cuda',attn_mode='torch')
    server.transformer.eval().requires_grad_(False)
    server.streaming_vae=SimpleNamespace(clear_cache=lambda:None)
    server.scheduler=FlowMatchScheduler(shift=config.snr_shift,sigma_min=0.,extra_one_step=True)
    server.action_scheduler=FlowMatchScheduler(shift=config.action_snr_shift,sigma_min=0.,extra_one_step=True)
    # The input is the already verified causal first Query latent; no future
    # Query latents or actions enter the inference model.
    first=sample['latents'][None,:,:1].to(device='cuda',dtype=torch.bfloat16)
    server._encode_obs=MethodType(lambda self,obs:first,server)
    def decode(self,actions):
        values=actions[0,dataset.channels,1:,:,0].permute(1,2,0).flatten(0,1).float().cpu()
        return dataset.space.decode_action(values,origin)
    server.postprocess_action=MethodType(decode,server)
    server._reset(prompt=None)
    server.prompt_embeds=sample['text_emb'][None].cuda()
    server.negative_prompt_embeds=dataset.text['empty'][None].cuda()
    torch.cuda.synchronize()
    tick=time.perf_counter()
    with torch.inference_mode():
        actions,latents=server._infer({},frame_st_id=0)
    torch.cuda.synchronize()
    assert list(actions.shape)==[16,16] and torch.isfinite(actions).all()
    assert torch.isfinite(latents).all()
    torch.save(actions,root/'inference/actions.pt')
    result=dict(smoke_only=True,robot_executed=False,episode_index=index,
                human_repo_id=record['human_repo_id'],action_shape=list(actions.shape),
                conditioning='fixed human text and first query latent only',
                action_coordinates='absolute dual-arm joint/gripper commands after IG origin decoding',
                video_denoising_steps=config.num_inference_steps,
                action_denoising_steps=config.action_num_inference_steps,
                chunk_seconds=time.perf_counter()-tick,total_seconds=time.perf_counter()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    (root/'inference/verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
