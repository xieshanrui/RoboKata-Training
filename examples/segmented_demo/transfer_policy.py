"""Native16D segmented pi0 policy, usable directly or over the binary pipe.

Reset accepts only RGB and its recorded timestamps. Predict accepts current
Query RGB and measured native state. No task names enter model observations.
"""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

# Select the checkpoint-frozen implementation before importing model modules.
# Library callers must set their import path to the same source explicitly.
if __name__ == '__main__':
    source_parser = argparse.ArgumentParser(add_help=False)
    source_parser.add_argument('--source', type=Path)
    selected_source = source_parser.parse_known_args()[0].source
    if selected_source is not None:
        sys.path[:0] = [str(selected_source/'src'), str(selected_source/'examples/segmented_demo'),
                       str(selected_source/'packages/openpi-client/src')]

import numpy as np
import safetensors.torch
import torch

from cache_vision import encode_patches
from data import PROMPT, dense_demo_indices, read, require, sha
import openpi
from openpi.models.model import Observation
from openpi.models.pi0_frame_demo_config import RoboICLConfig
from openpi.models.tokenizer import PaligemmaTokenizer


class TransferPolicy:
    def __init__(self, checkpoint, *, source=None, training_data=None, tokenizer=None, device='cuda:0'):
        checkpoint=Path(checkpoint);marker=read(checkpoint/'complete.json');contract=marker['contract']
        source=Path(source) if source else Path(__file__).resolve().parents[2]
        require(Path(openpi.__file__).resolve().parent == (source/'src/openpi').resolve(),
                'Imported OpenPI differs from --source; select the frozen source before importing the policy')
        for relative,digest in contract['identities']['model_sources_sha256'].items():
            require(sha(source/relative)==digest, f'Checkpoint source differs: {relative}')
        require(sha(checkpoint/'model.safetensors')==marker['model_sha256'], 'Checkpoint model SHA differs')
        training_data=Path(training_data or contract['data']);tokenizer=Path(tokenizer or contract['tokenizer'])
        require(sha(training_data/'normalization.json')==contract['identities']['normalization_sha256'], 'Normalization differs')
        require(sha(tokenizer)==contract['identities']['tokenizer_sha256'], 'Tokenizer differs')
        self.stats=read(training_data/'normalization.json')
        self.config=RoboICLConfig(**contract['model_config'])
        require(self.config.action_encoding in ('robotwin_eef16','ig_joint16'), 'Use legacy worker for TCP10')
        self.device=torch.device(device);self.marker=marker
        self.model=self.config.create_pytorch()
        safetensors.torch.load_model(self.model,str(checkpoint/'model.safetensors'),strict=True)
        self.model.to(self.device).eval()
        self.tokens,self.token_mask=[torch.from_numpy(x)[None].to(self.device) for x in
            PaligemmaTokenizer(48,tokenizer_path=tokenizer).tokenize(PROMPT)]
        self.demo=None

    @torch.no_grad()
    def reset(self, rgb, recorded_times, duration, seed):
        rgb=np.asarray(rgb);times=np.asarray(recorded_times,np.float64)
        require(rgb.dtype==np.uint8 and rgb.shape==(len(times),self.config.demo_views,224,224,3),'Demo RGB layout')
        indices,_=dense_demo_indices(times,float(duration),self.config.demo_sample_hz)
        # Preserve complete long demonstrations; the model's terminal learned
        # time embedding is reused after its configured training capacity.
        n=max(self.config.demo_frames,len(indices))
        valid=torch.zeros((1,n,self.config.demo_views),dtype=torch.bool,device=self.device)
        valid[:,:len(indices)]=True
        images=torch.zeros((1,n,self.config.demo_views,3,224,224),device=self.device)
        images[0,:len(indices)]=torch.from_numpy(np.array(rgb[indices],copy=True)).to(self.device).permute(0,1,4,2,3).float()/127.5-1
        ts=torch.zeros((1,n),device=self.device);ts[0,:len(indices)]=torch.as_tensor(times[indices],device=self.device,dtype=torch.float32)
        vision=self.model.paligemma_with_expert.paligemma.model.vision_tower
        patches=encode_patches(vision,images,valid,self.config.demo_encode_batch)
        self.demo=dict(demo_patches=patches,demo_valid=valid,demo_timestamps=ts)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

    def observation(self, rgb, state):
        require(self.demo is not None,'Reset Demo before prediction')
        rgb=np.asarray(rgb);state=np.asarray(state,np.float32)
        keys=self.config.query_image_keys
        require(rgb.dtype==np.uint8 and rgb.shape==(len(keys),224,224,3),'Query RGB layout')
        mean=np.asarray(self.stats['state']['mean'],np.float32);std=np.asarray(self.stats['state']['std'],np.float32)
        require(state.shape==mean.shape and np.isfinite(state).all(),'Measured state layout')
        state=np.pad((state-mean)/std,(0,32-len(state)))
        images=torch.from_numpy(np.array(rgb,copy=True)).to(self.device).permute(0,3,1,2).float()/127.5-1
        return Observation(images={k:images[i:i+1] for i,k in enumerate(keys)},
            image_masks={k:torch.ones(1,dtype=torch.bool,device=self.device) for k in keys},
            state=torch.from_numpy(state)[None].to(self.device),tokenized_prompt=self.tokens,
            tokenized_prompt_mask=self.token_mask,source_id=torch.zeros(1,dtype=torch.long,device=self.device),**self.demo)

    @torch.no_grad()
    def predict(self,rgb,state,*,noise=None):
        observation=self.observation(rgb,state)
        normalized=self.model.sample_actions(self.device,observation,noise=noise,num_steps=10)[0].float().cpu().numpy()
        require(np.isfinite(normalized).all(),'Nonfinite prediction')
        require(np.max(np.abs(normalized[:,16:]))<1e-4,'Untrained padding bridge did not reach zero')
        s=self.stats['action'];actions=normalized[:,:16]*np.asarray(s['std'],np.float32)+np.asarray(s['mean'],np.float32)
        if self.config.action_encoding=='robotwin_eef16':
            for start in (3,11):
                q=actions[:,start:start+4];norm=np.linalg.norm(q,axis=-1,keepdims=True)
                require((norm>1e-6).all(),'Degenerate predicted quaternion');actions[:,start:start+4]=q/norm
            actions[:,[7,15]]=np.clip(actions[:,[7,15]],0,1)
        return actions,normalized


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--training-data',type=Path)
    parser.add_argument('--tokenizer',type=Path)
    args=parser.parse_args()
    from protocol import receive,send
    stdin,stdout=sys.stdin.buffer,sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        torch.set_num_threads(4)
        policy=TransferPolicy(args.checkpoint,source=args.source,training_data=args.training_data,tokenizer=args.tokenizer)
        send(stdout,dict(ready=True,step=policy.marker['step'],action_encoding=policy.config.action_encoding,
                         action_horizon=50,query_image_keys=policy.config.query_image_keys,demo_views=policy.config.demo_views))
        while True:
            try:meta,arrays=receive(stdin)
            except EOFError:break
            if meta['kind']=='close':
                send(stdout,dict(closed=True));break
            if meta['kind']=='reset':
                policy.reset(arrays['rgb'],arrays['timestamps'],meta['duration'],meta['seed']);send(stdout,dict(reset=True))
            elif meta['kind']=='predict':
                actions,_=policy.predict(arrays['rgb'],arrays['state']);send(stdout,dict(predicted=True),actions=actions)
            else:raise ValueError('Unknown policy command')


if __name__=='__main__':
    main()
