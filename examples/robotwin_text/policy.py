"""Inference counterpart of the native text pi0.5 RoboTwin training adapter."""
from pathlib import Path
import json

import numpy as np
import safetensors.torch
import torch

from train import ROOT, NativeConfig, Observation, PaligemmaTokenizer, sha


class Policy:
    def __init__(self, checkpoint, device='cuda'):
        checkpoint=Path(checkpoint)
        receipt=json.loads((checkpoint/'complete.json').read_text(encoding='utf-8'))
        contract=receipt['contract'];self.cfg=contract['config'];self.device=device
        assert contract['model']=='native_pi05_full_finetune'
        assert sha(checkpoint/'model.safetensors')==receipt['model_sha256']
        for name,digest in contract['source_sha256'].items():
            assert sha(ROOT/name)==digest, f'Use frozen training source: {name}'
        assert sha(self.cfg['assets'])==contract['assets_sha256']
        assert sha(self.cfg['tokenizer'])==contract['tokenizer_sha256']
        self.assets=json.loads(Path(self.cfg['assets']).read_text(encoding='utf-8'))
        self.tokenizer=PaligemmaTokenizer(200,tokenizer_path=Path(self.cfg['tokenizer']))
        self.model=NativeConfig(pi05=True,pytorch_compile_mode=None).create_pytorch()
        safetensors.torch.load_model(self.model,str(checkpoint/'model.safetensors'),strict=True)
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, rgb, state, prompt, *, num_steps=10):
        """RGB: ordered head/left/right224x224 uint8; state: native EEF16."""
        rgb=np.asarray(rgb);state=np.asarray(state,np.float32)
        assert rgb.dtype==np.uint8 and rgb.shape==(3,224,224,3)
        assert state.shape==(16,) and np.isfinite(state).all() and prompt.strip()
        stats=self.assets['normalization']['state']
        normalized=(state-np.asarray(stats['q01']))/(np.asarray(stats['q99'])-np.asarray(stats['q01'])+1e-6)*2-1
        normalized=np.pad(normalized.astype(np.float32),(0,16))
        tokens,mask=self.tokenizer.tokenize(prompt,normalized)
        assert mask.sum()<200,'Prompt/state truncated'
        def tensor(x):return torch.as_tensor(x,device=self.device)[None]
        images=tensor(rgb).float().permute(0,1,4,2,3)/127.5-1
        keys=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
        obs=Observation(images={k:images[:,i] for i,k in enumerate(keys)},
            image_masks={k:torch.ones(1,dtype=torch.bool,device=self.device) for k in keys},
            state=tensor(normalized),tokenized_prompt=tensor(tokens),tokenized_prompt_mask=tensor(mask))
        predicted=self.model.sample_actions(self.device,obs,num_steps=num_steps)[0,:,:16].float().cpu().numpy()
        stats=self.assets['normalization']['action']
        action=(predicted+1)/2*(np.asarray(stats['q99'])-np.asarray(stats['q01'])+1e-6)+np.asarray(stats['q01'])
        assert np.isfinite(action).all()
        for start in (3,11):
            norm=np.linalg.norm(action[:,start:start+4],axis=-1,keepdims=True)
            assert np.all(norm>1e-6),'Degenerate quaternion'
            action[:,start:start+4]/=norm
        action[:,[7,15]]=np.clip(action[:,[7,15]],0,1)
        return action.astype(np.float32)
