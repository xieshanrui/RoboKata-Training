"""RoboTwin A+B1 adapter on the official Zero-WAM streaming/action interface.

ROBOTWIN_B1_RUN supplies immutable weights, Demo RGB and an execution receipt.
ROBOKATA_ROOT supplies the published A architecture; no single-arm adapters are used.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ['ROBOTWIN_B1_RUN'])
ROBOKATA = Path(os.environ['ROBOKATA_ROOT'])
CHECKPOINT = Path(os.environ['ROBOTWIN_B1_CHECKPOINT'])
sys.path[:0] = [str(ROOT), str(ROBOKATA / 'backend'), str(ROBOKATA / 'roboicl'), str(ROBOKATA / 'robotwin_b')]

from safetensors.torch import load_file
from examples.baselines.zero_wam.zero_wam.lora import LoRALinear, target_names
from examples.baselines.zero_wam.zero_wam.wan_va.modules.icl_model import WanICLTransformer3DModel
from examples.baselines.zero_wam.zero_wam.wan_va.semantic_ifp import SemanticIFPHead
from demo_attention import SPEC, attach_demo_attention
from demo_features import encode, load_encoder
from data import read_rgb as read_training_rgb
from wan_va import wan_va_server as native


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load_b1(transformer_path, torch_dtype, torch_device, **kwargs):
    meta = json.loads(CHECKPOINT.with_suffix('.json').read_text(encoding='utf-8'))
    path = CHECKPOINT
    assert sha(path) == meta['sha256']
    adapter = load_file(str(path), device='cpu')
    model = WanICLTransformer3DModel.from_pretrained(transformer_path, torch_dtype=torch_dtype)
    model.disable_mcp_modules()
    model.requires_grad_(False)
    targets = target_names(model)
    saved_targets = {name.rsplit('.', 1)[0] for name in adapter if name.endswith('.lora_a') and name.startswith('blocks.')}
    assert set(targets) == saved_targets
    for name in targets:
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, LoRALinear(
            model.get_submodule(name), meta['contract']['rank'], meta['contract']['lora_alpha']))
    attach_demo_attention(model, SPEC)
    params = dict(model.named_parameters())
    expected = {name for name in params if '.lora_' in name or '.demo_attention.' in name}
    training_only = {name for name in adapter if name.startswith(('mcp_mlp_hidden.', 'semantic_ifp.'))}
    assert set(adapter) == expected | training_only
    with torch.device('meta'):
        semantic = SemanticIFPHead(model.inner_dim, 4, 768, 2)
        mcp = torch.nn.Sequential(
            LoRALinear(torch.nn.Linear(model.inner_dim * 4, model.inner_dim), meta['contract']['rank'], meta['contract']['lora_alpha']),
            torch.nn.SiLU(), LoRALinear(torch.nn.Linear(model.inner_dim, model.inner_dim), meta['contract']['rank'], meta['contract']['lora_alpha']))
    extra_shapes = {'semantic_ifp.' + n: tuple(p.shape) for n, p in semantic.named_parameters()}
    extra_shapes.update({'mcp_mlp_hidden.' + n: tuple(p.shape) for n, p in mcp.named_parameters() if '.lora_' in n})
    assert {n: tuple(adapter[n].shape) for n in training_only} == extra_shapes
    with torch.no_grad():
        for name in expected:
            assert adapter[name].shape == params[name].shape, name
            assert torch.isfinite(adapter[name]).all(), name
            params[name].copy_(adapter[name])
    model.eval().requires_grad_(False)
    for index in SPEC['layers']:
        model.blocks[index].demo_attention.to(dtype=torch_dtype)
    model.to(device=torch_device)
    # Match native load_transformer: RoPE buffers also need the inference dtype.
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point():
                setattr(module, name, buffer.to(torch_dtype))
    receipt = dict(epoch=meta['epoch'], step=meta['step'], checkpoint_sha256=meta['sha256'],
                   inference_tensors=len(expected), training_only_tensors=len(training_only),
                   lora_rank=meta['contract']['rank'], lora_alpha=meta['contract']['lora_alpha'],
                   demo_attention=SPEC, action_dim=30, single_arm_adapter=False)
    output = Path(os.environ['B1_SERVER_RECEIPT'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return model


def single_gpu_model(model):
    # Each evaluation lane owns one GPU. Keep FP32 LoRA residuals as in
    # RoboKata inference instead of wrapping mixed base/adapter dtypes in FSDP.
    assert torch.distributed.get_world_size() == 1
    return model


class B1Server(native.VA_Server):
    def __init__(self, config):
        super().__init__(config)
        encoder = RUN / 'dino'
        meta = json.loads(CHECKPOINT.with_suffix('.json').read_text(encoding='utf-8'))
        assert sha(encoder / 'model.safetensors') == meta['contract']['features']['encoder_sha256']
        self.dino = load_encoder(encoder, self.device)
        self.demo_features = None
        self.feature_cache = {}
        self.demo_files = json.loads((RUN / 'demo_files.json').read_text(encoding='utf-8'))

    def _load_or_encode_icl(self, video_path, latent_path):
        latent, _ = super()._load_or_encode_icl(video_path, latent_path)
        # Both branches use RGB only. HumanGen cached text is never a model input.
        return latent, self.empty_text_emb

    def _cache_icl_context(self, video_path, latent_path):
        key = Path(latent_path).parent.name
        info = self.demo_files[key]
        if key not in self.feature_cache:
            path = RUN / info['file']
            assert sha(path) == info['sha256']
            frames = read_training_rgb(path)
            assert len(frames) == info['frames'] and len(frames) >= 8
            indices = ((np.arange(8) + 0.5) * len(frames) / 8).astype(int)
            sampled = frames[indices]
            # The published feature contract specifies FP32 DINO inference.
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                features = encode(self.dino, sampled)
            assert features.shape == (1, 32, 780) and torch.isfinite(features).all()
            self.feature_cache[key] = features.to(self.device)
        self.demo_features = self.feature_cache[key]
        super()._cache_icl_context(video_path, latent_path)

    def _prepare_icl_stream(self, data, timestep, mode, cache_type):
        result = super()._prepare_icl_stream(data, timestep, mode, cache_type)
        if mode == 'action':
            assert self.demo_features is not None
            result['demo_features'] = self.demo_features
        return result


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    native.load_transformer = load_b1
    native.shard_model = single_gpu_model
    native.VA_Server = B1Server
    config = native.VA_CONFIGS['robotwin']
    config.host = '127.0.0.1'
    config.rgb_only = True
    native.init_logger()
    native.main()
