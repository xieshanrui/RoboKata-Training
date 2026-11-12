"""RoboICL world-hand10 LoRA checkpoint on the pinned IG native streaming policy.

The supplied training adapter uses channels0..8,28 and pads measured state10
to18. This bridge also applies that contract to executed history, which the
supplied single-query inference script does not exercise.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from roboicl_worker import receive, send

CHANNELS = (*range(9), 28)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def main(args, stdin, stdout):
    import torch
    from easydict import EasyDict
    from torch.nn import functional as F
    version_a = args.version_a_source is not None
    sys.path.insert(0, str(args.version_a_source/'backend' if version_a else args.ig_source))
    from examples.baselines.zero_wam.zero_wam import policy
    from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig
    from examples.baselines.zero_wam.zero_wam.action_space import EpisodeActionSpace
    from examples.baselines.zero_wam.zero_wam.cache import sample_plan

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    checkpoint_sha = sha(args.checkpoint)
    assert checkpoint_sha == args.checkpoint_sha256
    # Training checkpoints contain NumPy RNG state as well as tensors.
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
               np.dtypes.UInt32DType, np.dtypes.Float64DType, EasyDict]
    if version_a:
        assert args.dino
        sys.path.insert(0, str(args.version_a_source/'roboicl'))
        from demo_attention import SPEC, attach_demo_attention
        from demo_features import FRAMES, SCHEMA, encode
        from safetensors.torch import load_file
        from transformers import Dinov2Model
        from query_views import QueryViewsMixin
        if args.checkpoint.suffix == '.safetensors':
            assert args.baseline_contract and args.adapter_metadata
            with torch.serialization.safe_globals(allowed):
                baseline = torch.load(args.baseline_contract, map_location='cpu', mmap=True, weights_only=True)
            metadata = json.loads(args.adapter_metadata.read_text(encoding='utf-8'))
            assert metadata['sha256'] == checkpoint_sha
            saved = dict(native_config=baseline['native_config'], global_step=metadata['global_step'],
                         adapter_state_dict=load_file(str(args.checkpoint), device='cpu'))
            assert sum(x.numel() for x in saved['adapter_state_dict'].values()) == metadata['trainable_parameters']
            del baseline
        else:
            with torch.serialization.safe_globals(allowed):
                saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)

    else:
        with torch.serialization.safe_globals(allowed):
            saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    settings = saved['native_config']
    query_views = 2 if settings.get('ig_query_views', 'head') == 'head_wrist' else 1
    if not version_a:
        class QueryViewsMixin:
            pass
    assert settings['ig_roboicl_channels'] == list(CHANNELS)
    stats = json.loads(args.stats.read_text(encoding='utf-8'))
    assert settings['ig_normalization'] == stats
    assert settings['ig_action_representation'] == stats['representation'] == 'roboicl_world_hand10_absolute'
    adapter_rank = settings['ig_lora_rank']
    assert isinstance(adapter_rank, int) and adapter_rank > 0
    manifest = json.loads((args.eval_data/'manifest.json').read_text(encoding='utf-8'))
    demos = {r['id']: r for r in manifest['rows']}

    def pack(actions, _offset):
        assert actions.shape[-1] == 10
        result = actions.new_zeros(*actions.shape[:-1], 30)
        result[..., list(CHANNELS)] = actions
        return result

    policy.pack_actions = pack
    policy.unpack_actions = lambda x, _offset: x[..., list(CHANNELS)]

    class RoboICLPolicy(QueryViewsMixin, policy.ZeroWAMAgent):
        def clear_cache(self):
            super().clear_cache()
            self.demo_features = None

        def _stream(self, data, timestep, action, *positional, **kwargs):
            inputs = super()._stream(data, timestep, action, *positional, **kwargs)
            if version_a and action and self.demo_features is not None:
                inputs['demo_features'] = self.demo_features
            return inputs

        def _state(self, obs):
            assert obs['state'].shape[-1] == 10
            return F.pad(self.roboicl_space.normalize(obs['state'].to(self.device), 'state'), (0, 8))

        def accept_executed(self, commands, frames):
            expected = 16 if self.chunk_idx == 0 else 32
            assert commands.shape == (expected, 10)
            assert frames.shape == ((expected//4, 224, 224, 3) if query_views == 1 else (expected//4, 2, 224, 224, 3))
            assert not self.executed_actions and not self.observed_frames
            raw = torch.as_tensor(commands.copy(), device=self.device).float()
            # The simulator's gripper applies this same physical [-1,1] bound.
            raw[:, 9].clamp_(-1, 1)
            lo, hi = self.roboicl_space.bounds(raw, 'action')
            normalized = 2 * (raw-lo) / (hi-lo+1e-6) - 1
            self.executed_actions.extend(x[None].clone() for x in normalized)
            rgb = torch.from_numpy(frames.copy()).to(self.device).float().div_(255)
            self.observed_frames.extend(x[None, None] for x in rgb)
            self.total_executed_steps += expected

    actor = RoboICLPolicy(ZeroWAMConfig(model_path=str(args.model),
        empty_text_embedding=str(args.empty_text), cache_dir=str(args.cache), action_dim=10,
        adapter_rank=adapter_rank, video_inference_steps=50, action_inference_steps=50,
        video_snr_shift=settings.get('snr_shift',5.),
        action_snr_shift=settings.get('action_snr_shift',1.),
        icl_rope_h=settings.get('icl_rope_h',24)))
    actor.query_views = query_views
    actor.action_channel_ids = CHANNELS
    actor.roboicl_space = EpisodeActionSpace(stats)
    if version_a:
        attach_demo_attention(actor.transformer, SPEC)
        for index in SPEC['layers']:
            actor.transformer.blocks[index].demo_attention.to(device=actor.device, dtype=actor.dtype)
        dino = Dinov2Model.from_pretrained(args.dino, local_files_only=True).to(actor.device)
        dino.eval().requires_grad_(False)
        assert (dino.config.hidden_size,dino.config.patch_size)==(768,14)
    expected = {n: p for n, p in actor.transformer.named_parameters() if p.requires_grad}
    adapter = saved['adapter_state_dict']
    assert not (set(expected)-set(adapter))
    extras = set(adapter)-set(expected)
    semantic = {n: adapter[n] for n in extras if n.startswith('semantic_ifp.')}
    if semantic:
        assert version_a, 'B1 requires the Version A inference architecture'
        from examples.baselines.zero_wam.zero_wam.wan_va.semantic_ifp import SemanticIFPHead
        with torch.device('meta'):
            head = SemanticIFPHead(actor.transformer.inner_dim, 4, 768, 2)
        shapes = {'semantic_ifp.'+n: tuple(p.shape) for n,p in head.named_parameters()}
        assert {n: tuple(p.shape) for n,p in semantic.items()} == shapes
        del head
    assert all(n.startswith(('mcp_blocks.', 'mcp_mlp_hidden.', 'mcp_projections.', 'semantic_ifp.')) for n in extras), sorted(extras)
    assert all(tuple(adapter[n].shape) == tuple(p.shape) for n,p in expected.items())
    actor.load_adapter({n: adapter[n] for n in expected})
    step = saved['global_step']
    del adapter, saved
    actor.eval()
    send(stdout, dict(ready=True, step=step, execute_tokens='native_first16_then32',
        checkpoint_sha256=checkpoint_sha, geometry_sha256=manifest['geometry_sha256'],
        model='Zero-WAM RoboICL LoRA', demo_views=1, query_views=query_views, demo_hz=12, action_hz=50,
        action_channels=list(CHANNELS), adapter_parameters_loaded=len(expected),
        training_only_mcp_parameters=len(extras)-len(semantic),
        training_only_semantic_parameters=len(semantic), version_b1=bool(semantic), source_sha256=sha(__file__),
        version_a=version_a, demo_attention_schema=SCHEMA if version_a else None,
        dino_sha256=sha(args.dino/'model.safetensors') if version_a else None,
        baseline_contract_sha256=sha(args.baseline_contract) if args.baseline_contract else None))
    pending_demo = None
    pending_features = None
    predictions = 0
    last_commands = None
    while True:
        try: meta, arrays = receive(stdin)
        except EOFError: break
        started = time.perf_counter()
        try:
            if meta['kind'] == 'close':
                send(stdout, dict(closed=True)); break
            if meta['kind'] == 'reset':
                assert meta['condition'] == 'correct'
                torch.manual_seed(int(meta['seed'])); torch.cuda.manual_seed_all(int(meta['seed']))
                actor.clear_cache(); predictions=0; last_commands=None
                row=demos[meta['demo']];path=args.eval_data/'records'/row['id']/'demo_rgb.npz'
                assert sha(path)==row['demo_sha256']
                with np.load(path,allow_pickle=False) as d: frames=d['rgb'].copy()
                assert frames.dtype==np.uint8 and frames.shape[1:]==(224,224,3)
                feature_ids = None
                if version_a:
                    feature_ids = ((np.arange(FRAMES)+.5)*len(frames)/FRAMES).astype(int)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pending_features = encode(dino,torch.from_numpy(frames[feature_ids]))
                    assert pending_features.shape==(1,32,780) and torch.isfinite(pending_features).all()
                plan=sample_plan(len(frames),50.,'human',{'video':{'human':{'target_fps':12.}}})
                ids=plan['sampled_frame_indices']
                cache=args.cache/(row['demo_sha256']+'.pt');cache.parent.mkdir(parents=True,exist_ok=True)
                if cache.exists(): latent=torch.load(cache,map_location='cpu',weights_only=True)
                else:
                    # Same causal 1+16-frame chunks, resize, and moments as supplied pilot.encode.
                    vae=actor.vae;wrapper=actor.streaming_vae;wrapper.clear_cache();chunks=[ids[:1],*[ids[i:i+16] for i in range(1,len(ids),16)]];latents=[]
                    mean=torch.tensor(vae.config.latents_mean,device=actor.device).view(1,-1,1,1,1)
                    std=torch.tensor(vae.config.latents_std,device=actor.device).view(1,-1,1,1,1)
                    with torch.inference_mode():
                        for chunk in chunks:
                            video=torch.from_numpy(frames[chunk]).permute(0,3,1,2).float()
                            video=F.interpolate(video,size=(320,480),mode='bilinear',align_corners=False)
                            video=(video/127.5-1).permute(1,0,2,3)[None].to(actor.device,actor.dtype)
                            mu=wrapper.encode_chunk(video).chunk(2,dim=1)[0].float()
                            latents.append(((mu-mean)/std)[0].to(actor.dtype).cpu())
                    latent=torch.cat(latents,dim=1).contiguous();torch.save(latent,cache)
                assert latent.shape[1]==len(plan['latent_anchor_indices']) and torch.isfinite(latent).all()
                pending_demo=latent[None]; actor.preview_human_latent=pending_demo
                send(stdout,dict(reset=True,demo=row['id'],source_record_sha256=row['recording_sha256'],
                    input_indices=ids.tolist(),latent_frames=latent.shape[1],head_rgb_only=True,
                    demo_attention_indices=feature_ids.tolist() if feature_ids is not None else None,
                    demo_attention_tokens=32 if version_a else 0))
                continue
            if meta['kind']=='executed':
                assert last_commands is not None and np.array_equal(arrays['actions'],last_commands)
                actor.accept_executed(arrays['actions'],arrays['frames'])
                send(stdout,dict(executed=True,actions=len(last_commands),observed_frames=len(arrays['frames']),
                    query_views=query_views,feedback_sha256=hashlib.sha256(arrays['frames'].tobytes()).hexdigest(),
                    total_executed_steps=actor.total_executed_steps));last_commands=None;continue
            assert meta['kind']=='predict' and pending_demo is not None and last_commands is None
            rgb=arrays['rgb'];state=arrays['history']
            assert rgb.shape==(2,224,224,3) and state.shape==(1,10)
            observed = rgb[:1] if query_views == 1 else rgb
            obs=dict(rgb=torch.from_numpy(observed.copy()).float().div_(255)[None],state=torch.from_numpy(state.copy()))
            if predictions==0:
                actor.prepare_for_eval(torch.empty(1),obs)
                if version_a:
                    actor.demo_features=pending_features.to(device=actor.device,dtype=actor.dtype)
            with torch.inference_mode():
                actions=actor.roboicl_space.denormalize(actor.get_action(obs),'action')[0].float().cpu().numpy()
            expected_len=16 if predictions==0 else 32
            if actions.shape!=(expected_len,10) or not np.isfinite(actions).all():raise FloatingPointError('Invalid policy action chunk')
            predictions+=1;last_commands=actions.copy();torch.cuda.synchronize()
            send(stdout,dict(predicted=True,seconds=time.perf_counter()-started,chunk=predictions,
                committed_history_actions=actor.total_executed_steps,history_latent_frames=actor.history_frame_offset),actions=actions)
        except Exception as e:
            traceback.print_exc();send(stdout,dict(error=repr(e),error_kind='policy_output' if isinstance(e,FloatingPointError) else 'infrastructure'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['ig-source','model','checkpoint','stats','empty-text','eval-data','cache']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    for name in ['version-a-source','baseline-contract','dino','adapter-metadata']:
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    stdin=sys.stdin.buffer;stdout=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    with redirect_stdout(sys.stderr):main(args,stdin,stdout)
