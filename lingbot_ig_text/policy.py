"""Native LingBot streaming inference with the frozen IG text/action contract."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch


class Policy:
    def __init__(self, settings, output):
        sys.path.insert(0, str(Path(settings['official_source'])/'wan_va'))
        import wan_va_server as native
        from modules.utils import load_transformer, load_vae, WanVAEStreamingWrapper
        from configs.va_robotwin_cfg import va_robotwin_cfg
        from utils import FlowMatchScheduler
        spec = importlib.util.spec_from_file_location('ig_action_space', settings['action_space_source'])
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.space = module.EpisodeActionSpace(json.loads(Path(settings['statistics']).read_text(encoding='utf-8')))
        self.channels = list(module.channel_ids(14))
        self.text = torch.load(settings['text_cache'], map_location='cpu', weights_only=False)
        self.output = Path(output)
        # Disable only upstream debug tensor dumps; prediction/cache logic stays native.
        native.save_async = lambda *args, **kwargs: None
        server = native.VA_Server.__new__(native.VA_Server)
        server.device = torch.device('cuda'); server.dtype = torch.bfloat16
        server.cache_name = 'ig_text'; server.save_root = str(self.output/'native')
        server.job_config = copy.deepcopy(va_robotwin_cfg)
        server.job_config.update(height=224, width=288, env_type='ig_single_zed2i',
                                 obs_cam_keys=['zed2i'], used_action_channel_ids=self.channels)
        server.env_type = server.job_config.env_type
        server.transformer = load_transformer(settings['checkpoint'], torch.bfloat16, 'cuda', attn_mode='torch')
        server.transformer.eval().requires_grad_(False)
        server.vae = load_vae(str(Path(settings['model'])/'vae'), torch.bfloat16, 'cuda')
        server.vae.eval().requires_grad_(False)
        server.streaming_vae = WanVAEStreamingWrapper(server.vae)
        server.scheduler = FlowMatchScheduler(shift=server.job_config.snr_shift, sigma_min=0., extra_one_step=True)
        server.action_scheduler = FlowMatchScheduler(shift=server.job_config.action_snr_shift, sigma_min=0., extra_one_step=True)
        server.postprocess_action = MethodType(lambda this, action: action.detach(), server)
        server.preprocess_action = MethodType(lambda this, action: action, server)
        self.server = server
        self.clear_cache()

    def eval(self): return self
    def train(self): return self

    def select_case(self, case):
        self.human_id = case['human_id']

    def clear_cache(self):
        self.frames = []; self.executed = []; self.chunk = 0; self.origin = None

    @staticmethod
    def image(value):
        x = torch.as_tensor(value).detach()
        if x.ndim == 5: x = x[:, -1]
        assert x.ndim == 4 and x.shape[0] == 1 and x.shape[-1] == 3
        return x[0].float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()

    def prepare_for_eval(self, human_video, robot_obs, human_desc):
        self.clear_cache()
        state = torch.as_tensor(robot_obs['states']).float().reshape(-1, 18)
        assert state.shape == (1, 18)
        self.origin = state[0].clone()
        self.initial_rgb = self.image(robot_obs['view_1'])
        self.server._reset(prompt=None)
        self.server.prompt_embeds = self.text['embeddings'][self.human_id][None].cuda()
        self.server.negative_prompt_embeds = self.text['empty'][None].cuda()

    def record_executed_action(self, action):
        raw = action.detach().float().reshape(1, 16).clone()
        raw[:, [7, 15]] = raw[:, [7, 15]].clamp(-1, 1)
        if self.space.relative:
            raw[:, [*range(7), *range(8, 15)]] -= self.origin[[*range(7), *range(9, 16)]].to(raw)
        lo, hi = self.space.bounds(raw, 'action')
        self.executed.append((2*(raw-lo)/(hi-lo+1e-6)-1)[0])

    def observe_step(self, obs):
        if self.executed and len(self.executed) % 4 == 0:
            self.frames.append({'zed2i': self.image(obs['rgb'])})

    @torch.inference_mode()
    def get_action(self, obs):
        server = self.server
        if self.chunk:
            expected = 16 if self.chunk == 1 else 32
            assert len(self.executed) == expected and len(self.frames) == expected//4
            packed = torch.zeros(1, 30, expected//16, 16, 1, device='cuda', dtype=torch.bfloat16)
            values = torch.stack(self.executed).reshape(-1, 16, 16).permute(2, 0, 1)
            packed[0, self.channels, :, :, 0] = values.to(packed)
            if self.chunk == 1:
                packed = torch.cat([torch.zeros_like(packed), packed], dim=2)
            server._compute_kv_cache({'obs': self.frames, 'state': packed})
            assert server.frame_st_id == self.chunk*2
            self.frames.clear(); self.executed.clear()
        normalized, latents = server._infer({'obs': [{'zed2i': self.initial_rgb}]}, frame_st_id=server.frame_st_id)
        start = 1 if self.chunk == 0 else 0
        values = normalized[0, self.channels, start:, :, 0].permute(1, 2, 0).flatten(0, 1).float()
        result = self.space.decode_action(values, self.origin)
        assert result.shape == (16 if self.chunk == 0 else 32, 16)
        assert torch.isfinite(result).all() and torch.isfinite(latents).all()
        self.chunk += 1
        return result[None]
