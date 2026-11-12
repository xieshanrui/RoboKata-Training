"""IG rollout interface for the native-joint OSVI adapter."""
import numpy as np
import torch
from .policy import Policy as NativePolicy, interpolate


class Policy:
    def __init__(self, settings, output):
        self.model = NativePolicy(settings['checkpoint'], settings['official_source'])
        self.clips = torch.load(settings['demo_clips'], map_location='cpu', weights_only=True)
        self.clear_cache()

    def eval(self): return self
    def train(self): return self
    def clear_cache(self): pass
    def record_executed_action(self, action): pass
    def observe_step(self, obs): pass

    def select_case(self, case):
        self.model.set_demo_clip(self.clips[case['human_record_sha256']].numpy())

    def prepare_for_eval(self, human_video, robot_obs, human_desc): pass

    def get_action(self, obs):
        rgb = obs['rgb']; state = obs['state'].reshape(1, 18)[0].float().cpu().numpy()
        if rgb.ndim == 5: rgb = rgb[:, -1]
        frame = rgb[0].float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        targets = self.model.predict(frame)
        current = np.r_[state[:7], targets[0, 7], state[9:16], targets[0, 15]]
        actions = []
        for target in targets:
            actions.extend(interpolate(current, target, 12)); current = target
        result = np.stack(actions)
        assert result.shape == (60, 16) and np.isfinite(result).all()
        return torch.as_tensor(result, device='cuda')[None]
