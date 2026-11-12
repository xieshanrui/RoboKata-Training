"""The shared RGB-only Demo reader used for training caches and inference."""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


def read_rgb_demo_npz(policy, path):
    with np.load(path, allow_pickle=False) as item:
        frames, fps = item['rgb'], float(item['fps'])
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError('Demo must contain only a single RGB video')
    cfg = policy.job_config
    indices = policy._sample_icl_frame_indices(len(frames), fps, cfg.icl_fps)
    frames = frames[indices]
    permutation = np.arange(len(frames))
    if policy.demo_control == 'shuffled':
        permutation = np.random.default_rng(policy.control_seed).permutation(len(frames))
        frames = frames[permutation]
    elif policy.demo_control == 'last_frame':
        frames = np.repeat(frames[-1:], len(frames), axis=0)
    elif policy.demo_control == 'masked':
        frames = np.zeros_like(frames)
    elif policy.demo_control != 'correct':
        raise ValueError(policy.demo_control)
    policy.demo_sampling = dict(indices=indices.tolist(), permutation=permutation.tolist(),
        control=policy.demo_control, source_fps=fps, sampled_frames=len(frames),
        rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
    tensor = torch.from_numpy(frames.copy()).permute(0,3,1,2).float()
    tensor = F.interpolate(tensor, size=(cfg.icl_height,cfg.icl_width), mode='bilinear',align_corners=False)
    return tensor.permute(1,0,2,3).unsqueeze(0).contiguous()
