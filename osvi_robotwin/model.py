"""Keep the official latent world model; extend only the waypoint output."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange

REVISION = 'ebd1fa7131078d10dbb101c8f0e3c9b3b86edb81'

def build(official):
    sys.path.insert(0, str(Path(official).resolve()))
    from models.model import StateSpaceModel
    model = StateSpaceModel(latent_dim=256, waypoints=5, sub_waypoints=True, metaworld=True)
    # 1+2+3+4+5 waypoint supervision, two native xyz/wxyz/gripper poses.
    model.waypoint_head = nn.Sequential(nn.Dropout(.2), nn.ReLU(),
        nn.Linear(256*5, 15*16), Rearrange('... (w d) -> ... w d', d=16))
    return model

def encode_pose(pose):
    value = np.array(pose, dtype=np.float32, copy=True)
    for start in (3,11):
        q = value[..., start:start+4]
        q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        # Canonicalize along time to avoid sign discontinuities in path supervision.
        if q.ndim == 2:
            if q[0,0] < 0: q[0] *= -1
            for i in range(1,len(q)):
                if np.dot(q[i-1], q[i]) < 0: q[i] *= -1
        value[..., start:start+4] = q*.2
    value[..., [7,15]] = (1-value[..., [7,15]])*.2
    return value

def decode_pose(prediction):
    value = np.array(prediction, dtype=np.float32, copy=True)
    for start in (3,11):
        q = value[...,start:start+4]
        norm = np.linalg.norm(q,axis=-1,keepdims=True)
        if np.any(norm<1e-6): raise ValueError('Degenerate predicted quaternion')
        value[...,start:start+4] = q/norm
    value[...,[7,15]] = 1-(value[...,[7,15]]>.1).astype(np.float32)
    if not np.isfinite(value).all(): raise ValueError('Nonfinite prediction')
    return value

def trajectory_loss(pred, target):
    from losses.loss import sdtw_trajectory_loss
    # Reuse upstream interpolation and SoftDTW per arm: each arm's gripper is last.
    losses = [sdtw_trajectory_loss(pred[...,i:i+8].float(),target[...,i:i+8].float(),
              None,60,1e-3,5,image_waypoints=False)[0] for i in (0,8)]
    return sum(losses)/2
