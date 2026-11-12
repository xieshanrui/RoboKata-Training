"""Recorded-pair and native RoboTwin inference using predicted dual-arm waypoints."""
from pathlib import Path
import cv2
import numpy as np
import torch
from act_robotwin.data import rgb_at, decoder
from act_robotwin.protocol import read, EVAL_TASKS
from .data import normalize
from .model import build,decode_pose

class Policy:
    def __init__(self,checkpoint,official,device='cuda',allow_smoke=False):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['smoke_only'] and not allow_smoke:raise ValueError('Smoke checkpoint cannot supply a benchmark score')
        self.device=device;self.model=build(official).to(device).eval()
        self.model.load_state_dict(saved['model'],strict=True);self.context=None

    def reset(self,video):
        ids=np.linspace(0,len(decoder(str(video)))-1,10).astype(int)
        self.context=normalize(rgb_at(video,ids))[None].to(self.device)

    @torch.inference_mode()
    def predict(self,rgb):
        assert self.context is not None
        frame=cv2.resize(np.asarray(rgb),(224,224),interpolation=cv2.INTER_AREA)
        # Repeat the current observation only; no future Query frames at inference.
        images=normalize(np.repeat(frame[None],6,axis=0))[None].to(self.device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
            out=self.model(images,self.context)
        return decode_pose(out['waypoints'][0,-5:].float().cpu().numpy())

def interpolate(current,target,steps=12):
    """Bounded native EEF execution, quaternion sign-aware interpolation."""
    current=np.asarray(current,dtype=np.float32);target=np.asarray(target,dtype=np.float32).copy()
    for i in (3,11):
        if np.dot(current[i:i+4],target[i:i+4])<0:target[i:i+4]*=-1
    for t in np.linspace(0,1,steps+1)[1:]:
        pose=current*(1-t)+target*t
        for i in (3,11):pose[i:i+4]/=max(np.linalg.norm(pose[i:i+4]),1e-8)
        pose[[7,15]]=target[[7,15]]
        yield pose

def get_model(args):
    task=args['task_name'];assert task in EVAL_TASKS
    model=Policy(args['osvi_checkpoint'],args['osvi_official'])
    mapping=read(args['osvi_demo_map']);assert set(mapping)==set(EVAL_TASKS)
    model.demo_video=Path(args['osvi_demo_root'])/mapping[task]
    return model

def reset_model(model):model.reset(model.demo_video)

def eval(TASK_ENV,model,observation):
    if model.context is None:reset_model(model)
    pose=observation['endpose']
    current=np.r_[pose['left_endpose'],pose['left_gripper'],pose['right_endpose'],pose['right_gripper']]
    predictions=model.predict(observation['observation']['head_camera']['rgb'])
    for target in predictions:
        for command in interpolate(current,target):
            if TASK_ENV.eval_success or TASK_ENV.take_action_cnt>=TASK_ENV.step_lim:return
            TASK_ENV.take_action(command,action_type='ee')
        current=target
