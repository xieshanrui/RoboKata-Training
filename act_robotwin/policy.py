"""Native RoboTwin deployment hooks with RGB Demo and absolute EEF actions."""
from pathlib import Path
import cv2
import numpy as np
import torch
from .model import ACT,demo_encoder
from .data import demo_rgb
from .protocol import read,sha,EVAL_TASKS,DINO_REVISION,verify_local_dino

class Policy:
    def __init__(self,checkpoint,ig_root,dino_path,device='cuda',allow_smoke=False):
        verify_local_dino(dino_path)
        self.device=device;c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if c['smoke_only'] and not allow_smoke:raise ValueError('Smoke checkpoint is not an evaluation model')
        self.model=ACT(ig_root,c['config']).to(device).eval();self.model.load_state_dict(c['model'],strict=True)
        self.stats=c['stats'];self.encoder=demo_encoder(ig_root).to(device).eval()
        self.encoder._meta=dict(self.encoder._meta,hf_model=str(dino_path))
        self.encoder._load_backbone();self.encoder._backbone.to(device);self.demo_cls=None

    @torch.no_grad()
    def reset(self,video):
        self.demo_cls=self.encoder._encode_dino(demo_rgb(video).unsqueeze(0))[0]
        assert self.demo_cls.shape==(1,1024)
        return {'demo_sha256':sha(video),'encoder_revision':DINO_REVISION}

    @torch.no_grad()
    def predict(self,images,state):
        if self.demo_cls is None:raise RuntimeError('Set the episode Demo before inference')
        rgb=np.stack([cv2.resize(x,(224,224),interpolation=cv2.INTER_AREA) for x in images])
        state=np.asarray(state,dtype=np.float32);assert state.shape==(16,)
        norm=(state-np.asarray(self.stats['state']['mean']))/np.asarray(self.stats['state']['std'])
        batch={'rgb':torch.from_numpy(rgb).permute(0,3,1,2).unsqueeze(0).float().to(self.device)/255,
               'state':torch.as_tensor(norm,device=self.device,dtype=torch.float32)[None],
               'demo_cls':self.demo_cls}
        pred=self.model(batch)[0][0].cpu().numpy()
        action=pred*np.asarray(self.stats['action']['std'])+np.asarray(self.stats['action']['mean'])
        assert action.shape==(24,16) and np.isfinite(action).all()
        # Native source quaternion ordering is retained, without scipy reordering.
        for start in [3,11]:
            norm=np.linalg.norm(action[:,start:start+4],axis=1,keepdims=True)
            if np.any(norm<1e-6):raise ValueError('Degenerate predicted quaternion')
            action[:,start:start+4]/=norm
        action[:,[7,15]]=np.clip(action[:,[7,15]],0,1)
        return action

def get_model(usr_args):
    task=usr_args['task_name']
    if task not in EVAL_TASKS:raise ValueError('Only the frozen six held-out tasks are allowed')
    model=Policy(usr_args['act_checkpoint'],usr_args['ig_root'],usr_args['dino_path'])
    mapping=read(usr_args['act_demo_map']);assert set(mapping)==set(EVAL_TASKS)
    model.demo_video=Path(usr_args['act_demo_root'])/mapping[task]
    return model

def reset_model(model):model.reset(model.demo_video)

def eval(TASK_ENV,model,observation):
    if model.demo_cls is None:reset_model(model)
    camera=observation['observation'];pose=observation['endpose']
    images=[camera[k]['rgb'] for k in ['head_camera','left_camera','right_camera']]
    state=np.r_[pose['left_endpose'],pose['left_gripper'],pose['right_endpose'],pose['right_gripper']]
    actions=model.predict(images,state)
    for action in actions:
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt>=TASK_ENV.step_lim:break
        TASK_ENV.take_action(action,action_type='ee')
