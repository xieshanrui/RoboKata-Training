"""Native joint-action waypoint decoding; no pose conversion or IK."""
import numpy as np
import cv2
import torch
from osvi_robotwin.model import build
from osvi_robotwin.data import normalize

def decode_action(normalized,statistics):
    result=np.asarray(normalized,dtype=np.float32)*np.asarray(statistics['std'],np.float32)+np.asarray(statistics['mean'],np.float32)
    assert result.shape[-1]==16 and np.isfinite(result).all()
    return result

def interpolate(current,target,steps):
    current=np.asarray(current,dtype=np.float32);target=np.asarray(target,dtype=np.float32)
    assert current.shape==target.shape==(16,) and steps>0
    for t in np.linspace(0,1,steps+1)[1:]:
        action=current*(1-t)+target*t;action[[7,15]]=target[[7,15]]
        yield action

class Policy:
    def __init__(self,checkpoint,official,device='cuda',allow_smoke=False):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert allow_smoke or not saved['smoke_only']
        assert saved['config']['action_representation']=='native_joint'
        self.statistics=saved['config']['normalization'];self.device=device
        self.model=build(official).to(device).eval();self.model.load_state_dict(saved['model']);self.context=None

    def set_demo_clip(self,rgb):
        assert len(rgb)==10
        self.context=normalize(rgb)[None].to(self.device)

    @torch.inference_mode()
    def predict(self,rgb):
        assert self.context is not None
        frame=cv2.resize(np.asarray(rgb),(224,224),interpolation=cv2.INTER_AREA)
        images=normalize(np.repeat(frame[None],6,axis=0))[None].to(self.device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
            out=self.model(images,self.context)['waypoints'][0,-5:].float().cpu().numpy()
        return decode_action(out,self.statistics)
