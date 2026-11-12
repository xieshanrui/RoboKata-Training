"""Non-actuating inference on a recorded pair; never a task-success evaluation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .protocol import read,write,sha,video_paths
from .data import episode,rgb_at
from .policy import Policy

def main():
    p=argparse.ArgumentParser()
    for k in ['data','prepared','checkpoint','ig-root','dino-path','output']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pair-index',type=int,default=0)
    p.add_argument('--frame',type=int,default=0)
    p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    prepared=read(a.prepared);row=prepared['rows'][a.pair_index]
    for name in [row['robot_data'],row['human_video'],*video_paths(row)]:
        assert sha(a.data/name)==prepared['files'][name],name
    _,state=episode(a.data,row)
    images=[rgb_at(a.data/n,[a.frame])[0] for n in video_paths(row)]
    model=Policy(a.checkpoint,a.ig_root,a.dino_path,a.device,a.allow_smoke)
    provenance=model.reset(a.data/row['human_video'])
    actions=model.predict(images,state[a.frame])
    assert np.isfinite(actions).all() and actions.shape==(24,16)
    for start in [3,11]:assert np.allclose(np.linalg.norm(actions[:,start:start+4],axis=1),1)
    assert np.all((actions[:,[7,15]]>=0)&(actions[:,[7,15]]<=1))
    write(a.output,{'pair_id':row['pair_id'],'frame':a.frame,'checkpoint_sha256':sha(a.checkpoint),
                    'demo':provenance,'actions':actions.tolist(),'action_type':'ee',
                    'non_actuating':True,'task_success_evaluated':False,
                    'smoke_only':torch.load(a.checkpoint,map_location='cpu',weights_only=True)['smoke_only']})
    print('Finite recorded-pair inference:',list(actions.shape),flush=True)

if __name__=='__main__':main()
