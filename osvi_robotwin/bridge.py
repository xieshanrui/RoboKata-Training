"""Finite file bridge for a separate SAPIEN process; at most two predictions."""
import argparse,time
from pathlib import Path
import numpy as np
from act_robotwin.protocol import write
from .policy import Policy

def main():
    p=argparse.ArgumentParser()
    for key in ['checkpoint','official','demo','exchange']:p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.exchange);root.mkdir(parents=True,exist_ok=True)
    model=Policy(a.checkpoint,a.official,allow_smoke=True);model.reset(a.demo)
    write(root/'ready.json',dict(smoke_only=True,max_predictions=2))
    deadline=time.monotonic()+600
    for i in range(2):
        req=root/f'request_{i}.npz'
        while not req.exists():
            if time.monotonic()>deadline:raise TimeoutError('Finite simulator bridge timeout')
            time.sleep(.1)
        with np.load(req,allow_pickle=False) as data:rgb=data['rgb']
        started=time.perf_counter();waypoints=model.predict(rgb)
        tmp=root/f'response_{i}.tmp.npy';np.save(tmp,waypoints);tmp.replace(root/f'response_{i}.npy')
        write(root/f'prediction_{i}.json',dict(seconds=time.perf_counter()-started,shape=list(waypoints.shape)))
    write(root/'complete.json',dict(predictions=2,smoke_only=True))

if __name__=='__main__':main()
