"""Formal bounded bridge, retaining the existing OSVI native EEF controller."""
import argparse
import time
from pathlib import Path
import numpy as np
from act_robotwin.protocol import read, write, sha
from .policy import Policy, interpolate


def main():
    p = argparse.ArgumentParser()
    for key in ['checkpoint', 'official', 'demo-map', 'demo-root', 'exchange']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--wall-seconds', type=int, required=True)
    a = p.parse_args(); root = a.exchange; root.mkdir(parents=True, exist_ok=True)
    model = Policy(a.checkpoint, a.official); mapping = read(a.demo_map)
    deadline = time.monotonic()+a.wall_seconds
    write(root/'ready.json', dict(checkpoint_sha256=sha(a.checkpoint)))
    while time.monotonic()<deadline and not (root/'STOP').exists():
        req = root/'request.json'
        if not req.exists(): time.sleep(.01); continue
        message = read(req); started = time.perf_counter()
        if message['kind']=='reset':
            model.reset(a.demo_root/mapping[message['task']]); result = {}
        elif message['kind']=='predict':
            with np.load(root/'input.npz', allow_pickle=False) as data:
                current = data['state']; targets = model.predict(data['images'][0])
            actions = []
            for target in targets:
                actions.extend(interpolate(current, target)); current = target
            actions = np.stack(actions); assert actions.shape==(60,16) and np.isfinite(actions).all()
            tmp = root/'output.tmp.npy'; np.save(tmp, actions); tmp.replace(root/'output.npy')
            result = dict(shape=list(actions.shape))
        else: raise ValueError(message['kind'])
        req.unlink(); tmp = root/'response.tmp'
        write(tmp, dict(id=message['id'], seconds=time.perf_counter()-started, **result)); tmp.replace(root/'response.json')


if __name__=='__main__': main()
