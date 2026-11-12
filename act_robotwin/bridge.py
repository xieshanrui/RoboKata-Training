"""Bounded inference bridge for an isolated RoboTwin simulator environment."""
import argparse
import time
from pathlib import Path

import numpy as np

from .policy import Policy
from .protocol import read, write, sha


def main():
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'ig-root', 'dino-path', 'exchange', 'demo-map', 'demo-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    args = parser.parse_args()
    root = args.exchange
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wall_seconds
    model = Policy(args.checkpoint, args.ig_root, args.dino_path)
    demos = read(args.demo_map)
    write(root / 'ready.json', {'checkpoint_sha256': sha(args.checkpoint)})
    request_id = 0
    while time.monotonic() < deadline and not (root / 'STOP').exists():
        request = root / 'request.json'
        if not request.exists():
            time.sleep(.01)
            continue
        command = read(request)
        assert isinstance(command['id'], str) and command['id']
        started = time.perf_counter()
        if command['kind'] == 'reset':
            result = model.reset(args.demo_root / demos[command['task']])
        elif command['kind'] == 'predict':
            with np.load(root / 'input.npz', allow_pickle=False) as data:
                actions = model.predict(list(data['images']), data['state'])
            temporary = root / 'output.tmp.npy'
            np.save(temporary, actions)
            temporary.replace(root / 'output.npy')
            result = {'shape': list(actions.shape)}
        else:
            raise ValueError(command['kind'])
        request.unlink()
        temporary = root / 'response.tmp'
        write(temporary, {'id': command['id'], 'seconds': time.perf_counter() - started, **result})
        temporary.replace(root / 'response.json')
        request_id += 1
    write(root / 'stopped.json', {'requests': request_id, 'time': time.time()})


if __name__ == '__main__':
    main()
