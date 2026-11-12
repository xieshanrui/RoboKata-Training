"""Atomic sharded recovery, including optimizer, sampler cursor and rank RNG."""
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def rng_state():
    n = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                python=random.getstate(), numpy=(n[0], n[1].tolist(), *n[2:]))


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))


def state(trainer):
    model, optimizer = get_state_dict(trainer.transformer, trainer.optimizer,
                                      options=StateDictOptions(cpu_offload=True))
    return dict(model=model, optimizer=optimizer)


def digest(value):
    """Hash every local model/Adam tensor and metadata without gathering ranks."""
    h = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            if hasattr(obj, 'to_local'):
                obj = obj.to_local()
            obj = obj.detach().cpu().contiguous()
            assert torch.isfinite(obj).all(), 'Nonfinite checkpoint tensor'
            h.update(str((obj.dtype, tuple(obj.shape))).encode())
            h.update(memoryview(obj.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                h.update(str(key).encode()); visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        else:
            h.update(repr(obj).encode())
    visit(value)
    return h.hexdigest()


def save(trainer, storage, identity, cursor, verify=False):
    storage = Path(storage)
    rank = dist.get_rank()
    parent = storage / 'recovery'
    pending = parent / f'step_{trainer.step:06d}'
    previous = None
    if rank == 0:
        assert shutil.disk_usage(storage).free > 75 * 2**30, 'Recovery needs 75GiB free'
        parent.mkdir(exist_ok=True)
        assert not pending.exists(), pending
        pending.mkdir()
    dist.barrier()
    rng = rng_state()
    payload = state(trainer)
    fingerprint = digest(payload) if verify else None
    dcp.save(payload, checkpoint_id=pending/'distributed')
    torch.save(dict(identity=identity, step=trainer.step, cursor=cursor,
                    scheduler=trainer.lr_scheduler.state_dict(), rng=rng,
                    fingerprint=fingerprint), pending/f'rank{rank}.pt')
    del payload
    dist.barrier()
    if rank == 0:
        assert all((pending/f'rank{r}.pt').is_file() for r in range(dist.get_world_size()))
        write(pending/'complete.json', dict(step=trainer.step, cursor=cursor, identity=identity))
        pointer = storage/'latest.json'
        if pointer.exists():
            previous = parent/json.loads(pointer.read_text(encoding='utf-8'))['directory']
        write(pointer, dict(directory=pending.name, step=trainer.step, cursor=cursor))
        if previous is not None and previous != pending:
            assert previous.resolve().parent == parent.resolve() and previous.is_dir()
            shutil.rmtree(previous)
    dist.barrier()
    restore_rng(rng)


def load(trainer, storage, identity):
    storage = Path(storage)
    directory = json.loads((storage/'latest.json').read_text(encoding='utf-8'))['directory']
    folder = storage/'recovery'/directory
    assert folder.resolve().parent == (storage/'recovery').resolve()
    assert (folder/'complete.json').is_file()
    saved = torch.load(folder/f'rank{dist.get_rank()}.pt', map_location='cpu', weights_only=True)
    assert saved['identity'] == identity, 'Changed source/config/data at resume'
    payload = state(trainer)
    # Adam only creates slots for parameters that received gradients. Fresh
    # state_dict helpers may initialize extra slots for inactive native branches.
    metadata = dcp.FileSystemReader(folder/'distributed').read_metadata()
    active = {path[2] for path in metadata.planner_data.values()
              if len(path) >= 3 and tuple(path[:2]) == ('optimizer', 'state')}
    assert active and active <= payload['optimizer']['state'].keys()
    payload['optimizer']['state'] = {k:v for k,v in payload['optimizer']['state'].items() if k in active}
    dcp.load(payload, checkpoint_id=folder/'distributed')
    # PyTorch 2.9's canonical-name setter requires an entry for every parameter,
    # even when Adam never created slots. Empty entries are only loader scaffolding.
    groups = payload['optimizer']['param_groups']
    for group in groups:
        for name in group['params']:
            payload['optimizer']['state'].setdefault(name, {})
    set_state_dict(trainer.transformer, trainer.optimizer,
                   model_state_dict=payload['model'], optim_state_dict=payload['optimizer'])
    for group, saved_group in zip(trainer.optimizer.param_groups, groups):
        for parameter, name in zip(group['params'], saved_group['params']):
            if name not in active:
                trainer.optimizer.state.pop(parameter, None)
    del payload
    trainer.lr_scheduler.load_state_dict(saved['scheduler'])
    trainer.step = saved['step']
    if saved['fingerprint'] is not None:
        payload = state(trainer)
        assert digest(payload) == saved['fingerprint'], 'Model/optimizer readback mismatch'
        del payload
    dist.barrier()
    restore_rng(saved['rng'])
    return saved
