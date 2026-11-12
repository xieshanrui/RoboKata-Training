"""Production C15 text fine-tuning through the native LingBot trainer."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import types

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from data import IGTextDataset
from batching import collate, install_trainer
from checkpoint import write, save, load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    root, storage = Path(settings['run_root']), Path(settings['storage'])
    rank, local, world = [int(os.environ[k]) for k in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']]
    assert world == settings['world_size'] == 8 and settings['batch_per_gpu'] == 8
    assert settings['epochs'] == 60 and settings['updates'] == 2820
    logs = root/'train'; logs.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(settings['seed']+rank)
    random.seed(settings['seed']+rank)
    np.random.seed(settings['seed']+rank)
    torch.cuda.set_device(local)
    sys.path.insert(0, str(Path(settings['official_source'])/'wan_va'))
    dataset_module = types.ModuleType('dataset')
    dataset_module.MultiLatentLeRobotDataset = lambda config: IGTextDataset(
        settings['manifest'], settings['statistics'], settings['text_cache'],
        settings['action_space_source'], cfg_prob=config.cfg_prob)
    sys.modules['dataset'] = dataset_module
    spec = importlib.util.spec_from_file_location('native_lingbot_train', Path(settings['official_source'])/'wan_va/train.py')
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    install_trainer(native)
    from configs.va_robotwin_train_cfg import va_robotwin_train_cfg
    native.init_distributed(world, local, rank)
    config = copy.deepcopy(va_robotwin_train_cfg)
    config.update(rank=rank, local_rank=local, world_size=world,
                  wan22_pretrained_model_name_or_path=settings['model'],
                  save_root=str(storage/'exports'), enable_wandb=False, load_worker=0,
                  batch_size=8, gradient_accumulation_steps=1, num_steps=2820,
                  empty_emb_path=settings['text_cache'])
    assert config.learning_rate == 1e-5 and config.warmup_steps == 10
    trainer = native.Trainer(config)
    wrappers = sum('CheckpointWrapper' in type(m).__name__ for m in trainer.transformer.modules())
    assert wrappers == 30
    dataset = trainer.train_loader.dataset
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=42, drop_last=False)
    assert len(dataset) == 3000 and len(sampler) == 375
    identity = dict(config_sha256=sha(args.config),
                    source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                    upstream={str(p.relative_to(settings['official_source'])):sha(p)
                              for p in Path(settings['official_source']).rglob('*.py')},
                    inputs={k:sha(settings[k]) for k in ['manifest','statistics','text_cache','action_space_source']})
    cursor = dict(epoch=0, next_batch=0)
    if args.resume:
        restored = load(trainer, storage, identity)
        cursor = restored['cursor']
        sampler.set_epoch(cursor['epoch'])
        next_ids = list(sampler)[cursor['next_batch']*8:(cursor['next_batch']+1)*8]
        write(logs/f'resume_rank{rank}.json', dict(step=trainer.step, cursor=cursor,
              next_indices=next_ids, learning_rate=trainer.lr_scheduler.get_last_lr()[0],
              state_readback_exact=restored['fingerprint'] is not None,
              rng_restored=True, optimizer_restored=True, time=time.time()))
    else:
        assert not (storage/'latest.json').exists()
    write(logs/f'ready_rank{rank}.json', dict(rank=rank, step=trainer.step,
          wrappers=wrappers, batch_per_gpu=8, global_batch=64, updates=2820,
          epochs=60, selection='final epoch60', identity=identity))
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    started = time.monotonic()
    while trainer.step < settings['updates']:
        epoch = cursor['epoch']
        sampler.set_epoch(epoch)
        order = list(sampler)
        for batch_index in range(cursor['next_batch'], 47):
            tick = time.monotonic()
            indices = order[batch_index*8:(batch_index+1)*8]
            assert len(indices) == (7 if batch_index == 46 else 8)
            batch = collate([dataset[i] for i in indices])
            used_lr = trainer.optimizer.param_groups[0]['lr']
            losses = trainer._train_step(batch, trainer.step)
            norm = losses['total_norm']
            if hasattr(norm, 'full_tensor'): norm = norm.full_tensor()
            torch.cuda.synchronize()
            values = torch.tensor([float(losses['latent_loss']), float(losses['action_loss']),
                                   float(norm), time.monotonic()-tick,
                                   torch.cuda.max_memory_allocated()/2**30], device=trainer.device)
            valid = torch.tensor(int(torch.isfinite(values).all()), device=trainer.device)
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            assert int(valid) == 1, 'Nonfinite update on a rank'
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            trainer.step += 1
            cursor = dict(epoch=epoch+1, next_batch=0) if batch_index == 46 else dict(epoch=epoch, next_batch=batch_index+1)
            row = dict(step=trainer.step, total_updates=2820, epoch=epoch, batch_index=batch_index,
                       rank=rank, indices=indices, batch_per_gpu=len(indices), global_batch=len(indices)*8,
                       video_loss=float(losses['latent_loss']), action_loss=float(losses['action_loss']),
                       grad_norm=float(norm), learning_rate=used_lr, seconds=float(values[3]),
                       peak_allocated_gib=float(values[4]), time=time.time())
            with (logs/f'metrics_rank{rank}.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row)+'\n')
            if rank == 0:
                write(logs/'latest_metrics.json', row); print(json.dumps(row), flush=True)
            stop = torch.tensor(int(stopping[0]), device=trainer.device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            end_phase = bool(stop) or (args.stop_after is not None and trainer.step >= args.stop_after)
            boundary = trainer.step in [940,1880,2820]
            if trainer.step in [2,20] or trainer.step % 100 == 0 or boundary or end_phase:
                save(trainer, storage, identity, cursor, verify=trainer.step == 2)
                if rank == 0:
                    write(logs/'checkpoint_status.json', dict(step=trainer.step,cursor=cursor,storage=str(storage),time=time.time()))
            if boundary:
                trainer.save_checkpoint()
                if rank == 0:
                    export = storage/'exports/checkpoints'/f'checkpoint_step_{trainer.step}'/'transformer'
                    assert (export/'diffusion_pytorch_model.safetensors').is_file()
                    write(export.parent/'training_contract.json', dict(identity=identity,epoch=cursor['epoch'],step=trainer.step))
                dist.barrier()
            if end_phase:
                write(logs/f'phase_complete_rank{rank}.json', dict(step=trainer.step,cursor=cursor,wall_seconds=time.monotonic()-started))
                dist.barrier(); dist.destroy_process_group(); return
        assert cursor['next_batch'] == 0
    write(logs/f'complete_rank{rank}.json',dict(completed=True,step=trainer.step,epochs=60,wall_seconds=time.monotonic()-started))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        error = traceback.format_exc()
        try:
            settings = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text(encoding='utf-8'))
            root = Path(settings['run_root'])
            root.mkdir(parents=True, exist_ok=True)
            write(root/f'failure_rank{os.getenv("RANK","unknown")}.json',
                  dict(error=error, time=time.time()))
        except Exception:
            pass  # Failure reporting must not replace the original exception.
        raise
