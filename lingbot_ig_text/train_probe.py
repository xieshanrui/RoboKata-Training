"""Bounded full-finetuning probe, reusing native LingBot training and FSDP.

The resulting checkpoint is smoke-only, not a trained benchmark submission.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from data import IGTextDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(settings["run_root"])
    rank, local, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == settings.get('world_size', 4)
    batch_size = settings.get('batch_per_gpu', 1)
    mixed_batch = settings.get('mixed_batch', False)
    torch.set_num_threads(4)
    torch.manual_seed(20261110 + rank)
    torch.cuda.set_device(local)
    sys.path.insert(0, str(Path(settings["official_source"]) / "wan_va"))
    # Inject the dataset dependency without changing upstream training/model code.
    dataset_module = types.ModuleType("dataset")
    dataset_module.MultiLatentLeRobotDataset = lambda config: IGTextDataset(
        settings["manifest"], settings["statistics"], settings["text_cache"],
        settings["action_space_source"], cfg_prob=config.cfg_prob)
    sys.modules["dataset"] = dataset_module
    spec = importlib.util.spec_from_file_location("native_lingbot_train", Path(settings["official_source"]) / "wan_va/train.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if mixed_batch:
        from batching import collate, install_trainer
        install_trainer(native)
    else:
        collate = default_collate
    if not settings.get('activation_checkpointing', True):
        native.apply_ac = lambda model: None
    from configs.va_robotwin_train_cfg import va_robotwin_train_cfg
    native.init_distributed(world, local, rank)
    config = copy.deepcopy(va_robotwin_train_cfg)
    config.update(rank=rank, local_rank=local, world_size=world,
                  wan22_pretrained_model_name_or_path=settings["model"],
                  save_root=str(root / "train"), enable_wandb=False, load_worker=0,
                  batch_size=batch_size, gradient_accumulation_steps=1, num_steps=settings["steps"],
                  empty_emb_path=settings["text_cache"])
    t0 = time.perf_counter()
    trainer = native.Trainer(config)
    dataset = trainer.train_loader.dataset
    trainable = sum(p.numel() for p in trainer.transformer.parameters() if p.requires_grad)
    wrappers = sum("CheckpointWrapper" in type(m).__name__ for m in trainer.transformer.modules())
    result = dict(rank=rank, world_size=world, full_finetuning=True, trainable_parameters=trainable,
                  batch_per_gpu=batch_size, effective_batch=batch_size*world, accumulation=1, checkpoint_wrappers=wrappers,
                  load_seconds=time.perf_counter() - t0, steps=[], smoke_only=True)
    out = root / "train"
    out.mkdir(exist_ok=True)
    (out / f"ready_rank{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    trainer.optimizer.zero_grad(set_to_none=True)
    indices = settings["probe_indices"][rank]
    tracked = next(p for n, p in trainer.transformer.named_parameters() if "action_proj_out.weight" in n)
    before = tracked.to_local().detach().clone()
    if settings.get('verify_padding', False):
        from padding_verify import verify_padding
        result['padding_check'] = verify_padding(trainer, dataset, indices[0], collate)
        (out / f'padding_rank{rank}.json').write_text(json.dumps(result['padding_check'], indent=2), encoding='utf-8')
    for step, index in enumerate(indices):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        chosen = index if isinstance(index, list) else [index]
        batch = collate([dataset[i] for i in chosen])
        losses = trainer._train_step(batch, step)
        torch.cuda.synchronize()
        norm = losses["total_norm"]
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        row = dict(step=step + 1, phase="long_stress" if step < 2 else "sampled_full_episodes",
                   episode_index=chosen, query_latents=int(batch["latents"].shape[2]),
                   seconds=time.perf_counter() - tick,
                   video_loss=float(losses["latent_loss"]), action_loss=float(losses["action_loss"]),
                   grad_norm=float(norm), learning_rate=trainer.lr_scheduler.get_last_lr()[0],
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        assert all(torch.isfinite(torch.tensor(row[k])).item() for k in ["video_loss", "action_loss", "grad_norm"]), row
        result["steps"].append(row)
        trainer.step = step + 1
        (out / f"metrics_rank{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if rank == 0:
            print(json.dumps(row), flush=True)
    changed = (tracked.to_local().detach() - before).abs().max().item()
    assert changed > 0, "Action output parameters did not update"
    result["action_head_max_change"] = changed
    if not settings.get('save_checkpoint', True):
        result['total_seconds'] = time.perf_counter() - t0
        (out / f'complete_rank{rank}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        dist.barrier()
        dist.destroy_process_group()
        return
    tick = time.perf_counter()
    trainer.save_checkpoint()
    dist.barrier()
    result["save_seconds"] = time.perf_counter() - tick
    result["total_seconds"] = time.perf_counter() - t0
    if rank == 0:
        checkpoint = out / "checkpoints" / f"checkpoint_step_{trainer.step}" / "transformer"
        assert (checkpoint / "config.json").is_file()
        from safetensors import safe_open
        with safe_open(checkpoint / "diffusion_pytorch_model.safetensors", framework="pt", device="cpu") as f:
            keys = list(f.keys())
            assert "action_proj_out.weight" in keys
            for name in keys:
                assert torch.isfinite(f.get_tensor(name)).all(), name
            result["checkpoint_tensor_count"] = len(keys)
        (checkpoint.parent / "probe_contract.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out / f"complete_rank{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        import traceback
        config_path = Path(sys.argv[sys.argv.index('--config') + 1])
        config = json.loads(config_path.read_text(encoding='utf-8'))
        target = Path(config['run_root']) / 'train'
        target.mkdir(parents=True, exist_ok=True)
        failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                       is_oom=isinstance(error, torch.OutOfMemoryError) or 'out of memory' in str(error).lower())
        if torch.cuda.is_initialized():
            failure.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (target / f"failure_rank{os.getenv('RANK', 'unknown')}.json").write_text(json.dumps(failure, indent=2), encoding='utf-8')
        raise
