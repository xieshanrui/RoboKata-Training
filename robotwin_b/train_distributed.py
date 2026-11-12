"""Four-rank RoboTwin A+B1 training with adapter and optimizer checkpoints."""
import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.utils.data import DataLoader, Dataset

from data import CAMERAS, read_latent, robot_episode, train_rows
from feature_cache import contract, entry_path, features_for_pair
from train import make_model, restore, trainer_for
from examples.baselines.zero_wam.zero_wam.training import atomic_json, atomic_torch, restore_rng, rng_state
from examples.baselines.zero_wam.zero_wam.wan_va.distributed.fsdp import apply_ac, shard_model
from examples.baselines.zero_wam.zero_wam.wan_va.semantic_ifp import semantic_ifp_losses
from examples.baselines.zero_wam.zero_wam.wan_va.utils import data_seq_to_patch

GIB = 1024**3
BATCH = 16
WORLD = 4
EPOCHS = 20


class Episodes(Dataset):
    def __init__(self, root, cache, rows, lengths, cap):
        self.root, self.cache, self.rows = root, cache, rows
        self.lengths, self.cap = lengths, cap
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start = 0
        if self.cap is not None and self.lengths[index] > self.cap:
            last = self.lengths[index] - self.cap
            starts = list(range(0, last, self.cap // 2)) + [last]
            start = starts[(self.epoch + index) % len(starts)]
        batch, anchors = robot_episode(self.root, row, self.cap, start)
        demo, targets = features_for_pair(self.cache, row, anchors, start)
        batch["icl_latents"], _ = read_latent(self.root / row["human_latent"])
        return batch, demo, targets


def collate(samples):
    def pad_stack(items, dim):
        shape = list(items[0].shape)
        shape[dim] = max(item.shape[dim] for item in items)
        padded = []
        for item in items:
            if any(size != shape[i] for i, size in enumerate(item.shape) if i != dim):
                raise ValueError("Incompatible spatial or channel dimensions within a RoboTwin batch")
            out = item.new_zeros(shape)
            out.narrow(dim, 0, item.shape[dim]).copy_(item)
            padded.append(out)
        return torch.stack(padded)

    items, demos, targets = zip(*samples)
    lengths = torch.tensor([item["latents"].shape[1] for item in items])
    human_lengths = torch.tensor([item["icl_latents"].shape[1] for item in items])
    batch = {key: pad_stack([item[key] for item in items], 1) for key in items[0]}
    return batch, torch.stack(demos), pad_stack(targets, 0), lengths, human_lengths


def batches(lengths, epoch, seed, rank):
    rng = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(lengths), generator=rng).tolist()
    width = WORLD * BATCH
    # Choose the tail before bucketing so long episodes are represented across epochs.
    order = order[:len(order) // width * width]
    order.sort(key=lengths.__getitem__)
    groups = [order[start:start + width] for start in range(0, len(order), width)]
    permutation = torch.randperm(len(groups), generator=rng).tolist()
    return [groups[index][rank * BATCH:(rank + 1) * BATCH] for index in permutation]


def reserve(device):
    target = 121 * GIB
    free, _ = torch.cuda.mem_get_info(device)
    extra = max(0, target - torch.cuda.memory_reserved(device))
    ready = torch.tensor(int(free >= extra + GIB), device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        raise RuntimeError("All four GPUs must have enough space for the 121 GiB allocator pool")
    if extra:
        pool = torch.empty(extra, dtype=torch.uint8, device=device)
        del pool
    torch.cuda.synchronize(device)
    if torch.cuda.memory_reserved(device) <= 120 * GIB:
        raise RuntimeError("Allocator reservation did not reach the requested floor")
    print(f"rank {dist.get_rank()} reserved {torch.cuda.memory_reserved(device) / GIB:.2f} GiB", flush=True)


def losses(trainer, inputs, predictions, valid):
    video, action, semantic = predictions
    v = inputs["latent_dict"]
    a = inputs["action_dict"]
    target = v["targets"]
    video = data_seq_to_patch(trainer.patch_size, video, *target.shape[-3:], batch_size=len(valid))
    weights = trainer.train_scheduler_latent.training_weight(v["timesteps"].flatten()).reshape_as(valid)
    error = (video.float() - target.float()).square().mean(dim=(1, 3, 4))
    video_loss = (error * weights).masked_fill(~valid, 0).sum() / valid.sum()
    action = action.reshape(len(valid), valid.shape[1], a["targets"].shape[3], -1)
    action = action.permute(0, 3, 1, 2).unsqueeze(-1)
    error = (action.float() - a["targets"].float()).square().masked_fill(~a["actions_mask"], 0)
    action_loss = error.sum() / (valid.sum() * math.prod(error.shape[1:2] + error.shape[3:]))
    b1 = semantic_ifp_losses(semantic, inputs["semantic_ifp_features"], valid, 2, 2)
    return torch.stack([video_loss, action_loss, *(value * 0.5 for value in b1)])


def checkpoint(model, optimizer, schedule, run, config, epoch, cursor, step, device):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
    adapter = get_model_state_dict(model, options=options)
    optim = get_optimizer_state_dict(model, optimizer, options=options)
    states = [None] * WORLD
    dist.all_gather_object(states, rng_state(device))
    if dist.get_rank() == 0:
        path = run / "checkpoints/latest.pt"
        atomic_torch(path, dict(contract=config, adapter=adapter, optimizer=optim,
                               scheduler=schedule.state_dict(), rng=states,
                               epoch=epoch, next_batch=cursor, step=step))
        if cursor == 0 and epoch % 5 == 0:
            target = path.with_name(f"epoch_{epoch:03d}.pt")
            if not target.exists():
                os.link(path, target)
        atomic_json(run / "checkpoint-status.json", dict(epoch=epoch, next_batch=cursor, step=step))
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "encoder", "empty-text", "run", "feature-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--benchmark-from", type=Path,
                      help="Measure three long-trajectory updates in a separate run directory")
    parser.add_argument("--no-reserve", action="store_true", help="Use the CUDA allocator without a reservation floor")
    parser.add_argument("--max-robot-frames", type=int)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.max_robot_frames is not None and args.max_robot_frames < 15:
        parser.error("The window must support the 14-frame B1 horizon")
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD:
        raise ValueError("This experiment requires four ranks")
    args.device = torch.device("cuda", local)
    args.rank, args.seed, args.compact = 106, 17, False
    torch.set_num_threads(2)
    rows, revision = train_rows(args.data)
    identity = dict(verified=True, demos=854, queries=len(rows), **contract(args.data, args.encoder, revision))
    if json.loads((args.feature_cache / "verified.json").read_text()) != identity:
        raise ValueError("Verified feature cache differs from this training run")
    lengths = [len(torch.load(entry_path(args.feature_cache, "query", row["pair_id"]),
                              map_location="cpu", weights_only=True)["anchors"]) for row in rows]
    updates = len(rows) // (BATCH * WORLD)
    config = dict(features=identity, cameras=list(CAMERAS), model=str(args.model.resolve()),
                  rank=106, lora_alpha=212, world=WORLD, per_gpu=BATCH, epochs=args.epochs,
                  updates_per_epoch=updates, tail="shuffle then drop before length bucketing",
                  learning_rate=5e-5, min_lr=1e-6, warmup=0, weight_decay=0.01,
                  seed=args.seed, demo_dropout=0.3, b1_weight=0.5, chunk_size=2,
                  horizons=[1, 3, 5, 7], allocator_floor_gib=121)
    if args.max_robot_frames is not None:
        config["max_robot_frames"] = args.max_robot_frames
    previous_config = {k: v for k, v in config.items() if k != "max_robot_frames"}
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=args.device)
    if rank == 0:
        if args.resume:
            disk_config = json.loads((args.run / "config.json").read_text())
            if disk_config not in (config, previous_config):
                raise ValueError("Resume contract differs")
        else:
            args.run.mkdir(parents=True, exist_ok=False)
            atomic_json(args.run / "config.json", config)
    dist.barrier()

    device = args.device
    args.device = torch.device("cpu")
    model = make_model(args, torch.bfloat16)
    args.device = device
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 465675523:
        raise ValueError(f"Unexpected trainable parameter count: {trainable}")
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    saved = None
    if args.resume or args.benchmark_from:
        source = args.benchmark_from or args.run / "checkpoints/latest.pt"
        saved = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
        if saved["contract"] != config:
            if not (args.resume and args.max_robot_frames and saved["step"] == 1
                    and saved["contract"] == previous_config):
                raise ValueError("Checkpoint contract differs")
        restore(model, {name: saved["adapter"][name] for name, param in model.named_parameters()
                        if param.requires_grad})
    trainer = trainer_for(args, model)
    apply_ac(model)
    shard_model(model)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=5e-5, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates * args.epochs, eta_min=1e-6)
    start, cursor, step = 0, 0, 0
    torch.manual_seed(args.seed + rank)
    if saved is not None:
        options = StateDictOptions(full_state_dict=True, strict=False)
        set_optimizer_state_dict(model, optimizer, saved["optimizer"], options=options)
        schedule.load_state_dict(saved["scheduler"])
        start, cursor, step = saved["epoch"], saved["next_batch"], saved["step"]
        if step != start * updates + cursor:
            raise ValueError("Invalid checkpoint training cursor")
        restore_rng(saved["rng"][rank], device)
        del saved
    empty = torch.load(args.empty_text, map_location="cpu", weights_only=True).squeeze(0)
    empty = empty.to(device=device, dtype=torch.bfloat16).expand(BATCH, -1, -1)
    dataset = Episodes(args.data, args.feature_cache, rows, lengths, args.max_robot_frames)
    capped_lengths = [min(length, args.max_robot_frames or length) for length in lengths]
    if rank == 0 and args.resume:
        atomic_json(args.run / "config.json", config)
    if rank == 0:
        atomic_json(args.run / "allocator-runtime.json",
                    dict(reservation_enabled=not args.no_reserve, floor_gib=0 if args.no_reserve else 121))
    if not args.no_reserve:
        reserve(device)
    stopping = False

    def stop_at_update(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_at_update)
    signal.signal(signal.SIGINT, stop_at_update)
    benchmark = args.benchmark_from is not None
    if benchmark:
        groups = []
        for low, high in ((17, 18), (19, 21), (28, 30)):
            candidates = [i for i, length in enumerate(lengths) if low <= length <= high]
            group = sorted(candidates, key=lengths.__getitem__)[:WORLD * BATCH]
            if len(group) != WORLD * BATCH:
                raise ValueError(f"Too few trajectories in benchmark band {low}–{high}")
            groups.append(group[rank * BATCH:(rank + 1) * BATCH])
        start, cursor = 0, 0
    for epoch in range(start, 1 if benchmark else args.epochs):
        begin = cursor if epoch == start else 0
        dataset.epoch = epoch
        plan = groups if benchmark else batches(capped_lengths, epoch, args.seed, rank)
        if args.resume and args.max_robot_frames and epoch == 0 and begin:
            # Keep the same epoch-zero ordering as the pre-window checkpoint.
            plan = batches(lengths, epoch, args.seed, rank)
        loader = DataLoader(dataset, batch_sampler=plan[begin:], collate_fn=collate,
                            num_workers=2, pin_memory=True, multiprocessing_context="spawn")
        tick = time.monotonic()
        for index, (batch, demo, targets, lengths_robot, lengths_human) in enumerate(loader, begin):
            torch.cuda.reset_peak_memory_stats(device)
            if benchmark:
                print(json.dumps(dict(rank=rank, benchmark_batch=index + 1,
                                      robot_frames=lengths_robot.tolist(),
                                      human_frames=lengths_human.tolist())), flush=True)
            batch = {k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype,
                            non_blocking=True) for k, v in batch.items()}
            robot, human = batch["latents"], batch["icl_latents"]
            lengths_robot, lengths_human = lengths_robot.to(device), lengths_human.to(device)
            valid = torch.arange(robot.shape[2], device=device)[None] < lengths_robot[:, None]
            batch.update(text_emb=empty, icl_text_emb=empty)
            inputs = trainer.prepare_native_input(batch)
            _, ph, pw = model.patch_size
            inputs["ig_batch_lengths"] = (
                lengths_robot, lengths_human, robot.shape[2], human.shape[2],
                (robot.shape[3] // ph) * (robot.shape[4] // pw),
                (human.shape[3] // ph) * (human.shape[4] // pw),
                trainer.config.action_per_frame,
            )
            keep = torch.rand((BATCH, 1, 1), device=device) >= 0.3
            inputs.update(demo_features=demo.to(device, dtype=torch.bfloat16) * keep,
                          semantic_ifp_features=targets.to(device))
            values = losses(trainer, inputs, model(inputs, train_mode=True), valid)
            with torch.no_grad():
                metrics = values.detach().clone()
                dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                if not torch.isfinite(metrics).all():
                    raise FloatingPointError(f"Nonfinite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            values.sum().backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                                 error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            step += 1
            memory = torch.tensor([torch.cuda.memory_reserved(device) / GIB,
                                   torch.cuda.max_memory_allocated(device) / GIB], device=device)
            memories = [torch.empty_like(memory) for _ in range(WORLD)]
            dist.all_gather(memories, memory)
            if not args.no_reserve and any(value[0].item() <= 120 for value in memories):
                reserve(device)
                memory[0] = torch.cuda.memory_reserved(device) / GIB
                dist.all_gather(memories, memory)
            elapsed = time.monotonic() - tick
            grad_norm = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)
            if rank == 0:
                print(json.dumps(dict(epoch=epoch + 1, step=step, seconds=elapsed,
                                      losses=metrics.tolist(), grad_norm=grad_norm,
                                      memory=[value.tolist() for value in memories])), flush=True)
            stop = torch.tensor(int(stopping), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            at_end = index + 1 == updates
            if not benchmark and (at_end or step in (1, 2) or stop.item()):
                checkpoint(model, optimizer, schedule, args.run, config,
                           epoch + int(at_end), 0 if at_end else index + 1, step, device)
            if stop.item():
                dist.destroy_process_group()
                return
            tick = time.monotonic()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
