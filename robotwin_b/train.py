"""Isolated RoboTwin A+B1 fine-tuning and checkpoint smoke test."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from data import CAMERAS, build_features, read_latent, robot_episode, train_rows
from demo_attention import SPEC, attach_demo_attention
from demo_features import load_encoder
from feature_cache import contract as feature_contract, features_for_pair
from examples.baselines.zero_wam.zero_wam.lora import LoRALinear, target_names
from examples.baselines.zero_wam.zero_wam.training import atomic_torch
from examples.baselines.zero_wam.zero_wam.wan_va.modules import icl_model
from examples.baselines.zero_wam.zero_wam.wan_va.semantic_ifp import SemanticIFPHead, future_anchor_pairs
from examples.baselines.zero_wam.zero_wam.wan_va.train import Trainer
from examples.baselines.zero_wam.zero_wam.wan_va.utils import FlowMatchScheduler
from native import va_robotwin_train_cfg


class HighViewIFP(SemanticIFPHead):
    def pool(self, states, frames, height, width):
        if width % len(CAMERAS):
            raise ValueError("Expected equal-width RoboTwin camera tiles")
        high_width = width // len(CAMERAS)
        high = states.reshape(states.shape[0], frames, height, width, -1)[..., :high_width, :]
        return super().pool(high.flatten(1, 3), frames, height, high_width)


def enable_cpu_attention():
    # Evaluate exactly the same Zero-WAM mask predicates using CPU SDPA.
    def dense_mask(predicate, batch, heads, queries, keys, device, **_):
        b = torch.arange(batch, device=device)[:, None, None, None]
        h = torch.arange(heads, device=device)[None, :, None, None]
        q = torch.arange(queries, device=device)[None, None, :, None]
        k = torch.arange(keys, device=device)[None, None, None, :]
        return predicate(b, h, q, k)

    def apply(cls, query, key, value, *, cross_attention, block_mask=None):
        if block_mask is None:
            block_mask = cls.cross_mask if cross_attention else cls.self_mask
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2).to(value.dtype), key.transpose(1, 2).to(value.dtype),
            value.transpose(1, 2), attn_mask=block_mask,
        )
        return output.transpose(1, 2)

    icl_model._compiled_create_block_mask = dense_mask
    icl_model.ICLAttentionBackend.apply = classmethod(apply)


def make_model(args, dtype):
    torch.manual_seed(args.seed)
    model_type = icl_model.WanICLTransformer3DModel
    if args.compact:
        model = model_type(num_attention_heads=2, attention_head_dim=32, action_inner_dim=64,
                           ffn_dim=128, action_ffn_dim=128).to(dtype=dtype)
    else:
        model = model_type.from_pretrained(args.model / "transformer", torch_dtype=dtype,
                                           local_files_only=True)
    model.requires_grad_(False)
    del model.mcp_blocks, model.mcp_projections
    for name in target_names(model):
        parent, child = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        setattr(module, child, LoRALinear(getattr(module, child), args.rank, 2 * args.rank))
    attach_demo_attention(model, SPEC)
    model.semantic_ifp = HighViewIFP(model.inner_dim, 4, 768, 2)
    return model.to(device=args.device, dtype=dtype)


def trainer_for(args, model):
    trainer = Trainer.__new__(Trainer)
    trainer.config = va_robotwin_train_cfg.copy()
    trainer.config.update(frame_chunk_size=2, attn_window=64, drop_icl=0.0)
    # Preserve EasyDict access even with versions whose copy() returns dict.
    trainer.config = type(va_robotwin_train_cfg)(trainer.config)
    trainer.device = torch.device(args.device)
    trainer.patch_size = model.patch_size
    trainer.enable_mcp = True
    trainer.semantic_ifp = {"weight": 0.5}
    trainer.gradient_accumulation_steps = 1
    for stream in ("latent", "action"):
        shift = trainer.config.snr_shift if stream == "latent" else trainer.config.action_snr_shift
        scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        setattr(trainer, f"train_scheduler_{stream}", scheduler)
    return trainer


def load_batch(args, row, empty, encoder, demos, cache_dir):
    batch, anchors = robot_episode(args.data, row, args.frames)
    cache = cache_dir / (row["pair_id"].replace("/", "__") + ".pt")
    if args.feature_cache is not None:
        demo, targets = features_for_pair(args.feature_cache, row, anchors)
    elif cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["pair"] != row or saved["anchors"] != anchors.tolist():
            raise ValueError(f"Feature cache does not match {row['pair_id']}")
        demo, targets = saved["demo"], saved["targets"]
    else:
        demo, targets = build_features(encoder, args.data, row, anchors, demos)
        atomic_torch(cache, dict(pair=row, anchors=anchors.tolist(), demo=demo, targets=targets))
    human, _ = read_latent(args.data / row["human_latent"])
    if args.compact:
        # Keep all three camera tiles and their full temporal timeline.
        camera_width = batch["latents"].shape[-1] // len(CAMERAS)
        batch["latents"] = torch.cat([part[:, :, :4, :4]
                                      for part in batch["latents"].split(camera_width, dim=-1)], dim=-1)
        human = human[:, :4, :4, :4]
    batch.update(icl_latents=human, text_emb=empty, icl_text_emb=empty)
    return batch, demo, targets, anchors


def state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()
            if param.requires_grad}


def restore(model, values):
    expected = {name: tuple(param.shape) for name, param in model.named_parameters()
                if param.requires_grad}
    if {name: tuple(value.shape) for name, value in values.items()} != expected:
        raise ValueError("Checkpoint adapter names or dimensions differ")
    model.load_state_dict(values, strict=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "encoder", "empty-text", "run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compact", action="store_true", help="Use a reduced random backbone for integration QA")
    args = parser.parse_args()
    if min(args.steps, args.rank, args.threads) < 1 or (args.frames is not None and args.frames < 3):
        parser.error("Positive steps, rank and threads required, with at least three frames")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    args.run.mkdir(parents=True, exist_ok=False)
    cache_dir = args.run / "features"
    cache_dir.mkdir()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        enable_cpu_attention()
    dtype = torch.float32 if args.compact else torch.bfloat16
    rows, revision = train_rows(args.data)
    if args.steps > len(rows):
        parser.error("This launcher supports at most one pass through the training pairs")
    rng = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(rows), generator=rng).tolist()
    if args.steps == 1:
        order = list(range(len(rows)))
    contract = dict(dataset_revision=revision, cameras=CAMERAS, b1_camera=CAMERAS[0],
                    model=str(args.model.resolve()), encoder=str(args.encoder.resolve()),
                    rank=args.rank, compact=args.compact, seed=args.seed,
                    max_frames=args.frames, frame_chunk_size=2, horizons=[1, 3, 5, 7])
    (args.run / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    empty = torch.load(args.empty_text, map_location="cpu", weights_only=True)
    if empty.ndim == 3 and empty.shape[0] == 1:
        empty = empty.squeeze(0)
    if empty.ndim != 2:
        raise ValueError(f"Expected a token-by-channel empty embedding, got {empty.shape}")
    encoder = None
    if args.feature_cache is None:
        encoder = load_encoder(args.encoder, torch.device("cpu"))
    else:
        verified = json.loads((args.feature_cache / "verified.json").read_text())
        expected = dict(verified=True, demos=854, queries=len(rows),
                        **feature_contract(args.data, args.encoder, revision))
        if verified != expected:
            raise ValueError("Feature cache verification does not match this run")
    demos = {}
    model = make_model(args, dtype).train()
    trainer = trainer_for(args, model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate, foreach=False)

    for step, index in enumerate(order[:args.steps], 1):
        row = rows[index]
        if args.steps == 1:
            # Choose a real trajectory long enough to exercise all four B1 horizons.
            for candidate in rows:
                _, anchors = robot_episode(args.data, candidate, args.frames)
                if len(anchors) > 14:
                    row = candidate
                    break
            else:
                raise ValueError("Smoke test requires at least 15 latent frames")
        batch, demo, targets, anchors = load_batch(args, row, empty, encoder, demos, cache_dir)
        batch = {key: value[None].to(device=args.device, dtype=(dtype if value.is_floating_point()
                                                                 else value.dtype))
                 for key, value in batch.items()}
        inputs = trainer.prepare_native_input(batch)
        inputs.update(demo_features=demo[None].to(device=args.device, dtype=dtype),
                      semantic_ifp_features=targets[None].to(device=args.device, dtype=dtype),
                      semantic_ifp_valid=torch.ones((1, len(anchors)), device=args.device, dtype=torch.bool))
        predictions = model(inputs, train_mode=True)
        video, action, semantic = trainer.compute_loss(inputs, predictions)
        loss = video + action + sum(semantic)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at {row['pair_id']}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        groups = {"demo_gate": [], "b1_head": [], "lora": []}
        for name, value in model.named_parameters():
            if value.grad is None:
                continue
            key = ("demo_gate" if name.endswith("demo_attention.gate") else
                   "b1_head" if name.startswith("semantic_ifp.") else
                   "lora" if name.endswith("lora_b") else None)
            if key is not None:
                groups[key].append(value.grad.detach().float().norm())
        norms = {name: torch.stack(values).norm().item() for name, values in groups.items()}
        if not all(torch.isfinite(torch.tensor(value)) and value > 0 for value in norms.values()):
            raise RuntimeError(f"Missing, zero or nonfinite gradients: {norms}")
        optimizer.step()
        counts = [int(mask.sum()) for _, mask in future_anchor_pairs(inputs["semantic_ifp_valid"], 4, 2, 2)]
        print(json.dumps(dict(step=step, pair=row["pair_id"], anchors=anchors.tolist(),
                              video_shape=list(batch["latents"].shape), action_shape=list(batch["actions"].shape),
                              camera_count=len(CAMERAS), horizons_valid=counts, video_loss=video.item(),
                              action_loss=action.item(), b1_loss=[value.item() for value in semantic],
                              gradient_norms=norms)), flush=True)

    path = args.run / "adapter.pt"
    atomic_torch(path, dict(contract=contract, adapter=state(model), step=step))
    # Reload against a clean base and check outputs on the exact same noised batch.
    model.eval()
    with torch.no_grad():
        reference = model(inputs, train_mode=True)
    del model, optimizer, predictions, loss, video, action, semantic, groups
    reloaded = make_model(args, dtype).eval()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["contract"] != contract or saved["step"] != step:
        raise ValueError("Checkpoint provenance differs")
    restore(reloaded, saved["adapter"])
    with torch.no_grad():
        actual = reloaded(inputs, train_mode=True)
    for before, after in zip((*reference[:2], *reference[2]), (*actual[:2], *actual[2])):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    print(json.dumps(dict(checkpoint=str(path), reloaded=True, outputs_match=True)), flush=True)


if __name__ == "__main__":
    main()
