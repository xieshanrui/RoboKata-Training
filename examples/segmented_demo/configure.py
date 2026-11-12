"""Create a local training configuration from the checked-in experiment defaults."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "base", "tokenizer", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch-per-gpu", type=int, default=32)
    parser.add_argument("--steps", type=int, help="Total updates; freeze before formal training")
    parser.add_argument("--lr-decay-steps", type=int, help="Absolute cosine schedule horizon")
    parser.add_argument("--checkpoint-strategy", choices=("nested", "layer", "selective", "none"))
    parser.add_argument("--checkpoint-layers", type=int, nargs='+')
    parser.add_argument("--pool-valid-only", action='store_true')
    parser.add_argument("--allow-tf32", action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(__file__).with_name("config.example.json").read_text(encoding="utf-8"))
    for path in (args.data / "manifest.json", args.data / "normalization.json", args.base, args.tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.gpus < 1 or args.batch_per_gpu < 1:
        raise ValueError("GPU count and real batch must be positive")
    cfg.update(data=str(args.data.resolve()), base_checkpoint=str(args.base.resolve()),
               tokenizer=str(args.tokenizer.resolve()), demo_cache=str(args.cache.resolve()),
               world_size=args.gpus, per_gpu_batch=args.batch_per_gpu,
               global_batch=args.gpus * args.batch_per_gpu,
               normalization_sha256=hashlib.sha256((args.data / "normalization.json").read_bytes()).hexdigest())
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") == "segmented_transfer_v1":
        cfg.update(dataset=manifest["dataset"], smoke_only=manifest["smoke_only"],
                   action_encoding=manifest["action_encoding"], demo_views=manifest["demo_views"],
                   query_image_keys=manifest["query_image_keys"],
                   demo_frames=max(cfg["demo_frames"], manifest["maximum_demo_frames_5hz"]))
    for name in ("steps", "lr_decay_steps"):
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                raise ValueError("Training/schedule boundaries must be positive")
            cfg[name] = value
    for name in ("checkpoint_strategy",):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if args.checkpoint_layers is not None:
        if args.checkpoint_strategy != 'selective':
            raise ValueError('--checkpoint-layers requires --checkpoint-strategy selective')
        cfg['checkpoint_layers'] = args.checkpoint_layers
    if args.pool_valid_only:
        cfg['pool_valid_only'] = True
    if args.allow_tf32:
        cfg['allow_tf32'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
