"""Thin segmented Demo adapter for OpenPI's PyTorch training loop."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import time

import safetensors.torch
import torch
import torch.distributed as dist

from data import read, write, sha
from frame_data import FrameLoader
from openpi.models.pi0_frame_demo_config import RoboICLConfig
from openpi.training.config import TrainConfig, DataConfig
from openpi.training.optimizer import AdamW, CosineDecaySchedule


def import_legacy(checkpoint, target, cfg, model_cfg, rank, world, loader):
    """Validated one-time format conversion; historical files stay read-only."""
    marker = read(checkpoint/'complete.json')
    old = marker['contract']
    for key in ('global_batch', 'per_gpu_batch', 'world_size', 'warmup', 'lr_peak', 'lr_end',
                'lr_decay_steps', 'demo_frames', 'demo_sample_hz', 'demo_compression_layer', 'demo_window_seconds'):
        assert old[key] == cfg[key], f'Legacy setting changed: {key}'
    for key, path in [('manifest_sha256',Path(cfg['data'])/'manifest.json'),
                      ('normalization_sha256',Path(cfg['data'])/'normalization.json'),
                      ('tokenizer_sha256',Path(cfg['tokenizer'])),('base_checkpoint_sha256',Path(cfg['base_checkpoint']))]:
        assert old['identities'][key] == sha(path), key
    cache = read(Path(cfg['demo_cache'])/'cache_manifest.json')
    assert cache['identity']['base_checkpoint_sha256'] == old['identities']['base_checkpoint_sha256']
    for key, value in json.loads(json.dumps(dataclasses.asdict(model_cfg))).items():
        if key in old['model_config']:
            assert old['model_config'][key] == value, f'Model configuration changed: {key}'
    if rank == 0:
        assert sha(checkpoint/'model.safetensors') == marker['model_sha256']
        assert sha(checkpoint/'training.pt') == marker['training_sha256']
        saved = torch.load(checkpoint/'training.pt', map_location='cpu', weights_only=False)
        assert saved['step'] == marker['step'] and saved['contract'] == old
        assert len(saved['rng']) == world
        model = model_cfg.create_pytorch()
        safetensors.torch.load_model(model, str(checkpoint/'model.safetensors'), strict=True)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        legacy_names = read(Path(cfg['legacy_parameter_names']))
        assert legacy_names['checkpoint_marker_sha256'] == sha(checkpoint/'complete.json')
        assert names == legacy_names['names'], 'Optimizer parameter order differs from frozen model'
        params = [p for p in model.parameters() if p.requires_grad]
        ids = [i for g in saved['optimizer']['param_groups'] for i in g['params']]
        assert len(ids) == len(params)
        for i, p in zip(ids, params, strict=True):
            state = saved['optimizer']['state'].get(i)
            if state is None:
                continue  # Adam creates state only for parameters that have received gradients.
            assert tuple(state['exp_avg'].shape) == tuple(p.shape)
            assert tuple(state['exp_avg_sq'].shape) == tuple(p.shape)
            assert int(state['step']) == saved['step']
        optim = torch.optim.AdamW(params, lr=cfg['lr_peak'], betas=(.9,.95), eps=1e-8, weight_decay=1e-10)
        optim.load_state_dict(saved['optimizer'])
        pending, final = target/('.import_'+str(saved['step'])), target/str(saved['step'])
        assert not pending.exists() and not final.exists()
        pending.mkdir(parents=True)
        os.link(checkpoint/'model.safetensors', pending/'model.safetensors')
        torch.save(saved['optimizer'], pending/'optimizer.pt')
        runtime = [dict(loader=loader.state_dict(), torch=r['torch'], cuda=r['cuda']) for r in saved['rng']]
        torch.save(dict(global_step=saved['step'], runtime_states=runtime, trainable_parameter_names=names,
                        source_checkpoint=str(checkpoint), source_marker_sha256=sha(checkpoint/'complete.json')),
                   pending/'metadata.pt')
        write(pending/'migration.json',dict(passed=True,step=saved['step'],parameter_count=len(params),
              source=str(checkpoint),source_marker_sha256=sha(checkpoint/'complete.json'),
              source_model_sha256=marker['model_sha256'],source_training_sha256=marker['training_sha256'],
              sampler_before='pair/phase weighted with stationary cap',sampler_after='uniform Query frames',
              sampler_reset=True,optimizer_restored=True,torch_rng_restored=True,
              model_strict_load=True,model_config_compared=True,
              optimizer_sha256=sha(pending/'optimizer.pt'),metadata_sha256=sha(pending/'metadata.pt')))
        pending.rename(final)
        del model, optim, saved
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--import-legacy', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, help='Total updates including imported updates')
    args = parser.parse_args(); cfg = read(args.config)
    world, rank, local = (int(os.environ.get(k, d)) for k, d in [('WORLD_SIZE',1),('RANK',0),('LOCAL_RANK',0)])
    assert world == cfg['world_size'] and cfg['global_batch'] == world*cfg['per_gpu_batch']
    assert cfg['gradient_accumulation'] == 1 and not (args.import_legacy and args.resume)
    torch.set_num_threads(cfg.get('cpu_threads',4))
    if 'allow_tf32' in cfg:
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg['allow_tf32'])
    if cfg.get('cpu_affinity_per_local_rank'):
        os.sched_setaffinity(0, cfg['cpu_affinity_per_local_rank'][local])
    model_cfg = RoboICLConfig(action_dim=32, action_encoding=cfg.get('action_encoding', 'world_tcp10'), demo_frames=cfg['demo_frames'],
                              demo_views=cfg.get('demo_views', 2),
                              query_image_keys=tuple(cfg.get('query_image_keys', ('base_0_rgb', 'left_wrist_0_rgb'))),
                              checkpoint_strategy=cfg.get('checkpoint_strategy', 'nested'),
                              checkpoint_layers=tuple(cfg['checkpoint_layers']) if cfg.get('checkpoint_layers') is not None else None,
                              pool_valid_only=cfg.get('pool_valid_only', False),
                              demo_sample_hz=cfg['demo_sample_hz'], demo_compression_layer=cfg['demo_compression_layer'],
                              demo_window_seconds=cfg['demo_window_seconds'])
    loader = FrameLoader(cfg, rank, world)
    if cfg.get('dataset', 'roboicl') != 'roboicl':
        manifest = loader.dataset.records.manifest
        assert manifest['dataset'] == cfg['dataset'] and manifest['action_encoding'] == model_cfg.action_encoding
        assert tuple(manifest['query_image_keys']) == model_cfg.query_image_keys and manifest['demo_views'] == model_cfg.demo_views
        assert not manifest['smoke_only'] or cfg.get('smoke_only'), 'Integration subset cannot start formal training'
    base_sha = sha(cfg['base_checkpoint'])
    assert loader.dataset.records.cache_manifest['identity']['base_checkpoint_sha256'] == base_sha
    args.output = args.output.resolve(); steps = args.steps or cfg['steps']
    native = TrainConfig(name=args.output.parent.name, exp_name=args.output.name,
                         checkpoint_base_dir=str(args.output.parent.parent), model=model_cfg,
                         batch_size=cfg['global_batch'], num_train_steps=steps, seed=cfg['seed'],
                         num_workers=0, save_interval=5000, log_interval=10, wandb_enabled=False,
                         resume=bool(args.import_legacy or args.resume), ema_decay=None,
                         pytorch_weight_path=None if args.import_legacy or args.resume else str(Path(cfg['base_checkpoint']).parent),
                         optimizer=AdamW(), lr_schedule=CosineDecaySchedule(warmup_steps=cfg['warmup'],
                             peak_lr=cfg['lr_peak'], decay_steps=cfg['lr_decay_steps'], decay_lr=cfg['lr_end']))
    assert native.checkpoint_dir == args.output
    root = Path(__file__).resolve().parents[2]
    source_paths = [root/'scripts/train_pytorch.py', *Path(__file__).parent.glob('*.py'),
                    root/'src/openpi/models/model.py',root/'src/openpi/models/pi0_config.py',
                    root/'src/openpi/models/pi0_frame_demo_config.py', root/'src/openpi/models/tokenizer.py']
    source_paths += list((root/'src/openpi/models_pytorch').glob('*.py'))
    loader.contract = dict(data=cfg['data'],tokenizer=cfg['tokenizer'],vision_frozen=True,
        demo_frames=cfg['demo_frames'],demo_sample_hz=cfg['demo_sample_hz'],model_config=dataclasses.asdict(model_cfg),
        sampler='uniform_query_frames_v1',global_batch=cfg['global_batch'],
        identities=dict(base_checkpoint_sha256=base_sha,
          model_sources_sha256={str(p.relative_to(root)):sha(p) for p in source_paths},
          manifest_sha256=sha(Path(cfg['data'])/'manifest.json'),normalization_sha256=sha(Path(cfg['data'])/'normalization.json'),
          tokenizer_sha256=sha(cfg['tokenizer']),cache_manifest_sha256=sha(Path(cfg['demo_cache'])/'cache_manifest.json')))
    loader.contract['training'] = {k:cfg[k] for k in ('world_size','per_gpu_batch','global_batch',
        'gradient_accumulation','seed','warmup','lr_peak','lr_end','lr_decay_steps')}
    loader.contract['training']['allow_tf32'] = bool(torch.backends.cuda.matmul.allow_tf32)
    loader.contract = json.loads(json.dumps(loader.contract))
    spec = importlib.util.spec_from_file_location('openpi_native_trainer', root/'scripts/train_pytorch.py')
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    trainer.init_logging()
    if args.import_legacy:
        assert steps > read(args.import_legacy/'complete.json')['step']
        trainer.setup_ddp()
        import_legacy(args.import_legacy, args.output, cfg, model_cfg, rank, world, loader)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        sources = {str(p.relative_to(root)):sha(p) for p in [Path(__file__), Path(__file__).with_name('frame_data.py'),
                   Path(__file__).with_name('data.py'),root/'scripts/train_pytorch.py']}
        write(args.output/f'uniform_entry_{time.time_ns()}.json',dict(config=cfg,total_steps=steps,sources=sources,
              sampler=loader.state_dict(),import_legacy=str(args.import_legacy) if args.import_legacy else None,
              resume=args.resume,model_config=dataclasses.asdict(model_cfg),
              epoch_definition='Query frames traversed once; distributed drop_last tail omitted; legal Demo per frame/epoch'))
    trainer.train_loop(native, data_loader=(loader, DataConfig()))
    if rank == 0:
        write(args.output/'finished.json', dict(passed=True,steps=steps,sampler='uniform_query_frames_v1'))


if __name__ == '__main__':
    main()
