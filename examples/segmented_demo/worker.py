"""Finite RGB-only segmented-Demo policy over the existing binary pipe protocol."""
from contextlib import redirect_stdout
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2], help='Pinned implementation matching checkpoint source hashes')
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--eval-data', type=Path, required=True)
parser.add_argument('--training-data', type=Path, help='Relocated training normalization with the same SHA')
parser.add_argument('--tokenizer', type=Path, help='Relocated tokenizer with the same SHA')
parser.add_argument('--source-amendment', type=Path, help='Explicit original/new source hashes for an audited inference-only change')
args = parser.parse_args()
SOURCE = args.source.resolve()
sys.path[:0] = [str(SOURCE/'src'), str(SOURCE/'examples/segmented_demo'), str(SOURCE/'packages/openpi-client/src')]


from data import PROMPT, collate, dense_demo_indices, normalize, read, sha
from cache_vision import encode_patches


def main(args, stdin, stdout):
    import torch
    import safetensors.torch
    from openpi.models.pi0_frame_demo_config import RoboICLConfig
    from openpi.models.tokenizer import PaligemmaTokenizer
    # Protocol belongs to this entry; --source only selects checkpoint-frozen model code.
    import importlib.util
    spec = importlib.util.spec_from_file_location("segmented_protocol", Path(__file__).with_name("protocol.py"))
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    receive, send = protocol.receive, protocol.send

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    marker = read(args.checkpoint/'complete.json')
    cfg = marker['contract']
    assert sha(args.checkpoint/'model.safetensors') == marker['model_sha256']
    root = SOURCE
    amendment = read(args.source_amendment) if args.source_amendment else None
    if amendment:
        assert amendment['checkpoint_sha256'] == marker['model_sha256']
    for relative, digest in cfg['identities']['model_sources_sha256'].items():
        actual = sha(root/relative)
        if actual != digest:
            assert amendment and amendment['files'].get(relative) == {'original_sha256': digest, 'inference_sha256': actual}, relative
    tokenizer_path = args.tokenizer or Path(cfg['tokenizer'])
    training_data = args.training_data or Path(cfg['data'])
    assert sha(tokenizer_path) == cfg['identities']['tokenizer_sha256']
    assert sha(training_data/'normalization.json') == cfg['identities']['normalization_sha256']
    stats = read(training_data/'normalization.json')
    model_config = RoboICLConfig(**cfg['model_config'])
    model = model_config.create_pytorch()
    safetensors.torch.load_model(model, str(args.checkpoint/'model.safetensors'), strict=True)
    model.to(device).eval()
    assert cfg['vision_frozen'] and model.phase=='A'
    tokens, mask = PaligemmaTokenizer(48, tokenizer_path=tokenizer_path).tokenize(PROMPT)
    tokens, mask = torch.from_numpy(tokens).to(device), torch.from_numpy(mask).to(device)
    manifest = read(args.eval_data/'manifest.json')
    rows = {row['id']: row for row in manifest['rows']}
    geometry = manifest['geometry_sha256']
    send(stdout, dict(ready=True, step=marker['step'], checkpoint_sha256=marker['model_sha256'],
        source_sha256=sha(Path(__file__)), model_sources_sha256=cfg['identities']['model_sources_sha256'],
        source_amendment_sha256=sha(args.source_amendment) if args.source_amendment else None,
        geometry_sha256=geometry, query_history_mode='current', query_history_tokens=0,
        action_hz=50, execute_tokens=50, query_rgb_views=2, demo_rgb_views=2,
        demo_sample_hz=cfg['demo_sample_hz'], demo_representation='shared_siglip_4tokens_view_frame_then_1second_mean_after_2layers',
        action_sampler='known_zero_padding_bridge', vae_model_parameters=0,vision_frozen=True,
        demo_cache_boundary='raw_siglip_patches_before_trainable_pooling_and_projection'))
    demo = None
    while True:
        try:
            meta, arrays = receive(stdin)
        except EOFError:
            break
        started = time.perf_counter()
        try:
            if meta['kind'] == 'close':
                send(stdout, dict(closed=True))
                break
            if meta['kind'] == 'reset':
                seed = int(meta['seed'])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                row = rows[meta['demo']]
                path = args.eval_data/'records'/row['id']/'demo_rgb.npz'
                assert sha(path) == row['demo_sha256']
                with np.load(path, allow_pickle=False) as item:
                    indices, targets = dense_demo_indices(item['recorded_times'], row['seconds'], cfg['demo_sample_hz'])
                    assert np.array_equal(indices,item['indices'])
                    times=item['timestamps'].copy()
                    assert np.array_equal(times,item['recorded_times'][indices])
                    rgb=item['rgb'].copy()
                assert meta['condition']=='correct', 'Only frozen correct-Demo panel authorized'
                demo_frames = max(cfg['demo_frames'], len(times))
                condition = meta['condition']
                permutation = np.arange(len(rgb))
                demo_rgb = np.zeros((demo_frames, 2, 224, 224, 3), np.uint8)
                demo_valid = np.zeros((demo_frames, 2), bool)
                demo_times = np.zeros(demo_frames, np.float32)
                demo_rgb[:len(rgb)] = rgb
                demo_valid[:len(rgb)] = True
                demo_times[:len(rgb)] = times
                images=torch.from_numpy(demo_rgb).to(device).float().permute(0,1,4,2,3)[None]/127.5-1
                valid=torch.from_numpy(demo_valid).to(device)[None]
                patches=encode_patches(model.paligemma_with_expert.paligemma.model.vision_tower,images,valid,4)[0]
                demo = dict(demo_patches=patches, demo_valid=demo_valid, demo_timestamps=demo_times)
                send(stdout, dict(reset=True, condition=condition, seed=seed, valid_frames=int(demo_valid[:,0].sum()),
                    source_record_sha256=row['recording_sha256'], query_history_tokens=0,
                    input_indices=indices.tolist(), permutation=permutation.tolist(), control_seed=20261017,
                    raw_siglip_patches_cached=True,vision_frozen=True))
                continue
            assert meta['kind'] == 'predict' and demo is not None
            assert set(arrays) == {'rgb', 'history', 'history_times', 'history_valid'}
            assert arrays['rgb'].shape == (2,224,224,3) and arrays['rgb'].dtype == np.uint8
            assert arrays['history'].shape == (1,10), 'Segmented architecture consumes current measured state only'
            assert arrays['history_times'].tolist() == [0.] and arrays['history_valid'].tolist() == [True]
            state = np.pad(normalize(arrays['history'][0], stats), (0,22))
            sample = dict(demo, image=arrays['rgb'], state=state,
                actions=np.zeros((50,32), np.float32), action_valid=np.ones((50,32), bool))
            observation, _, _ = collate([sample], device, tokens, mask)
            with torch.inference_mode():
                actions = model.sample_actions(device, observation, num_steps=10)[0,:,:10].float().cpu().numpy()
            actions[:,:3] = (actions[:,:3]+1)/2*(np.asarray(stats['high_xyz'])-stats['low_xyz'])+stats['low_xyz']
            if not np.isfinite(actions).all():
                raise FloatingPointError('Nonfinite policy output')
            torch.cuda.synchronize()
            send(stdout, dict(predicted=True, seconds=time.perf_counter()-started, query_history_tokens=0), actions=actions)
        except Exception as error:
            import traceback
            traceback.print_exc()
            send(stdout, dict(error=repr(error), error_kind='policy_output' if isinstance(error,FloatingPointError) else 'infrastructure'))


if __name__ == '__main__':
    input_stream = sys.stdin.buffer
    output_stream = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with redirect_stdout(sys.stderr):
        main(args, input_stream, output_stream)
