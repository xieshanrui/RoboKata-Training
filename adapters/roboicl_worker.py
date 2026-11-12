"""RGB-only Zero-WAM worker over a length-prefixed NPZ stdin/stdout pipe.

The adapter keeps released model weights, normalization and denoising settings.
Only head/left-wrist query views are supplied; right-arm predictions are returned
for provenance but never executed by the single-arm simulator client.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import time
import traceback

import numpy as np


def read_exact(stream, count):
    blocks = []
    while count:
        block = stream.read(count)
        if not block:
            raise EOFError('RPC stream closed')
        blocks.append(block)
        count -= len(block)
    return b''.join(blocks)


def receive(stream):
    length = struct.unpack('!Q', read_exact(stream, 8))[0]
    if length > 256 * 1024 * 1024:
        raise ValueError('RPC packet too large')
    with np.load(io.BytesIO(read_exact(stream, length)), allow_pickle=False) as item:
        values = {key: item[key] for key in item.files}
    meta = json.loads(values.pop('meta').tobytes().decode('utf-8'))
    return meta, values


def send(stream, meta, **arrays):
    packet = io.BytesIO()
    np.savez_compressed(packet, meta=np.frombuffer(json.dumps(meta).encode('utf-8'), dtype=np.uint8), **arrays)
    payload = packet.getvalue()
    stream.write(struct.pack('!Q', len(payload)) + payload)
    stream.flush()


def main(args, input_stream, output_stream):
    import torch
    from wan_va.configs import VA_CONFIGS
    from wan_va.utils import init_logger
    from wan_va.wan_va_server import VA_Server
    from roboicl_video import read_rgb_demo_npz

    init_logger()

    class RGBPolicy(VA_Server):
        def _read_icl_video(self, video_path):
            if Path(video_path).suffix != '.npz':
                return super()._read_icl_video(video_path)
            return read_rgb_demo_npz(self, video_path)

    config = deepcopy(VA_CONFIGS['robotwin'])
    config.model_path = str(args.model)
    config.save_root = str(args.output)
    config.local_rank, config.rank, config.world_size = 0, 0, 1
    config.rgb_only = True
    adaptation_path = args.model / 'adaptation.json'
    adaptation = None
    if adaptation_path.exists():
        from roboicl_single_arm import configure_single_arm
        adaptation = json.loads(adaptation_path.read_text(encoding='utf-8'))
        configure_single_arm(config, adaptation)
    config.obs_cam_keys = ['observation.images.cam_high', 'observation.images.cam_left_wrist']
    config.num_inference_steps = args.denoise_steps
    config.action_num_inference_steps = args.denoise_steps
    config.empty_text_emb_path = str(Path(__file__).resolve().parents[1] / 'wan_va/assets/empty_text_emb.pt')
    config.enable_offload = False
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    policy = RGBPolicy(config)
    policy.vae.eval().requires_grad_(False)
    torch.cuda.synchronize()
    send(output_stream, dict(ready=True, torch=torch.__version__, device=torch.cuda.get_device_name(),
                             load_seconds=time.perf_counter() - started,
                             query_views=config.obs_cam_keys, demo_views=1, rgb_only=True,
                             denoise_steps=args.denoise_steps, checkpoint=str(args.model),
                             action_dim=config.action_dim, adaptation=adaptation,
                             trainable_parameters=sum(p.numel() for p in policy.transformer.parameters() if p.requires_grad),
                             source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in [Path(__file__).resolve(),
                                                      Path(__file__).resolve().parents[1]/'wan_va/wan_va_server.py',
                                                      Path(__file__).resolve().parents[1]/'wan_va/modules/model.py',
                                                      Path(__file__).resolve().parents[1]/'wan_va/modules/icl_model.py',
                                                      Path(__file__).with_name('roboicl_video.py'),
                                                      Path(__file__).with_name('roboicl_single_arm.py')]},
                             gpu_allocated_bytes=torch.cuda.memory_allocated()))

    def observation(rgb):
        if rgb.ndim == 4:
            rgb = rgb[None]
        if rgb.ndim != 5 or rgb.shape[1] != 2 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Query must be [time,2,height,width,3] uint8')
        return [{name: frame[i] for i, name in enumerate(config.obs_cam_keys)} for frame in rgb]

    while True:
        try:
            meta, arrays = receive(input_stream)
        except EOFError:
            break
        started = time.perf_counter()
        try:
            kind = meta['kind']
            if kind == 'close':
                send(output_stream, dict(closed=True))
                break
            if kind == 'reset':
                seed = int(meta['seed'])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                policy.demo_control = meta.get('control', 'correct')
                policy.control_seed = seed
                policy.infer(dict(reset=True, use_icl=True, prompt='Follow the demonstration.',
                                  icl_video_path=meta['demo'], video_guidance_scale=-1,
                                  icl_guidance_scale=5))
                result = dict(reset=True, demo_sampling=policy.demo_sampling)
            elif kind == 'predict':
                result_arrays = policy.infer(dict(obs=observation(arrays['rgb']), prompt='Follow the demonstration.'))
                result = dict(predicted=True, chunk_index=policy.chunk_idx)
            elif kind == 'cache':
                policy.infer(dict(obs=observation(arrays['rgb']), compute_kv_cache=True))
                result = dict(cached=True, chunk_index=policy.chunk_idx)
            else:
                raise ValueError(f'Unknown RPC operation {kind}')
            torch.cuda.synchronize()
            result.update(seconds=time.perf_counter() - started,
                          gpu_allocated_bytes=torch.cuda.memory_allocated(),
                          gpu_peak_bytes=torch.cuda.max_memory_allocated())
            send(output_stream, result, **(result_arrays if kind == 'predict' else {}))
        except Exception as error:
            traceback.print_exc()
            send(output_stream, dict(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--denoise-steps', type=int, default=50)
    cli = parser.parse_args()
    with redirect_stdout(sys.stderr):
        main(cli, sys.stdin.buffer, sys.__stdout__.buffer)
