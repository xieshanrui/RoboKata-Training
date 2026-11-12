"""Frozen IG environments/rollout, with native model-specific policy adapters."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2), encoding='utf-8'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--indices', required=True)
    p.add_argument('--output', required=True); p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args(); config = read(args.config); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, config['ig_source'])
    sys.path.insert(0, str(Path(config['ig_source'])/'ManiSkill'))
    from examples.baselines.diffusion_policy.eval_dp_imitator import make_eval_envs_with_level, clear_l_level
    from examples.baselines.zero_wam.zero_wam.rollout import evaluate_with_task_encoder
    from examples.baselines.zero_wam.zero_wam.evaluation import RGBRolloutRecorder
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
    from examples.baselines.lerobot_dataset.evaluate_processor import HumanVideoSimEvaluateProcessor
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    torch.set_num_threads(4)
    if config['model_type'] == 'lingbot_text':
        from lingbot_ig_text.policy import Policy
        agent = Policy(config['policy'], output)
        mode = InputMode.LANGUAGE_ONLY
    else:
        from osvi_ig.evaluate_policy import Policy
        agent = Policy(config['policy'], output)
        mode = InputMode.VIDEO_ONLY

    class Processor:
        config = SimpleNamespace(normalization_method='native')
        normalizer = SimpleNamespace(normalize_state=lambda state, *args, **kwargs: state)
        def _resolve_dataset_idx(self, env_id): return 0
        normalize_state_rgb = HumanVideoSimEvaluateProcessor.normalize_state_rgb
        def denormalize_action(self, action, env_id): return action
        def get_task_description(self, env_id): return agent.text['prompts'][agent.human_id]
        def get_video(self, env_id, count): return torch.zeros(count, 1, 3, 224, 224)

    processor = Processor(); cases = read(config['case_plan'])['cases']
    for index in read(args.indices):
        case = cases[index]; agent.select_case(case)
        seed = case['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        lane = output/'smoke' if args.smoke_steps else Path(config['results'])/case['env_id']/f"episode_{case['episode']:03d}"
        lane.mkdir(parents=True, exist_ok=False)
        write(output/'current.json', dict(index=index, case=case, time=time.time()))
        level, base = case['env_id'].split('_', 1)
        steps = args.smoke_steps or 500
        env = make_eval_envs_with_level(base, level, 1, 'physx_cpu',
            dict(control_mode='pd_joint_pos', reward_mode='dense', obs_mode='rgb', render_mode='rgb_array',
                 max_episode_steps=steps, sensor_configs={'shader_pack':'rt-fast'}, human_render_camera_configs={'shader_pack':'rt-fast'}),
            {'obs_horizon':1}, None, [FlattenRGBDObservationWrapper])
        recorder = RGBRolloutRecorder(lane/'query.mp4', 30)
        trace = []; reset_count = [0]; started = time.time()

        class Recorded:
            def __getattr__(self, key): return getattr(env, key)
            def reset(self, **kwargs):
                obs, info = env.reset(seed=seed, **kwargs)
                reset_count[0] += 1
                if reset_count[0] == 1:
                    recorder.begin({'rgb':torch.as_tensor(obs['rgb']).float()/255.})
                    np.savez_compressed(lane/'initial.npz', rgb=np.asarray(obs['rgb']), state=np.asarray(obs['state']))
                return obs, info
            def step(self, action):
                result = env.step(action); recorder.actual_step(result)
                trace.append(np.concatenate([np.asarray(action['panda_wristcam-0']), np.asarray(action['panda_wristcam-1'])], axis=-1).reshape(16))
                assert np.isfinite(trace[-1]).all()
                if len(trace)%16 == 0:
                    write(lane/'progress.json', dict(actions=len(trace), seconds=time.time()-started, time=time.time()))
                return result
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                metrics = evaluate_with_task_encoder(1, agent, Recorded(),
                    dict(env_id=case['env_id'], delta_control=False, pred_horizon=config['prediction_horizon'],
                         temporal_agg=False, light_temporal_agg=False, dynamic_action_chunks=True,
                         max_timesteps=steps, device='cuda', sim_backend='physx_cpu'), processor, mode, progress_bar=False)
            assert len(trace)==steps
            np.save(lane/'actions.npy', np.stack(trace))
            row = dict(**case, max_episode_steps=steps, smoke_only=bool(args.smoke_steps),
                       checkpoint_sha256=config['checkpoint_sha256'], seconds=time.time()-started,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       metrics={k:np.asarray(v).reshape(-1).tolist() for k,v in metrics.items()})
            for key in ['success_once', 'success_at_end']:
                assert len(row['metrics'][key])==1 and row['metrics'][key][0] in [0,1]
            write(lane/'metrics.json', row)
        finally:
            recorder.finish(None); env.close(); clear_l_level(); agent.clear_cache()
        if args.smoke_steps:
            write(output/'smoke_complete.json', dict(actions=len(trace), elapsed_seconds=time.time()-started, finite=True)); break
    write(output/'complete.json', dict(time=time.time(), smoke_only=bool(args.smoke_steps)))


if __name__ == '__main__': main()
