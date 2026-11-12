"""Run the frozen RGB-only Zero-WAM pilot through the existing 50Hz executor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import traceback
from contextlib import nullcontext
from collections import Counter
from filelock import FileLock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_strategy_policy import roboicl_strategy_policy
from envs.robot.roboicl_ee import FrankaKinematics, decode10
from collect_roboicl_random_pick import Collector, RECIPE, observe, measurements_array, validate_pick_geometry
from rollout_roboicl_ee_policy import drive_command


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    def scalar(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=scalar)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_for_local_gpu(args):
    while args.min_free_gpu_mib:
        free = float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], encoding='utf-8').splitlines()[0])
        if free >= args.min_free_gpu_mib:
            return
        print(json.dumps(dict(waiting_for_local_gpu=True,free_mib=free,
                              required_mib=args.min_free_gpu_mib)),flush=True)
        time.sleep(10)


def score(rows, config):
    positions = np.asarray([r['object_poses'][0][:3] for r in rows])
    bottom = np.asarray([r['object_bottom_z'][0] for r in rows])
    gripper = np.asarray([r['gripper_command'] for r in rows])
    lifted = float(positions[:, 2].max()-positions[0, 2])
    physical = bool(len(rows) >= 251 and lifted >= .10
        and np.all(np.linalg.norm(positions[-251:, :2]-config['destination'], axis=1) < .03)
        and np.all(np.abs(bottom[-251:]-.74) < .003)
        and np.linalg.norm(np.ptp(positions[-251:], axis=0)) < .001
        and np.all(gripper[-251:] > .8))
    if config['strategy'].get('sequence_rule') in ('requested_projection_exact','ordered_region_stops'):
        from score_roboicl_open_routes import score_open_route
        measured = dict(object_poses=np.asarray([r['object_poses'] for r in rows]),
            object_bottom_z=bottom[:, None], gripper_command=gripper)
        route = score_open_route(measured, config)
        requested = config['strategy']['sequence']
        return dict(physical_success=physical, route_success=route['route_success'],
            joint_success=bool(physical and route['route_success']), sequence_metrics=route,
            # Count required visits, including a requested return to the same region.
            # Order is scored separately; surplus visits cannot increase recall.
            region_recall=sum((Counter(route['requested_events']) & Counter(requested)).values())/len(requested),
            visits=route['regions'], route_window_steps=route['scoring_window_steps'],
            maximum_lift_m=lifted,
            final_destination_distance_m=float(np.linalg.norm(positions[-1, :2]-config['destination'])),
            object_path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))
    carried = (bottom > .79) & (gripper < .2)
    begins = np.flatnonzero(carried)
    start = int(begins[0]) if len(begins) else 0
    releases = np.flatnonzero((np.arange(len(rows)) > start) & (gripper > .8)) if len(begins) else []
    end = int(releases[0]) if len(releases) else len(rows)-1
    visits = []
    for region in config['strategy']['regions']:
        inside = carried[start:end+1] & np.all(np.abs(positions[start:end+1]-region['center_m']) <= region['half_size_m'], axis=1)
        hits = np.flatnonzero(np.convolve(inside.astype(int), np.ones(5, dtype=int), mode='valid') == 5) if len(inside) >= 5 else []
        visits.append(dict(region=region['id'], first_step=int(hits[0]+start) if len(hits) else None,
                           minimum_center_distance_m=float(np.linalg.norm(positions-np.asarray(region['center_m']), axis=1).min())))
    first = [v['first_step'] for v in visits]
    ordered = bool(all(v is not None for v in first) and all(a < b for a, b in zip(first, first[1:])))
    return dict(physical_success=physical, route_success=ordered, joint_success=bool(physical and ordered),
                region_recall=sum(v is not None for v in first)/len(first), visits=visits,
                route_window_steps=[start, end], maximum_lift_m=lifted,
                final_destination_distance_m=float(np.linalg.norm(positions[-1, :2]-config['destination'])),
                object_path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))


def run(args, trial, worker):
    from roboicl_frames import pose_matrix, hand_to_relative, relative_to_hand
    mode = 'preflight_expert' if args.expert_replay else args.mode
    folder = args.output / mode / trial['id']
    if (folder/'receipt.json').exists():
        old = read(folder/'receipt.json')
        if old.get('finished'):
            assert old['panel_sha256'] == sha(args.panel), 'Existing receipt belongs to another panel'
            print(json.dumps(dict(skipped=trial['id'])), flush=True)
            return old
        raise RuntimeError(f'Incomplete attempt needs explicit recovery: {folder}')
    folder.mkdir(parents=True)
    if worker is not None:
        write(folder/'worker_identity.json', worker.ready)
    config_path = Path(trial['query']['folder'])/'config.json'
    assert sha(config_path) == trial['query']['config_sha256']
    config = read(config_path)
    config.update(need_plan=False, save_data=False, record_control=False, save_path=str(folder))
    task = roboicl_strategy_policy()
    collector = None
    receipt = dict(trial=trial, panel_sha256=sha(args.panel), mode='expert_replay' if args.expert_replay else args.mode,
                   finished=False, valid_policy_outputs=False, worker_calls=[], initial_verified=False,
                   evaluator_sha256=sha(__file__))
    started = time.perf_counter()
    stage = 'setup'
    try:
        # Serialize admission through the first render so simultaneous workers
        # cannot all pass the same free-memory snapshot before allocating.
        admission = FileLock(str(args.output/'simulator_start.lock')) if args.min_free_gpu_mib else nullcontext()
        with admission:
            wait_for_local_gpu(args)
            task.setup_demo(**config)
            kin = FrankaKinematics(args.geometry)
            validate_pick_geometry(kin, task.robot, folder)
            initial = task.object_state()
            rgb, _ = observe(task)
            state = kin.state10(drive_command(task.robot)).astype(np.float32)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == trial['query']['first_rgb_sha256'], 'Query reset RGB mismatch'
            assert np.array_equal(state, np.asarray(trial['query']['state0'], dtype=np.float32)), 'Query reset state mismatch'
            reference = read(Path(trial['query']['folder'])/'receipt.json')['initial']
            checks = {key: float(np.max(np.abs(np.asarray(initial[key])-np.asarray(reference[key]))))
                      for key in ['object_poses', 'left_qpos', 'left_qvel', 'left_drive_position',
                                  'left_drive_velocity', 'object_linear_velocity', 'object_angular_velocity']}
            assert max(checks.values()) < 1e-6, checks
            receipt['initial_verified'] = True
            receipt['physical_reset_max_errors'] = checks
            initial_api = np.asarray(task.robot.get_left_ee_pose(), dtype=float)
            api_to_hand = np.linalg.inv(pose_matrix(initial_api)) @ np.asarray(initial['hand_pose'])
            recording = Path(trial['query']['folder'])/'policy_recording.npz'
            with np.load(recording, allow_pickle=False) as item:
                recorded_actions = item['action']
            errors = []
            for action in recorded_actions:
                pose, grip = decode10(action)
                recovered = relative_to_hand(hand_to_relative(pose, grip, initial_api, api_to_hand), initial_api, api_to_hand)
                errors.append(float(np.max(np.abs(recovered-action))))
            assert max(errors) < 1e-6
            write(folder/'calibration.json', dict(initial_api=initial_api.tolist(), api_to_hand=api_to_hand.tolist(),
                 maximum_recorded_action_roundtrip_error=max(errors), initial=initial))
            recipe = dict(RECIPE, control_hz=50, query_chunk_tokens=32, max_policy_physics_steps=6000,
                          enforce_predictive_clearance=False, write_preview_frames=False)
            collector = Collector(task, kin, folder, recipe)
        if worker is not None:
            stage = 'reset'
            ack, _ = worker.request('reset', demo=args.remote_root+'/demos/'+Path(trial['demo']).name,
                                     control=trial['control'], seed=trial['seed'])
            receipt['demo_ack'] = ack
        first, cache_frames, chunks, predictions = True, [], 0, []
        while task.physics_step < 6000:
            if args.expert_replay:
                if chunks >= len(recorded_actions):
                    break
                actions = recorded_actions[chunks:chunks+32]
                chunks += len(actions)
            else:
                stage = 'predict'
                meta, values = worker.request('predict', rgb=rgb)
                raw = values['action']
                assert raw.shape == ((8 if args.mode == 'finetuned' else 16), 2, 16), raw.shape
                receipt['valid_policy_outputs'] = True
                stage = 'validate_prediction'
                assert np.isfinite(raw).all(), 'Non-finite model prediction'
                np.save(folder/f'raw_actions_{chunks:03d}.npy', raw)
                left = raw[:8, 1 if first else 0:].transpose(1, 2, 0).reshape(-1, 8)
                actions = np.stack([relative_to_hand(a, initial_api, api_to_hand) for a in left]).astype(np.float32)
                receipt['worker_calls'].append(meta)
                chunks += 1
                first = False
            cache_frames = []
            for n, action in enumerate(actions):
                if task.physics_step+5 > 6000:
                    break
                stage = 'execute'
                collector.action(action)
                if (n+1) % 4 == 0:
                    frame, _ = observe(task)
                    cache_frames.append(frame)
                if not args.expert_replay and score(collector.rows, config)['physical_success']:
                    break
            metrics = score(collector.rows, config)
            print(json.dumps(dict(trial=trial['id'], physics_step=task.physics_step, chunks=chunks,
                                  maximum_lift_m=metrics['maximum_lift_m'], physical=metrics['physical_success'])), flush=True)
            if not args.expert_replay and metrics['physical_success']:
                break
            if worker is not None and task.physics_step < 6000:
                assert len(cache_frames) in [4, 8], len(cache_frames)
                stage = 'cache'
                meta, _ = worker.request('cache', rgb=np.stack(cache_frames))
                receipt.setdefault('cache_calls', []).append(meta)
                rgb = cache_frames[-1]
        receipt['status'] = 'completed'
    except Exception as error:
        traceback.print_exc()
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                       failure_stage=stage, infrastructure_failure=stage in ['setup', 'reset', 'predict', 'cache'])
    finally:
        if collector is not None:
            active_steps = task.physics_step
            hold_rgb = []
            for _ in range(50):
                collector.hold(5)
                hold_rgb.append(observe(task)[0])
            measurements = measurements_array(collector.rows, collector.applied)
            np.savez_compressed(folder/'measurements.npz', **measurements)
            np.savez_compressed(folder/'recording.npz', rgb=np.asarray(collector.rgb), state=np.asarray(collector.states),
                                action=np.asarray(collector.actions), hold_rgb=np.asarray(hold_rgb))
            write(folder/'contacts.json', task.contact_events)
            receipt['metrics'] = score(collector.rows, config)
            if receipt.get('status') != 'completed':
                receipt['metrics']['joint_success'] = False
            receipt.update(active_physics_steps=active_steps, actions_executed=len(collector.actions),
                           controller_audits=collector.audits,
                           robot_table_contacts=sum('table' in e['bodies'] and any(n.startswith('panda_') for n in e['bodies']) for e in task.contact_events))
            write(folder/'head_calibration.json', task.cameras.get_config()['head_camera'])
        receipt.update(finished=True, wall_seconds=time.perf_counter()-started)
        write(folder/'receipt.json', receipt)
        write(folder/'config.json', config)
        if hasattr(task, 'scene'):
            task.close_env()
    print(json.dumps(dict(finished=trial['id'], status=receipt['status'], metrics=receipt.get('metrics'))), flush=True)
    return receipt


def main(args):
    sys.path.insert(0, str(args.model_repo/'adapters'))
    panel = read(args.panel)
    trials = [t for i, t in enumerate(panel['trials']) if i % args.shards == args.shard]
    if args.trial:
        trials = [t for t in panel['trials'] if t['id'] == args.trial]
    worker = None
    try:
        if not args.expert_replay:
            wait_for_local_gpu(args)
            from roboicl_client import Worker
            worker = Worker(args.host, args.remote_python, args.remote_source, args.remote_model or args.remote_root+'/weights',
                            args.remote_root+f'/worker_{args.mode}_{args.gpu}', args.output/f'worker_{args.mode}_{args.gpu}.log', gpu=args.gpu)
            write(args.output/f'worker_{args.mode}_{args.gpu}_ready.json', worker.ready)
        for trial in trials:
            run(args, trial, worker)
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['panel', 'geometry', 'model-repo', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    for name in ['host', 'remote-python', 'remote-source', 'remote-root']:
        p.add_argument('--'+name)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--trial')
    p.add_argument('--mode', choices=['zero_shot', 'finetuned'], default='zero_shot')
    p.add_argument('--remote-model')
    p.add_argument('--min-free-gpu-mib',type=int,default=0)
    p.add_argument('--expert-replay', action='store_true')
    main(p.parse_args())
