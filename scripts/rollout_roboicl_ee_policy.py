"""Run 25 Hz world-TCP10 policies through the qualified Franka Pink adapter."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_pick_place import roboicl_pick_place
from envs.robot.roboicl_ee import FrankaKinematics, PinkController, CONTROL_RECIPE
from rollout_roboicl_policy import Worker, read, write, digest, observe as observe_rgb, apply


class EEWorker(Worker):
    def predict(self, query, output):
        self.process.stdin.write(json.dumps(dict(input=str(query), output=str(output))) + '\n')
        self.process.stdin.flush()
        response = self.receive()
        if Path(response.get('output', '')).resolve() != output.resolve():
            raise ValueError(f'Unexpected worker response: {response}')
        actions = np.load(output, allow_pickle=False)
        if actions.shape != (50, 10) or not np.isfinite(actions).all():
            raise ValueError('Expected finite [50,10] world-TCP actions')
        return actions


def drive_command(robot):
    q = np.asarray([j.get_drive_target()[0] for j in robot.left_entity.get_active_joints()])
    return np.r_[q[:7], np.clip((q[7:9].mean() + .01) / .06, 0, 1)]


def observe(task, kinematics):
    rgb, _ = observe_rgb(task)
    return rgb, kinematics.state10(drive_command(task.robot)).astype(np.float32)


def run(args, worker=None):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'queries', 'actions', 'frames'):
        (output/name).mkdir()
    config, source = read(args.episode/'config.json'), read(args.episode/'result.json')
    if not source.get('replay_success'):
        raise ValueError('Expected an accepted replay source')
    geometry = read(args.geometry)
    geometry_urdf = (args.geometry.parent/geometry['urdf']).resolve()
    representation = read(args.checkpoint/'complete.json')['run_contract']['action_representation']
    expected_geometry = dict(tcp=geometry['tcp'], geometry_sha256=digest(args.geometry),
        geometry_urdf_sha256=digest(geometry_urdf))
    if any(representation.get(key) != value for key, value in expected_geometry.items()):
        raise ValueError('Checkpoint geometry differs from the requested controller geometry')
    config.update(save_data=False, record_control=False, need_plan=False, save_path=str(output))
    if args.target_index is not None:
        config['target_index'] = args.target_index
    spec = importlib.util.spec_from_file_location('roboicl_scorer', args.scorer)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    contract = read(args.acceptance_contract)
    scorer.validate_contract(contract)
    provenance = {}
    for path in (Path(__file__), Path(__file__).parents[1]/'envs/robot/roboicl_ee.py',
                 args.worker_script, args.geometry, args.scorer, args.acceptance_contract):
        shutil.copyfile(path, output/'source'/path.name)
        provenance[str(path.resolve())] = digest(path)
    for path in (args.episode/'result.json', args.episode/'config.json', args.demo,
                 args.normalization, args.checkpoint/'model.safetensors', args.checkpoint/'complete.json'):
        provenance[str(path.resolve())] = digest(path)
    write(output/'config.json', config)
    receipt = dict(kind='world_tcp10_closed_loop_trial', status='starting',
        formal_evaluation_requested=args.formal_evaluation, formal_success=None,
        policy_inputs=['Demo RGB', 'query RGB', 'query world TCP from current drive command'],
        action_encoding='world_tcp10_absolute_25hz_v1', controller=CONTROL_RECIPE,
        controller_recipe_sha256=hashlib.sha256(json.dumps(CONTROL_RECIPE, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest(),
        geometry_binding=expected_geometry,
        episode=str(args.episode), demo=str(args.demo), checkpoint=str(args.checkpoint),
        target_index_evaluator_only=config['target_index'],
        destination_xy_evaluator_only=config['destination'],
        initial_state_verified=False, expert_clearance_executed=False,
        max_policy_steps=args.max_steps, requested_stability_steps=args.stability_steps,
        completed_policy_steps=0, completed_stability_steps=0, inference_calls=0,
        execute_tokens=args.execute_tokens, physics_steps_per_token=10,
        bound_controls=True, actuator_projection='Pink configuration/velocity limits; signed gripper clipped to[-1,1]; 6D Gram-Schmidt; no Cartesian clipping',
        gripper_clipped_values=0, source_sha256=provenance, seed=args.seed,
        demo_control=args.demo_control, shuffle_seed=args.shuffle_seed, error=None)
    write(output/'receipt.json', receipt)
    task, measurements, applied, audits, latencies = None, [], [], [], []
    owns_worker = worker is None
    try:
        task = roboicl_pick_place()
        task.setup_demo(**config)
        initial = task.object_state()
        measurements.append(initial)
        errors = {}
        for key in ('object_poses', 'left_qpos', 'left_qvel', 'right_qpos', 'object_linear_velocity',
                    'object_angular_velocity', 'gripper_command', 'left_drive_position', 'left_drive_velocity'):
            value, reference = np.asarray(initial[key]), np.asarray(source['replay_initial'][key])
            if not np.allclose(value, reference, atol=1e-6, rtol=0):
                raise ValueError('Initial state mismatch: '+key)
            errors[key] = float(np.max(np.abs(value-reference)))
        kin = FrankaKinematics(args.geometry)
        kin.validate_robot(task.robot)
        controller = PinkController(kin, drive_command(task.robot))
        if abs(task.scene.get_timestep()-.004) > 1e-9:
            raise ValueError('Qualified physics timestep required')
        receipt.update(initial_state_verified=True, initial_state_max_abs_errors=errors,
            physics_timestep_s=float(task.scene.get_timestep()))
        if worker is None:
            worker = EEWorker(args, output)
        else:
            worker.validate_configuration(args)
        receipt['worker_demo_acknowledgement'] = worker.set_demo(args.demo, args.demo_control, args.seed, args.shuffle_seed)
        receipt['status'] = 'running'
        joints = task.robot.left_entity.get_active_joints()
        while receipt['completed_policy_steps'] < args.max_steps:
            index = receipt['inference_calls']
            rgb, state = observe(task, kin)
            query = output/'queries'/f'{index:05d}.npz'
            np.savez_compressed(query, rgb=rgb, state=state)
            if index == 0:
                receipt['initial_query_rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
            Image.fromarray(np.concatenate(rgb, axis=1)).save(output/'frames'/f'{task.physics_step:06d}.png')
            started = time.monotonic()
            actions = worker.predict(query, output/'actions'/f'{index:05d}.npy')
            latencies.append(time.monotonic()-started)
            receipt['inference_calls'] += 1
            controller.reset(drive_command(task.robot))
            for action in actions[:args.execute_tokens]:
                if receipt['completed_policy_steps'] >= args.max_steps:
                    break
                receipt['gripper_clipped_values'] += int(abs(action[-1]) > 1)
                positions, velocities, grippers, audit = controller.command(action)
                audits.append(audit)
                for q, dq, g in zip(positions, velocities, grippers, strict=True):
                    if receipt['completed_policy_steps'] >= args.max_steps:
                        break
                    task.robot.set_arm_joints(q, dq, 'left')
                    task.robot.set_gripper(g, 'left', gripper_eps=0)
                    applied.append([float(j.get_drive_target()[0]) for j in joints]+
                        [float(j.get_drive_velocity_target()[0]) for j in joints])
                    task._step_scene()
                    measurements.append(task.object_state())
                    receipt['completed_policy_steps'] += 1
            write(output/'receipt.json', receipt)
        hold = np.asarray([j.get_drive_target()[0] for j in joints])
        for _ in range(args.stability_steps):
            apply(task, joints, hold, np.zeros(9))
            applied.append(np.r_[hold, np.zeros(9)])
            measurements.append(task.object_state())
            receipt['completed_stability_steps'] += 1
        receipt['status'] = 'completed'
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if measurements:
                telemetry = dict(object_poses=np.asarray([r['object_poses'] for r in measurements]),
                    physics_step=np.asarray([r['physics_step'] for r in measurements]),
                    gripper_command=np.asarray([r['gripper_command'] for r in measurements]),
                    qpos=np.asarray([r['left_qpos'] for r in measurements]),
                    qvel=np.asarray([r['left_qvel'] for r in measurements]),
                    applied_controls=np.asarray(applied).reshape(-1, 18))
                np.savez_compressed(output/'measurements.npz', **telemetry)
                receipt.update(initial=measurements[0], final=measurements[-1], inference_latency_s=latencies)
                score = scorer.score_rollout(receipt, telemetry, contract)
                write(output/'score.json', score)
                receipt.update(score, task_success=score['physical_complete_success'])
                write(output/'ik_audit.json', audits)
            write(output/'receipt.json', receipt)
        finally:
            if worker is not None and owns_worker:
                worker.close()
            if task is not None and hasattr(task, 'scene'):
                task.close_env()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('episode', 'output', 'worker-python', 'worker-script', 'worker-source', 'checkpoint', 'normalization', 'tokenizer', 'demo', 'geometry'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute-tokens', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--stability-steps', type=int, default=0)
    parser.add_argument('--worker-timeout', type=float, default=600)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--denoise-steps', type=int, default=10)
    parser.add_argument('--demo-control', choices=('correct', 'masked', 'last_frame', 'shuffled'), default='correct')
    parser.add_argument('--shuffle-seed', type=int, default=20261004)
    parser.add_argument('--target-index', type=int, choices=(0, 1, 2))
    parser.add_argument('--formal-evaluation', action='store_true')
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--scorer', type=Path, default=root/'scripts/score_roboicl_rollout.py')
    parser.add_argument('--acceptance-contract', type=Path, default=root/'experiments/policy_acceptance_v2_20261002.json')
    args = parser.parse_args()
    if not 1 <= args.execute_tokens <= 50 or args.max_steps <= 0 or args.stability_steps < 0:
        parser.error('Invalid token or physics-step budgets')
    if args.formal_evaluation and (args.max_steps != 5000 or args.stability_steps != 250):
        parser.error('Formal trials require5000 policy and250 stability steps')
    receipt = run(args)
    print(json.dumps({k: receipt[k] for k in ('status', 'task_success', 'object_choice_correct')}))


if __name__ == '__main__':
    main()
