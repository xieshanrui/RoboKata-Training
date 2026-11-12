"""Qualify a saved dense reference and a model-owned control adapter in physics."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import shutil

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_pick_place import roboicl_pick_place


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def run(task, folder, output, mode, adapter, control_steps=10):
    trace = read(folder/'control_trace.json')
    source = read(folder/'result.json')
    rows = trace['frames']
    if trace['version'] != 2 or not source['replay_success']:
        raise ValueError('A successful v2 dense reference is required')
    if any(r['setter_calls']['right_arm'] or r['setter_calls']['right_gripper'] for r in rows):
        raise ValueError('This adapter qualifies only the active-left, parked-right contract')
    config = read(folder/'config.json')
    config.update(save_data=False, record_control=False, save_path=str(output), need_plan=False)
    output.mkdir(parents=True, exist_ok=False)
    measurements = []
    try:
        task.setup_demo(**config)
        initial = task.object_state()
        if not np.allclose(initial['object_poses'], source['replay_initial']['object_poses'], atol=1e-5, rtol=0):
            raise ValueError('Replay initial objects differ from collection')
        if not np.allclose(initial['left_qpos'], rows[0]['measured_qpos'], atol=1e-6, rtol=0):
            raise ValueError('Replay initial robot state differs from collection')
        joints = task.robot.left_entity.get_active_joints()
        if [j.name for j in joints] != trace['joint_names']:
            raise ValueError('Joint order mismatch')
        def knot(row):
            return np.r_[row['drive_position'][:7], row['drive_velocity'][:7], row['gripper_command']] if mode == 'hermite_25hz' else np.r_[row['drive_position'][:7], row['gripper_command']]
        start = knot(rows[0])
        for step, row in enumerate(rows):
            if mode == 'reference':
                # Repeating a setter during a native hold can wake constraints.
                # Reproduce native setter timing rather than forcing every drive.
                for key, indices in (('left_arm', range(7)), ('left_gripper', range(7, 9))):
                    if row['setter_calls'][key]:
                        for i in indices:
                            joints[i].set_drive_target(row['drive_position'][i])
                            joints[i].set_drive_velocity_target(row['drive_velocity'][i])
                if row['setter_calls']['left_arm'] or row['setter_calls']['left_gripper']:
                    task.robot.left_entity.set_qf(np.array(row['applied_qf']))
                    task.robot.right_entity.set_qf(task.robot.right_entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
                task.robot.left_gripper_val = row['gripper_command']
            elif mode == 'dense_25hz':
                if step % 10 == 0:
                    block = [rows[min(step + i, len(rows) - 1)] for i in range(10)]
                    controls = np.array([r['drive_position'] + r['drive_velocity'] for r in block]).ravel()
                    positions, velocities = adapter.dense_drive_block(controls)
                for joint, position, velocity in zip(joints, positions[step % 10], velocities[step % 10], strict=True):
                    joint.set_drive_target(position)
                    joint.set_drive_velocity_target(velocity)
                # Deployment uses current physics, never reference forces or segment flags.
                for entity in (task.robot.left_entity, task.robot.right_entity):
                    entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
                task.robot.left_gripper_val = float(np.clip((positions[step % 10, 7:9].mean() + .01) / .06, 0, 1))
            else:
                if step % control_steps == 0:
                    length = min(control_steps, len(rows) - step)
                    endrow = rows[min(step + length, len(rows) - 1)]
                    endpoint = knot(endrow)
                    endpoint[-1] = row['gripper_command']
                    interpolate = adapter.joint_hermite_segment if mode == 'hermite_25hz' else adapter.joint_linear_segment
                    positions, velocities = interpolate(start, endpoint, physics_dt=trace['timestep_s'], steps=length)
                    start = endpoint
                task.robot.set_arm_joints(positions[step % control_steps, :7], velocities[step % control_steps], 'left')
                task.robot.set_gripper(positions[step % control_steps, 7], 'left')
            measurements.append(dict(physics_step=step, qpos=task.robot.left_entity.get_qpos().tolist(),
                                     qvel=task.robot.left_entity.get_qvel().tolist(),
                                     gripper_command=float(task.robot.get_left_gripper_val()),
                                     object_poses=[a.get_pose().p.tolist()+a.get_pose().q.tolist() for a in task.objects]))
            task._step_scene()
        final = task.object_state()
        poses = np.array([r['object_poses'] for r in measurements] + [final['object_poses']])
        target = config['target_index']
        others = [i for i in range(3) if i != target]
        initial_xyz = np.array(initial['positions'])
        lift_step = source['replay_stages']['lift']['physics_step']
        lift_height = float(poses[lift_step, target, 2] - initial_xyz[target, 2])
        task.target_lifted = lift_height > .06
        clearance = source['replay_stages']['clearance']['physics_step']
        closing = next(i for i, r in enumerate(measurements) if r['gripper_command'] < .999999)
        pre = poses[clearance:closing, target]
        q = pre[:, 3:] / np.linalg.norm(pre[:, 3:], axis=1, keepdims=True)
        displacement = float(np.linalg.norm(pre[:, :3] - pre[0, :3], axis=1).max())
        rotation = float(np.degrees(2*np.arccos(np.clip(np.abs(q @ q[0]), 0, 1))).max())
        qpos = np.array([r['qpos'] for r in measurements])
        reported_mode = mode.replace('25hz', '50hz') if control_steps == 5 else mode
        report = dict(mode=reported_mode, source=str(folder.resolve()), physics_steps=len(rows),
                      control_period_physics_steps=control_steps,
                      source_sha256=hashlib.sha256((folder/'control_trace.json').read_bytes()).hexdigest(),
                      adapter_sha256=adapter.source_sha256,
                      source_lift_stage_height_m=lift_height,
                      max_lift_height_m=float((poses[:, target, 2] - initial_xyz[target, 2]).max()),
                      task_success=task.check_success(), preclose_displacement_m=displacement,
                      preclose_rotation_deg=rotation,
                      qpos_max_error=float(np.abs(qpos - np.array([r['measured_qpos'] for r in rows])).max()),
                      object_position_max_error_m=float(np.abs(poses[:-1, :, :3] - np.array([r['object_poses'] for r in rows])[:, :, :3]).max()),
                      nontarget_max_displacement_m=float(np.linalg.norm(poses[:, others, :3] - initial_xyz[others], axis=-1).max()),
                      initial=initial, final=final)
        report['passed'] = bool(report['task_success'] and displacement <= .005 and rotation <= 10 and report['nontarget_max_displacement_m'] < .03)
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        np.savez_compressed(output/'measurements.npz', qpos=qpos, object_poses=poses)
        print(json.dumps({k: v for k, v in report.items() if k not in ('initial', 'final')}), flush=True)
        return report
    finally:
        if hasattr(task, 'scene'):
            task.close_env()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--episode', type=Path)
    source.add_argument('--collection', type=Path)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('linear_25hz', 'hermite_25hz', 'dense_25hz'), default='linear_25hz')
    parser.add_argument('--control-steps', type=int, choices=(5, 10), default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('model_control_adapter', args.adapter)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.source_sha256 = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    (args.output/'source').mkdir()
    for path in (args.adapter, Path(__file__), Path(__file__).resolve().parents[1]/'envs/roboicl_pick_place.py'):
        shutil.copyfile(path, args.output/'source'/path.name)
    episodes = [(args.episode, args.output)] if args.episode else []
    if args.collection:
        for case in read(args.collection/'cases.json'):
            result = read(args.collection/case['name']/'result.json')
            if not result.get('accepted_attempt'):
                raise ValueError(f'Collection contains a failed cell: {case["name"]}')
            episodes.append((args.collection/case['name']/result['accepted_attempt'], args.output/case['name']))
    reports = []
    # Match the collector's native reset path and retain its planner instances.
    task = roboicl_pick_place()
    for episode, output in episodes:
        reference = run(task, episode, output/'reference', 'reference', adapter, args.control_steps)
        if not reference['passed'] or reference['qpos_max_error'] > 1e-5 or reference['object_position_max_error_m'] > 1e-5:
            raise RuntimeError('Exact reference is not equivalent; candidate replay is not interpretable')
        reports.append(run(task, episode, output/'candidate', args.mode, adapter, args.control_steps))
    (args.output/'summary.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in reports),
        count=len(reports), reports=reports), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
