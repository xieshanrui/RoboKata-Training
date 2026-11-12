"""Qualify EE expert compression through Pink and real Franka physics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_pick_place import roboicl_pick_place
from envs.robot.roboicl_ee import FrankaKinematics, PinkController, CONTROL_RECIPE
from replay_roboicl_controls import run as reference_run


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_command(row):
    g = float(np.clip((np.mean(row['drive_position'][7:9]) + .01) / .06, 0, 1))
    return np.r_[row['drive_position'][:7], g]


def candidate(task, folder, output, geometry):
    output.mkdir()
    source, trace = read(folder/'result.json'), read(folder/'control_trace.json')
    rows = trace['frames']
    config = read(folder/'config.json')
    config.update(save_data=False, record_control=False, need_plan=False, save_path=str(output))
    measurements, audits, controls, actions = [], [], [], []
    try:
        task.setup_demo(**config)
        initial = task.object_state()
        for key in ('object_poses', 'left_qpos', 'left_qvel', 'left_drive_position', 'left_drive_velocity'):
            if not np.allclose(initial[key], source['replay_initial'][key], atol=1e-6, rtol=0):
                raise ValueError('Reset mismatch: ' + key)
        if abs(task.scene.get_timestep() - .004) > 1e-9:
            raise ValueError('Physics clock mismatch')
        if not geometry.exists():
            urdf = Path(task.robot.left_urdf_path).resolve()
            copied = geometry.parent/'panda.urdf'
            shutil.copyfile(urdf, copied)
            write(geometry, dict(version=1, urdf='panda.urdf', urdf_sha256=sha(urdf),
                frame='world', state_source='drive_command', tcp='panda_hand',
                tcp_offset_from_panda_hand=np.eye(4).tolist(),
                tcp_reason='Native Franka move_group and measured hand_pose use panda_hand; no additional grasp offset in policy geometry.',
                arm_joint_names=[j.name for j in task.robot.left_arm_joints],
                world_from_base=task.robot.left_entity.get_root_pose().to_transformation_matrix().tolist(),
                rotation6d='concatenated first two matrix columns', signed_gripper='2*native_open_fraction-1',
                state_timing='observation before applying row k: replay_initial drive at k=0, row k-1 otherwise',
                action_timing=f'at observation physics_step k, token h uses drive row min(k+{CONTROL_RECIPE["integration_steps"]}*(h+1)-1,last)',
                control_recipe=CONTROL_RECIPE))
        kin = FrankaKinematics(geometry)
        fk_error = kin.validate_robot(task.robot)
        initial_g = float(np.clip((np.mean(initial['left_drive_position'][7:9])+.01)/.06, 0, 1))
        initial_command = np.r_[initial['left_drive_position'][:7], initial_g]
        controller = PinkController(kin, initial_command)
        joints = task.robot.left_entity.get_active_joints()
        period = CONTROL_RECIPE['integration_steps']
        for start in range(0, len(rows), period):
            endpoint = native_command(rows[min(start + period-1, len(rows)-1)])
            action = kin.state10(endpoint).astype(np.float32)
            positions, velocities, grippers, audit = controller.command(action)
            audits.append(audit)
            actions.append(action)
            for sub in range(period):
                g = grippers[sub]
                # Only decoded EE and the IK state reach the controller; no source q/dq/forces.
                task.robot.set_arm_joints(positions[sub], velocities[sub], 'left')
                task.robot.set_gripper(g, 'left', gripper_eps=0)
                controls.append(np.r_[positions[sub], velocities[sub], g])
                measurements.append(task.object_state())
                task._step_scene()
        final = task.object_state()
        measurements.append(final)
        poses = np.asarray([r['object_poses'] for r in measurements])
        initial_xyz = poses[0, :, :3]
        target = config['target_index']
        other = [i for i in range(3) if i != target]
        lift = poses[:, :, 2] - initial_xyz[:, 2]
        ever_lifted = np.flatnonzero((lift > .06).any(axis=0)).tolist()
        task.target_lifted = lift[:, target].max() > .06
        clearance = source['replay_stages']['clearance']['physics_step']
        closing = next(i for i, r in enumerate(measurements) if r['gripper_command'] < .999999)
        pre = poses[clearance:closing, target]
        q = pre[:, 3:] / np.linalg.norm(pre[:, 3:], axis=1, keepdims=True)
        displacement = float(np.linalg.norm(pre[:, :3]-pre[0, :3], axis=1).max())
        rotation = float(np.degrees(2*np.arccos(np.clip(np.abs(q @ q[0]), 0, 1))).max())
        nontarget = float(np.linalg.norm(poses[:, other, :3]-initial_xyz[other], axis=-1).max())
        commands = np.asarray([r['gripper_command'] for r in measurements])
        last = poses[-251:, target, :3]
        stable = bool(np.all(np.linalg.norm(last[:, :2]-task.destination, axis=1) < .065)
            and np.all(np.abs(last[:, 2]-initial_xyz[target, 2]) < .035)
            and np.all(commands[-251:] > .8))
        report = dict(source=str(folder), source_sha256=sha(folder/'control_trace.json'),
            native_gripper_metadata_max_difference=max(abs(native_command(r)[-1]-r['gripper_command']) for r in rows),
            geometry_sha256=sha(geometry), source_physics_steps=len(rows),
            physics_steps=len(controls), tail_hold_padding_steps=len(controls)-len(rows), fk_sim_max_abs_error=fk_error,
            task_success=task.check_success(), object_choice_correct=ever_lifted == [target],
            ever_lifted_objects=ever_lifted, max_lift_m=lift.max(axis=0).tolist(),
            preclose_displacement_m=displacement, preclose_rotation_deg=rotation,
            nontarget_max_displacement_m=nontarget, released_stable_250=stable,
            ik_max_position_error_m=max(r['position_error_m'] for r in audits),
            ik_max_rotation_error_rad=max(r['rotation_error_rad'] for r in audits),
            initial=initial, final=final)
        report['passed'] = bool(report['task_success'] and report['object_choice_correct']
            and displacement <= .005 and rotation <= 10 and nontarget < .03 and stable)
        write(output/'result.json', report)
        np.savez_compressed(output/'measurements.npz', object_poses=poses, gripper_command=commands,
            qpos=np.asarray([r['left_qpos'] for r in measurements]), applied_controls=np.asarray(controls),
            ee_actions=np.asarray(actions), physics_step=np.arange(len(measurements)))
        return report
    finally:
        if hasattr(task, 'scene'):
            task.close_env()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--control-hz', type=int, choices=(25, 50, 250), default=25,
        help='50/250 are qualification diagnostics only; production contract remains25Hz')
    args = parser.parse_args()
    CONTROL_RECIPE.update(control_hz=args.control_hz, integration_steps=250//args.control_hz)
    args.output.mkdir(parents=True, exist_ok=False)
    panel = read(args.panel)
    trials = panel['trials']
    if panel.get('kind') == 'selected990_train_state_first_object_choice_diagnostic':
        # Policy panel shares target0 reset sources; expert qualification needs
        # each target's own accepted expert trajectory in that same initial state.
        episodes = []
        for trial in trials:
            cell = Path(trial['episode']).parent.parent / (trial['initial_state_id'] + '_t' + str(trial['target_index']))
            accepted = read(cell/'result.json')['accepted_attempt']
            episodes.append(str(cell/accepted))
        episodes = list(dict.fromkeys(episodes))
    else:
        episodes = list(dict.fromkeys(t['episode'] for t in trials))
    if args.limit:
        episodes = episodes[:args.limit]
    write(args.output/'frozen_qualification.json', dict(episodes=episodes,
        ee_action_serialization='float32 matching exported labels and inference actions',
        selection_source=str(args.panel), selection_sha256=sha(args.panel),
        controller=CONTROL_RECIPE, task_success='native task.check_success',
        choice='only target ever lifted >.06m', preclose_displacement_max_m=.005,
        preclose_rotation_max_deg=10, nontarget_displacement_max_m=.03,
        stability='final251states all placement XY<.065m height<.035m gripper>.8',
        exact_joint_match_required=False, require_all_pass=True))
    source_dir = args.output/'source'
    source_dir.mkdir()
    for path in (Path(__file__), Path(__file__).parents[1]/'envs/robot/roboicl_ee.py'):
        shutil.copyfile(path, source_dir/path.name)
    task = roboicl_pick_place()
    reports = []
    adapter = SimpleNamespace(source_sha256=sha(source_dir/'roboicl_ee.py'))
    try:
        for index, episode in enumerate(episodes):
            case = args.output/f'case_{index:03d}'
            case.mkdir()
            reference = reference_run(task, Path(episode), case/'reference', 'reference', adapter, CONTROL_RECIPE['integration_steps'])
            if not reference['passed'] or reference['qpos_max_error'] > 1e-5 or reference['object_position_max_error_m'] > 1e-5:
                raise RuntimeError('Exact reference failed')
            result = candidate(task, Path(episode), case/'candidate', args.output/'geometry.json')
            reports.append(result)
            write(args.output/'summary.json', dict(status='running', count=len(reports), passed=all(r['passed'] for r in reports), reports=reports))
            print(json.dumps(dict(case=index, passed=result['passed'], task_success=result['task_success'],
                choice=result['object_choice_correct'], preclose_m=result['preclose_displacement_m'],
                ik_error_m=result['ik_max_position_error_m'])), flush=True)
        write(args.output/'summary.json', dict(status='completed', count=len(reports), passed=all(r['passed'] for r in reports), reports=reports))
    except Exception:
        write(args.output/'failure.json', dict(traceback=traceback.format_exc(), completed=len(reports)))
        raise


if __name__ == '__main__':
    main()
