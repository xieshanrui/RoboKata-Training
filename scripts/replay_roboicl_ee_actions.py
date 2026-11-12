"""Independently replay recorded10D EE actions through Pink and fresh physics."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_pick_place import roboicl_pick_place
from envs.robot.roboicl_ee import FrankaKinematics, PinkController, CONTROL_RECIPE
from rollout_roboicl_policy import observe as observe_rgb
from rollout_roboicl_ee_policy import drive_command
from qualify_roboicl_ee import read, write, sha
from collect_roboicl_pink_expert import score_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = read(args.recording/'frozen_recipe.json')
    if recipe['controller'] != CONTROL_RECIPE or recipe['geometry_sha256'] != sha(args.geometry):
        raise ValueError('Recorded controller or geometry differs from replay')
    if recipe['expert'].get('query_chunk_tokens') != 10 or recipe['expert'].get('ik_chunk_seed') != 'actual_current_drive_command':
        raise ValueError('Recording must use the deployed10token current-drive reset convention')
    with np.load(args.recording/'policy_recording.npz', allow_pickle=False) as data:
        actions = data['action'].copy()
        reference_rgb = data['rgb'].copy()
        reference_state = data['state'].copy()
        observation_steps = data['observation_step'].copy()
    if actions.shape != (len(actions), 10) or not np.array_equal(observation_steps, np.arange(len(actions))*10):
        raise ValueError('Expected finite25Hz action sequence')
    receipt = dict(status='starting', passed=False, generation='EE action -> Pink -> physics; source q/dq never used as input',
        recording=str(args.recording), policy_recording_sha256=sha(args.recording/'policy_recording.npz'),
        controller_sha256=sha(Path(__file__).parents[1]/'envs/robot/roboicl_ee.py'),
        geometry_sha256=sha(args.geometry), error=None)
    config = read(args.recording/'config.json')
    config.update(save_path=str(args.output), save_data=False, record_control=False, need_plan=False)
    write(args.output/'config.json', config)
    shutil.copyfile(Path(__file__), args.output/Path(__file__).name)
    task = roboicl_pick_place()
    measurements, applied, state_errors, rgb_errors = [], [], [], []
    try:
        task.setup_demo(**config)
        measurements.append(task.object_state())
        kin = FrankaKinematics(args.geometry)
        kin.validate_robot(task.robot)
        controller = PinkController(kin, drive_command(task.robot))
        joints = task.robot.left_entity.get_active_joints()
        for index, action in enumerate(actions):
            rgb, _ = observe_rgb(task)
            state = kin.state10(drive_command(task.robot)).astype(np.float32)
            state_errors.append(float(np.max(np.abs(state-reference_state[index]))))
            rgb_errors.append(int(np.max(np.abs(rgb.astype(int)-reference_rgb[index].astype(int)))))
            if index % 10 == 0:
                controller.reset(drive_command(task.robot))
            positions, velocities, grippers, _ = controller.command(action)
            for q, dq, g in zip(positions, velocities, grippers, strict=True):
                task.robot.set_arm_joints(q, dq, 'left')
                task.robot.set_gripper(g, 'left', gripper_eps=0)
                applied.append([float(j.get_drive_target()[0]) for j in joints]+
                    [float(j.get_drive_velocity_target()[0]) for j in joints])
                task._step_scene()
                measurements.append(task.object_state())
        original = read(args.recording/'receipt.json')
        hold_steps = original.get('stability_physics_steps', 0)
        hold = np.asarray([j.get_drive_target()[0] for j in joints])
        for _ in range(hold_steps):
            for joint, position in zip(joints, hold, strict=True):
                joint.set_drive_target(position)
                joint.set_drive_velocity_target(0.)
            for entity in (task.robot.left_entity, task.robot.right_entity):
                entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            applied.append(np.r_[hold, np.zeros(9)])
            task._step_scene()
            measurements.append(task.object_state())
        result = dict(object_poses=np.asarray([r['object_poses'] for r in measurements]),
            physics_step=np.asarray([r['physics_step'] for r in measurements]),
            gripper_command=np.asarray([r['gripper_command'] for r in measurements]),
            qpos=np.asarray([r['left_qpos'] for r in measurements]), qvel=np.asarray([r['left_qvel'] for r in measurements]),
            hand_pose=np.asarray([r['hand_pose'] for r in measurements]), applied_controls=np.asarray(applied))
        # Reference physics is loaded only after execution and cannot guide IK.
        with np.load(args.recording/'measurements.npz', allow_pickle=False) as reference:
            errors = {key:float(np.max(np.abs(value-reference[key]))) for key, value in result.items()}
        np.savez_compressed(args.output/'measurements.npz', **result)
        receipt.update(status='completed', tokens=len(actions), physics_steps=task.physics_step,
            max_abs_errors=errors, state_max_abs_error=max(state_errors), rgb_max_abs_error=max(rgb_errors))
        receipt['numerical_equivalence_within_1e6'] = bool(all(v <= 1e-6 for v in errors.values()) and max(state_errors) <= 1e-6 and max(rgb_errors) == 0)
        receipt.update(score_trajectory(measurements, original['stages'], config['target_index'],
            np.asarray(config['destination']), len(actions)*10, hold_steps))
        target = config['target_index']
        initial_z = measurements[0]['positions'][target][2]
        task.target_lifted = any(r['positions'][target][2]-initial_z > .06 for r in measurements)
        receipt['native_task_success'] = bool(task.check_success())
        receipt['qualified'] = bool(receipt['qualified'] and receipt['native_task_success'])
        reset_errors = {key:float(np.max(np.abs(np.asarray(measurements[0][key])-original['initial'][key])))
            for key in ('object_poses', 'left_qpos', 'left_qvel', 'right_qpos', 'left_drive_position', 'left_drive_velocity')}
        receipt['initial_state_max_abs_errors'] = reset_errors
        receipt['passed'] = bool(receipt['qualified'] and max(reset_errors.values()) <= 1e-6)
        receipt['qualified_trajectory_reproduced'] = bool(original.get('qualified') and receipt['passed'])
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        write(args.output/'receipt.json', receipt)
        if hasattr(task, 'scene'):
            task.close_env()
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
