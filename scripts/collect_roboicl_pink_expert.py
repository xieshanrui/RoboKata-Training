"""Generate one fresh Pink expert using native high-level grasp/place goals."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import transforms3d as t3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_pick_place import roboicl_pick_place
from envs.robot.roboicl_ee import FrankaKinematics, PinkController, CONTROL_RECIPE, encode10
from rollout_roboicl_policy import observe as observe_rgb
from rollout_roboicl_ee_policy import drive_command
from qualify_roboicl_ee import read, write, sha


EXPERT_RECIPE = dict(version=2, control_hz=25, physics_dt=.004,
    translation_speed_m_s=.30, rotation_speed_rad_s=1.5,
    minimum_move_duration_s=.20, trajectory='quintic ease-in/out in translation and shortest SO3',
    arrival_position_tolerance_m=.002, arrival_rotation_tolerance_rad=.02,
    arrival_max_arm_velocity_rad_s=.10, arrival_consecutive_tokens=3,
    arrival_timeout_s=3., gripper_duration_s=1.2,
    close_contact_target='smoothly move from reached native preclose measured pose to original static drive grasp goal while closing',
    final_stability_physics_steps=250,
    policy_time_limit_physics_steps=5000,
    query_chunk_tokens=10, ik_chunk_seed='actual_current_drive_command',
    high_level_grasp_source='accepted clearance/pregrasp endpoint FK; grasp from native measured hand at start of close; no dense path or source timing',
    placement_source='native get_place_pose from actual held object and measured robot pose')


def matrix(pose):
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
    return result


def score_trajectory(measurements, stages, target, destination, policy_steps, stability_steps):
    poses = np.asarray([r['object_poses'] for r in measurements])
    xyz = poses[:, :, :3]
    commands = np.asarray([r['gripper_command'] for r in measurements])
    lift = xyz[:, :, 2]-xyz[0, :, 2]
    ever_lifted = np.flatnonzero((lift > .06).any(axis=0)).tolist()
    others = [i for i in range(3) if i != target]
    start = stages.get('clearance', {}).get('end_step')
    end = stages.get('close', {}).get('start_step', len(poses)-1)
    observed_preclose = start is not None and 0 <= start <= end < len(poses)
    displacement = rotation = None
    if observed_preclose:
        pre = poses[start:end+1, target]
        q = pre[:, 3:]/np.linalg.norm(pre[:, 3:], axis=1, keepdims=True)
        displacement = float(np.linalg.norm(pre[:, :3]-pre[0, :3], axis=1).max())
        rotation = float(np.degrees(2*np.arccos(np.clip(np.abs(q@q[0]), 0, 1))).max())
    non_target = float(np.linalg.norm(xyz[:, others]-xyz[0, others], axis=-1).max())
    xy_error = np.linalg.norm(xyz[:, target, :2]-destination, axis=1)
    height_error = np.abs(xyz[:, target, 2]-xyz[0, target, 2])
    stable = bool(stability_steps == 250 and len(poses) == policy_steps+251
        and np.all(xy_error[policy_steps:] < .065)
        and np.all(height_error[policy_steps:] < .035)
        and np.all(commands[policy_steps:] > .8))
    geometric_success = bool(ever_lifted == [target] and xy_error[-1] < .065 and height_error[-1] < .035
        and commands[-1] > .8 and non_target < .03)
    stages_complete = all(stages.get(name, {}).get('status') == 'completed'
        and stages[name]['end_step'] <= len(poses)-1
        for name in ('clearance', 'pregrasp', 'grasp', 'close', 'lift', 'preplace', 'place', 'release', 'retreat'))
    task_success = bool(geometric_success and stages_complete and stable)
    return dict(task_success=task_success, object_choice_correct=ever_lifted == [target],
        geometric_task_predicates=geometric_success, complete_stage_sequence=stages_complete,
        ever_lifted_objects=ever_lifted, max_object_lift_m=lift.max(axis=0).tolist(),
        preclose_displacement_m=displacement, preclose_rotation_deg=rotation,
        preclose_observed=observed_preclose, preclose_window_complete=observed_preclose and 'close' in stages,
        nontarget_max_displacement_m=non_target,
        target_final_xy_error_m=float(xy_error[-1]), target_final_height_error_m=float(height_error[-1]),
        final_gripper_open=bool(commands[-1] > .8), released_stable_250=stable,
        policy_physics_steps=policy_steps, stability_physics_steps=stability_steps,
        policy_elapsed_s=policy_steps*.004,
        within_policy_time_budget=0 < policy_steps <= 5000,
        qualified=bool(task_success and observed_preclose and 'close' in stages and displacement <= .005 and rotation <= 10
            and non_target < .03 and stable and 0 < policy_steps <= 5000))

class Expert:
    def __init__(self, task, kin, output):
        self.task, self.kin, self.output = task, kin, output
        self.controller = PinkController(kin, drive_command(task.robot))
        self.measurements = [task.object_state()]
        self.rgb, self.states, self.actions, self.steps, self.applied, self.audit = [], [], [], [], [], []
        self.stages = {}
        self.pending_token = None
        self.goal = kin.pose(drive_command(task.robot))
        self.g = drive_command(task.robot)[-1]

    def measured_pose(self):
        return self.task.robot.left_entity.find_link_by_name('panda_hand').pose.to_transformation_matrix()

    def token(self, pose, gripper):
        if self.task.physics_step + 10 > EXPERT_RECIPE['policy_time_limit_physics_steps']:
            raise RuntimeError('Expert exhausted the frozen20s policy budget before complete placement/retreat')
        rgb, _ = observe_rgb(self.task)
        state = self.kin.state10(drive_command(self.task.robot)).astype(np.float32)
        start_step = self.task.physics_step
        action = encode10(pose, gripper).astype(np.float32)
        if len(self.actions) % EXPERT_RECIPE['query_chunk_tokens'] == 0:
            self.controller.reset(drive_command(self.task.robot))
        self.pending_token = dict(index=len(self.actions), start_step=start_step,
            action=action.tolist(), executed_physics_steps=0)
        positions, velocities, grippers, audit = self.controller.command(action)
        joints = self.task.robot.left_entity.get_active_joints()
        for q, dq, g in zip(positions, velocities, grippers, strict=True):
            self.task.robot.set_arm_joints(q, dq, 'left')
            self.task.robot.set_gripper(g, 'left', gripper_eps=0)
            applied = [float(j.get_drive_target()[0]) for j in joints]+[float(j.get_drive_velocity_target()[0]) for j in joints]
            self.task._step_scene()
            self.applied.append(applied)
            self.measurements.append(self.task.object_state())
            self.pending_token['executed_physics_steps'] += 1
        self.rgb.append(rgb)
        self.states.append(state)
        self.steps.append(start_step)
        self.actions.append(action)
        self.audit.append(audit)
        Image.fromarray(np.concatenate(rgb, axis=1)).save(self.output/'frames'/f'{len(self.actions)-1:06d}.png')
        self.pending_token = None

        self.goal, self.g = pose.copy(), float(gripper)

    def move(self, target, name):
        start = self.kin.pose(drive_command(self.task.robot))
        rotation = self.kin.pin.log3(start[:3, :3].T @ target[:3, :3])
        # Quintic's maximum normalized speed is1.875; scale to respect limits.
        duration = max(EXPERT_RECIPE['minimum_move_duration_s'],
            1.875*np.linalg.norm(target[:3, 3]-start[:3, 3])/EXPERT_RECIPE['translation_speed_m_s'],
            1.875*np.linalg.norm(rotation)/EXPERT_RECIPE['rotation_speed_rad_s'])
        count = int(np.ceil(duration*25))
        start_step = self.task.physics_step
        self.stages[name] = dict(status='started', start_step=start_step, target=target.tolist())
        write(self.output/'stages.json', self.stages)
        for i in range(1, count+1):
            t = i/count
            alpha = 10*t**3-15*t**4+6*t**5
            intermediate = np.eye(4)
            intermediate[:3, 3] = start[:3, 3]+alpha*(target[:3, 3]-start[:3, 3])
            intermediate[:3, :3] = start[:3, :3] @ self.kin.pin.exp3(alpha*rotation)
            self.token(intermediate, self.g)
        consecutive = 0
        for _ in range(int(EXPERT_RECIPE['arrival_timeout_s']*25)):
            pose = self.measured_pose()
            poserr = float(np.linalg.norm(pose[:3, 3]-target[:3, 3]))
            roterr = float(np.linalg.norm(self.kin.pin.log3(pose[:3, :3].T @ target[:3, :3])))
            speed = float(np.abs(self.task.robot.left_entity.get_qvel()[:7]).max())
            arrived = (poserr <= EXPERT_RECIPE['arrival_position_tolerance_m']
                and roterr <= EXPERT_RECIPE['arrival_rotation_tolerance_rad']
                and speed <= EXPERT_RECIPE['arrival_max_arm_velocity_rad_s'])
            consecutive = consecutive+1 if arrived else 0
            if consecutive >= EXPERT_RECIPE['arrival_consecutive_tokens']:
                self.stages[name] = dict(status='completed', start_step=start_step, end_step=self.task.physics_step,
                    measured_position_error_m=poserr, measured_rotation_error_rad=roterr,
                    max_arm_velocity_rad_s=speed, target=target.tolist())
                write(self.output/'stages.json', self.stages)
                print(json.dumps(dict(stage=name, physics_step=self.task.physics_step, position_error_m=poserr)), flush=True)
                return
            self.token(target, self.g)
        raise RuntimeError(f'{name} did not reach measured arrival criteria: position={poserr}, rotation={roterr}, speed={speed}')

    def gripper(self, target, name, contact_pose=None):
        start_step, g0 = self.task.physics_step, self.g
        self.stages[name] = dict(status='started', start_step=start_step)
        write(self.output/'stages.json', self.stages)
        pose0 = self.goal.copy()
        count = int(EXPERT_RECIPE['gripper_duration_s']*25)
        rotation = None if contact_pose is None else self.kin.pin.log3(pose0[:3, :3].T @ contact_pose[:3, :3])
        for index, g in enumerate(np.linspace(g0, target, count+1)[1:], 1):
            pose = pose0.copy()
            if contact_pose is not None:
                fraction = index/count
                pose[:3, 3] += fraction*(contact_pose[:3, 3]-pose0[:3, 3])
                pose[:3, :3] = pose0[:3, :3] @ self.kin.pin.exp3(fraction*rotation)
            self.token(pose, g)
        self.stages[name] = dict(status='completed', start_step=start_step, end_step=self.task.physics_step)
        write(self.output/'stages.json', self.stages)

    def save(self):
        np.savez_compressed(self.output/'policy_recording.npz', rgb=np.asarray(self.rgb, dtype=np.uint8),
            state=np.asarray(self.states, dtype=np.float32), action=np.asarray(self.actions, dtype=np.float32),
            observation_step=np.asarray(self.steps), timestamp_s=np.asarray(self.steps)*.004)
        rows = self.measurements
        np.savez_compressed(self.output/'measurements.npz',
            object_poses=np.asarray([r['object_poses'] for r in rows]),
            physics_step=np.asarray([r['physics_step'] for r in rows]),
            gripper_command=np.asarray([r['gripper_command'] for r in rows]),
            qpos=np.asarray([r['left_qpos'] for r in rows]), qvel=np.asarray([r['left_qvel'] for r in rows]),
            hand_pose=np.asarray([r['hand_pose'] for r in rows]),
            applied_controls=np.asarray(self.applied).reshape(-1, 18))
        write(self.output/'ik_audit.json', self.audit)
        if self.pending_token is not None:
            write(self.output/'incomplete_token.json', self.pending_token)

    def stability_hold(self):
        joints = self.task.robot.left_entity.get_active_joints()
        positions = np.asarray([j.get_drive_target()[0] for j in joints])
        for _ in range(EXPERT_RECIPE['final_stability_physics_steps']):
            for joint, position in zip(joints, positions, strict=True):
                joint.set_drive_target(position)
                joint.set_drive_velocity_target(0.)
            for entity in (self.task.robot.left_entity, self.task.robot.right_entity):
                entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.applied.append(np.r_[positions, np.zeros(9)])
            self.task._step_scene()
            self.measurements.append(self.task.object_state())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source').mkdir()
    (args.output/'frames').mkdir()
    for path in (Path(__file__), Path(__file__).parents[1]/'envs/robot/roboicl_ee.py'):
        shutil.copyfile(path, args.output/'source'/path.name)
    write(args.output/'frozen_recipe.json', dict(expert=EXPERT_RECIPE, controller=CONTROL_RECIPE,
        acceptance=dict(preclose_displacement_max_m=.005, preclose_rotation_max_deg=10,
            nontarget_max_displacement_m=.03, require_target_only_lift=True,
            require_native_complete_success=True, require_final250_stability=True,
            max_policy_physics_steps=5000, stability_gripper_open_threshold=.8),
        geometry_sha256=sha(args.geometry), source_control_sha256=sha(args.episode/'control_trace.json'),
        policy_inputs='Fresh RGB plus query current drive FK only; expert target/object information is generation-only'))
    config = read(args.episode/'config.json')
    config.update(save_data=False, record_control=False, need_plan=False, save_path=str(args.output))
    write(args.output/'config.json', config)
    kin = FrankaKinematics(args.geometry)
    frames = read(args.episode/'control_trace.json')['frames']
    arm_segments = []
    for _, group in itertools.groupby(frames, key=lambda r:r['segment']):
        rows = list(group)
        if rows[0]['setter_calls']['left_arm']:
            arm_segments.append(kin.pose(np.r_[rows[-1]['drive_position'][:7], 1.]))
    if len(arm_segments) < 3:
        raise ValueError('Expected clearance/pregrasp/grasp high-level goals')
    close_row = next(r for r in frames if r['setter_calls']['left_gripper'])
    original_grasp_drive = arm_segments[2].copy()
    arm_segments[2] = kin.pose(np.r_[close_row['measured_qpos'][:7], 1.])
    write(args.output/'high_level_goals.json', dict(clearance=arm_segments[0].tolist(),
        pregrasp=arm_segments[1].tolist(), grasp=arm_segments[2].tolist(),
        grasp_source_physics_step=close_row['physics_step'],
        grasp_source='measured_qpos FK, original native physics replay verified joint-for-joint identical',
        replaced_drive_grasp=original_grasp_drive.tolist(),
        measured_vs_drive_position_gap_m=float(np.linalg.norm(original_grasp_drive[:3, 3]-arm_segments[2][:3, 3])),
        measured_vs_drive_rotation_gap_rad=float(np.linalg.norm(kin.pin.log3(original_grasp_drive[:3, :3].T @ arm_segments[2][:3, :3]))),
        reason='Use native measured preclose pose for arrival; only during closing move toward native drive endpoint for contact preload',
        interpretation='Three static world-hand goals only; source intermediate controls and time are never executed'))
    task, expert = roboicl_pick_place(), None
    policy_steps, stability_steps = None, 0
    receipt = dict(status='starting', task_success=False, qualified=False, source=str(args.episode),
        target_index_evaluator_only=config['target_index'], generation='fresh_Pink_expert_not_old_control_replay')
    try:
        task.setup_demo(**config)
        initial = task.object_state()
        original = read(args.episode/'result.json')['replay_initial']
        for key in ('object_poses', 'left_qpos', 'left_qvel', 'left_drive_position', 'left_drive_velocity', 'right_qpos'):
            if not np.allclose(initial[key], original[key], atol=1e-6, rtol=0):
                raise ValueError('Initial state differs: '+key)
        kin.validate_robot(task.robot)
        expert = Expert(task, kin, args.output)
        # Native high-level grip frame has a fixed transform to measured hand.
        high_to_hand = np.linalg.inv(matrix(task.robot.get_left_ee_pose())) @ expert.measured_pose()
        write(args.output/'high_level_to_hand.json', high_to_hand.tolist())
        expert.move(arm_segments[0], 'clearance')
        expert.move(arm_segments[1], 'pregrasp')
        expert.move(arm_segments[2], 'grasp')
        expert.gripper(0., 'close', contact_pose=original_grasp_drive)
        lift = expert.measured_pose().copy()
        lift[2, 3] += .12
        expert.move(lift, 'lift')
        target_initial_z = initial['positions'][config['target_index']][2]
        task.target_lifted = bool(task.target.get_pose().p[2]-target_initial_z > .06)
        if not task.target_lifted:
            raise RuntimeError('Fresh Pink expert did not retain target lift at the lift endpoint')
        for distance, name in ((.1, 'preplace'), (.02, 'place')):
            pose = task.get_place_pose(task.target, task.arm_tag,
                [*task.destination, .742], constrain=task.expert_place_constraint, pre_dis=distance)
            expert.move(matrix(pose) @ high_to_hand, name)
        expert.gripper(1., 'release')
        retreat = expert.measured_pose().copy()
        retreat[2, 3] += .10
        expert.move(retreat, 'retreat')
        policy_steps = task.physics_step
        expert.stability_hold()
        stability_steps = 250
        receipt.update(status='completed', native_task_success=task.check_success(),
            physics_steps=task.physics_step, initial=initial, final=task.object_state())
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        try:
            if expert is not None:
                expert.save()
                receipt['physics_steps'] = task.physics_step
                receipt['stages'] = expert.stages
                receipt.update(initial=expert.measurements[0], final=expert.measurements[-1],
                    completed_tokens=len(expert.actions), incomplete_token=expert.pending_token)
                stability_steps = 0 if policy_steps is None else task.physics_step-policy_steps
                receipt.update(score_trajectory(expert.measurements, expert.stages, config['target_index'],
                    task.destination, task.physics_step if policy_steps is None else policy_steps, stability_steps))
                target = config['target_index']
                initial_z = expert.measurements[0]['positions'][target][2]
                task.target_lifted = any(r['positions'][target][2]-initial_z > .06 for r in expert.measurements)
                receipt['native_task_success'] = bool(task.check_success())
                receipt['qualified'] = bool(receipt['qualified'] and receipt['native_task_success'])
                if receipt['status'] != 'completed':
                    receipt['qualified'] = False
            write(args.output/'receipt.json', receipt)
        finally:
            if hasattr(task, 'scene'):
                task.close_env()
    print(json.dumps({k:receipt[k] for k in ('status', 'task_success', 'qualified')}), flush=True)


if __name__ == '__main__':
    main()
