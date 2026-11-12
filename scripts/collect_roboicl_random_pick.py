"""Collect native pure-pick EE commands through Pink, or independently replay them."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import transforms3d as t3d
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_random_pick import roboicl_random_pick, SPAWN_ADMISSION
from envs.robot.roboicl_ee import FrankaKinematics, PinkController, CONTROL_RECIPE, encode10, decode10
from rollout_roboicl_ee_policy import drive_command
from qualify_roboicl_ee import read, write, sha

RECIPE = dict(version=4, control_hz=25, physics_dt=.004, query_chunk_tokens=10,
    initialization_fk_matrix_tolerance=1e-5, spawn_admission=SPAWN_ADMISSION,
    ik_chunk_seed='actual_current_drive_command', translation_speed_m_s=.30,
    rotation_speed_rad_s=1.5, close_duration_s=.40, lift_distance_m=.15,
    final_stability_physics_steps=125, max_policy_physics_steps=2500,
    trajectory='quintic moves, no arrival waits; approach -> close -> lift',
    grasp_source='official create_target_pose_list contact tilt candidates; nearest start-hand rotation; collision-derived grasp_dis; no native planner trajectory',
    planned_gripper_table_clearance_m=.003, commanded_gripper_table_clearance_m=.001,
    pre_grasp_dis_m=.10, candidate_max_hand_z_world_z=-.7,
    training_excludes_final_stability=True)
ACCEPTANCE = dict(lift_min_m=.10, final_hold_steps=125, gripper_max_open_fraction=.2,
    require_no_object_table_support=True, two_finger_contact_diagnostic_only=True,
    require_no_robot_table_contact=True, contact_impulse_threshold_ns=1e-6,
    expert_quality_version=1, actual_finger_penetration_tolerance_m=.001,
    require_ee_replay_equivalence=True, success_rate_target=.90,
    development_only=True)


def matrix(pose):
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
    return result


def observe(task):
    task._update_render()
    task.cameras.update_picture()
    raw = task.cameras.get_rgb()
    native = [raw[name]['rgb'] for name in ('head_camera', 'left_camera')]
    rgb = np.stack([np.asarray(ImageOps.pad(Image.fromarray(frame), (224, 224),
        method=Image.Resampling.BILINEAR)) for frame in native])
    return rgb, native


def preview_image(native):
    return Image.fromarray(np.concatenate([np.asarray(ImageOps.pad(Image.fromarray(frame),
        (640, 480), method=Image.Resampling.BILINEAR)) for frame in native], axis=1))


def validate_pick_geometry(kin, robot, output):
    """Use the established SAPIEN float32 FK audit tolerance for this task only."""
    root_error = float(np.max(np.abs(robot.left_entity.get_root_pose().to_transformation_matrix()-kin.world_from_base)))
    names = [joint.name for joint in robot.left_arm_joints]
    fk = kin.pose(np.r_[robot.left_entity.get_qpos()[:7],1.])
    hand = robot.left_entity.find_link_by_name('panda_hand').entity_pose.to_transformation_matrix()
    rotation = fk[:3,:3].T @ hand[:3,:3]
    u, _, vh = np.linalg.svd(rotation)
    diagnostic = dict(root_matrix_max_abs_error=root_error, arm_joint_names=names,
        expected_arm_joint_names=kin.joint_names, urdf_sha256=sha(kin.urdf),
        fk=fk.tolist(), measured_hand=hand.tolist(), matrix_max_abs_error=float(np.max(np.abs(fk-hand))),
        translation_l2_error_m=float(np.linalg.norm(fk[:3,3]-hand[:3,3])),
        rotation_angle_rad=float(np.linalg.norm(kin.pin.log3(u @ vh))),
        matrix_tolerance=RECIPE['initialization_fk_matrix_tolerance'])
    write(output/'initial_geometry_validation.json',diagnostic)
    if root_error > 1e-6 or names != kin.joint_names or diagnostic['matrix_max_abs_error'] > diagnostic['matrix_tolerance']:
        raise ValueError('Pure-pick robot geometry validation failed; inspect initial_geometry_validation.json')
    return diagnostic


def measurements_array(rows, applied):
    return dict(object_poses=np.asarray([r['object_poses'] for r in rows]),
        physics_step=np.asarray([r['physics_step'] for r in rows]),
        gripper_command=np.asarray([r['gripper_command'] for r in rows]),
        qpos=np.asarray([r['left_qpos'] for r in rows]),
        qvel=np.asarray([r['left_qvel'] for r in rows]),
        hand_pose=np.asarray([r['hand_pose'] for r in rows]),
        object_bottom_z=np.asarray([r['object_bottom_z'] for r in rows]),
        finger_bottom_z=np.asarray([r['finger_bottom_z'] for r in rows]),
        applied_controls=np.asarray(applied).reshape(-1, 18))


def score(task, rows, policy_steps, stability_steps):
    xyz = np.asarray([r['object_poses'] for r in rows])[:, :, :3]
    lift = xyz[:, :, 2]-xyz[0, :, 2]
    tail = lift[policy_steps:]
    held = np.flatnonzero((tail >= ACCEPTANCE['lift_min_m']).all(axis=0)).tolist()
    commands = np.asarray([r['gripper_command'] for r in rows])
    events = [e for e in task.contact_events if e['physics_step'] >= policy_steps]
    supported, finger_contacts = [], {}
    for index in held:
        name = task.object_specs[index]['asset']
        contacts = [e for e in events if name in e['bodies']]
        if any('table' in e['bodies'] for e in contacts):
            supported.append(index)
        steps = {step:set() for step in range(policy_steps, policy_steps+stability_steps+1)}
        for event in contacts:
            steps[event['physics_step']].update(n for n in event['bodies'] if 'finger' in n)
        finger_contacts[str(index)] = bool(steps and all(len(names) >= 2 for names in steps.values()))
    success = bool(stability_steps == 125 and len(rows) == policy_steps+126 and held
        and (commands[policy_steps:] < .2).all() and not supported)
    robot_table_contacts = [e for e in task.contact_events if 'table' in e['bodies']
        and any(n.startswith('panda_') for n in e['bodies'])]
    minimum_finger_z = np.min([r['finger_bottom_z'] for r in rows],axis=0)
    quality = bool(not robot_table_contacts and minimum_finger_z.min() >=
        SPAWN_ADMISSION['table_height_m']-ACCEPTANCE['actual_finger_penetration_tolerance_m'])
    return dict(task_success=success, data_quality_passed=quality, expert_quality_version=1,
        qualified=success and quality, stable_lifted_objects=held,
        maximum_lift_m=lift.max(axis=0).tolist(), final_lift_m=lift[-1].tolist(),
        table_supported_objects=supported, continuous_two_finger_contact=finger_contacts,
        robot_table_contact_count=len(robot_table_contacts),
        robot_table_contact_max_impulse_ns=max([e['impulse_ns'] for e in robot_table_contacts], default=0.),
        minimum_finger_bottom_z_m=minimum_finger_z.tolist(),
        minimum_object_bottom_z_during_hold_m=np.min([r['object_bottom_z'] for r in rows[policy_steps:]],axis=0).tolist(),
        policy_physics_steps=policy_steps, stability_physics_steps=stability_steps,
        policy_elapsed_s=policy_steps*.004, simulation_elapsed_s=(len(rows)-1)*.004)


class Collector:
    def __init__(self, task, kin, output, recipe, arm_table_guard=None):
        self.task, self.kin, self.output, self.recipe = task, kin, output, recipe
        self.record_rgb = recipe.get('record_rgb', True)
        if recipe.get('write_preview_frames', False) and not self.record_rgb:
            raise ValueError('Preview frames require RGB recording')
        self.arm_table_guard = arm_table_guard
        self.controller = PinkController(kin, drive_command(task.robot), control_hz=recipe.get('control_hz',25))
        self.period = self.controller.integration_steps
        self.rows, self.applied = [task.object_state()], []
        self.rgb, self.states, self.actions, self.audits, self.stages = [], [], [], [], {}
        self.native_rgb = [[], []]
        self.timing = dict(observation_render_s=0., ik_physics_s=0., image_write_s=0.)
        self.goal = kin.pose(drive_command(task.robot))
        self.g = drive_command(task.robot)[-1]
        self.pending_token = None
        self.high_to_hand = np.linalg.inv(matrix(task.robot.get_left_ee_pose())) @ np.asarray(self.rows[0]['hand_pose'])
        self.high_to_hand_max_error = 0.
        self.commanded_gripper_bottom_z, self.target_gripper_bottom_z = [], []
        self.pink_target_hand_poses, self.commanded_hand_poses = [], []
        self.planned_hand_poses = []
        self.previous_serialized_target = self.goal.copy()

    def token(self, pose, gripper):
        action = encode10(pose, gripper).astype(np.float32)
        self.action(action)
        self.goal, self.g = pose.copy(), float(gripper)

    def action(self, action):
        if self.task.physics_step+self.period > self.recipe['max_policy_physics_steps']:
            raise RuntimeError('Pure-pick policy budget exhausted')
        started = time.perf_counter()
        if self.record_rgb:
            rgb, native = observe(self.task)
        self.timing['observation_render_s'] += time.perf_counter()-started
        started = time.perf_counter()
        state = self.kin.state10(drive_command(self.task.robot)).astype(np.float32)
        if len(self.actions) % self.recipe.get('query_chunk_tokens',10) == 0:
            self.controller.reset(drive_command(self.task.robot))
        target_pose, _ = decode10(action)
        enforce_clearance = self.recipe.get('enforce_predictive_clearance', True)
        bottom = self.task.gripper_bottom_at_hand_pose(target_pose)
        if enforce_clearance and bottom < SPAWN_ADMISSION['table_height_m']+self.recipe['planned_gripper_table_clearance_m']-1e-6:
            raise RuntimeError('Serialized EE target violates gripper/table clearance')
        self.target_gripper_bottom_z.append(bottom)
        start_pose = self.kin.pose(np.r_[self.controller.configuration.q, self.controller.gripper])
        delta = self.kin.pin.log3(start_pose[:3,:3].T @ target_pose[:3,:3])
        pink_targets = []
        planned_targets = []
        planned_delta = self.kin.pin.log3(self.previous_serialized_target[:3,:3].T @ target_pose[:3,:3])
        for fraction in np.arange(1,self.period+1)/self.period:
            pose = np.eye(4)
            pose[:3,3] = start_pose[:3,3]+fraction*(target_pose[:3,3]-start_pose[:3,3])
            pose[:3,:3] = start_pose[:3,:3] @ self.kin.pin.exp3(fraction*delta)
            pink_targets.append(pose)
            planned = np.eye(4)
            planned[:3,3] = self.previous_serialized_target[:3,3]+fraction*(target_pose[:3,3]-self.previous_serialized_target[:3,3])
            planned[:3,:3] = self.previous_serialized_target[:3,:3] @ self.kin.pin.exp3(fraction*planned_delta)
            planned_targets.append(planned)
        planned_min = min(self.task.gripper_bottom_at_hand_pose(pose) for pose in planned_targets)
        pink_min = min(self.task.gripper_bottom_at_hand_pose(pose) for pose in pink_targets)
        if enforce_clearance and planned_min < SPAWN_ADMISSION['table_height_m']+self.recipe['planned_gripper_table_clearance_m']-1e-6:
            raise RuntimeError('Interpolated EE path violates gripper/table clearance')
        if enforce_clearance and pink_min < SPAWN_ADMISSION['table_height_m']+self.recipe['commanded_gripper_table_clearance_m']-1e-6:
            raise RuntimeError('Pink interpolation target violates gripper/table clearance')
        self.pending_token = dict(index=len(self.actions), action=action.tolist(), executed_physics_steps=0)
        positions, velocities, grippers, audit = self.controller.command(action)
        commanded_poses = [self.kin.pose(np.r_[q,g]) for q,g in zip(positions,grippers)]
        commanded_bottom = [self.task.gripper_bottom_at_hand_pose(pose) for pose in commanded_poses]
        minimum = min(commanded_bottom)
        if enforce_clearance and minimum < SPAWN_ADMISSION['table_height_m']+self.recipe['commanded_gripper_table_clearance_m']-1e-6:
            write(self.output/'rejected_command_geometry.json', dict(token=self.pending_token,
                commanded_gripper_bottom_z=commanded_bottom, minimum_z=minimum))
            raise RuntimeError('Pink drive FK violates gripper/table clearance')
        if self.arm_table_guard is not None:
            collision = self.arm_table_guard.check(positions,measured=self.task.robot.left_entity.get_qpos()[:7])
            if collision is not None:
                write(self.output/'rejected_arm_table_command.json',dict(token=self.pending_token,
                    current_physics_step=self.task.physics_step,collision=collision,
                    command_modified=False,token_physics_steps_executed=0))
                raise RuntimeError('Whole-arm/table guard rejected token before execution')
        joints = self.task.robot.left_entity.get_active_joints()
        for q, dq, g, bottom, pink_pose, command_pose, planned_pose in zip(positions, velocities, grippers,
                commanded_bottom, pink_targets, commanded_poses, planned_targets):
            self.task.robot.set_arm_joints(q, dq, 'left')
            self.task.robot.set_gripper(g, 'left', gripper_eps=0)
            self.applied.append([float(j.get_drive_target()[0]) for j in joints]+
                [float(j.get_drive_velocity_target()[0]) for j in joints])
            self.task._step_scene()
            self.rows.append(self.task.object_state())
            self.commanded_gripper_bottom_z.append(bottom)
            self.pink_target_hand_poses.append(pink_pose)
            self.commanded_hand_poses.append(command_pose)
            self.planned_hand_poses.append(planned_pose)
            transform = np.linalg.inv(matrix(self.task.robot.get_left_ee_pose())) @ np.asarray(self.rows[-1]['hand_pose'])
            self.high_to_hand_max_error = max(self.high_to_hand_max_error, float(np.max(np.abs(transform-self.high_to_hand))))
            self.pending_token['executed_physics_steps'] += 1
        index = len(self.actions)
        if self.record_rgb:
            self.rgb.append(rgb)
        self.states.append(state)
        self.actions.append(action)
        self.audits.append(audit)
        self.timing['ik_physics_s'] += time.perf_counter()-started
        if self.record_rgb:
            for stream, frame in zip(self.native_rgb, native):
                stream.append(np.asarray(frame).copy())
        started = time.perf_counter()
        if self.recipe.get('write_preview_frames', False):
            preview_image(native).save(self.output/'frames'/f'{index:06d}.png')
        self.timing['image_write_s'] += time.perf_counter()-started
        self.pending_token = None
        self.previous_serialized_target = target_pose

    def move(self, target, name, duration=None, gripper=None):
        start = self.goal.copy()
        rotation = self.kin.pin.log3(start[:3,:3].T @ target[:3,:3])
        if duration is None:
            duration = max(.16, 1.875*np.linalg.norm(target[:3,3]-start[:3,3])/self.recipe['translation_speed_m_s'],
                1.875*np.linalg.norm(rotation)/self.recipe['rotation_speed_rad_s'])
        count = int(np.ceil(duration*self.controller.control_hz))
        start_step, initial_g = self.task.physics_step, self.g
        for index in range(1, count+1):
            fraction = index/count
            alpha = 10*fraction**3-15*fraction**4+6*fraction**5
            pose = np.eye(4)
            pose[:3,3] = start[:3,3]+alpha*(target[:3,3]-start[:3,3])
            pose[:3,:3] = start[:3,:3] @ self.kin.pin.exp3(alpha*rotation)
            bottom = self.task.gripper_bottom_at_hand_pose(pose)
            if bottom < SPAWN_ADMISSION['table_height_m']+self.recipe['planned_gripper_table_clearance_m']-1e-6:
                raise RuntimeError(f'Planned {name} path violates gripper/table clearance: {bottom}')
            self.token(pose, initial_g if gripper is None else initial_g+fraction*(gripper-initial_g))
        measured = self.task.robot.left_entity.find_link_by_name('panda_hand').pose.to_transformation_matrix()
        self.stages[name] = dict(start_step=start_step, end_step=self.task.physics_step,
            duration_s=(self.task.physics_step-start_step)*.004,
            measured_position_error_m=float(np.linalg.norm(measured[:3,3]-target[:3,3])),
            target=target.tolist())
        print(json.dumps(dict(stage=name, **self.stages[name])), flush=True)

    def hold(self, count):
        joints = self.task.robot.left_entity.get_active_joints()
        positions = np.asarray([j.get_drive_target()[0] for j in joints])
        for index in range(count):
            for joint, position in zip(joints, positions):
                joint.set_drive_target(position)
                joint.set_drive_velocity_target(0.)
            for entity in (self.task.robot.left_entity, self.task.robot.right_entity):
                entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.applied.append(np.r_[positions, np.zeros(9)])
            self.task._step_scene()
            self.rows.append(self.task.object_state())
            if self.recipe.get('write_preview_frames', False) and index % 10 == 0:
                _, native = observe(self.task)
                preview_image(native).save(self.output/'hold_frames'/f'{index:06d}.png')

    def save(self):
        steps = np.arange(len(self.actions))*self.period
        rgb_payload = dict(rgb=np.asarray(self.rgb, dtype=np.uint8)) if self.record_rgb else {}
        np.savez_compressed(self.output/'policy_recording.npz', **rgb_payload,
            state=np.asarray(self.states, dtype=np.float32), action=np.asarray(self.actions, dtype=np.float32),
            observation_step=steps, timestamp_s=steps*.004)
        np.savez_compressed(self.output/'measurements.npz', **measurements_array(self.rows, self.applied))
        if self.record_rgb:
            for name, frames in zip(('head_camera', 'left_camera'), self.native_rgb):
                np.savez_compressed(self.output/f'{name}_rgb.npz', rgb=np.asarray(frames, dtype=np.uint8))
        write(self.output/'ik_audit.json', self.audits)
        write(self.output/'stages.json', self.stages)
        write(self.output/'contacts.json', self.task.contact_events)
        np.savez_compressed(self.output/'command_geometry.npz',
            gripper_collision_envelope_in_hand=self.task.gripper_collision_envelope,
            target_gripper_bottom_z=np.asarray(self.target_gripper_bottom_z),
            commanded_gripper_bottom_z=np.asarray(self.commanded_gripper_bottom_z),
            pink_target_hand_pose=np.asarray(self.pink_target_hand_poses),
            planned_hand_pose=np.asarray(self.planned_hand_poses),
            commanded_hand_pose=np.asarray(self.commanded_hand_poses),
            physics_step=np.arange(1,len(self.commanded_gripper_bottom_z)+1))
        if self.pending_token:
            write(self.output/'incomplete_token.json', self.pending_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=.30)
    parser.add_argument('--close-duration', type=float, default=.40)
    parser.add_argument('--preview-frames', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'frames', 'hold_frames'):
        (args.output/name).mkdir()
    sources = [Path(__file__), Path(__file__).parents[1]/'envs/roboicl_random_pick.py',
            Path(__file__).parents[1]/'envs/robot/roboicl_ee.py',
            Path(__file__).parents[1]/'envs/roboicl_pick_place.py',
            Path(__file__).parents[1]/'envs/_base_task.py',
            Path(__file__).parents[1]/'envs/robot/robot.py',
            Path(__file__).parent/'collect_roboicl_pink_expert.py',
            Path(__file__).parent/'rollout_roboicl_ee_policy.py',
            Path(__file__).parent/'qualify_roboicl_ee.py']
    for source in sources:
        shutil.copyfile(source, args.output/'source'/source.name)
    write(args.output/'source_hashes.json', {str(p.relative_to(Path(__file__).parents[1])):sha(p) for p in sources})
    recipe = dict(RECIPE, translation_speed_m_s=args.speed, close_duration_s=args.close_duration,
        write_preview_frames=args.preview_frames)
    if args.replay:
        frozen = read(args.replay/'frozen_recipe.json')
        if frozen['controller'] != CONTROL_RECIPE or frozen['geometry_sha256'] != sha(args.geometry):
            raise ValueError('Replay contract mismatch')
        recipe = frozen['expert']
    config = read(args.replay/'config.json' if args.replay else args.config)
    config.update(save_data=False, record_control=False, need_plan=False, save_path=str(args.output),
        task_name='roboicl_random_pick')
    write(args.output/'config.json', config)
    write(args.output/'frozen_recipe.json', dict(expert=recipe, acceptance=ACCEPTANCE,
        controller=CONTROL_RECIPE, geometry_sha256=sha(args.geometry),
        generation='independent native static contact goals -> float32 EE -> Pink -> physics',
        policy_input='RGB plus current drive FK; target/object metadata generation/evaluation only'))
    task, collector = roboicl_random_pick(), None
    receipt = dict(status='starting', task_success=False, qualified=False, development_only=True)
    policy_steps, hold_steps = None, 0
    try:
        init_start = time.perf_counter()
        task.setup_demo(**config)
        receipt['initialization_wall_s'] = time.perf_counter()-init_start
        kin = FrankaKinematics(args.geometry)
        write(args.output/'initial_state.json', task.object_state())
        validate_pick_geometry(kin, task.robot, args.output)
        collector = Collector(task, kin, args.output, recipe)
        if args.replay:
            with np.load(args.replay/'policy_recording.npz', allow_pickle=False) as source:
                actions = source['action'].copy()
            for action in actions:
                collector.action(action)
            collector.stages = read(args.replay/'stages.json')
        else:
            measured = task.robot.left_entity.find_link_by_name('panda_hand').pose.to_transformation_matrix()
            high_to_hand = np.linalg.inv(matrix(task.robot.get_left_ee_pose())) @ measured
            candidates = []
            candidate_audit = []
            rotation_projections = []
            contact_to_high = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
            required_z = SPAWN_ADMISSION['table_height_m']+recipe['planned_gripper_table_clearance_m']
            for contact_id, _ in task.target.iter_contact_points():
                contact = task.target.get_contact_point(contact_id, 'matrix')
                high = contact @ contact_to_high
                u, _, vh = np.linalg.svd(high[:3,:3])
                if np.linalg.det(u @ vh) < 0:
                    raise ValueError(f'Asset contact {contact_id} is a reflection, not a rigid rotation')
                rotation_projections.append(dict(contact_id=contact_id,
                    original_determinant=float(np.linalg.det(high[:3,:3])),
                    max_abs_projection_error=float(np.max(np.abs(high[:3,:3]-u @ vh)))))
                high[:3,:3] = u @ vh
                high[:3,3] += high[:3,:3] @ np.array([-.12,0.,0.])
                base_pose = high[:3,3].tolist()+t3d.quaternions.mat2quat(high[:3,:3]).tolist()
                center_pose = task.target.get_contact_point(contact_id, 'list')
                # Reuse native Franka tilt generation; execute the resulting EE
                # goals through Pink, never relabel a joint planner trajectory.
                for tilt_index, native_pose in enumerate(task.robot.create_target_pose_list(base_pose,center_pose,'left')):
                    hand = matrix(native_pose) @ high_to_hand
                    if hand[2,2] >= recipe['candidate_max_hand_z_world_z']:
                        continue
                    uncorrected_bottom = task.gripper_bottom_at_hand_pose(hand)
                    retreat = max(0.,(required_z-uncorrected_bottom)/(-hand[2,2]))
                    hand[:3,3] -= retreat*hand[:3,2]
                    rotation_error = float(np.linalg.norm(kin.pin.log3(collector.goal[:3,:3].T @ hand[:3,:3])))
                    info = dict(contact_id=contact_id,tilt_index=tilt_index,
                        hand_z_world_z=float(hand[2,2]),rotation_from_start_rad=rotation_error,
                        uncorrected_gripper_bottom_z_m=uncorrected_bottom,grasp_dis_m=retreat,
                        corrected_gripper_bottom_z_m=task.gripper_bottom_at_hand_pose(hand))
                    candidate_audit.append(info)
                    candidates.append((rotation_error,retreat,contact_id,tilt_index,hand,info))
            if not candidates:
                raise RuntimeError('No downward official tilt candidate; keep this generation failure')
            _, _, contact_id, tilt_index, grasp, selected = min(candidates,key=lambda c:(c[0],c[1]))
            pre = grasp.copy()
            pre[:3,3] -= grasp[:3,2]*recipe['pre_grasp_dis_m']
            clearance = collector.goal.copy()
            clearance[2,3] = max(clearance[2,3],pre[2,3],grasp[2,3])+.06
            write(args.output/'high_level_goals.json',dict(high_to_hand=high_to_hand.tolist(),
                pregrasp=pre.tolist(),grasp=grasp.tolist(),clearance=clearance.tolist(),
                selected_contact_id=contact_id,selected_tilt_index=tilt_index,
                selected_candidate=selected,contact_rotation_projections=rotation_projections,
                candidates=candidate_audit,grasp_geometry='actual SAPIEN hand/finger collision shapes; q=0/.04 envelope'))
            collector.move(clearance, 'approach_clearance')
            collector.move(pre, 'approach_pregrasp')
            collector.move(grasp, 'approach_grasp')
            collector.move(grasp, 'close', duration=recipe['close_duration_s'], gripper=0.)
            lift = collector.goal.copy()
            lift[2,3] += recipe['lift_distance_m']
            collector.move(lift, 'lift')
        policy_steps = task.physics_step
        hold_start = time.perf_counter()
        collector.hold(recipe['final_stability_physics_steps'])
        receipt['stability_hold_wall_s'] = time.perf_counter()-hold_start
        hold_steps = recipe['final_stability_physics_steps']
        receipt.update(status='completed')
        if args.replay:
            with np.load(args.replay/'policy_recording.npz', allow_pickle=False) as source:
                errors = {key:float(np.max(np.abs(np.asarray(value).astype(float)-source[key].astype(float))))
                    for key,value in (('rgb',collector.rgb), ('state',collector.states))}
            with np.load(args.replay/'measurements.npz', allow_pickle=False) as source:
                errors.update({key:float(np.max(np.abs(value-source[key])))
                    for key,value in measurements_array(collector.rows, collector.applied).items()})
            receipt.update(replay_max_abs_errors=errors, replay_bitwise_equal=all(v == 0 for v in errors.values()))
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        try:
            if collector:
                save_start = time.perf_counter()
                collector.save()
                receipt.update(save_wall_s=time.perf_counter()-save_start, component_wall_s=collector.timing,
                    high_to_hand_max_abs_error=collector.high_to_hand_max_error)
                receipt.update(score(task, collector.rows, task.physics_step if policy_steps is None else policy_steps, hold_steps))
                receipt.update(completed_tokens=len(collector.actions), initial=collector.rows[0], final=collector.rows[-1],
                    initialization_quality=task.initialization_quality, stages=collector.stages)
            if receipt['status'] != 'completed':
                receipt['qualified'] = receipt['task_success'] = False
                receipt['data_quality_passed'] = False
            receipt['initialization_quality'] = task.initialization_quality
            receipt['wall_elapsed_s'] = time.perf_counter()-started
            write(args.output/'receipt.json', receipt)
        finally:
            if hasattr(task, 'scene'):
                task.close_env()
    print(json.dumps({key:receipt.get(key) for key in ('status','qualified','policy_elapsed_s','wall_elapsed_s','replay_bitwise_equal')}), flush=True)


if __name__ == '__main__':
    main()
