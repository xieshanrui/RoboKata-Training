"""Record fresh native cuRobo joint execution as synchronized 25 Hz EE labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import sapien
import transforms3d as t3d
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_random_pick_curobo import roboicl_random_pick_curobo
from envs.robot.roboicl_ee import FrankaKinematics
from envs.utils import ArmTag

RECIPE = dict(version=3, expert_controller='curobo_native',
    action_source='fk_of_executed_joint_drive', recording_schema='curobo_dense25_v1',
    expert_quality_version=2, physics_dt=.004, control_hz=25,
    grasp_mesh_margin_m=.003, grasp_buffered_sphere_margin_m=.001,
    grasp_palm_object_margin_m=.001,
    drive_mesh_clearance_min_m=.001, geometry_allowance_m=1e-6,
    max_policy_physics_steps=2500, lift_distance_m=.15, final_stability_physics_steps=125,
    tail_alignment='Execute at most9 additional physics steps with last drive positions and zero drive velocity',
    labels='obs before step s; state from previous actual drive; action from actual drive at s+9',
    planner_world='All five SAPIEN table collision boxes; pregrasp additionally includes actual object world AABBs; only object boxes removed before grasp',
    planner_reuse='Retain MotionGen; update scene world and reset internal seeds before each episode')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if hasattr(x, 'p') and hasattr(x, 'q'):
            return dict(p=x.p.tolist(), q=x.q.tolist())
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=encode)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(position):
    return np.r_[position[:7], np.clip((np.mean(position[7:9])+.01)/.06, 0, 1)]


class RecordingTask(roboicl_random_pick_curobo):
    recorder = None

    def execute_arm_path(self, path):
        if path['status'] != 'Success':
            raise RuntimeError('Validated arm path is unavailable')
        self.take_dense_action(dict(left_arm=path,left_gripper=None,right_arm=None,right_gripper=None))

    def move_from_drive(self, stage, displacement=None, target_hand=None):
        """Plan from measured current joints in continuous mode; keep legacy drives otherwise."""
        q = np.asarray([j.get_drive_target()[0] for j in self.robot.left_entity.get_active_joints()])
        pose = self.recorder.kin.pose(command(q))
        measured = self.object_state()['hand_pose']
        error = float(np.linalg.norm(pose[:3,3]-np.asarray(measured)[:3,3]))
        if error > .015 and not getattr(self, 'measured_segment_chaining', False):
            raise RuntimeError(f'Arm did not track previous endpoint: {error:.6f}m')
        if getattr(self,'continuous_pick_place',False):
            q=self.robot.left_entity.get_qpos().copy()
            pose=self.recorder.kin.pose(command(q))
        if target_hand is not None:
            pose = np.asarray(target_hand).copy()
        else:
            pose[:3,3] += np.asarray(displacement)
        with self.cartesian_motion(ArmTag('left'),stage):
            path = self.robot.left_planner.plan_path(q,sapien.Pose(p=pose[:3,3],q=t3d.quaternions.mat2quat(pose[:3,:3])))
        self.execute_arm_path(path)
        return error

    def set_gripper(self, set_tag='together', left_pos=None, right_pos=None):
        if self.recorder is None or not getattr(self, 'fast_gripper', False) or (getattr(self,'continuous_pick_place',False) and left_pos is not None and left_pos>.5):
            return super().set_gripper(set_tag, left_pos, right_pos)
        if set_tag != 'left':
            raise ValueError('Fast collection gripper currently supports left arm only')
        now = self.robot.get_left_gripper_val()
        count, settle = 100, 20
        values = np.r_[np.linspace(now, left_pos, count), np.full(settle, left_pos)]
        return dict(num_step=len(values), per_step=(left_pos-now)/count, result=values)

    def _step_scene(self):
        if self.recorder is not None:
            profile=getattr(self,'_release_finger_profile',None)
            if profile is not None:
                index=self._dense_context['step']
                position,velocity=profile
                for joint,q,dq in zip(self.robot.left_entity.get_active_joints()[7:9],position[index],velocity[index]):
                    joint.set_drive_target(float(q));joint.set_drive_velocity_target(float(dq))
                self.robot.left_gripper_val=float(np.clip((position[index].astype(np.float64).mean()+.01)/.06,0,1))
            if hasattr(self,'prepare_execution_forces'):
                self.prepare_execution_forces()
            self.recorder.before_step()
        super()._step_scene()
        if self.recorder is not None:
            self.recorder.after_step()

    def take_dense_action(self, sequence, save_freq=-1):
        gripper=sequence.get('left_gripper')
        release=(self.recorder is not None and getattr(self,'continuous_pick_place',False)
                 and not getattr(self, 'official_robotwin_control', False)
                 and gripper is not None and sequence.get('left_arm') is None
                 and float(gripper['result'][-1])>self.robot.get_left_gripper_val()+1e-6)
        if release:
            start=self.robot.left_entity.get_qpos()[7:9].copy()
            u=np.minimum(np.arange(gripper['num_step'])/199.,1.)
            smooth=10*u**3-15*u**4+6*u**5
            derivative=np.where(u<1,(30*u**2-60*u**3+30*u**4)/(199*.004),0.)
            position=(start[None]+(.05-start)[None]*smooth[:,None]).astype(np.float32)
            velocity=((.05-start)[None]*derivative[:,None]).astype(np.float32)
            self._release_finger_profile=(position,velocity)
            sequence=dict(sequence,left_gripper=dict(gripper,result=np.clip((position.astype(np.float64).mean(axis=1)+.01)/.06,0,1),joint_position=position,joint_velocity=velocity,
                release_control='measured independent finger quintic opening with original drive properties'))
        try:
            if self.recorder is not None:self.recorder.execution(sequence)
            return super().take_dense_action(sequence, save_freq)
        finally:
            if release:
                self._release_finger_profile=None


class NativeRecording:
    def __init__(self, task, kin, output, max_policy_physics_steps=None):
        self.task, self.kin, self.output = task, kin, output
        self.recording_substeps = getattr(task, 'recording_substeps', 10)
        if self.recording_substeps not in (5, 10):
            raise ValueError('Native recording supports 25 or 50 Hz')
        self.max_policy_physics_steps = (RECIPE['max_policy_physics_steps']
                                         if max_policy_physics_steps is None else max_policy_physics_steps)
        self.phase = 0  # 0 native execution, 1 real alignment steps, 2 verification
        self.rows = [task.object_state()]
        self.initial_drive = np.asarray(self.rows[0]['left_drive_position'])
        self.previous_drive = self.initial_drive.copy()
        self.rgb, self.head, self.wrist, self.states, self.observation_steps = [], [], [], [], []
        self.verification_head,self.verification_wrist,self.verification_steps=[],[],[]
        self.controls, self.qf, self.right_qf, self.gripper, self.modes, self.calls = [], [], [], [], [], []
        self.drive_fk, self.commanded_bottom = [], []
        self.plans, self.executions = [], []
        self.wrapped = []
        self.velocity_reset_pending = False
        self.velocity_reset = []
        self.explicit_calls = None
        self.render_wall_s = 0.

    def install_plan_logging(self):
        planner = self.task.robot.left_planner
        for name in ('plan_path', 'plan_batch', 'plan_home_path'):
            if name == 'plan_home_path' and not hasattr(planner, name):
                continue
            original = getattr(planner, name)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                planner.last_trajectory=None
                started = time.perf_counter()
                result = _original(*args, **kwargs)
                row = dict(method=_name, status=result.get('status'), wall_s=time.perf_counter()-started,
                    current_joint_position=np.asarray(args[0]), target=args[1], kwargs=kwargs,
                    world_obstacle_names=list(planner.active_world['cuboid']))
                if 'cartesian_path' in result:
                    row['cartesian_path'] = result['cartesian_path']
                if getattr(self.task,'motion_diagnostics',False) and planner.last_trajectory is not None:
                    directory=self.output/'planned_paths';directory.mkdir(exist_ok=True)
                    file=directory/f'plan_{len(self.plans):04d}.npz'
                    trajectory=planner.last_trajectory
                    np.savez_compressed(file,position=trajectory.position.detach().cpu().numpy(),
                        velocity=trajectory.velocity.detach().cpu().numpy())
                    row['planned_path_file']=str(file.relative_to(self.output))
                for key in ('position', 'velocity'):
                    if key in result:
                        value = np.asarray(result[key])
                        row[key+'_shape'] = list(value.shape)
                        row[key+'_sha256'] = hashlib.sha256(value.tobytes()).hexdigest()
                self.plans.append(row)
                write(self.output/'planner_calls.json', self.plans)
                return result
            self.wrapped.append((planner, name, original))
            setattr(planner, name, wrapped)

    def restore_plan_logging(self):
        for planner, name, original in self.wrapped:
            setattr(planner, name, original)

    def execution(self, sequence):
        arrays, meta = {}, dict(start_step=self.task.physics_step)
        for part, data in sequence.items():
            if data is None:
                continue
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    arrays[part+'_'+key] = np.asarray(value)
                else:
                    meta[part+'_'+key] = value
        index = len(self.executions)
        np.savez_compressed(self.output/f'native_execution_{index:03d}.npz', **arrays)
        meta['file'] = f'native_execution_{index:03d}.npz'
        self.executions.append(meta)

    def before_step(self):
        step = self.task.physics_step
        if self.phase != 2 and step >= self.max_policy_physics_steps:
            raise RuntimeError('Native expert policy budget exhausted')
        joints = self.task.robot.left_entity.get_active_joints()
        position = np.asarray([joint.get_drive_target()[0] for joint in joints])
        velocity = np.asarray([joint.get_drive_velocity_target()[0] for joint in joints])
        fk = self.kin.pose(command(position))
        bottom = self.task.gripper_bottom_at_hand_pose(fk)
        if self.phase != 2 and bottom < .74+RECIPE['drive_mesh_clearance_min_m']-RECIPE['geometry_allowance_m']:
            write(self.output/'rejected_native_drive.json',dict(physics_step=step,drive_position=position,
                hand_pose=fk,gripper_bottom_z_m=bottom))
            raise RuntimeError('Native cuRobo drive violates real gripper/table clearance')
        if step % self.recording_substeps == 0 and (self.phase != 2 or getattr(self.task,'motion_diagnostics',False)):
            started = time.perf_counter()
            self.task._update_render()
            self.task.cameras.update_picture()
            raw = self.task.cameras.get_rgb()
            head, wrist = (raw[name]['rgb'].copy() for name in ('head_camera','left_camera'))
            if self.phase==2:
                self.verification_head.append(head);self.verification_wrist.append(wrist);self.verification_steps.append(step)
            else:
                self.head.append(head)
                self.wrist.append(wrist)
                self.rgb.append(np.stack([np.asarray(ImageOps.pad(Image.fromarray(frame),(224,224),
                    method=Image.Resampling.BILINEAR)) for frame in (head,wrist)]))
                self.states.append(self.kin.state10(command(self.previous_drive)).astype(np.float32))
                self.observation_steps.append(step)
            self.render_wall_s += time.perf_counter()-started
        context = self.task._dense_context
        calls = []
        for name in ('left_arm','left_gripper','right_arm','right_gripper'):
            sequence = context['sequence'][name] if context else None
            count = (len(sequence['position']) if name.endswith('arm') else sequence['num_step']) if sequence is not None else 0
            calls.append(bool(context and context['step'] < count))
        if self.explicit_calls is not None:
            calls = self.explicit_calls
        self.controls.append(np.r_[position,velocity])
        self.qf.append(self.task.robot.left_entity.get_qf().copy())
        self.right_qf.append(self.task.robot.right_entity.get_qf().copy())
        self.gripper.append(self.task.robot.get_left_gripper_val())
        self.modes.append(self.phase)
        self.calls.append(calls)
        self.velocity_reset.append(self.velocity_reset_pending)
        self.velocity_reset_pending = False
        self.drive_fk.append(fk)
        self.commanded_bottom.append(bottom)
        self.previous_drive = position.copy()

    def after_step(self):
        self.rows.append(self.task.object_state())

    def zero_velocity(self):
        for joint in self.task.robot.left_entity.get_active_joints():
            joint.set_drive_velocity_target(0.)
        self.velocity_reset_pending = True

    def hold(self, count, phase):
        if count == 0:
            return
        self.phase = phase
        self.zero_velocity()
        for _ in range(count):
            for entity in (self.task.robot.left_entity,self.task.robot.right_entity):
                entity.set_qf(entity.compute_passive_force(gravity=True,coriolis_and_centrifugal=True))
            self.task._step_scene()

    def replay(self, source):
        """Execute saved physics-rate controls without planning or relabeling."""
        with np.load(Path(source)/'native_controls.npz') as trace:
            if np.any(trace['setter_calls'][:,2:]):
                raise ValueError('This single-arm replay requires an inactive right arm')
            if not np.array_equal(self.initial_drive, trace['initial_drive_position']):
                raise ValueError('Replay initial drives differ from source')
            joints = self.task.robot.left_entity.get_active_joints()
            release_ranges=[]
            for execution in read(Path(source)/'native_executions.json'):
                if 'left_gripper_release_motor_force_limit_n' in execution:
                    release_ranges.append((execution['start_step'],execution['start_step']+execution['left_gripper_num_step'],execution['left_gripper_release_motor_force_limit_n']))
            saved_properties=None
            for index, control in enumerate(trace['applied_controls']):
                for begin,end,limit in release_ranges:
                    if index==begin:
                        saved_properties=[(j.stiffness,j.damping,j.force_limit,j.drive_mode) for j in joints[7:9]]
                        for j in joints[7:9]:j.set_drive_property(stiffness=j.stiffness,damping=j.damping,force_limit=limit,mode=j.drive_mode)
                    if index==end:
                        for j,(k,d,f,mode) in zip(joints[7:9],saved_properties):j.set_drive_property(stiffness=k,damping=d,force_limit=f,mode=mode)

                self.phase = int(trace['phase'][index])
                calls = trace['setter_calls'][index]
                self.explicit_calls = calls.tolist()
                if trace['velocity_zeroed'][index]:
                    self.zero_velocity()
                indices = list(range(7)) if calls[0] else []
                if calls[1]:
                    indices.extend((7,8))
                for joint_index in indices:
                    joints[joint_index].set_drive_target(float(control[joint_index]))
                    joints[joint_index].set_drive_velocity_target(float(control[joint_index+9]))
                self.task.robot.left_entity.set_qf(trace['left_qf'][index])
                self.task.robot.right_entity.set_qf(trace['right_qf'][index])
                self.task.robot.left_gripper_val = float(trace['gripper_command'][index])
                self.task._step_scene()
        self.explicit_calls = None

    def save(self, policy_steps):
        controls = np.asarray(self.controls).reshape(-1,18)
        complete = policy_steps//self.recording_substeps
        endpoints = np.arange(complete)*self.recording_substeps+self.recording_substeps-1
        actions = np.asarray([self.kin.state10(command(controls[index,:9])) for index in endpoints],dtype=np.float32)
        np.savez_compressed(self.output/'policy_recording.npz',rgb=np.asarray(self.rgb[:complete],dtype=np.uint8),
            state=np.asarray(self.states[:complete],dtype=np.float32),action=actions,
            observation_step=np.asarray(self.observation_steps[:complete],dtype=np.int32),
            timestamp_s=np.asarray(self.observation_steps[:complete])*.004)
        for name, frames in (('head_camera',self.head),('left_camera',self.wrist)):
            np.savez_compressed(self.output/f'{name}_rgb.npz',rgb=np.asarray(frames[:complete],dtype=np.uint8))
        if self.verification_steps:
            np.savez_compressed(self.output/'verification_rgb.npz',head=np.asarray(self.verification_head),
                wrist=np.asarray(self.verification_wrist),physics_step=np.asarray(self.verification_steps))
        rows = self.rows
        np.savez_compressed(self.output/'measurements.npz',
            physics_step=np.asarray([r['physics_step'] for r in rows]),
            object_poses=np.asarray([r['object_poses'] for r in rows]),
            qpos=np.asarray([r['left_qpos'] for r in rows]),qvel=np.asarray([r['left_qvel'] for r in rows]),
            hand_pose=np.asarray([r['hand_pose'] for r in rows]),
            gripper_command=np.asarray([r['gripper_command'] for r in rows]),
            object_bottom_z=np.asarray([r['object_bottom_z'] for r in rows]),
            finger_bottom_z=np.asarray([r['finger_bottom_z'] for r in rows]),applied_controls=controls)
        np.savez_compressed(self.output/'native_controls.npz',physics_step=np.arange(len(controls)),
            applied_controls=controls,initial_drive_position=self.initial_drive,
            left_qf=np.asarray(self.qf),right_qf=np.asarray(self.right_qf),
            gripper_command=np.asarray(self.gripper),setter_calls=np.asarray(self.calls,dtype=bool),
            phase=np.asarray(self.modes,dtype=np.uint8),velocity_zeroed=np.asarray(self.velocity_reset,dtype=bool),
            drive_hand_pose=np.asarray(self.drive_fk),gripper_bottom_z=np.asarray(self.commanded_bottom),
            gripper_collision_envelope_in_hand=self.task.gripper_collision_envelope)
        write(self.output/'contacts.json',self.task.contact_events)
        write(self.output/'native_executions.json',self.executions)


def run_episode(task, config_path, output, geometry, worker_index=0, episode_index=0, replay_source=None):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'source').mkdir()
    root = Path(__file__).parents[1]
    sources = [Path(__file__),root/'envs/roboicl_random_pick_curobo.py',root/'envs/roboicl_random_pick.py',
        root/'envs/roboicl_pick_place.py',root/'envs/_base_task.py',root/'envs/robot/robot.py',
        root/'envs/robot/planner.py',root/'envs/robot/roboicl_ee.py']
    for source in sources:
        shutil.copyfile(source,output/'source'/source.name)
    write(output/'source_hashes.json',{str(path.relative_to(root)):sha(path) for path in sources})
    config = read(config_path)
    task.fast_gripper = bool(config.get('fast_gripper', False))
    config.update(need_plan=replay_source is None,save_data=False,record_control=False,save_freq=None,save_path=str(output),
        task_name='roboicl_random_pick_curobo')
    write(output/'config.json',config)
    write(output/'frozen_recipe.json',dict(expert=RECIPE,geometry_sha256=sha(geometry),
        cartesian_pick_paths=bool(config.get('cartesian_pick_paths',False)),
        direct_pick_approach=bool(config.get('direct_pick_approach',False)),
        trajectory_policy='Prefer collision-checked continuous approach; otherwise validated pregrasp plus straight approach; continue from actual drive endpoint'
            if config.get('direct_pick_approach',False) else 'Original native stages',
        gripper_ramp_and_settle_steps=[100,20] if config.get('fast_gripper',False) else [200,100],
        cartesian_maximum_deviation_m=.005 if config.get('cartesian_pick_paths',False) else None))
    receipt = dict(status='starting',task_success=False,data_quality_passed=False,qualified=False,
        expert_quality_version=2,expert_controller='curobo_native',recording_schema='curobo_dense25_v1',
        worker_index=worker_index,episode_index=episode_index)
    if replay_source is not None:
        receipt.update(execution_mode='recorded_native_controls_replay',replay_source=str(Path(replay_source).resolve()),
            replay_trace_sha256=sha(Path(replay_source)/'native_controls.npz'))
    recorder = None
    policy_steps, native_steps = 0, 0
    task.recorder = None
    if hasattr(task,'grasp_clearance_audit'):
        del task.grasp_clearance_audit
    try:
        init_started = time.perf_counter()
        task.setup_demo(**config)
        receipt.update(initialization_wall_s=time.perf_counter()-init_started,curobo_reused=task.curobo_reused)
        kin = FrankaKinematics(geometry)
        initial = task.object_state()
        fk = kin.pose(np.r_[task.robot.left_entity.get_qpos()[:7],1.])
        fk_error = float(np.max(np.abs(fk-np.asarray(initial['hand_pose']))))
        if fk_error > 1e-5:
            raise ValueError('Native collection geometry differs from actual Franka')
        receipt['initial_fk_matrix_max_abs_error'] = fk_error
        write(output/'initial_state.json',initial)
        world = {}
        for side in ('left','right'):
            planner = getattr(task.robot,side+'_planner')
            world[side] = dict(planner_class=type(planner).__module__+'.'+type(planner).__qualname__,
                motion_gen_class=type(planner.motion_gen).__module__+'.'+type(planner.motion_gen).__qualname__,
                world=planner.scene_world,collision_cache=planner.motion_gen.collision_cache,
                batch_collision_cache=planner.motion_gen_batch.collision_cache,
                motion_gen_identity=id(planner.motion_gen),motion_gen_batch_identity=id(planner.motion_gen_batch),
                planner_yml_sha256=sha(planner.yml_path))
        write(output/'planner_world.json',world)
        recorder = NativeRecording(task,kin,output)
        task.recorder = recorder
        recorder.install_plan_logging()
        if replay_source is None:
            world_stages = [task.set_pregrasp_obstacles(True)]
            write(output/'planner_world_stages.json',world_stages)
            grasp = task.clearance_grasp_actor(task.target,ArmTag('left'),
                mesh_margin=RECIPE['grasp_mesh_margin_m'],sphere_margin=RECIPE['grasp_buffered_sphere_margin_m'],
                palm_margin=RECIPE['grasp_palm_object_margin_m'])
            write(output/'grasp_clearance.json',task.grasp_clearance_audit)
            receipt['execution_approach'] = task.grasp_clearance_audit['execution_approach']
            if task.qualified_direct_path is not None:
                world_stages.append(task.set_pregrasp_obstacles(False))
                write(output/'planner_world_stages.json',world_stages)
                task.execute_arm_path(task.qualified_direct_path)
                receipt['preclose_physics_step'] = task.physics_step
                before_close = np.asarray([r['object_poses'] for r in recorder.rows])[:,:, :3]
                receipt['preclose_object_max_displacement_m'] = float(np.linalg.norm(before_close-before_close[:1],axis=2).max())
                if receipt['preclose_object_max_displacement_m'] > .01:
                    raise RuntimeError('Continuous approach disturbed object before closing')
                if not task.move(task.close_gripper(ArmTag('left')),save_freq=None):
                    raise RuntimeError('Native cuRobo gripper close failed')
            else:
                if task.cartesian_pick_paths:
                    task.execute_arm_path(task.qualified_pregrasp_path)
                elif not task.move((grasp[0],grasp[1][:1]),save_freq=None) or not task.plan_success:
                    raise RuntimeError('Native cuRobo pregrasp planning failed')
                world_stages.append(task.set_pregrasp_obstacles(False))
                write(output/'planner_world_stages.json',world_stages)
                if task.cartesian_pick_paths:
                    task.execute_arm_path(task.qualified_grasp_path)
                    if not task.move(task.close_gripper(ArmTag('left')),save_freq=None):
                        raise RuntimeError('Native cuRobo gripper close failed')
                else:
                    if not task.move((grasp[0],grasp[1][1:]),save_freq=None) or not task.plan_success:
                        raise RuntimeError('Native cuRobo grasp planning failed')
            receipt['after_grasp_physics_step'] = task.physics_step
            if task.cartesian_pick_paths:
                receipt['lift_start_tracking_error_m'] = task.move_from_drive('lift',displacement=[0,0,RECIPE['lift_distance_m']])
            else:
                if not task.move(task.move_by_displacement(ArmTag('left'),z=RECIPE['lift_distance_m']),save_freq=None) or not task.plan_success:
                    raise RuntimeError('Native cuRobo lift planning failed')
            native_steps = task.physics_step
            recorder.hold((-native_steps)%10,phase=1)
            policy_steps = task.physics_step
            recorder.hold(RECIPE['final_stability_physics_steps'],phase=2)
        else:
            source_receipt = read(Path(replay_source)/'receipt.json')
            if source_receipt['status'] != 'completed':
                raise ValueError('Independent native replay requires completed source execution')
            native_steps, policy_steps = (source_receipt[key] for key in ('native_physics_steps','policy_physics_steps'))
            recorder.replay(replay_source)
        receipt['status'] = 'completed'
    except Exception as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        task.recorder = None
        if recorder is not None:
            recorder.restore_plan_logging()
            if not policy_steps:
                policy_steps = task.physics_step
            recorder.save(policy_steps)
            poses = np.asarray([row['object_poses'] for row in recorder.rows])
            lift = poses[:,:,2]-poses[0,:,2]
            final_hold = receipt['status']=='completed' and len(recorder.rows)==policy_steps+126
            held = np.flatnonzero((lift[policy_steps:] >= .10).all(axis=0)).tolist() if final_hold else []
            support = [event for event in task.contact_events if event['physics_step']>=policy_steps
                and 'table' in event['bodies'] and any(spec['asset'] in event['bodies'] for spec in task.object_specs)]
            robot_table = [event for event in task.contact_events if 'table' in event['bodies']
                and any(name.startswith('panda_') for name in event['bodies'])]
            physical = bool(final_hold and held and not support and all(row['gripper_command']<.2 for row in recorder.rows[policy_steps:]))
            quality = bool(receipt['status']=='completed' and not robot_table
                and np.min([row['finger_bottom_z'] for row in recorder.rows])>=.739
                and min(recorder.commanded_bottom[:policy_steps])>=.741-1e-6)
            receipt.update(task_success=physical,data_quality_passed=quality,qualified=physical and quality,
                stable_lifted_objects=held,final_lift_m=lift[-1].tolist(),maximum_lift_m=lift.max(axis=0).tolist(),
                robot_table_contact_count=len(robot_table),robot_table_contact_max_impulse_ns=max([e['impulse_ns'] for e in robot_table],default=0),
                minimum_finger_bottom_z_m=float(np.min([row['finger_bottom_z'] for row in recorder.rows])),
                minimum_commanded_gripper_bottom_z_m=min(recorder.commanded_bottom[:policy_steps],default=None),
                native_physics_steps=native_steps,alignment_physics_steps=policy_steps-native_steps,
                policy_physics_steps=policy_steps,policy_elapsed_s=policy_steps*.004,
                stability_physics_steps=125 if final_hold else 0,completed_tokens=policy_steps//10,
                render_wall_s=recorder.render_wall_s,initial=recorder.rows[0],final=recorder.rows[-1])
        receipt.update(initialization_quality=task.initialization_quality,wall_elapsed_s=time.perf_counter()-started)
        if replay_source is None and hasattr(task,'grasp_clearance_audit'):
            write(output/'grasp_clearance.json',task.grasp_clearance_audit)
        write(output/'receipt.json',receipt)
        if hasattr(task,'scene'):
            task.close_env()
    print(json.dumps({key:receipt.get(key) for key in ('status','qualified','policy_elapsed_s','wall_elapsed_s','curobo_reused')}) ,flush=True)
    return receipt


def compare_replay(source, replay):
    comparisons = {}
    for name in ('measurements.npz','native_controls.npz','policy_recording.npz','head_camera_rgb.npz','left_camera_rgb.npz'):
        with np.load(Path(source)/name) as a, np.load(Path(replay)/name) as b:
            comparisons[name] = {key: bool(key in b and np.array_equal(a[key],b[key])) for key in a.files}
    contact_key = lambda event: (event['physics_step'],tuple(event['bodies']),event['impulse_ns'])
    comparisons['contacts'] = sorted(read(Path(source)/'contacts.json'),key=contact_key) == sorted(read(Path(replay)/'contacts.json'),key=contact_key)
    source_receipt, replay_receipt = read(Path(source)/'receipt.json'), read(Path(replay)/'receipt.json')
    passed = all(all(row.values()) if isinstance(row,dict) else row for row in comparisons.values())
    result = dict(passed=passed,comparisons=comparisons,
        task_success_match=source_receipt['task_success']==replay_receipt['task_success'],
        data_quality_match=source_receipt['data_quality_passed']==replay_receipt['data_quality_passed'])
    result['passed'] = result['passed'] and result['task_success_match'] and result['data_quality_match']
    write(Path(source)/'native_replay_comparison.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--jobs-file',type=Path)
    parser.add_argument('--replay-source',type=Path)
    args = parser.parse_args()
    task = RecordingTask()
    jobs = read(args.jobs_file) if args.jobs_file else dict(worker_index=0,
        episodes=[dict(config=str(args.config),output=str(args.output))])
    results = []
    for index,episode in enumerate(jobs['episodes']):
        for path, expected in jobs.get('source_sha256',{}).items():
            if sha(path) != expected:
                raise RuntimeError('Frozen native collector source changed: '+path)
        if episode.get('config_sha256') and sha(episode['config']) != episode['config_sha256']:
            raise RuntimeError('Frozen native episode configuration changed')
        receipt = run_episode(task,episode['config'],episode['output'],args.geometry,jobs['worker_index'],index,
            replay_source=args.replay_source)
        row = dict(output=episode['output'],**{key:receipt.get(key) for key in
            ('status','qualified','task_success','data_quality_passed','wall_elapsed_s','initialization_wall_s',
             'curobo_reused','policy_elapsed_s','expert_quality_version','error')})
        if episode.get('replay_selected'):
            row['replay_selected'] = True
            if receipt['status']=='completed':
                replay_output = Path(episode['output'])/'independent_native_replay'
                replay_receipt = run_episode(task,episode['config'],replay_output,args.geometry,jobs['worker_index'],index,
                    replay_source=episode['output'])
                row['native_replay_status'] = replay_receipt['status']
                row['native_replay_passed'] = compare_replay(episode['output'],replay_output)['passed']
            else:
                row.update(native_replay_status='source_incomplete_no_replacement',native_replay_passed=False)
        results.append(row)
        if args.jobs_file:
            write(args.jobs_file.with_suffix('.results.json'),results)


if __name__=='__main__':
    main()
