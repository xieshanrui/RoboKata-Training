"""Record corrected three-object pick/place as continuous 25 Hz EE labels."""
import argparse
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import transforms3d as t3d

from collect_roboicl_curobo_pick import RecordingTask,NativeRecording,RECIPE,command,read,write,sha
from envs.roboicl_select_curobo import roboicl_select_curobo
from envs.robot.roboicl_ee import FrankaKinematics
from envs.robot.roboicl_mesh_sweep import carried_mesh_sweep
from score_roboicl_open_routes import physical_hold_mask
from envs.utils import ArmTag


class RecordingSelectTask(RecordingTask,roboicl_select_curobo):
    carried_vertices_in_hand=None

    def move_from_drive(self, stage, displacement=None, target_hand=None):
        if self.carried_vertices_in_hand is not None:
            # The attachment transform and the planner start must use the same
            # frame: measured joints in continuous mode, legacy drives otherwise.
            state=self.object_state()
            hand=self.recorder.kin.pose(command(np.asarray(state['left_qpos' if self.continuous_pick_place else 'left_drive_position'])))
            transform=np.linalg.inv(hand) @ self.target.get_pose().to_transformation_matrix()
            self.carried_object_in_hand=transform
            self.carried_vertices_in_hand=self.object_collision_vertices[self.target_index] @ transform[:3,:3].T+transform[:3,3]
        rejections=[]
        for attempt in range(4):
            before=self.physics_step
            try:
                return super().move_from_drive(stage,displacement,target_hand)
            except RuntimeError as error:
                if self.physics_step!=before or str(error) not in (
                    'Validated arm path is unavailable','Carried object sweep intersects a non-target or table'):
                    raise
                sweep=self.recorder.output/'rejected_carried_sweep.json'
                rejections.append(dict(attempt=attempt,physics_step=before,error=str(error),
                    sweep=read(sweep) if str(error).startswith('Carried object') and sweep.exists() else None))
                write(self.recorder.output/f'{stage}_{before}_planning_rejections.json',rejections)
                if attempt==3:raise

    def execute_arm_path(self,path):
        if self.carried_vertices_in_hand is not None and path['status']=='Success':
            # The held object's actual collision geometry must also clear all
            # non-targets; checking only the robot spheres is insufficient.
            poses=np.stack([self.recorder.kin.pose(np.r_[q[:7],0.]) for q in path['position']])
            sweep=carried_mesh_sweep(poses,self.carried_object_in_hand,self.object_collision_parts,
                [actor.get_pose().to_transformation_matrix() for actor in self.objects],self.target_index)
            sweep['actual_start_bottom_m']=float(self.object_state()['object_bottom_z'][self.target_index])
            path['carried_object_sweep']=sweep
            if any(r['overlapping_steps'] for r in sweep['non_target_overlap']) or sweep['minimum_object_bottom_m']<.739:
                write(self.recorder.output/'rejected_carried_sweep.json',path['carried_object_sweep'])
                raise RuntimeError('Carried object sweep intersects a non-target or table')
        return super().execute_arm_path(path)


def score(task,recorder,receipt):
    rows=recorder.rows;target=task.target_index;others=[i for i in range(len(task.objects)) if i!=target]
    poses=np.asarray([r['object_poses'] for r in rows]);xyz=poses[:,:,:3]
    lift=xyz[:,:,2]-xyz[0,:,2]
    complete=receipt['status']=='completed' and len(rows)==receipt['policy_physics_steps']+251
    tail=xyz[-251:,target] if complete else xyz[-1:,target]
    destination_error=float(np.linalg.norm(tail[:,:2]-task.destination,axis=1).max())
    non_target_shift=float(np.linalg.norm(xyz[:,others]-xyz[0,others],axis=2).max()) if others else 0.
    held_objects=np.flatnonzero(lift.max(axis=0)>=.10).tolist()
    if getattr(task,'strategy',{}).get('motion_profile')=='native_curobo_v2':
        measured=dict(object_bottom_z=np.asarray([r['object_bottom_z'] for r in rows]),
            gripper_command=np.asarray([r['gripper_command'] for r in rows]))
        held_objects=[i for i in range(len(task.objects)) if np.any(np.convolve(
            physical_hold_mask(measured,i,task.contact_events).astype(int),np.ones(5,dtype=int),'valid')==5)]
    robot_table=[e for e in task.contact_events if 'table' in e['bodies'] and any(n.startswith('panda_') for n in e['bodies'])]
    final_bottom=np.asarray([r['object_bottom_z'][target] for r in rows[-251:]])
    stable_range=float(np.linalg.norm(np.ptp(tail,axis=0)))
    physical=bool(complete and held_objects==[target] and destination_error<.03 and non_target_shift<.03
        and np.all(np.abs(final_bottom-.74)<.003) and stable_range<.001
        and all(r['gripper_command']>.8 for r in rows[-251:]))
    quality=bool(complete and not robot_table and np.min([r['finger_bottom_z'] for r in rows])>=.739
        and min(recorder.commanded_bottom)>=.741-1e-6)
    return dict(task_success=physical,data_quality_passed=quality,qualified=physical and quality,
        maximum_lift_m=lift.max(axis=0).tolist(),lifted_object_indices=held_objects,
        maximum_destination_error_m=destination_error,non_target_maximum_displacement_m=non_target_shift,
        final_bottom_z_m=float(final_bottom[-1]),released_tail_position_range_m=stable_range,
        robot_table_contact_count=len(robot_table),minimum_commanded_gripper_bottom_z_m=min(recorder.commanded_bottom,default=None))


def run_episode(task,config_path,output,geometry,replay_source=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'source').mkdir()
    repo=Path(__file__).parents[1]
    sources=[Path(__file__),Path(__file__).with_name('collect_roboicl_curobo_pick.py'),
        *[repo/p for p in ('envs/roboicl_select_curobo.py','envs/roboicl_random_pick_curobo.py',
            'envs/roboicl_random_pick.py','envs/roboicl_pick_place.py','envs/_base_task.py',
            'envs/robot/robot.py','envs/robot/planner.py','envs/robot/roboicl_ee.py',
            'envs/camera/camera.py','env_cfg/task_config/_camera_config.yml')]]
    sources.append(repo/'envs/robot/roboicl_mesh_sweep.py')
    sources.append(repo/'envs/robot/roboicl_continuous_transfer.py')
    for p in sources:shutil.copyfile(p,output/'source'/p.name)
    write(output/'source_hashes.json',{str(p.relative_to(repo)):sha(p) for p in sources})
    config=read(config_path);config.update(need_plan=True,save_data=False,record_control=False,save_freq=None,
        save_path=str(output),task_name=getattr(task,'task_name','roboicl_select_curobo'))
    policy_budget=getattr(task,'policy_budget_steps',3000 if config.get('continuous_pick_place',False) else RECIPE['max_policy_physics_steps'])
    selected=config['object_specs'][config['target_index']]
    identity=[selected['asset'],selected['model_id']]
    if config.get('target_identity')!=identity:
        config['inherited_target_identity']=config.get('target_identity')
        config['target_identity']=identity
    assert config.get('cartesian_pick_paths') is True
    assert config.get('fast_gripper') is True or config.get('official_robotwin_control') is True
    assert config.get('direct_pick_approach') is True or (
        config.get('direct_pick_approach') is False and getattr(task,'supports_separate_pregrasp',False))
    write(output/'config.json',config)
    write(output/'frozen_recipe.json',dict(expert=dict(RECIPE,recording_schema='curobo_select_dense25_v1',
        expert_quality_version=1,final_stability_physics_steps=250,
        planner_world='All five table boxes and oriented boxes of every non-target convex collision part retained; only target removed for contact'),task='three_object_pick_place',
        control_hz=25,final_stability_physics_steps=250,max_policy_physics_steps=policy_budget,
        gripper_ramp_and_settle_steps=[100,20],geometry_sha256=sha(geometry),
        place_path_chord_limit=1.75,xy_overshoot_limit_m=.02,backtracking_limit_m=.02,
        preclose_translation_limit_m=.005,preclose_rotation_limit_deg=10,
        direct_pick_approach=bool(config['direct_pick_approach']),
        carried_mesh_anchor='actual object relative to command FK before each carried segment',
        parallel_jaw_flip=bool(config.get('parallel_jaw_flip',False)),
        prefer_native_jaw_orientation=bool(config.get('prefer_native_jaw_orientation',False)),
        place_hold_orientation=bool(config.get('place_hold_orientation',False)),
        non_target_collision_obstacles='retained throughout',carried_object_sweep=True,
        obstacle_representation='one conservative OBB per physical convex collision part; 1um outward padding',
        motion_plan_attempts=4,straight_segment_limits=dict(path_chord_ratio=1.5,backtracking_m=.02,deviation_m=.005),
        grasp_candidate_fallback='original contact ordering, then remaining official rotation candidates; same initial state'))
    recipe=read(output/'frozen_recipe.json')
    recipe['grasp_planning_palm_margin_m']=.005 if config.get('continuous_pick_place',False) else .001
    recipe['release_object_bottom_m']=.744 if config.get('continuous_pick_place',False) else .742
    recipe['continuous_pick_place']=bool(config.get('continuous_pick_place',False))
    recipe['return_to_origin']=bool(config.get('return_to_origin',False))
    recipe['return_home_api']='Official release/retreat/return sequence; native plan_pose to recorded initial hand pose; all objects retained as obstacles'
    recipe['return_home_tolerance']=dict(position_m=.002,rotation_deg=1.)
    recipe['destination_plate_in_planner']=bool(config.get('continuous_pick_place',False))
    recipe['gripper_operation_steps']=dict(close=[100,20],release=[200,100] if config.get('continuous_pick_place',False) else [100,20])
    recipe['post_retreat_settling']=dict(window_steps=250,maximum_position_range_m=.0008,
        maximum_extra_steps=1000,policy_step_budget=policy_budget) if config.get('continuous_pick_place',False) else None
    recipe['arm_joint_drive']=dict(stiffness=5000.,damping=500.) if config.get('continuous_pick_place',False) else dict(stiffness=1000.,damping=200.)
    recipe['planning_state_source']='measured current joints' if config.get('continuous_pick_place',False) else 'previous drive command'
    if config.get('continuous_pick_place',False):recipe['carried_mesh_anchor']='actual object relative to measured current hand FK before sole carried plan'
    if config.get('continuous_pick_place',False):recipe['release_finger_control']=dict(start='measured independent finger positions',profile='quintic position and analytic velocity',ramp_steps=200,settle_steps=100,drive_properties='original upstream properties unchanged')
    recipe['joint_acceleration_feedforward']=bool(config.get('continuous_pick_place',False))
    recipe['feedforward_model']='mass matrix exported from actual SAPIEN articulation; passive gravity/Coriolis retained'
    if config.get('official_robotwin_control', False):
        recipe.update(gripper_ramp_and_settle_steps=[200, 100],
            gripper_operation_steps=dict(close=[200, 100], release=[200, 100]),
            arm_joint_drive=dict(stiffness=1000., damping=200.),
            release_finger_control=dict(start='previous command', profile='official linear200 plus100 hold'),
            joint_acceleration_feedforward=False,
            feedforward_model='Official passive gravity/Coriolis only')
    write(output/'frozen_recipe.json',recipe)
    receipt=dict(status='starting',qualified=False,task_success=False,data_quality_passed=False,
        expert_controller='curobo_native',recording_schema='curobo_select_dense25_v1',
        action_source='fk_of_executed_joint_drive',expert_quality_version=1,stages={})
    started=time.perf_counter();recorder=None;task.recorder=None;task.carried_vertices_in_hand=None
    task.grasp_clearance_audit=None
    try:
        init_started=time.perf_counter()
        task.fast_gripper=bool(config.get('fast_gripper', False));task.setup_demo(**config)
        if getattr(task, 'measured_segment_chaining', False):
            recipe['post_retreat_settling'] = None
        receipt.update(initialization_wall_s=time.perf_counter()-init_started,curobo_reused=task.curobo_reused)
        if task.continuous_pick_place:
            recipe['release_clearance_m']=.003
            recipe['release_object_bottom_m']=task.destination_top_z_m+recipe['release_clearance_m']
            recipe['placement_support']='top of actual destination collision geometry'
            write(output/'frozen_recipe.json',recipe)
            write(output/'physical_gripper_parts.json',task.robot.left_planner.target_mesh_sweep.snapshot)
        write(output/'object_collision_parts.json',task.object_collision_parts)
        kin=FrankaKinematics(geometry);initial=task.object_state();write(output/'initial_state.json',initial)
        assert np.max(np.abs(kin.pose(command(np.asarray(initial['left_qpos'])))-np.asarray(initial['hand_pose'])))<1e-5
        recorder=NativeRecording(task,kin,output,max_policy_physics_steps=policy_budget);task.recorder=recorder;recorder.install_plan_logging()
        if recorder.recording_substeps != 10 or len(task.objects) != 3:
            receipt['recording_schema']=f'curobo_regions_native224_{250//recorder.recording_substeps}hz_v1'
            recipe.update(task=config['task_name'],control_hz=250//recorder.recording_substeps,
                object_count=len(task.objects),recording_substeps=recorder.recording_substeps)
            recipe['expert'].update(control_hz=recipe['control_hz'],recording_schema=receipt['recording_schema'],
                labels=f'obs before step s; state from previous actual drive; action from actual drive at s+{recorder.recording_substeps-1}')
            write(output/'frozen_recipe.json',recipe)
        task._update_render();task.cameras.update_picture()
        initial_rgb=task.cameras.get_rgb()
        np.savez_compressed(output/'initial_rgb.npz',head=initial_rgb['head_camera']['rgb'],wrist=initial_rgb['left_camera']['rgb'])
        if replay_source is not None:
            original=read(Path(replay_source)/'receipt.json');assert original['status']=='completed'
            recorder.replay(replay_source)
            receipt.update(policy_physics_steps=original['policy_physics_steps'],native_physics_steps=original['native_physics_steps'],
                stages=original['stages'],replay_source=str(replay_source),execution_mode='recorded_native_controls_replay',
                replay_trace_sha256=sha(Path(replay_source)/'native_controls.npz'))
        else:
            worlds=[task.set_pregrasp_obstacles(True)]
            contact_id = config['object_specs'][task.target_index]['contact_id']
            contact_ids = None if contact_id is None else [contact_id] + [
                index for index, _ in task.target.iter_contact_points() if index != contact_id]
            grasp_actions = task.clearance_grasp_actor(task.target,ArmTag('left'),mesh_margin=.003,sphere_margin=.001,
                palm_margin=.005 if task.continuous_pick_place else .001,
                contact_point_ids=contact_ids)
            write(output/'grasp_clearance.json',task.grasp_clearance_audit)
            receipt['execution_approach']=task.grasp_clearance_audit['execution_approach']
            if task.qualified_direct_path is not None:
                worlds.append(task.set_pregrasp_obstacles(False));task.execute_arm_path(task.qualified_direct_path)
            else:
                task.execute_arm_path(task.qualified_pregrasp_path)
                worlds.append(task.set_pregrasp_obstacles(False))
                if getattr(task, 'measured_segment_chaining', False):
                    action = [a for a in grasp_actions[1] if a.action == 'move'][-1]
                    task.execute_grasp_approach(action)
                    receipt['execution_approach'] = task.grasp_clearance_audit['execution_approach']
                else:
                    task.execute_arm_path(task.qualified_grasp_path)
            receipt['stages']['close']=task.physics_step
            poses=np.asarray([r['object_poses'][task.target_index] for r in recorder.rows])
            translation=float(np.linalg.norm(poses[:,:3]-poses[0,:3],axis=1).max())
            q=poses[:,3:]/np.linalg.norm(poses[:,3:],axis=1,keepdims=True)
            rotation=float(np.degrees(2*np.arccos(np.clip(np.abs(q @ q[0]),0,1))).max())
            receipt.update(preclose_translation_m=translation,preclose_rotation_deg=rotation)
            if translation>.005 or rotation>10:raise RuntimeError('Target moved before closing')
            assert task.move(task.close_gripper(ArmTag('left')),save_freq=None)
            receipt['stages']['lift']=task.physics_step
            # Match the physical attachment frame to the same joint state used
            # at the plan start. Command-space anchoring remains the legacy mode.
            closed=task.object_state()
            hand=kin.pose(command(np.asarray(closed['left_qpos' if task.continuous_pick_place else 'left_drive_position'])))
            transform=np.linalg.inv(hand) @ task.target.get_pose().to_transformation_matrix()
            task.carried_object_in_hand=transform
            task.carried_vertices_in_hand=task.object_collision_vertices[task.target_index] @ transform[:3,:3].T+transform[:3,3]
            if not task.continuous_pick_place:
                task.move_from_drive('lift',displacement=[0,0,.15])
            receipt['stages']['transfer']=task.physics_step
            worlds.append(task.set_pregrasp_obstacles(False))
            current=task.object_state();q=np.asarray(current['left_drive_position']);goal=kin.pose(command(q))
            goal[:2,3]+=task.destination-np.asarray(current['object_poses'][task.target_index][:2])
            goal[2,3]+=.742-current['object_bottom_z'][task.target_index]
            if task.continuous_pick_place:
                planning_hand=kin.pose(command(np.asarray(current['left_qpos'])))
                object_in_hand=np.linalg.inv(planning_hand) @ task.target.get_pose().to_transformation_matrix()
                initial_object=np.asarray(recorder.rows[0]['object_poses'][task.target_index])
                desired_object=np.eye(4)
                desired_object[:3,:3]=t3d.quaternions.quat2mat(initial_object[3:])
                desired_object[:3,3]=initial_object[:3]
                desired_object[:2,3]=task.destination
                # Retain a small, uniform release gap above the actual support.
                # Zero gap plus tracking/grip deflection can press a held object
                # into the table and kick it into the palm before opening.
                desired_object[2,3]+=recipe['release_object_bottom_m']-recorder.rows[0]['object_bottom_z'][task.target_index]
                goal=desired_object @ np.linalg.inv(object_in_hand)
                receipt['placement_goal']=dict(method='restore initially settled object orientation using measured closed-grasp transform',
                    desired_object_pose=desired_object.tolist(),object_in_planning_hand=object_in_hand.tolist(),planning_state_source='measured joints')
                highest=max((float((vertices @ actor.get_pose().to_transformation_matrix()[:3,:3].T+actor.get_pose().p)[:,2].max())
                    for i,(actor,vertices) in enumerate(zip(task.objects,task.object_collision_vertices)) if i!=task.target_index), default=.74)
                task.robot.left_planner.transfer_clearance_rise_m=max(.15,highest+.02-current['object_bottom_z'][task.target_index])
            task.move_from_drive('place',target_hand=goal)
            receipt['stages']['release']=task.physics_step
            assert task.move(task.open_gripper(ArmTag('left')),save_freq=None)
            task.carried_vertices_in_hand=None
            receipt['stages']['retreat']=task.physics_step
            task.move_from_drive('lift',displacement=[0,0,.10])
            if config.get('return_to_origin',False):
                # Official RoboTwin tasks lift clear before back_to_origin.
                # A released target is now an obstacle, not a contact exemption.
                receipt['stages']['return']=task.physics_step
                worlds.append(task.set_pregrasp_obstacles(True))
                home_path=task.robot.left_planner.plan_home_path(task.robot.left_entity.get_qpos(),initial['left_qpos'])
                task.execute_arm_path(home_path)
                receipt['stages']['home']=task.physics_step
                receipt['return_to_origin']=dict(api=home_path['cartesian_path']['planner_method'],upstream_reference='Base_Task.back_to_origin after retreat',initial_joint_reference=initial['left_qpos'][:7],
                    official_target_ee_pose=task.robot.left_original_pose,
                    initial_hand_pose=initial['hand_pose'],
                    placed_target_restored_as_obstacle=True)
            if task.continuous_pick_place and not getattr(task, 'measured_segment_chaining', False):
                # Completion includes letting the released object settle after
                # the hand has cleared it. This never interrupts approach or
                # carrying. The subsequent independent one-second audit stays
                # separate and retains its frozen 1mm acceptance threshold.
                start_settle=task.physics_step
                budget=min(1000,((policy_budget-task.physics_step)//10)*10)
                while True:
                    tail=np.asarray([r['object_poses'][task.target_index][:3]
                        for r in recorder.rows[-251:]])
                    spread=float(np.linalg.norm(tail-tail[0],axis=1).max())
                    if len(tail)==251 and spread<=.0008:break
                    if task.physics_step-start_settle>=budget:
                        raise RuntimeError('Released object did not settle within the frozen physics-step budget')
                    # Record completion waiting as an explicit native open-
                    # gripper hold. It is part of the expert operation ledger;
                    # alignment and the independent audit retain phases1/2.
                    task.take_dense_action(dict(left_arm=None,right_arm=None,right_gripper=None,
                        left_gripper=dict(result=np.full(10,task.robot.get_left_gripper_val()),
                            num_step=10,per_step=0.,task_completion_hold=True)))
                receipt['post_retreat_settling']=dict(start_step=start_settle,
                    extra_steps=task.physics_step-start_settle,preceding_one_second_position_range_m=spread,
                    completion_margin_m=.0008,independent_future_audit_steps=250)
            elif getattr(task, 'measured_segment_chaining', False):
                receipt['post_retreat_settling'] = dict(extra_steps=0,
                    independent_future_audit_steps=250, completion_hold_enabled=False)
            receipt['native_physics_steps']=task.physics_step
            recorder.hold((-task.physics_step)%10,phase=1)
            receipt['policy_physics_steps']=task.physics_step
            write(output/'planner_world_stages.json',worlds)
            recorder.hold(250,phase=2)
        receipt['status']='completed'
    except Exception as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        task.recorder=None
        if task.grasp_clearance_audit is not None:
            write(output/'grasp_clearance.json',task.grasp_clearance_audit)
        if recorder is not None:
            recorder.restore_plan_logging();steps=receipt.get('policy_physics_steps',task.physics_step)
            recorder.save(steps);receipt.update(score(task,recorder,receipt))
            receipt.update(initial=recorder.rows[0],final=recorder.rows[-1],policy_physics_steps=steps,
                policy_elapsed_s=steps*.004,completed_tokens=steps//recorder.recording_substeps,initialization_quality=task.initialization_quality,
                render_wall_s=recorder.render_wall_s)
        receipt['wall_elapsed_s']=time.perf_counter()-started;write(output/'receipt.json',receipt)
        if hasattr(task,'scene'):task.close_env()
    print({k:receipt.get(k) for k in ('status','qualified','policy_elapsed_s','error')},flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','geometry'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--replay-source',type=Path)
    a=parser.parse_args();run_episode(RecordingSelectTask(),a.config,a.output,a.geometry,a.replay_source)


if __name__=='__main__':main()
