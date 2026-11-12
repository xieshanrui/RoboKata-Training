"""Cross-controller diagnostic: saved cuRobo EE labels through policy Pink."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs.roboicl_random_pick import roboicl_random_pick
from envs.robot.roboicl_ee import FrankaKinematics, decode10
from collect_roboicl_random_pick import Collector, RECIPE, score, observe, validate_pick_geometry
from qualify_roboicl_ee import read, write, sha
from rollout_roboicl_random_pick import score_policy


def reset_probe(task, sources, geometry, output):
    output.mkdir()
    values, checks = [], []
    for index, source in enumerate((sources[0], sources[1], sources[0])):
        config = read(source/'config.json')
        config.update(need_plan=False,save_data=False,record_control=False,
            save_path=str(output),task_name='roboicl_random_pick')
        try:
            task.setup_demo(**config)
            kin = FrankaKinematics(geometry)
            initial = task.object_state()
            rgb, _ = observe(task)
            reference = read(source/'initial_state.json')
            initial_ok = all(np.allclose(initial[key],reference[key],atol=1e-6,rtol=0)
                for key in ('object_poses','left_qpos','left_qvel','left_drive_position',
                            'left_drive_velocity','hand_pose'))
            with np.load(source/'policy_recording.npz') as z:
                rgb_ok = bool(np.array_equal(rgb,z['rgb'][0]))
            np.save(output/f'{index}_rgb.npy',rgb)
            write(output/f'{index}_initial_state.json',initial)
            values.append((initial,rgb))
            checks.append(dict(source=str(source),initial_matches_source=bool(initial_ok),rgb_exact=rgb_ok))
        finally:
            if hasattr(task,'scene'):
                task.close_env()
    passed = bool(all(x['initial_matches_source'] and x['rgb_exact'] for x in checks)
        and values[0][0]==values[2][0] and np.array_equal(values[0][1],values[2][1]))
    write(output/'receipt.json',dict(passed=passed,sequence=checks,physical_a_return_exact=values[0][0]==values[2][0],
        rgb_a_return_exact=bool(np.array_equal(values[0][1],values[2][1])),executed_control_tokens=0))
    if not passed:
        raise ValueError('Persistent oracle scene reset validation failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--episodes',type=Path,nargs='+')
    group.add_argument('--panel',type=Path,help='Frozen policy dev35 panel; all original sources are replayed')
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    panel = read(args.panel) if args.panel else None
    if panel is not None:
        if panel['count'] != 35 or len(panel['cases']) != 35:
            raise ValueError('Oracle panel must contain all35 fixed development cases')
        args.episodes = [Path(case['source']) for case in panel['cases']]
        if any(source.name != case['initial_state_id'] for source,case in zip(args.episodes,panel['cases'])):
            raise ValueError('Oracle panel identity mismatch')
    root = Path(__file__).parents[1]
    sources = [Path(__file__),Path(__file__).parent/'collect_roboicl_random_pick.py',
        Path(__file__).parent/'rollout_roboicl_random_pick.py',root/'envs/roboicl_random_pick.py',
        root/'envs/roboicl_pick_place.py',root/'envs/_base_task.py',root/'envs/robot/roboicl_ee.py',
        root/'envs/robot/robot.py',root/'envs/robot/planner.py',args.geometry]
    pins = {str(path.resolve()):sha(path) for path in sources}
    source_pins = {str(source):{name:sha(source/name) for name in
        ('config.json','initial_state.json','policy_recording.npz','receipt.json','native_executions.json')}
        for source in args.episodes}
    (args.output/'source').mkdir()
    for path in sources:
        shutil.copyfile(path,args.output/'source'/path.name)
    write(args.output/'launch.json',dict(role='oracle execution diagnostic, not learned policy performance',
        panel=panel,panel_sha256=sha(args.panel) if args.panel else None,source_sha256=pins,
        episode_source_sha256=source_pins,source_order=[str(x) for x in args.episodes],
        extra_verification_steps=125,online_planning=False,action_modification=False))
    task=roboicl_random_pick()
    if panel is not None:
        reset_probe(task,args.episodes,args.geometry,args.output/'reset_probe')
    results=[]
    for source in args.episodes:
        if any(sha(Path(path)) != digest for path,digest in pins.items()):
            raise ValueError('Oracle execution source changed after freezing')
        if any(sha(source/name) != digest for name,digest in source_pins[str(source)].items()):
            raise ValueError('Native oracle source changed after freezing')
        started=time.perf_counter()
        output=args.output/source.name
        output.mkdir()
        config=read(source/'config.json')
        config.update(need_plan=False,save_data=False,record_control=False,save_path=str(output),task_name='roboicl_random_pick')
        write(output/'config.json',config)
        write(output/'native_stages.json',read(source/'native_executions.json'))
        recipe=dict(RECIPE,planned_gripper_table_clearance_m=.001,max_policy_physics_steps=5000,write_preview_frames=False)
        write(output/'frozen_recipe.json',dict(expert_source=str(source),pink_recipe=recipe,
            diagnostic='Unmodified saved EE sequence; no replan or target feedback; physical final-drive hold is separate verification',
            policy_ee_guard_floor_m=.001,post_sequence_verification_steps=125))
        collector=None
        report=dict(status='starting',source=str(source),task_success=False,
            role='oracle label execution diagnostic; never learned policy performance',
            action_origin='unmodified accepted native cuRobo25Hz EE sequence',policy_score_before_verification=None)
        policy_steps=0
        hold_steps=0
        try:
            task.setup_demo(**config)
            kin=FrankaKinematics(args.geometry)
            validate_pick_geometry(kin,task.robot,output)
            initial=task.object_state()
            write(output/'initial_state.json',initial)
            reference=read(source/'initial_state.json')
            initial_errors={key:float(np.max(np.abs(np.asarray(initial[key])-np.asarray(reference[key]))))
                for key in ('object_poses','left_qpos','left_qvel','left_drive_position','left_drive_velocity','hand_pose')}
            report['initial_max_abs_errors']=initial_errors
            if max(initial_errors.values())>1e-6:
                raise ValueError('Cross-controller initial state differs from native recording')
            with np.load(source/'policy_recording.npz') as z:
                actions=z['action'].copy()
                rgb,_=observe(task)
                report['first_rgb_exact']=bool(np.array_equal(rgb,z['rgb'][0]))
            if not report['first_rgb_exact']:
                raise ValueError('Oracle initial RGB differs from native recording')
            collector=Collector(task,kin,output,recipe)
            for index,action in enumerate(actions):
                try:
                    collector.action(action)
                except Exception as error:
                    pose,gripper=decode10(action)
                    write(output/'rejected_target_diagnostic.json',dict(index=index,raw_prediction=action.tolist(),
                        decoded_hand_pose=pose.tolist(),gripper=gripper,
                        target_bottom_z=task.gripper_bottom_at_hand_pose(pose),physics_step=task.physics_step,
                        error=f'{type(error).__name__}: {error}',action_modified=False))
                    raise
            policy_steps=task.physics_step
            report['policy_score_before_verification']=score_policy(collector)
            collector.hold(125)
            hold_steps=125
            report['status']='completed'
        except Exception as error:
            report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        finally:
            if collector is not None:
                collector.save()
                report.update(score(task,collector.rows,policy_steps or task.physics_step,hold_steps))
                errors=[float(np.linalg.norm(np.asarray(collector.rows[(i+1)*10]['hand_pose'])[:3,3]-decode10(action)[0][:3,3]))
                    for i,action in enumerate(collector.actions)]
                report.update(completed_tokens=len(collector.actions),tracking_position_error_max_m=max(errors,default=None),
                    tracking_position_error_last_m=errors[-1] if errors else None,
                    post_sequence_verification_steps=hold_steps,trajectory_bitwise_equality_required=False)
                write(output/'tracking_position_error.json',errors)
            if report['status']!='completed':
                report['task_success']=report['qualified']=False
            report['wall_elapsed_s']=time.perf_counter()-started
            write(output/'receipt.json',report)
            if hasattr(task,'scene'):
                task.close_env()
        results.append(report)
        write(args.output/'summary.json',results)
        print(json.dumps({key:report.get(key) for key in ('source','status','qualified','policy_elapsed_s',
            'tracking_position_error_max_m','robot_table_contact_count')}),flush=True)


if __name__=='__main__':
    main()
