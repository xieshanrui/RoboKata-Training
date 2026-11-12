"""Run a masked-Demo pure-pick policy through the native 25 Hz Pink interface."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.roboicl_random_pick import roboicl_random_pick
from envs.robot.roboicl_ee import FrankaKinematics, CONTROL_RECIPE, decode10
from collect_roboicl_random_pick import Collector, RECIPE, observe, validate_pick_geometry
from rollout_roboicl_ee_policy import EEWorker, drive_command
from qualify_roboicl_ee import read, write, sha


class PickWorker(EEWorker):
    """Use the existing file/pipe prediction protocol without a Demo argument."""
    def __init__(self, args, output):
        self.configuration = {name:str(getattr(args,name).resolve()) for name in
            ('worker_python','worker_script','worker_source','checkpoint','normalization','tokenizer')}
        self.denoise_steps, self.output = args.denoise_steps, output.resolve()
        self.artifact_digests = {}
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        env['PYTHONPATH'] = str(args.worker_source.resolve())+os.pathsep+env.get('PYTHONPATH','')
        self.stderr = (output/'worker.stderr.log').open('w',encoding='utf-8')
        self.messages, self.timeout = queue.Queue(), args.worker_timeout
        command = [str(args.worker_python), '-u', str(args.worker_script.resolve()),
            '--checkpoint', str(args.checkpoint.resolve()), '--normalization', str(args.normalization.resolve()),
            '--tokenizer', str(args.tokenizer.resolve()), '--seed', str(args.seed),
            '--denoise-steps', str(args.denoise_steps), '--device', args.worker_device]
        write(output/'worker_command.json',dict(argv=command,demo_argument_present=False))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, encoding='utf-8', env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.reader = threading.Thread(target=self._read,args=(output/'worker.stdout.log',),daemon=True)
        self.reader.start()
        try:
            if self.receive() != {'ready':True}:
                raise ValueError('Pure-pick worker did not report ready')
        except Exception:
            self.close()
            raise

    def reset_episode(self, seed):
        self.reset_request_id = getattr(self, 'reset_request_id', 0)+1
        request_id = str(self.reset_request_id)
        self.process.stdin.write(json.dumps(dict(reset_episode=True, seed=seed, request_id=request_id))+'\n')
        self.process.stdin.flush()
        response = self.receive()
        if response != dict(episode_ready=request_id, seed=seed):
            raise ValueError(f'Pure-pick worker reset was not acknowledged: {response}')
        return response


def sustained_policy_pick(collector):
    """Evaluator only: detect an already completed .5 s policy-controlled lift."""
    step = collector.task.physics_step
    if step < 125:
        return False
    tail = collector.rows[-126:]
    initial_z = np.asarray(collector.rows[0]['object_poses'])[:,2]
    lift = np.asarray([row['object_poses'] for row in tail])[:,:,2]-initial_z
    held = np.flatnonzero((lift >= .10).all(axis=0))
    if not len(held) or not all(row['gripper_command'] < .2 for row in tail):
        return False
    for index in held:
        name = collector.task.object_specs[index]['asset']
        if not any(event['physics_step'] >= step-125 and name in event['bodies'] and 'table' in event['bodies']
                   for event in collector.task.contact_events):
            return True
    return False


def score_policy(collector):
    """Score the final125steps inside the policy stream, with no extra hold."""
    step = collector.task.physics_step
    poses = np.asarray([row['object_poses'] for row in collector.rows])
    lift = poses[:,:,2]-poses[0,:,2]
    window_start = max(0,step-125)
    stable_objects = np.flatnonzero((lift[window_start:] >= .10).all(axis=0)).tolist()
    supported = []
    for index in stable_objects:
        name = collector.task.object_specs[index]['asset']
        if any(event['physics_step'] >= window_start and name in event['bodies'] and 'table' in event['bodies']
               for event in collector.task.contact_events):
            supported.append(index)
    closed = all(row['gripper_command'] < .2 for row in collector.rows[window_start:])
    success = bool(step >= 125 and stable_objects and not supported and closed)
    robot_table_contacts = [event for event in collector.task.contact_events if 'table' in event['bodies']
        and any(name.startswith('panda_') for name in event['bodies'])]
    quality = bool(not robot_table_contacts and np.min([row['finger_bottom_z'] for row in collector.rows]) >= .739)
    return dict(task_success=success and quality, physical_complete_success=success,
        data_quality_passed=quality,expert_quality_version=1,
        robot_table_contact_count=len(robot_table_contacts),
        stable_lifted_objects=stable_objects,table_supported_objects=supported,
        maximum_lift_m=lift.max(axis=0).tolist(),final_lift_m=lift[-1].tolist(),
        verification_window=[window_start,step],verification_window_complete=step >= 125,
        verification_window_source='actual policy actions, not a post-policy hold',
        policy_physics_steps=step,post_policy_hold_steps=0,policy_elapsed_s=step*.004)


def run(args, task=None, worker=None, initial_reference=None):
    started = time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('source','queries','actions','frames','hold_frames'):
        (args.output/name).mkdir()
    marker = read(args.checkpoint/'complete.json')
    contract, model = marker['run_contract'],marker['model_config']
    if (contract.get('task') != 'pure_pick' or contract.get('demo_conditioning') != 'masked'
            or model.get('demo_conditioning') != 'masked' or model.get('action_encoding') != 'world_tcp10'):
        raise ValueError('This entry point requires a masked-Demo pure-pick EE checkpoint')
    representation = contract['action_representation']
    geometry = read(args.geometry)
    if (representation.get('geometry_sha256') != sha(args.geometry)
            or representation.get('geometry_urdf_sha256') != sha(args.geometry.parent/geometry['urdf'])):
        raise ValueError('Checkpoint/controller geometry binding mismatch')
    config = read(args.config)
    config.update(save_data=False,record_control=False,need_plan=False,save_path=str(args.output),
        task_name='roboicl_random_pick')
    write(args.output/'config.json',config)
    # Expert planning keeps 3 mm clearance; the learned command domain keeps
    # a 1 mm hard floor, leaving 2 mm tolerance without projecting predictions.
    recipe = dict(RECIPE,max_policy_physics_steps=args.max_steps,write_preview_frames=args.preview_frames,
        expert_planning_margin_m=.003,policy_command_floor_m=.001,
        planned_gripper_table_clearance_m=.001)
    guard_v2 = getattr(args,'arm_table_guard_v2',False)
    guard_mode = 'whole_arm_table_reject_v2' if guard_v2 else 'hand_envelope_v1'
    write(args.output/'frozen_recipe.json',dict(expert=recipe,controller=CONTROL_RECIPE,
        task_collision_guard=guard_mode,
        geometry_sha256=sha(args.geometry),acceptance=dict(lift_min_m=.10,
            stable_window_steps=125,gripper_max_open_fraction=.2,require_no_table_support=True,
            verification_window='inside actual policy action stream',post_policy_hold_steps=0),
        action_source='learned pure-pick model; no expert goals or contact annotations read by control',
        control_feasibility=dict(planned_gripper_table_clearance_m=recipe['planned_gripper_table_clearance_m'],
            expert_planning_margin_m=recipe['expert_planning_margin_m'],
            policy_command_floor_m=recipe['policy_command_floor_m'],
            commanded_gripper_table_clearance_m=recipe['commanded_gripper_table_clearance_m'],
            geometry='constant Franka hand/finger collision envelope and table plane; no object labels',
            violation='reject command and score this rollout as failure; no expert correction'),
        policy_observation='RGB and actual previous drive FK10 only; no Demo',
        early_stop='After policy actions have already maintained lift>=.10m,closedcommand,no table support for125physics steps; stop at next token boundary; no additional hold'))
    sources = [Path(__file__),Path(__file__).parent/'collect_roboicl_random_pick.py',
        Path(__file__).parents[1]/'envs/roboicl_random_pick.py',
        Path(__file__).parents[1]/'envs/robot/roboicl_ee.py',args.worker_script,args.geometry]
    if guard_v2:
        sources.append(Path(__file__).parents[1]/'envs/robot/roboicl_table_guard.py')
    for source in sources:
        shutil.copyfile(source,args.output/'source'/source.name)
    write(args.output/'source_hashes.json',{str(path.resolve()):sha(path) for path in sources})
    task = roboicl_random_pick() if task is None else task
    collector, owns_worker = None, worker is None
    receipt = dict(status='starting',task_success=False,formal_evaluation=args.formal_evaluation,
        checkpoint=str(args.checkpoint),model_sha256=marker.get('model_sha256'),
        task_collision_guard=guard_mode,
        no_demo=True,policy_inputs=['query RGB','actual previous drive FK10'],
        seed=args.seed,max_policy_steps=args.max_steps,first_sustained_pick_step=None,
        worker_calls=0,inference_wall_s=[],initial_query_rgb_sha256=None)
    try:
        init_started = time.perf_counter()
        task.setup_demo(**config)
        receipt['initialization_wall_s'] = time.perf_counter()-init_started
        kin = FrankaKinematics(args.geometry)
        write(args.output/'initial_state.json',task.object_state())
        validate_pick_geometry(kin,task.robot,args.output)
        guard = None
        if guard_v2:
            from envs.robot.roboicl_table_guard import ArmTableGuard
            guard = ArmTableGuard(task,kin)
            guard.save_geometry(args.output)
        collector = Collector(task,kin,args.output,recipe,arm_table_guard=guard)
        if worker is None:
            worker = PickWorker(args,args.output)
        else:
            worker.validate_configuration(args)
            receipt['worker_reset'] = worker.reset_episode(args.seed)
            receipt['persistent_worker_output'] = str(worker.output)
        while task.physics_step < args.max_steps:
            index = receipt['worker_calls']
            rgb, _ = observe(task)
            state = kin.state10(drive_command(task.robot)).astype(np.float32)
            query = args.output/'queries'/f'{index:05d}.npz'
            np.savez_compressed(query,rgb=rgb,state=state)
            if index == 0:
                receipt['initial_query_rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
                if initial_reference is not None:
                    checks = dict(rgb=np.array_equal(rgb, initial_reference['rgb']),
                        state=np.array_equal(state, initial_reference['state']))
                    initial = collector.rows[0]
                    for key in ('object_poses', 'left_qpos', 'left_qvel', 'left_drive_position',
                                'left_drive_velocity', 'object_linear_velocity', 'object_angular_velocity'):
                        checks[key] = bool(np.allclose(initial[key], initial_reference['initial'][key],
                                                      atol=1e-6, rtol=0))
                    receipt['initial_reference_checks'] = checks
                    if not all(checks.values()):
                        raise ValueError(f'Initial state/query differs from admitted source: {checks}')
            prediction_started = time.perf_counter()
            actions = worker.predict(query,args.output/'actions'/f'{index:05d}.npy')
            receipt['inference_wall_s'].append(time.perf_counter()-prediction_started)
            receipt['worker_calls'] += 1
            for action_index, action in enumerate(actions[:10]):
                if task.physics_step >= args.max_steps:
                    break
                before_step = task.physics_step
                try:
                    collector.action(action)
                except Exception as error:
                    pose, gripper = decode10(action)
                    bottom = task.gripper_bottom_at_hand_pose(pose)
                    action_file = args.output/'actions'/f'{index:05d}.npy'
                    write(args.output/'rejected_target_diagnostic.json',dict(
                        overall_token_zero_based=len(collector.actions),chunk_index=index,
                        index_within_chunk=action_index,action_file=str(action_file),
                        action_file_sha256=sha(action_file),raw_prediction=action.tolist(),
                        decoded_hand_pose=pose.tolist(),decoded_gripper_open_fraction=gripper,
                        target_gripper_bottom_z_m=bottom,required_bottom_z_m=.741,
                        numerical_allowance_m=1e-6,deficit_to_nominal_floor_m=.741-bottom,
                        physics_step_before_token=before_step,execution_reached_physics_step=task.physics_step,
                        executed_token_physics_steps=task.physics_step-before_step,
                        prediction_modified=False,error=f'{type(error).__name__}: {error}'))
                    raise
                if sustained_policy_pick(collector):
                    receipt['first_sustained_pick_step'] = task.physics_step
                    break
            write(args.output/'receipt.json',receipt)
            if receipt['first_sustained_pick_step'] is not None:
                break
        receipt['status'] = 'completed'
    except Exception as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:
        try:
            if collector is not None:
                collector.save()
                receipt.update(score_policy(collector),
                    initial=collector.rows[0],final=collector.rows[-1],completed_tokens=len(collector.actions),
                    component_wall_s=collector.timing)
            if receipt['status'] != 'completed':
                receipt['task_success'] = receipt['physical_complete_success'] = False
            receipt.update(initialization_quality=task.initialization_quality,wall_elapsed_s=time.perf_counter()-started)
            write(args.output/'receipt.json',receipt)
        finally:
            if worker is not None and owns_worker:
                worker.close()
            if hasattr(task,'scene'):
                task.close_env()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','geometry','output','worker-python','worker-script','worker-source',
                 'checkpoint','normalization','tokenizer'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=5000)
    parser.add_argument('--worker-device',default='cuda')
    parser.add_argument('--worker-timeout',type=float,default=600.)
    parser.add_argument('--seed',type=int,default=20261004)
    parser.add_argument('--denoise-steps',type=int,default=10)
    parser.add_argument('--formal-evaluation',action='store_true')
    parser.add_argument('--preview-frames',action='store_true')
    parser.add_argument('--arm-table-guard-v2',action='store_true',help='Opt into separately versioned whole-arm table rejection')
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_steps % 10 or (args.formal_evaluation and args.max_steps != 5000):
        parser.error('Use complete25Hz tokens; formal policy budget is5000physics steps/20seconds')
    receipt = run(args)
    print(json.dumps({key:receipt.get(key) for key in ('status','task_success','policy_elapsed_s','first_sustained_pick_step')}))


if __name__ == '__main__':
    main()
