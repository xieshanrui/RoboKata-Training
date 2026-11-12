"""Execute fresh three-object RGB-Demo policies through the qualified Pink path."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs.roboicl_select_curobo import roboicl_select
from envs.robot.roboicl_ee import FrankaKinematics,CONTROL_RECIPE,decode10
from collect_roboicl_random_pick import Collector,RECIPE,observe,validate_pick_geometry,measurements_array
from rollout_roboicl_ee_policy import EEWorker,drive_command
from qualify_roboicl_ee import read,write,sha


class SelectWorker(EEWorker):
    def set_panel_demo(self,path,condition,seed,permutation,expected_frames):
        import json
        self.demo_request_id+=1
        control='correct' if condition=='wrong_target' else condition
        request=dict(set_demo=str(path.resolve()),demo_control=control,seed=seed,
            shuffle_seed=20261003,request_id=self.demo_request_id)
        if control=='shuffled':request['shuffle_indices']=permutation
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush();reply=self.receive()
        assert reply['demo_ready']==self.demo_request_id and reply['seed']==seed and reply['demo_control']==control
        assert Path(reply['demo']).resolve()==path.resolve()
        assert reply['demo_indices']==expected_frames,'Closed-loop Demo frames differ from frozen offline panel'
        if control=='shuffled':assert reply['permutation']==permutation
        return reply


def reset_probe(args,panel,worker,task):
    out=args.output/'reset_probe';out.mkdir();values=[];rows=[]
    for index,trial in enumerate((panel['trials'][0],panel['trials'][15],panel['trials'][0])):
        source=Path(trial['source']);config=read(source/'config.json')
        config.update(need_plan=False,save_data=False,record_control=False,save_path=str(out),task_name='roboicl_select')
        try:
            task.setup_demo(**config);kin=FrankaKinematics(args.geometry);rgb,_=observe(task)
            state=kin.state10(drive_command(task.robot)).astype(np.float32);initial=task.object_state()
            packed=Path(trial['query_packed'])
            assert np.array_equal(rgb,np.load(packed/'rgb.npy',mmap_mode='r')[0])
            assert np.array_equal(state,np.load(packed/'state.npy',mmap_mode='r')[0]) and task.physics_step==0
            query=out/f'{index}_query.npz';np.savez_compressed(query,rgb=rgb,state=state)
            ack=worker.set_panel_demo(Path(trial['support']),trial['condition'],trial['noise_seed'],trial['shuffle'],trial['support_frames'])
            actions=worker.predict(query,out/f'{index}_actions.npy')
            values.append((rgb,state,actions,initial));rows.append(dict(trial=trial['id'],ack=ack))
            write(out/f'{index}_initial.json',initial)
        finally:
            if hasattr(task,'scene'):task.close_env()
    exact=all(np.array_equal(values[0][i],values[2][i]) for i in range(3)) and values[0][3]==values[2][3]
    write(out/'receipt.json',dict(passed=exact,sequence=rows,control_tokens_executed=0,
        check='A-B-A actual scene reset, Demo/RNG reset and full50-token prediction equality'))
    assert exact,'ICL persistent worker reset failed'


def run(args,trial,worker,task):
    contract=read(args.acceptance_contract)
    online=contract['version']==3
    spec=importlib.util.spec_from_file_location('select_policy_scorer',args.scorer)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    scorer.validate_contract(contract)
    output=args.output/trial['id'];output.mkdir(exist_ok=False)
    for name in ('source','queries','actions','frames','hold_frames'):(output/name).mkdir()
    source=Path(trial['source']);config_path=source/'config.json'
    assert sha(config_path)==trial['config_sha256']
    config=read(config_path)
    reference_path=source/('source_receipt.json' if (source/'recording_metadata.json').exists() else 'receipt.json')
    reference=read(reference_path)['initial']
    config.update(need_plan=False,save_data=False,record_control=False,save_path=str(output),task_name='roboicl_select')
    assert config['target_index']==trial['target']
    write(output/'config.json',config)
    marker=read(args.checkpoint/'complete.json');representation=marker['run_contract']['action_representation']
    hz, execute = representation['action_hz'], representation['execute_steps']
    assert (hz,execute) in ((25,10),(50,25))
    controller_recipe=dict(CONTROL_RECIPE,control_hz=hz,integration_steps=250//hz)
    recipe=dict(RECIPE,control_hz=hz,query_chunk_tokens=execute,max_policy_physics_steps=5000,planned_gripper_table_clearance_m=.001,
        enforce_predictive_clearance=not online,
        write_preview_frames=trial['video'])
    write(output/'frozen_recipe.json',dict(controller=controller_recipe,collector=recipe,physics_dt=.004,
        execute_tokens=execute,max_policy_steps=5000,post_policy_hold_steps=0 if online else 250,
        post_policy_hold='None; stable placement checked during active policy' if online else 'Unchanged final drive only, no expert target or action',
        task_collision_guard='actual_contacts_only' if online else 'hand_envelope_v1',geometry_sha256=sha(args.geometry)))
    paths=[Path(__file__),Path(__file__).with_name('collect_roboicl_random_pick.py'),
        Path(__file__).parents[1]/'envs/roboicl_select_curobo.py',Path(__file__).parents[1]/'envs/robot/roboicl_ee.py',
        args.scorer,args.acceptance_contract,args.worker_script]
    if args.axis=='scene':paths.append(Path(__file__).parents[1]/'envs/roboicl_select_scene.py')
    for path in paths:shutil.copyfile(path,output/'source'/path.name)
    write(output/'source_hashes.json',{str(p.resolve()):sha(p) for p in paths})
    marker=read(args.checkpoint/'complete.json');representation=marker['run_contract']['action_representation']
    assert representation['geometry_sha256']==sha(args.geometry)
    receipt=dict(status='starting',task_success=False,formal_evaluation_requested=True,
        evaluation_protocol='physics_online_v3' if online else 'legacy_guarded_v2',
        termination_reason=None,
        checkpoint=str(args.checkpoint),model_sha256=marker['model_sha256'],
        policy_inputs=['Demo RGB only','Query RGB','Previous actual drive FK10'],
        expert_clearance_executed=False,target_index_evaluator_only=trial['target'],
        destination_xy_evaluator_only=config['destination'],initial_state_verified=False,
        max_policy_steps=5000,requested_stability_steps=250,completed_policy_steps=0,completed_stability_steps=0,
        action_hz=hz,execute_tokens=execute,bound_controls=True,seed=trial['noise_seed'],demo_condition=trial['condition'],
        source_recording=str(source),support=trial['support'],worker_calls=0,inference_wall_s=[])
    collector=None;started=time.perf_counter()
    try:
        task.setup_demo(**config);kin=FrankaKinematics(args.geometry);validate_pick_geometry(kin,task.robot,output)
        initial=task.object_state();rgb,_=observe(task);state=kin.state10(drive_command(task.robot)).astype(np.float32)
        packed=Path(trial['query_packed'])
        assert np.array_equal(rgb,np.load(packed/'rgb.npy',mmap_mode='r')[0])
        assert np.array_equal(state,np.load(packed/'state.npy',mmap_mode='r')[0])
        checks={k:bool(np.allclose(initial[k],reference[k],atol=1e-6,rtol=0)) for k in
            ('object_poses','left_qpos','left_qvel','left_drive_position','left_drive_velocity','object_linear_velocity','object_angular_velocity')}
        assert all(checks.values()),checks
        receipt.update(initial_state_verified=True,initial_reference_checks=checks,
            initial_query_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),physics_timestep_s=task.scene.get_timestep())
        collector=Collector(task,kin,output,recipe)
        monitor=scorer.OnlineSuccessMonitor(initial,trial['target'],config['destination']) if online else None
        if monitor is not None:monitor.update(initial)
        success=False
        support=Path(trial['support']);assert worker.file_digest(support)==trial['support_sha256']
        worker.validate_configuration(args)
        demo_begin=time.perf_counter()
        receipt['worker_demo_acknowledgement']=worker.set_panel_demo(support,trial['condition'],trial['noise_seed'],trial['shuffle'],trial['support_frames'])
        receipt['demo_setup_wall_s']=time.perf_counter()-demo_begin
        while task.physics_step<5000 and not success:
            index=receipt['worker_calls'];rgb,_=observe(task)
            state=kin.state10(drive_command(task.robot)).astype(np.float32)
            query=output/'queries'/f'{index:05d}.npz';np.savez_compressed(query,rgb=rgb,state=state)
            begin=time.perf_counter();actions=worker.predict(query,output/'actions'/f'{index:05d}.npy')
            receipt['inference_wall_s'].append(time.perf_counter()-begin);receipt['worker_calls']+=1
            for action_index,action in enumerate(actions[:execute]):
                before=task.physics_step
                try:collector.action(action)
                except Exception as error:
                    pose,gripper=decode10(action);action_file=output/'actions'/f'{index:05d}.npy'
                    write(output/'rejected_target_diagnostic.json',dict(overall_token_zero_based=len(collector.actions),
                        chunk_index=index,index_within_chunk=action_index,action_file=str(action_file),
                        action_file_sha256=sha(action_file),raw_prediction=action.tolist(),decoded_hand_pose=pose.tolist(),
                        decoded_gripper_open_fraction=gripper,target_gripper_bottom_z_m=task.gripper_bottom_at_hand_pose(pose),
                        required_bottom_z_m=.741,numerical_allowance_m=1e-6,physics_step_before_token=before,
                        execution_reached_physics_step=task.physics_step,executed_token_physics_steps=task.physics_step-before,
                        prediction_modified=False,error=f'{type(error).__name__}: {error}'))
                    raise
                if online:
                    # Check every executed physics sample, terminate on token boundaries.
                    ready=False
                    for row in collector.rows[before+1:]:ready=monitor.update(row)
                    if ready:
                        success=True
                        break
            receipt['completed_policy_steps']=task.physics_step
        if online:
            table_contact=any('table' in e['bodies'] and any(n.startswith('panda_') for n in e['bodies'])
                for e in task.contact_events)
            reason=('completed_with_contact' if table_contact else 'success') if success else 'timeout'
            receipt.update(status='completed',termination_reason=reason,
                physical_completion_time_s=task.physics_step*.004 if success else None,
                success_time_s=task.physics_step*.004 if success and not table_contact else None)
        else:
            collector.hold(250);receipt.update(completed_stability_steps=250,status='completed',
                termination_reason='fixed_budget_then_hold')
    except Exception as error:
        receipt.update(status='failed',termination_reason='execution_error',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        if collector is not None:
            receipt['completed_policy_steps']=min(task.physics_step,5000)
            receipt['completed_stability_steps']=max(0,task.physics_step-5000)
    finally:
        if collector is not None:
            collector.save();telemetry=measurements_array(collector.rows,collector.applied)
            receipt.update(initial=collector.rows[0],final=collector.rows[-1],completed_tokens=len(collector.actions))
            result=scorer.score_rollout(receipt,telemetry,contract)
            table=[e for e in task.contact_events if 'table' in e['bodies'] and any(n.startswith('panda_') for n in e['bodies'])]
            result.update(robot_table_contact_count=len(table),data_quality_passed=not table,
                task_success=bool(result['formal_success'] and not table))
            write(output/'score.json',result);receipt.update(result)
        receipt['wall_elapsed_s']=time.perf_counter()-started;write(output/'receipt.json',receipt)
        if hasattr(task,'scene'):task.close_env()
    return receipt


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','runtime','geometry','output'):p.add_argument('--'+name,type=Path,required=True)
    repo=Path(__file__).parents[2]
    p.add_argument('--scorer',type=Path,default=repo/'scripts/score_roboicl_rollout.py')
    p.add_argument('--acceptance-contract',type=Path,default=repo/'experiments/policy_acceptance_v3_20261007.json',
        help='Default: actual physics and online success. Pass v2 explicitly to reproduce guarded fixed-budget evaluation.')
    p.add_argument('--checkpoint-screen',action='store_true',help='Explicitly run a frozen correct-Demo dev subset for checkpoint screening')
    args=p.parse_args();panel=read(args.panel);runtime=read(args.runtime)
    planned=panel['sample_counts']['trials_per_memory']
    assert panel['task']=='icl_pick_place' and len(panel['trials'])==planned and runtime['passed']
    expected_controls=set(panel['controls'])
    assert 'correct' in expected_controls and expected_controls <= {'correct','masked','last_frame','shuffled','wrong_target'}
    assert len(expected_controls)==len(panel['controls'])
    if args.checkpoint_screen:assert expected_controls=={'correct'}
    assert planned==panel['sample_counts']['independent_initial_states']*3*len(expected_controls)
    assert planned>=18
    if args.checkpoint_screen:assert panel['evaluation_kind']=='checkpoint_screen'
    assert len({t['id'] for t in panel['trials']})==planned
    args.axis=panel.get('axis')
    assert args.axis in (None,'scene','view')
    if args.axis:
        eval_mode=panel.get('axis_eval_mode','held_out')
        assert eval_mode in ('held_out','in_distribution')
        train_variants=set(panel['axis_train_variants'])
        heldout=panel['axis_held_out_variant']
        assert heldout not in train_variants
        for trial in panel['trials']:
            config=read(Path(trial['source'])/'config.json')
            variant=config[f'roboicl_{args.axis}_preset']
            if eval_mode=='held_out':
                assert variant==heldout
            else:
                assert (panel['evaluation_kind']=='development'
                        and panel['controls']==['correct']
                        and variant in train_variants
                        and variant==trial['axis_variant'])
    for state in {t['initial_state_id'] for t in panel['trials']}:
        assert {(t['target'],t['condition']) for t in panel['trials'] if t['initial_state_id']==state}=={
            (target,condition) for target in range(3) for condition in panel['controls']}
    for name in ('worker_python','worker_script','worker_source','checkpoint','normalization','tokenizer'):
        setattr(args,name,Path(runtime[name]))
    marker=read(args.checkpoint/'complete.json')
    assert marker['global_step']==runtime['global_step']
    assert marker['global_step'] in panel.get('checkpoint_steps',[30000])
    assert marker['model_sha256']==runtime['model_sha256'] and marker['run_contract']['task']=='icl_pick_place'
    args.seed=panel['trials'][0]['noise_seed'];args.demo=Path(panel['trials'][0]['support'])
    args.demo_control='correct';args.shuffle_seed=20261003;args.denoise_steps=10;args.worker_timeout=600.
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('collect_roboicl_random_pick.py'),
        Path(__file__).parents[1]/'envs/roboicl_select_curobo.py',Path(__file__).parents[1]/'envs/roboicl_random_pick.py',
        Path(__file__).parents[1]/'envs/robot/roboicl_ee.py',args.scorer,args.acceptance_contract,args.worker_script,args.panel]
    if args.axis=='scene':sources.append(Path(__file__).parents[1]/'envs/roboicl_select_scene.py')
    pins={str(path.resolve()):sha(path) for path in sources}
    (args.output/'source').mkdir()
    for path in sources:shutil.copyfile(path,args.output/'source'/path.name)
    write(args.output/'launch.json',dict(panel=panel,panel_sha256=sha(args.panel),runtime=runtime,
        checkpoint_screen=args.checkpoint_screen,
        max_policy_steps=5000,post_policy_hold_steps=0 if read(args.acceptance_contract)['version']==3 else 250,
        evaluation_protocol='physics_online_v3' if read(args.acceptance_contract)['version']==3 else 'legacy_guarded_v2',
        action_hz=marker['run_contract']['action_representation']['action_hz'],
        execute_tokens=marker['run_contract']['action_representation']['execute_steps'],denoise_steps=10,
        scorer_sha256=sha(args.scorer),acceptance_sha256=sha(args.acceptance_contract),source_sha256=pins,
        runner_sha256=sha(Path(__file__))))
    if args.axis=='scene':
        from envs.roboicl_select_scene import roboicl_select_scene
        task=roboicl_select_scene()
    else:
        task=roboicl_select()
    worker=None;rows=[];consecutive_errors=0
    try:
        worker=SelectWorker(args,args.output)
        reset_probe(args,panel,worker,task)
        for trial in panel['trials']:
            assert all(sha(Path(path))==digest for path,digest in pins.items()),'Pinned runtime changed during evaluation'
            result=run(args,trial,worker,task)
            rows.append(dict(id=trial['id'],initial_state_id=trial['initial_state_id'],target=trial['target'],
                condition=trial['condition'],status=result['status'],object_choice_correct=result.get('object_choice_correct',False),
                termination_reason=result['termination_reason'],evaluation_protocol=result['evaluation_protocol'],
                task_success=result['task_success'],receipt=str(args.output/trial['id']/'receipt.json')))
            write(args.output/'summary.json',dict(status='running',planned=planned,completed=len(rows),rows=rows))
            print(json.dumps(rows[-1]),flush=True)
            consecutive_errors=consecutive_errors+1 if result['status']=='failed' else 0
            if consecutive_errors>=3:
                raise RuntimeError('Three consecutive execution errors; inspect the runtime before further trials')
        write(args.output/'summary.json',dict(status='completed',planned=planned,completed=len(rows),rows=rows))
    except Exception as error:
        write(args.output/'summary.json',dict(status='failed',planned=planned,completed=len(rows),rows=rows,
            error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        if worker is not None:worker.close()


if __name__=='__main__':main()
