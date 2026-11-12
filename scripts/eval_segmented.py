"""Single frozen scene: measured Query state, native50Hz Pink, execute50.

Ported from the executed rollout_finish_v2.py; physical scoring is unchanged.
The caller owns the model process. This program runs one scene and exits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

ROBOTWIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROBOTWIN), str(ROBOTWIN/'scripts')]

def measured_state(pose, qpos):
    pose=np.asarray(pose);qpos=np.asarray(qpos)
    if pose.shape[-2:]!=(4,4) or qpos.shape[-1]<9:
        raise ValueError('Expected measured hand poses and measured Franka qpos')
    grip=np.clip((qpos[...,7:9].mean(-1)+.01)/.06,0,1)*2-1
    result=np.concatenate((pose[...,:3,3],pose[...,:3,0],pose[...,:3,1],grip[...,None]),-1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite measured Query state')
    return result

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding='utf-8')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def score(m,c,contacts):
 t=c['target_index'];poses=m['object_poses'];n=len(poses);tail=min(n,251)
 body=poses[:,t,:3]+np.stack([t3d.quaternions.quat2mat(q)@c['gallery_contract']['placement']['base_center_local_m'] for q in poses[:,t,3:]])
 lifted=[]
 for i in range(poses.shape[1]):
  mask=physical_hold_mask(m,i,contacts)
  if len(mask)>=5 and np.any(np.convolve(mask.astype(int),np.ones(5,dtype=int),'valid')==5):lifted.append(i)
 others=[i for i in range(poses.shape[1]) if i!=t]
 shift=float(np.linalg.norm(poses[:,others,:3]-poses[0,others,:3],axis=2).max()) if others else 0.
 error=float(np.linalg.norm(body[-tail:,:2]-c['destination'],axis=1).max())
 spread=float(np.linalg.norm(np.ptp(body[-tail:],axis=0)))
 released_settled=bool(n>=251 and t in lifted and spread<.001 and np.all(abs(m['object_bottom_z'][-tail:,t]-.74)<.003) and np.all(m['gripper_command'][-tail:]>.8))
 placed=bool(n>=251 and error<=.01 and spread<.001 and np.all(abs(m['object_bottom_z'][-tail:,t]-.74)<.003) and np.all(m['gripper_command'][-tail:]>.8))
 target=f'roboicl_object_{t}'
 table=[e for e in contacts if 'table' in e['bodies'] and any(b.startswith('panda_') for b in e['bodies']) and e['impulse_ns']>0]
 cross=[e for e in contacts if target in e['bodies'] and any(b.startswith('roboicl_object_') and b!=target for b in e['bodies']) and e['impulse_ns']>0]
 robot_other=[e for e in contacts if any(b.startswith('panda_') for b in e['bodies']) and any(b==f'roboicl_object_{i}' for b in e['bodies'] for i in others) and e['impulse_ns']>0]
 route=score_open_route(m,c,contacts)
 physical=lifted==[t] and shift<.03 and placed
 return dict(released_settled=released_settled,waypoint_success=bool(route['requested_order_success']),physical=bool(physical),correct_object=lifted==[t],lifted_indices=lifted,placed=placed,
  route=bool(route['route_success']),route_metrics=route,non_target_displacement_mm=shift*1000,
  placement_error_mm=error*1000,settled_spread_mm=spread*1000,robot_table_contacts=len(table),
  target_non_target_contacts=len(cross),robot_non_target_contacts=len(robot_other),
  physical_route_success=bool(physical and route['route_success'] and not table and not cross and not robot_other),
  requested_grasp_semantics='not_classified; physical_route_success is not full semantic-grasp success')

def main(a):
 panel=read(a.bundle/'panel.json');trial=next(t for t in panel['trials'] if t['id']==a.trial)
 source=a.bundle/'queries'/trial['id'];cfg=read(source/'config.json');out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
 assert sha(source/'config.json')==trial['query_record']['config_sha256']
 geometry=a.bundle/'geometry/geometry.json'
 assert sha(geometry)==trial['query_record']['geometry_sha256']
 cfg.update(save_path=str(out),save_data=False,record_control=False)
 task=DatasetCollectionTask(geometry);task.fast_gripper=False;task.recorder=None;task.carried_vertices_in_hand=None
 receipt=dict(trial=trial['id'],pair=trial['pair_id'],stratum=trial['stratum'],task_id=trial['task_id'],
  status='starting',finished=False,kind='expert_interface_replay' if a.expert_replay else 'policy_closed_loop',
  panel_sha256=sha(a.bundle/'panel.json'),protocol_sha256=sha(a.bundle/'protocol.json'),source_sha256=sha(Path(__file__)),
  input='DemoRGB only; Query current RGB and measuredhand10',execute_tokens=50,initial_verified=False,calls=[],termination_policy='grasp_release_settle_v2',termination_reason='time_limit')
 collector=None;worker=None;queries=[];stage='setup';started=time.monotonic()
 try:
  task.setup_demo(**cfg);kin=FrankaKinematics(geometry);validate_pick_geometry(kin,task.robot,out)
  initial=task.object_state();reference=read(source/'initial_state.json')
  write(out/'platform_initialized_state.json',initial)
  import sapien
  task.robot.left_entity.set_qpos(reference['left_qpos'])
  task.robot.left_entity.set_qvel(reference['left_qvel'])
  for j,pos,vel in zip(task.robot.left_entity.get_active_joints(),reference['left_drive_position'],reference['left_drive_velocity']):
   j.set_drive_target(pos);j.set_drive_velocity_target(vel)
  for i,(actor,body) in enumerate(zip(task.objects,task.object_bodies)):
   pose=reference['object_poses'][i];actor.actor.set_pose(sapien.Pose(pose[:3],pose[3:]))
   body.set_linear_velocity(reference['object_linear_velocity'][i]);body.set_angular_velocity(reference['object_angular_velocity'][i])
  task.initial_positions=np.asarray(reference['positions']);task.initial_robot_qpos=list(reference['left_qpos'])
  initial=task.object_state()
  receipt['initial_restoration']='SAPIEN native pose,velocities,qpos,qvel,drive targets restored from original recorded Query; no physics advance before measurement'
  keys=['object_poses','left_qpos','left_qvel','left_drive_position','left_drive_velocity','object_linear_velocity','object_angular_velocity']
  errors={k:float(np.max(np.abs(np.asarray(initial[k])-np.asarray(reference[k])))) for k in keys}
  assert max(errors.values())<1e-6,errors
  rgb,_=observe(task)
  write(out/'head_camera_calibration.json',task.cameras.get_config()['head_camera'])
  with np.load(source/'initial_rgb.npz',allow_pickle=False) as z:
   ref_rgb=z['rgb'];receipt['cross_platform_initial_rgb']=dict(exact=bool(np.array_equal(rgb,ref_rgb)),mean_abs=float(np.abs(rgb.astype(float)-ref_rgb).mean()),max_abs=int(np.abs(rgb.astype(int)-ref_rgb).max()))
   assert rgb.shape==ref_rgb.shape and rgb.std()>4,'Invalid server render'
   expert=np.load(source/'expert_actions.npy',allow_pickle=False) if a.expert_replay else None
  write(out/'initial_state.json',initial);np.savez_compressed(out/'initial_query.npz',rgb=rgb,history=measured_state([initial['hand_pose']],[initial['left_qpos']]))
  np.savez_compressed(out/'initial_rgb.npz',head=rgb[0],wrist=rgb[1])
  receipt.update(initial_verified=True,initial_errors=errors,initial_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
  recipe=dict(RECIPE,control_hz=50,query_chunk_tokens=50,max_policy_physics_steps=6000,enforce_predictive_clearance=False,record_rgb=True,write_preview_frames=False)
  collector=Collector(task,kin,out,recipe);write(out/'recipe.json',recipe)
  if not a.expert_replay:
   stage='worker_load';worker=ParentWorker(sys.stdin.buffer,PIPE)
   assert worker.ready['step']==a.checkpoint_step and worker.ready['execute_tokens']==50 and worker.ready['geometry_sha256']==sha(geometry)
   write(out/'worker_identity.json',worker.ready)
   ack,_=worker.request('reset',demo=trial['demo'],condition='correct',seed=trial['seed']);receipt['demo_ack']=ack
  i=0
  while task.physics_step<6000:
   if a.expert_replay:
    if i*50>=len(expert):break
    actions=expert[i*50:(i+1)*50]
   else:
    stage='predict';rgb,_=observe(task);s=task.object_state()
    history=measured_state([s['hand_pose']],[s['left_qpos']]);ht=np.zeros(1,np.float32);hv=np.ones(1,bool)
    queries.append(dict(step=task.physics_step,rgb=rgb,history=history))
    meta,values=worker.request('predict',rgb=rgb,history=history,history_times=ht,history_valid=hv)
    actions=values['actions'];assert actions.shape==(50,10) and np.isfinite(actions).all()
    np.save(out/f'prediction_{i:03d}.npy',actions,allow_pickle=False);receipt['calls'].append(meta);actions=actions[:50]
   stage='execute'
   for action in actions:
    if task.physics_step+5>6000:break
    collector.action(action)
   i+=1
   if not a.expert_replay:
    current=score(measurements_array(collector.rows,collector.applied),cfg,task.contact_events)
    if current['released_settled']:
     receipt['termination_reason']='target_grasped_released_settled';break
  receipt['policy_steps']=task.physics_step;stage='hold'
  if receipt['termination_reason']=='time_limit':collector.hold(250)
  receipt['status']='completed';receipt['termination_step']=task.physics_step
 except Exception as e:
  traceback.print_exc();receipt.update(status='failed',error=repr(e),failure_stage=stage,
   infrastructure_failure=stage in ['setup','worker_load','predict'] and type(e).__name__!='PolicyOutputError')
 finally:
  if collector is not None:
   collector.save();m=measurements_array(collector.rows,collector.applied)
   receipt['metrics']=score(m,cfg,task.contact_events);receipt['actions_executed']=len(collector.actions)
   if receipt['status']!='completed':receipt['metrics']['physical_route_success']=False
   write(out/'strategy_metrics.json',receipt['metrics']['route_metrics'])
   if queries:np.savez_compressed(out/'queries.npz',physics_step=np.array([q['step'] for q in queries]),rgb=np.stack([q['rgb'] for q in queries]),state=np.concatenate([q['history'] for q in queries]))
  write(out/'config.json',cfg);receipt.update(finished=True,wall_seconds=time.monotonic()-started);write(out/'receipt.json',receipt)
  if hasattr(task,'head_calibration'):write(out/'head_camera_calibration.json',task.head_calibration)
  if worker:worker.close()
  if hasattr(task,'scene'):task.close_env()
 print(json.dumps({'status':receipt['status'],'initial_verified':receipt['initial_verified'],'metrics':receipt.get('metrics')}),flush=True)
 return 2 if receipt.get('infrastructure_failure') else 0


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--openpi',type=Path,required=True)
    p.add_argument('--checkpoint-step',type=int,required=True)
    p.add_argument('--curobo-source',type=Path)
    p.add_argument('--sim-dependencies',type=Path,help='Optional isolated pip --target runtime packages')
    p.add_argument('--expert-replay',action='store_true')
    a=p.parse_args()
    a.bundle=a.bundle.resolve();a.output=a.output.resolve();a.openpi=a.openpi.resolve()
    if a.curobo_source:sys.path.insert(0,str(a.curobo_source.resolve()))
    if a.sim_dependencies:sys.path.insert(0,str(a.sim_dependencies.resolve()))
    import numpy as np
    import transforms3d as t3d
    protocol_path=str(a.openpi/'examples/segmented_demo')
    sys.path.insert(0,protocol_path)
    from pipe_client import ParentWorker
    # Only the pipe module belongs to OpenPI in this process. Leaving its data.py
    # on sys.path shadows RoboTwin's data.decode_image_bit namespace package.
    sys.path.remove(protocol_path)
    PIPE=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    os.chdir(ROBOTWIN)
    import warp
    if os.environ.get('WARP_CACHE_PATH'):
        warp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH']
    import curobo._src.runtime as runtime
    if os.environ.get('CUROBO_CACHE_DIR'):
        runtime.cache_dir=os.environ['CUROBO_CACHE_DIR']
    from collect_roboicl_dataset import DatasetCollectionTask
    from collect_roboicl_random_pick import Collector,RECIPE,observe,measurements_array,validate_pick_geometry
    from envs.robot.roboicl_ee import FrankaKinematics
    from score_roboicl_open_routes import score_open_route,physical_hold_mask
    raise SystemExit(main(a))
