from pathlib import Path
import sys, os, json, importlib, traceback, subprocess, hashlib, time
import numpy as np
import yaml
import imageio_ffmpeg
import h5py, cv2
from PIL import Image

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--case',required=True)
parser.add_argument('--route-catalog',type=Path,help='Frozen canonical route definitions; verifies ordered world coordinates before simulation')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--embodiment-assets',type=Path,help='Optional embodiment asset root; object assets stay in this repository')
args_cli=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
case_id=args_cli.case
case=next(c for c in json.loads(args_cli.manifest.read_text(encoding='utf-8'))['cases'] if c['id']==case_id)
if case.get('alignment_revision') and args_cli.route_catalog is None:
    raise ValueError('Aligned collection requires an independent frozen route catalog')
if args_cli.route_catalog is not None:
    catalog_bytes=args_cli.route_catalog.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest()!=case['route_catalog_sha256']:
        raise ValueError('Frozen route catalog hash mismatch')
    route=next(r for r in json.loads(catalog_bytes) if r['name']==case['complete_task'].split('|')[-1])
    expected=[route['points_world_m'][i] for i in route['order']]
    if case['route']!=route['name'] or case['waypoints']!=expected:
        raise ValueError('Case waypoint order differs from the canonical task')
if case['category'] != 'choice_route':
    raise ValueError('This collector supports the validated choice_route manifest only')
out=args_cli.output/case_id
out.mkdir(parents=True,exist_ok=False)
row=dict(case,completed=False,success=False,reason=None,
         attempt_kind='first_attempt',
         execution_source_sha256=hashlib.sha256((ROOT/('envs/'+case['task']+'.py')).read_bytes()).hexdigest())
task=None;rec=None;start=time.monotonic()

class SceneRecorder:
    def __init__(self,scene,task,actors):
        self.scene=scene;self.task=task;self.actors=actors;self.samples=[];self.stages=[]
        self.step_index=0;self.frames=0;self.collisions=[];self.proc=None;self.calibration=None;self.contact_trace=[];self.joint_trace=[];self.frame_observations=[];self.frame_steps=[]
        def entity_ids(actor):
            body = actor.actor
            return {link.entity.per_scene_id for link in body.get_links()} if hasattr(body, 'get_links') else {body.per_scene_id}
        self.object_body_ids = [sorted(entity_ids(actor)) for actor in actors]
        self.inactive_body_ids = set().union(*(entity_ids(actor) for actor in task.inactive_objects)) if task.inactive_objects else set()
        self.active_body_ids = set().union(*(set(ids) for ids in self.object_body_ids)) - self.inactive_body_ids
        self.robot_body_ids = {link.entity.per_scene_id for side in ['left', 'right'] for link in getattr(task.robot, side+'_entity').get_links()}
    def __getattr__(self,name):return getattr(self.scene,name)
    def step(self):
        self.scene.step();self.step_index+=1;self.record()
    def record(self):
        robot=self.task.robot
        pose=[]
        for a in self.actors:
            p=a.get_pose();pose.extend([*p.p,*p.q])
        sample=[self.step_index/250,*pose,*robot.get_left_ee_pose(),*robot.get_right_ee_pose(),
                *robot.get_left_arm_real_jointState(),*robot.get_right_arm_real_jointState(),
                *robot.get_left_arm_jointState(),*robot.get_right_arm_jointState()]
        self.samples.append(sample);self.stages.append(getattr(self.task,'stage','native_expert'))
        self.joint_trace.append([self.step_index/250,*robot.left_entity.get_qpos(),*robot.left_entity.get_qvel(),*robot.right_entity.get_qpos(),*robot.right_entity.get_qvel()])
        for contact in self.scene.get_contacts():
            names_pair=[body.entity.name for body in contact.bodies]
            points=[]
            for p in contact.points:
                if np.linalg.norm(p.impulse)>1e-7:
                    points.append(dict(position=np.asarray(p.position).tolist(),normal=np.asarray(p.normal).tolist(),impulse=np.asarray(p.impulse).tolist(),separation=float(p.separation)))
            if points:self.contact_trace.append(dict(step=self.step_index,stage=getattr(self.task,'stage','unknown'),bodies=names_pair,body_ids=[b.entity.per_scene_id for b in contact.bodies],points=points))
        inactive, active, robot_ids = self.inactive_body_ids, self.active_body_ids, self.robot_body_ids
        if inactive:
            for c in self.scene.get_contacts():
                a,b=[body.entity.name for body in c.bodies]
                aid,bid=[body.entity.per_scene_id for body in c.bodies]
                if (aid in inactive and bid in active|robot_ids) or (bid in inactive and aid in active|robot_ids):
                    impulse=max([float(np.linalg.norm(p.impulse)) for p in c.points] or [0.])
                    if impulse>1e-4:self.collisions.append([self.step_index,a,b,impulse])
        if self.step_index%5==0:self.capture()
    def capture(self):
        full=self.task.get_obs();obs=full['observation']
        self.frame_observations.append(full);self.frame_steps.append(self.step_index)
        def video_panel(im):
            h,w=im.shape[:2];scale=min(320/w,240/h);vw,vh=round(w*scale),round(h*scale)
            panel=np.zeros((240,320,3),dtype=np.uint8);x,y=(320-vw)//2,(240-vh)//2
            panel[y:y+vh,x:x+vw]=cv2.resize(im,(vw,vh))
            return panel
        frame=np.concatenate([video_panel(obs[k]['rgb']) for k in ['head_camera','left_camera','right_camera']],axis=1)
        if self.proc is None:
            h,w=frame.shape[:2]
            self.proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','50','-i','-','-an','-c:v','libx264','-threads','1','-crf','22','-pix_fmt','yuv420p',str(out/'video.mp4')],stdin=subprocess.PIPE)
            Image.fromarray(frame).save(out/'initial.png')
            self.initial_rgb_sha=hashlib.sha256(b''.join(obs[k]['rgb'].tobytes() for k in ['head_camera','left_camera','right_camera'])).hexdigest()
            self.calibration=self.task.cameras.get_config()
        self.proc.stdin.write(np.asarray(frame,dtype=np.uint8).tobytes());self.frames+=1
        self.last_frame=frame
    def finish(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait(timeout=60)!=0:raise RuntimeError('FFmpeg export failed')
            Image.fromarray(self.last_frame).save(out/'final.png')
        if self.frame_observations:
            with h5py.File(out/'episode.hdf5','x') as h:
                h.attrs['schema']='RoboTwin observations plus explicit 250Hz state/action provenance'
                h.attrs['physics_dt']=.004;h.attrs['rgb_dt']=.02
                h.attrs['action_semantics']='native drive targets at timestamp; full next-step applied targets preserved in trace.npz'
                h.create_dataset('timestamp',data=np.asarray(self.frame_steps)*.004)
                h.create_dataset('physics_step',data=self.frame_steps)
                for camera in ['head_camera','left_camera','right_camera']:
                    frames=[cv2.imencode('.jpg',cv2.cvtColor(v['observation'][camera]['rgb'],cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])[1].tobytes() for v in self.frame_observations]
                    h.create_dataset('observation/'+camera+'/rgb',data=np.asarray(frames,dtype='S'+str(max(map(len,frames)))))
                    for name in ['intrinsic_cv','extrinsic_cv','cam2world_gl']:
                        h.create_dataset('observation/'+camera+'/'+name,data=np.asarray([v['observation'][camera][name] for v in self.frame_observations]))
                for section in ['joint_action','endpose']:
                    for name in self.frame_observations[0][section]:
                        h.create_dataset(section+'/'+name,data=np.asarray([v[section][name] for v in self.frame_observations]))
                for side in ['left','right']:
                    h.attrs[side+'_joint_names']=json.dumps([j.name for j in getattr(self.task.robot,side+'_entity').get_active_joints()])
        (out/'contacts.json').write_text(json.dumps(self.contact_trace),encoding='utf-8')
        np.savez_compressed(out/'actual_joints.npz',samples=np.asarray(self.joint_trace))
        np.savez_compressed(out/'trace.npz',samples=np.asarray(self.samples),stages=np.asarray(self.stages))
        def conv(v):
            if isinstance(v,np.ndarray):return v.tolist()
            if isinstance(v,np.generic):return v.item()
            raise TypeError(str(type(v)))
        (out/'calibration.json').write_text(json.dumps(self.calibration,default=conv),encoding='utf-8')

def observe_initial(actors):
    return [dict(name=a.get_name(),p=a.get_pose().p.tolist(),q=a.get_pose().q.tolist()) for a in actors]

def route_checks(rec):
    data=np.asarray(rec.samples);n=len(rec.actors);idx=case['intent_index']
    pos=data[:,1+7*idx:4+7*idx];dt=.004
    speed=np.full(len(pos),np.inf)
    speed[5:]=np.linalg.norm(pos[5:]-pos[:-5],axis=1)/.020
    cursor=0;checks=[]
    for wp in case['waypoints']:
        mask=(np.linalg.norm(pos-np.asarray(wp),axis=1)<=.031)&(speed<=.05)
        run=0;best=0;end=None
        for i in range(cursor,len(mask)):
            run=run+1 if mask[i] else 0;best=max(best,run)
            if run>=51:end=i;break
        checks.append(dict(point=wp,passed=end is not None,dwell_seconds=max(0,(51 if end is not None else best)-1)*dt))
        if end is None:break
        cursor=end+1
    return checks

try:
    T=ROOT;robot=case['robot']
    args=yaml.safe_load((T/'task_config/demo_clean.yml').read_text(encoding='utf-8'))
    cfgpath=(args_cli.embodiment_assets or T/'assets/embodiments')/robot;cfg=yaml.safe_load((cfgpath/'config.yml').read_text(encoding='utf-8'))
    camera=yaml.safe_load((T/'task_config/_camera_config.yml').read_text(encoding='utf-8'))['ROBOICL_SELECT_HEAD_224_F48']
    args.update(render_backend='raster',task_name=case['task'],task_config='demo_clean',ckpt_setting='0',save_root=str(out),
                left_robot_file=str(cfgpath),right_robot_file=str(cfgpath),dual_arm_embodied=robot=='aloha-agilex',
                embodiment_dis=.6,left_embodiment_config=cfg,right_embodiment_config=cfg,
                head_camera_h=camera['h'],head_camera_w=camera['w'],eval_mode=True,
                eval_video_log=False,policy_name='ACT',render_freq=0,save_data=False,save_freq=None)
    cls=getattr(importlib.import_module('envs.roboicl_choice_route_v2'),case['task']);args['case_manifest']=case
    task=cls();task.setup_demo(now_ep_num=0,seed=case['seed'],is_test=True,**args)
    row['setup_completed']=True
    # Save the actual camera profiles, including clipping and mount offsets.
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH
    camera_profiles=yaml.safe_load((Path(CONFIGS_PATH)/'_camera_config.yml').read_text(encoding='utf-8'))
    row['camera_configuration']={name:dict(type=getattr(task.cameras,name+'_camera_type'),
        **camera_profiles[getattr(task.cameras,name+'_camera_type')]) for name in ('head','wrist')}
    row['camera_configuration']['wrist']['mount']='native camera link plus fixed profile mount_offset_m, when present'
    actors=task.objects
    task.inactive_objects=[actors[1-case['intent_index']]]
    material_changes=[];finger_links=set()
    for side in ['left','right']:
        for joint,_,_ in getattr(task.robot,side+'_gripper'):
            link=joint.child_link
            if link.entity.per_scene_id in finger_links:continue
            finger_links.add(link.entity.per_scene_id)
            for shape in link.collision_shapes:
                old=shape.physical_material;before=[float(old.static_friction),float(old.dynamic_friction)]
                shape.set_physical_material(task.scene.create_physical_material(2.,2.,float(old.restitution)))
                actual=shape.physical_material
                material_changes.append(dict(link=link.name,before=before,after=[float(actual.static_friction),float(actual.dynamic_friction)]))
    row['finger_material']=material_changes
    row['initial_objects']=observe_initial(actors)
    row['initial_native_success']=bool(task.check_success())
    if row['initial_native_success']:raise RuntimeError('Selected native task already solved at initial state')
    row['pd']={k:v for k,v in cfg.items() if 'stiff' in k or 'damping' in k}
    row['arm']=str(task.arm)
    rec=SceneRecorder(task.scene,task,actors);task.scene=rec
    row['object_body_ids']=rec.object_body_ids
    row['robot_body_ids']=sorted(rec.robot_body_ids)
    row['contact_identity_revision']='entity_ids_v1'
    rec.record()
    row['planner_requests']=[]
    for side in ['left','right']:
        original=getattr(task.robot,side+'_plan_path')
        def logged(target_pose,*args,_original=original,_side=side,**kwargs):
            res=_original(target_pose,*args,**kwargs)
            row['planner_requests'].append(dict(side=_side,stage=getattr(task,'stage','native'),target=np.asarray(target_pose).tolist(),status=res.get('status')))
            return res
        setattr(task.robot,side+'_plan_path',logged)
    row['actual_joint_names']={side:[j.name for j in getattr(task.robot,side+'_entity').get_active_joints()] for side in ['left','right']}
    task.v2_plan_directory=out/'plans'
    from envs.robot.native_v2_planner import shapes, vertices, matrix
    from envs.robot.roboicl_mesh_sweep import convex, transform
    import hppfcl as fcl
    req=fcl.DistanceRequest();req.enable_signed_distance=True
    geoms=[[convex(vertices(s) @ matrix(s.local_pose)[:3,:3].T + matrix(s.local_pose)[:3,3]) for s in shapes(a.actor)] for a in actors]
    transforms=[transform(matrix(a.get_pose())) for a in actors]
    gap=min(float(fcl.distance(a,transforms[0],b,transforms[1],req,fcl.DistanceResult())) for a in geoms[0] for b in geoms[1])
    row['initial_object_mesh_clearance_m']=gap
    if gap < -1e-6:raise RuntimeError('Initial physical object meshes intersect')
    task.play_once()
    if case['category']=='choice_route':
        row['route_checks']=route_checks(rec)
    row['native_info']=task.info
    row.update(completed=True,plan_success=bool(task.plan_success),native_success=bool(task.check_success()))
    row['success']=row['plan_success'] and row['native_success']
    row['final_objects']=observe_initial(actors)
    target_pose=task.target_actor.get_pose().p
    row['placement_error_m']=float(np.linalg.norm(target_pose[:2]-np.asarray(case['placement_xy'])))
    row['success']=row['success'] and row['placement_error_m']<=.010

    data=np.asarray(rec.samples)
    shifts=[]
    for actor in task.inactive_objects:
        i=actors.index(actor);xyz=data[:,1+7*i:4+7*i]
        shifts.append(dict(name=actor.get_name(),max_shift_m=float(np.linalg.norm(xyz-xyz[0],axis=1).max())))
    row['inactive_object_motion']=shifts
    row['non_target_shift_m']=max([x['max_shift_m'] for x in shifts] or [0.])
    row['success']=row['success'] and row['non_target_shift_m']<=.01 and not rec.collisions
    if case['category']=='choice_route':
        row['success']=row['success'] and len(row['route_checks'])==len(case['waypoints']) and all(x['passed'] for x in row['route_checks'])
    row['object_contacts']=rec.collisions
    if not row['success']:
        reasons=[]
        if not row['plan_success']:reasons.append('native_planning')
        if not row['native_success']:reasons.append('grasp_or_place')
        if row.get('non_target_shift_m',0)>.01:reasons.append('non_target_moved')
        if row.get('object_contacts'):reasons.append('objects_contact')
        if any(not x['passed'] for x in row.get('route_checks',[])):reasons.append('waypoint_dwell')
        if row.get('final_speed_m_s',0)>.03:reasons.append('not_settled')
        row['reason']=','.join(reasons)
except Exception:
    row['reason']='runtime_failure';row['error']=traceback.format_exc()
    if task is not None:
        row['stage']=getattr(task,'stage','native_expert');row['plan_success']=bool(getattr(task,'plan_success',False))
finally:
    if rec:
        try:
            rec.finish();row.update(video='video.mp4' if rec.frames else None,frames=rec.frames,rgb_hz=50,physics_steps=rec.step_index,initial_rgb_sha256=rec.__dict__.get('initial_rgb_sha'),object_contacts=rec.collisions)
        except Exception:row['export_error']=traceback.format_exc();row['success']=False
    if rec:
        robot_names={l.name for side in ['left','right'] for l in getattr(task.robot,side+'_entity').get_links()}
        table_contacts=[c for c in rec.contact_trace if 'table' in c['bodies'] and any(n in robot_names for n in c['bodies']) and max(np.linalg.norm(p['impulse']) for p in c['points'])>1e-4]
        row['robot_table_contact_steps']=len(set(c['step'] for c in table_contacts))
        self_contacts=[c for c in rec.contact_trace if all(x in robot_names for x in c['bodies']) and max(np.linalg.norm(p['impulse']) for p in c['points'])>1e-4]
        row['robot_self_contact_steps']=len(set(c['step'] for c in self_contacts))
        if self_contacts:row['success']=False;row['reason']=(row.get('reason') or '')+',robot_self_contact'
        row['success']=row.get('success',False) and not table_contacts
        if table_contacts:row['reason']=(row.get('reason') or '')+',robot_table_contact'
    if task:
        row['v2_evidence']=getattr(task,'v2_evidence',[])
        try:task.close_env()
        except Exception:row['close_error']=traceback.format_exc()
    row['wall_seconds']=time.monotonic()-start
    (out/'result.json').write_text(json.dumps(row,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding='utf-8')
    print(json.dumps({k:row.get(k) for k in ['id','completed','success','reason','stage','wall_seconds']},ensure_ascii=False),flush=True)
