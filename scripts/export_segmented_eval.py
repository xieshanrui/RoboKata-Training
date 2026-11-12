"""Read-only postprocessing of fresh100 and training25; unannotated videos."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import collections,hashlib,html,json,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def display_title(title, spec):
 names=read(Path(__file__).resolve().parents[1]/'env_cfg/task_config/roboicl_object_names.json')
 name=names.get(f"{spec['asset']}/{spec['model_id']}")
 return name+(' · '+title.split(' · ',1)[1] if ' · ' in title else '') if name else title
def analyze(run,t):
 cfg=read(run/'config.json');contacts=read(run/'contacts.json');receipt=read(run/'receipt.json')
 with np.load(run/'measurements.npz',allow_pickle=False) as z:
  g=z['gripper_command'];poses=z['object_poses'].copy();steps=z['physics_step'].copy()
  closed=np.flatnonzero(g<.2);close=None;contact_objects=[];detail={}
  if len(closed):
   end=int(closed[0]);opened=np.flatnonzero(g[:end]>.8);start=int(opened[-1]) if len(opened) else 0
   stop=min(end+50,len(g)-1);close=[int(steps[start]),int(steps[stop])]
   for i in range(2):
    events=[e for e in contacts if close[0]<=e['physics_step']<=close[1] and e['impulse_ns']>1e-6 and f'roboicl_object_{i}' in e['bodies'] and any(b in ['panda_leftfinger','panda_rightfinger'] for b in e['bodies'])]
    unique=sorted({int(e['physics_step']) for e in events});detail[str(i)]=dict(steps=len(unique),impulse_ns=sum(e['impulse_ns'] for e in events),first_step=unique[0] if unique else None)
    if len(unique)>=3:contact_objects.append(i)
 m=receipt['metrics'];target=int(cfg['target_index']);identity={}
 for i,spec in enumerate(cfg['object_specs']):identity[str(i)]=dict(asset=spec['asset'],model_id=spec['model_id'])
 return dict(id=t['id'],category=t['stratum'],family=t['query_record']['initial_family_id'],target_index=target,target_object=t['query_record']['object_id'],target_grasp=t['query_record']['grasp_id'],split=t['split'],target_title=display_title(t['query_record']['title'],cfg['object_specs'][target]),first_closure_contact_objects=contact_objects,first_closure_window=close,contact_details=detail,first_contact_correct=contact_objects==[target],no_closure=close is None,identities=identity,metrics=m,duration_s=float(steps[-1]*.004)),poses,steps,cfg
def export(t):
 run=H/'runs'/t['id'];row,poses,steps,cfg=analyze(run,t)
 with np.load(run/'policy_recording.npz',allow_pickle=False) as z:
  q=z['rgb'].copy();obs=z['observation_step'];assert np.array_equal(obs,np.arange(len(q))*5)
 dp=B/t['demo_record']['source']/'policy_recording.npz'
 assert sha(dp)==t['demo_record']['recording_sha256']
 with np.load(dp,allow_pickle=False) as z:d=z['rgb'].copy();assert np.allclose(z['timestamp_s'],np.arange(len(d))*.02)
 def frame(i):
  dd=d[min(i,len(d)-1)];qq=q[min(i,len(q)-1)]
  return np.concatenate([np.concatenate(dd,axis=1),np.concatenate(qq,axis=1)],axis=0)
 n=max(len(d),len(q));mp=A/(t['id']+'.mp4');tmp=mp.with_suffix('.partial.mp4')
 p=subprocess.Popen([FF,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','448x448','-r','50','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
 try:
  for i in range(n):p.stdin.write(frame(i).tobytes())
 finally:p.stdin.close()
 err=p.stderr.read();assert p.wait()==0,err.decode('utf-8',errors='replace')
 dec=subprocess.run([FF,'-v','error','-i',str(tmp),'-f','null','-'],capture_output=True);assert dec.returncode==0 and not dec.stderr
 tmp.replace(mp);Image.fromarray(frame(0)).save(A/(t['id']+'.jpg'),quality=90)
 # Six raw policy frames for actual review; no claim of continuous human watching.
 ii=np.linspace(0,len(q)-1,6).astype(int);sheet=np.concatenate([np.concatenate(q[i],axis=0) for i in ii],axis=1)
 Image.fromarray(sheet).save(A/(t['id']+'_phases.jpg'),quality=90)
 cal=read(run/'head_camera_calibration.json');K=np.asarray(cal['intrinsic_cv']);E=np.asarray(cal['extrinsic_cv'])
 def project(points):
  xyz=np.c_[points,np.ones(len(points))]@E.T;p=xyz@K.T;return p[:,:2]/p[:,2:]
 im=Image.fromarray(q[0,0]).resize((672,672));draw=ImageDraw.Draw(im)
 regs=[next(r for r in cfg['strategy']['regions'] if r['id']==k) for k in cfg['strategy']['sequence']]
 for j,reg in enumerate(regs):
  for a,b in [(0,1),(0,2),(1,2)]:
   angles=np.linspace(0,2*np.pi,60);v=np.tile(reg['center_m'],(60,1));v[:,a]+=reg['radius_m']*np.cos(angles);v[:,b]+=reg['radius_m']*np.sin(angles)
   draw.line([tuple(x) for x in project(v)*3],fill='#ffb020',width=2)
  uv=project(np.array([reg['center_m']]))[0]*3;draw.text(tuple(uv),str(j+1),fill='#ff2020',stroke_width=1)
 for i,color in [(0,'#00d3ff'),(1,'#e56fff')]:
  uv=project(poses[::5,i,:3])*3
  draw.line([tuple(x) for x in uv if np.isfinite(x).all()],fill=color,width=2)
 im.save(A/(t['id']+'_waypoints.jpg'),quality=90)
 # Simple SVG measured XY/XZ, with task waypoint circles in matching metric coordinates.
 chunks=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="340" viewBox="0 0 760 340"><rect width="760" height="340" fill="#f8f6ef"/>']
 for col,pair in enumerate([(0,1),(0,2)]):
  v=poses[:,:,:3][:,:,pair].reshape(-1,2);rc=np.asarray([r['center_m'] for r in regs])[:,pair];allv=np.vstack([v,rc]);low=allv.min(0)-.04;high=allv.max(0)+.04
  scale=min(320/(high[0]-low[0]),260/(high[1]-low[1]));xoff=col*380+25
  def uv(v):return np.c_[xoff+(v[:,0]-low[0])*scale,300-(v[:,1]-low[1])*scale]
  chunks.append(f'<text x="{xoff}" y="20" font-family="sans-serif">{"XY" if col==0 else "XZ"} / meters</text>')
  for i,color in [(0,'#0088aa'),(1,'#ad48b0')]:
   xy=uv(poses[::5,i,:3][:,pair]);points=' '.join(f'{x:.2f},{y:.2f}' for x,y in xy);chunks.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
  for j,r in enumerate(regs):
   x,y=uv(np.asarray([r['center_m']])[:,pair])[0];chunks.append(f'<circle cx="{x}" cy="{y}" r="{r["radius_m"]*scale}" fill="none" stroke="#c08818"/><text x="{x}" y="{y}">{j+1}</text>')
 chunks.append('</svg>');(A/(t['id']+'_trajectory.svg')).write_text(''.join(chunks),encoding='utf-8')
 row.update(media=dict(video='assets/'+mp.name,poster='assets/'+t['id']+'.jpg',phases='assets/'+t['id']+'_phases.jpg',waypoints='assets/'+t['id']+'_waypoints.jpg',trajectory='assets/'+t['id']+'_trajectory.svg'),video=dict(full_decode=True,burned_text=False,overlays=False,fps=50,normal_speed=True,frames=n,demo_frames=len(d),policy_frames=len(q),demo_sha256=sha(dp),policy_sha256=sha(run/'policy_recording.npz')))
 write(H/'report'/(t['id']+'.json'),row);print(t['id'],'EXPORTED',flush=True);return row

def summarize(rr):
 return dict(n=len(rr),first_contact_correct=sum(r['first_contact_correct'] for r in rr),correct_lift=sum(r['metrics']['correct_object'] for r in rr),wrong_lift=sum(bool(r['metrics']['lifted_indices']) and not r['metrics']['correct_object'] for r in rr),no_lift=sum(not r['metrics']['lifted_indices'] for r in rr),released_settled=sum(r['metrics']['released_settled'] for r in rr),task_success=sum(r['metrics']['physical_route_success'] for r in rr))

def main(args):
 global H,A,B,FF
 H=args.run.resolve();B=args.bundle.resolve();A=H/'report/assets';A.mkdir(parents=True,exist_ok=True)
 FF=str(args.ffmpeg.resolve()) if args.ffmpeg else imageio_ffmpeg.get_ffmpeg_exe()
 completed=read(H/'complete.json');assert completed['completed']
 ids=set(completed['ids']);panel=[t for t in read(B/'panel.json')['trials'] if t['id'] in ids]
 assert len(panel)==len(ids)
 for t in panel:assert read(H/'runs'/t['id']/'execution50_verified.json')['passed']
 rows=[export(t) for t in panel]
 totals={s:summarize([r for r in rows if r['split']==s]) for s in sorted({r['split'] for r in rows})}
 write(H/'analysis.json',dict(rows=rows,totals=totals,checkpoint=completed['checkpoint_step'],execute_tokens=50))
 page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Segmented π0 evaluation</title><style>body{font:16px system-ui;margin:30px;background:#f5f3ed}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}article{background:white;padding:16px}video,img{width:100%}summary{cursor:pointer}</style><h1>分段π0评测</h1>']
 page.append(f"<p>检查点 {completed['checkpoint_step']}；预测50／执行50；{len(rows)}条首尝试。抓法语义未分类，抓起不等于完整任务成功。</p>")
 for label,v in totals.items():page.append(f"<p>{html.escape(label)}：{v['n']}条，正确抓起 {v['correct_lift']}，抓错 {v['wrong_lift']}，松爪落稳 {v['released_settled']}，物理路线完整 {v['task_success']}。</p>")
 page.append('<div class="grid">')
 for r in rows:
  media=r['media'];m=r['metrics']
  page.append(f'<article><h3>{html.escape(r["id"])}</h3><p>{html.escape(r["target_title"])} · {html.escape(r["split"])}</p><p>正确抓起 {m["correct_object"]}；路点 {m["waypoint_success"]}；目标放置 {m["placed"]}</p><video controls preload="none" poster="{media["poster"]}" src="{media["video"]}"></video><details><summary>标定路点、实测轨迹与阶段RGB</summary><img src="{media["waypoints"]}" loading="lazy"><img src="{media["trajectory"]}" loading="lazy"><img src="{media["phases"]}" loading="lazy"></details></article>')
 page.append('</div><p>视频无文字，Demo在上、策略在下，50fps正常仿真时间。完整解码通过；不代表已连续人工观看或完成浏览器交互检查。</p></html>')
 (H/'report/index.html').write_text(''.join(page),encoding='utf-8')
 write(H/'report/verification.json',dict(cases=len(rows),all_decoded=True,no_burned_text=True,normal_speed_50fps=True,browser_interaction_tested=False,continuous_human_watch=False))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--ffmpeg',type=Path)
 main(p.parse_args())
