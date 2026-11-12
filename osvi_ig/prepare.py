"""C15 RGB and native joint-action trajectories with train-only normalization."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from act_robotwin.protocol import sha,write
from act_robotwin.data import decoder

def main():
    p=argparse.ArgumentParser()
    for n in ['manifest','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    assert len(manifest['sim_episodes'])==3000 and len(manifest['human_episodes'])==750
    a.output.mkdir(parents=True,exist_ok=True)
    human=manifest['human_episodes'];sim=manifest['sim_episodes']
    if a.smoke:
        sim=[sim[i] for i in [0,199,400,1000,2000,2999]]
        tasks={x['human_repo_id'] for x in sim};human=[x for x in human if x['human_repo_id'] in tasks and x['episode_index']<2]
    total=np.zeros(16,np.float64);square=total.copy();count=0
    for record in sim:
        cached=torch.load(record['cache_path'],map_location='cpu',weights_only=False,mmap=True)
        actions=np.asarray(cached['actions'],dtype=np.float64)
        assert actions.shape==(record['source']['episode_length'],16) and np.isfinite(actions).all()
        total+=actions.sum(0);square+=(actions*actions).sum(0);count+=len(actions)
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,0)).clip(.01)
    stats=dict(mean=mean.tolist(),std=std.tolist(),rows=count,episodes=len(sim),
        representation='absolute native qpos targets and source grippers',clipping=False)
    def cache(record):
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        source=record['source'];n=source['episode_length'];kind=record['kind']
        name=f"{kind}_{record['repo_id']}_{record['episode_index']:06d}.pt";dest=a.output/name
        cols=['episode_index','frame_index','timestamp']+(['action.qpos_gripper_actions'] if kind=='sim' else [])
        table=pq.read_table(source['data_paths'],columns=cols)
        table=table.filter(pc.equal(table['episode_index'],record['episode_index'])).sort_by('frame_index')
        ids=table['frame_index'].to_numpy();times=table['timestamp'].to_numpy()
        assert len(table)==n and np.array_equal(ids,np.arange(n))
        assert np.max(np.abs(times-ids/source['fps']))<.001
        selected=np.linspace(0,n-1,6 if kind=='sim' else 10).astype(int)
        global_times=source['video_from_timestamp']+times[selected].astype(np.float64)
        d=decoder(source['video_path']);frames=d.get_frames_played_at(seconds=global_times.tolist())
        assert np.max(np.abs(frames.pts_seconds.numpy()-global_times))<=1/source['video_fps']+.001
        import cv2
        rgb=np.stack([cv2.resize(f,(224,224),interpolation=cv2.INTER_AREA) for f in frames.data.permute(0,2,3,1).numpy()])
        payload=dict(rgb=torch.from_numpy(rgb),record_sha256=record['record_sha256'],selected_indices=selected.tolist(),source_times=global_times.tolist())
        if kind=='sim':
            actions=np.asarray(table['action.qpos_gripper_actions'].to_pylist(),dtype=np.float32)
            original=torch.load(record['cache_path'],map_location='cpu',weights_only=False,mmap=True)
            assert np.array_equal(actions,np.asarray(original['actions']))
            assert actions.shape==(n,16) and np.isfinite(actions).all()
            normalized=((actions-mean)/std).astype(np.float32)
            assert np.max(np.abs(normalized*std+mean-actions))<1e-5
            ti=np.linspace(0,n-1,50).astype(int)
            payload.update(trajectory=torch.from_numpy(normalized[ti]),initial_action=torch.from_numpy(actions[0]),
                source_action_samples=torch.from_numpy(actions[ti]))
        torch.save(payload,dest)
        return dict(path=str(dest.resolve()),human_repo_id=record['human_repo_id'],sha256=sha(dest))
    result={}
    for kind,records in [('human',human),('sim',sim)]:
        result[kind]=[]
        for i,record in enumerate(records):
            result[kind].append(cache(record))
            if (i+1)%50==0:print(kind,i+1,len(records),flush=True)
    write(a.output/'prepared.json',dict(schema='osvi_ig_c15_native_joint_v1',smoke_only=a.smoke,
        manifest_sha256=sha(a.manifest),records_sha256=manifest['records_sha256'],normalization=stats,
        coordinate='native absolute joint commands; source gripper convention; no IK/FK',
        demo='RGB only; random same-task human per epoch/query; no human pose',**result))

if __name__=='__main__':main()
