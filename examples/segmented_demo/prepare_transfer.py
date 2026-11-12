"""Prepare verified native RGB/action records, never include held-out tasks.

--limit prepares a representative finite integration subset; normalization
still uses the complete selected training split. Such outputs are smoke-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import shutil
from pathlib import Path
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torchcodec.decoders import VideoDecoder

from data import dense_demo_indices, read, require, sha, write


def video_info(path):
    decoder = VideoDecoder(str(path), device='cpu', num_ffmpeg_threads=1)
    n, fps = len(decoder), float(decoder.metadata.average_fps)
    require(n > 0 and fps > 0, 'Invalid video clock')
    return n, fps


def export_rgb(paths, starts, frames, fps, target):
    output = np.lib.format.open_memmap(target, mode='w+', dtype=np.uint8, shape=(frames, len(paths), 224, 224, 3))
    for view, (path, start) in enumerate(zip(paths, starts, strict=True)):
        count, actual_fps = video_info(path)
        require(abs(actual_fps-fps) < .01 and start+frames <= count, 'Video/data timing mismatch')
        decoder = VideoDecoder(str(path), device='cpu', num_ffmpeg_threads=1)
        for begin in range(0,frames,64):
            ids=list(range(start+begin,start+min(begin+64,frames)))
            batch=decoder.get_frames_at(indices=ids)
            require(np.max(np.abs(batch.pts_seconds.numpy()-np.asarray(ids)/fps)) < 1/fps+.001, 'Decoder timestamps differ')
            for j,rgb in enumerate(batch.data.permute(0,2,3,1).numpy()):
                output[begin+j, view] = cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)
    output.flush()
    require(np.array_equal(output.shape, (frames, len(paths), 224,224,3)), 'RGB layout')
    del output


def native_rows(args):
    queries, humans = [], {}
    if args.dataset == 'robotwin':
        source = read(args.input)
        require(source['train_tasks'] == 43 and not source['smoke_only'], 'Full43 training manifest required')
        require(len(source['rows']) in (2143,2149), 'Unexpected training record count')
        for i, row in enumerate(source['rows']):
            table = pq.read_table(args.root/row['robot_data'], columns=['frame_index','timestamp','action','observation.state']).to_pydict()
            action, state = (np.asarray(table[k], np.float32) for k in ('action','observation.state'))
            n = len(action);times = np.asarray(table['timestamp'], np.float64)
            require(table['frame_index'] == list(range(n)) and np.allclose(times, np.arange(n)/50, atol=1e-4), 'RoboTwin recorded clock')
            require(action.shape == state.shape == (n,16), 'Native EEF16 dimensions')
            for value in (action,state):
                require(all(np.allclose(np.linalg.norm(value[:,s:s+4],axis=1),1,atol=.02) for s in (3,11)), 'Native quaternion layout')
            human = row['human_video'];hid = hashlib.sha256(human.encode()).hexdigest()[:24]
            if hid not in humans:
                hn,hfps = video_info(args.root/human)
                humans[hid] = dict(id='d_'+hid, task=row['task'], frames=hn, fps=hfps, paths=[str(args.root/human)], starts=[0])
            require(humans[hid]['task'] == row['task'], 'Human video paired across tasks')
            paths = [str(args.root/row['robot_video'].replace('observation.images.cam_high', name)) for name in
                     ('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist')]
            queries.append(dict(id=f'q_{i:06d}', task=row['task'], frames=n, fps=50., paths=paths, starts=[0]*3,
                                action=action, state=state, demo_ids=[humans[hid]['id']], source_id=row['pair_id']))
    else:
        source = read(args.input)
        require(len(source['sim_episodes']) == 3000 and len(source['human_episodes']) == 750, 'C15 split required')
        for i,row in enumerate(source['human_episodes']):
            s=row['source'];require(s['fps'] == s['video_fps'] == 30, 'Native IG video clock')
            humans[str(i)] = dict(id=f'd_{i:06d}',task=row['repo_id'],frames=s['episode_length'],fps=s['fps'],
                                 paths=[s['video_path']],starts=[int(round(s['video_from_timestamp']*s['fps']))])
        for i,row in enumerate(source['sim_episodes']):
            s=row['source'];require(s['fps'] == s['video_fps'] == 30, 'Native IG Query clock')
            cached = torch.load(row['cache_path'], map_location='cpu', mmap=True, weights_only=False)
            require(cached['record_sha256'] == row['record_sha256'], 'IG source identity')
            action = np.array(cached['actions'],np.float32,copy=True);state=np.array(cached['states'],np.float32,copy=True)
            n=s['episode_length'];require(action.shape == (n,16) and state.shape == (n,18), 'Native IG dimensions')
            require(np.allclose(np.asarray(cached['timestamps']),np.arange(n)/30,atol=1e-4), 'IG recorded clock')
            ids=[h['id'] for h in humans.values() if h['task']==row['human_repo_id']]
            require(len(ids)==50,'Each C15 task requires50 humans')
            queries.append(dict(id=f'q_{i:06d}',task=row['human_repo_id'],frames=n,fps=30.,paths=[s['video_path']],
                                starts=[int(round(s['video_from_timestamp']*30))],action=action,state=state,demo_ids=ids,
                                source_id=f"{row['repo_id']}/{row['episode_index']}",record_sha256=row['record_sha256']))
    return queries,list(humans.values())


def materialize(row, output):
    require(shutil.disk_usage(output).free > 150*2**30, 'Preparation disk reserve exhausted')
    folder=output/'records'/row['id'];folder.mkdir(parents=True)
    labels=dict(timestamp_s=np.arange(row['frames'],dtype=np.float64)/row['fps'])
    if 'action' in row:
        for k in ('action','state'):
            require(np.isfinite(row[k]).all(),'Nonfinite native labels');labels[k]=row.pop(k)
    np.savez(folder/'supervision.npz',**labels)
    export_rgb(row['paths'],row['starts'],row['frames'],row['fps'],folder/'rgb.npy')
    st=(folder/'rgb.npy').stat();row.update(seconds=row['frames']/row['fps'],labels_sha256=sha(folder/'supervision.npz'),
        recording_sha256=sha(folder/'rgb.npy'),rgb_bytes=st.st_size,rgb_mtime_ns=st.st_mtime_ns)
    row['source_videos']=[dict(path=x,bytes=Path(x).stat().st_size,mtime_ns=Path(x).stat().st_mtime_ns) for x in row['paths']]
    return row['id']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['robotwin','ig10k'],required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();cv2.setNumThreads(2);require(not a.output.exists(), 'Preserve existing preparation')
    require(1 <= a.workers <= 8, 'Use1–8 preparation workers')
    torch.set_num_threads(1)
    a.output.mkdir(parents=True);started=time.time()
    queries,humans=native_rows(a);full_count=len(queries);frames=sum(q['frames'] for q in queries)
    stats={}
    for k in ('action','state'):
        total=sum(np.asarray(q[k],np.float64).sum(0) for q in queries)
        square=sum(np.square(np.asarray(q[k],np.float64)).sum(0) for q in queries)
        mean=total/frames;std=np.sqrt(np.maximum(square/frames-mean*mean,0)).clip(.01)
        stats[k]=dict(mean=mean.tolist(),std=std.tolist())
    stats.update(training_frames=frames,training_episodes=full_count,source_sha256=sha(a.input),clipping=False)
    if a.limit:
        longest=max(humans,key=lambda h:h['frames']/h['fps'])
        required=[max(range(len(queries)),key=lambda i:queries[i]['frames']),next(i for i,q in enumerate(queries) if longest['id'] in q['demo_ids'])]
        indices=list(dict.fromkeys(required+np.linspace(0,len(queries)-1,a.limit,dtype=int).tolist()))[:a.limit]
        queries=[queries[i] for i in indices]
        by_id={h['id']:h for h in humans}
        for q in queries:
            q['demo_ids']=list(dict.fromkeys([q['demo_ids'][0],max(q['demo_ids'],key=lambda d:by_id[d]['frames']/by_id[d]['fps'])]))
        used={d for q in queries for d in q['demo_ids']};humans=[h for h in humans if h['id'] in used]
    pairs=[[d,q['id']] for q in queries for d in q['demo_ids']]
    write(a.output/'normalization.json',stats);write(a.output/'pairing.json',dict(pairs=pairs))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(materialize,row,a.output) for row in [*queries,*humans]]
        try:
            for completed,future in enumerate(as_completed(pending),1):
                identity=future.result()
                write(a.output/'progress.json',dict(record=identity,completed=completed,total=len(pending),seconds=time.time()-started))
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    maximum=max(len(dense_demo_indices(np.arange(h['frames'])/h['fps'],h['seconds'],5)[0]) for h in humans)
    manifest=dict(schema='segmented_transfer_v1',dataset=a.dataset,smoke_only=bool(a.limit),rows=queries,demos=humans,
        demo_views=1,query_image_keys=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'] if a.dataset=='robotwin' else ['base_0_rgb'],
        action_encoding='robotwin_eef16' if a.dataset=='robotwin' else 'ig_joint16',pairing_file='pairing.json',
        pairing_sha256=sha(a.output/'pairing.json'),input_manifest=str(a.input),input_sha256=sha(a.input),
        normalization_episodes=full_count,normalization_frames=frames,maximum_demo_frames_5hz=maximum)
    manifest['preparation_source_sha256']=sha(__file__)
    write(a.output/'manifest.json',manifest)
    write(a.output/'preparation_receipt.json',dict(passed=True,queries=len(queries),humans=len(humans),pairs=len(pairs),
        query_frames=sum(q['frames'] for q in queries),full_training_frames=frames,maximum_demo_frames_5hz=maximum,
        smoke_only=bool(a.limit),seconds=time.time()-started,manifest_sha256=sha(a.output/'manifest.json')))
    print((a.output/'preparation_receipt.json').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':
    main()
