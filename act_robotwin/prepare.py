"""Explicit manifest/RGB download and train-only statistics; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from .protocol import DATASET,REVISION,DINO_REVISION,training_rows,video_paths,write,sha,EVAL_TASKS
from .data import episode,decoder

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--download',action='store_true')
    p.add_argument('--smoke-pairs',type=int,default=0)
    p.add_argument('--exclude-pairs',type=Path,help='Frozen dataset-quality exclusions; never selected by policy outcomes')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    def fetch(name):
        if a.download:
            hf_hub_download(DATASET,name,repo_type='dataset',revision=REVISION,local_dir=a.data)
        if not (a.data/name).is_file():raise FileNotFoundError(a.data/name)
    for n in ['manifests/train.jsonl','splits/task_splits.json']:fetch(n)
    rows=training_rows(a.data)
    original_count=len(rows);excluded=[]
    if a.exclude_pairs:
        contract=json.loads(a.exclude_pairs.read_text(encoding='utf-8'))
        assert contract['dataset_revision']==REVISION
        excluded=contract['excluded_pairs'];assert len(set(excluded))==len(excluded)
        assert set(excluded)<={r['pair_id'] for r in rows}
        tasks={r['task'] for r in rows}
        rows=[r for r in rows if r['pair_id'] not in set(excluded)]
        assert len(rows)==contract['retained_pairs'] and {r['task'] for r in rows}==tasks
    if a.smoke_pairs:
        if not 1<=a.smoke_pairs<=8:raise ValueError('Smoke uses 1–8 fixed manifest rows')
        rows=rows[:a.smoke_pairs]
    all_action=[];all_state=[];files={}
    for i,row in enumerate(rows):
        names=[row['robot_data'],row['human_video'],*video_paths(row)]
        for n in names:
            fetch(n)
            if n not in files:files[n]=sha(a.data/n)
        action,state=episode(a.data,row)
        for n in video_paths(row):assert len(decoder(str(a.data/n)))==len(action),n
        assert len(decoder(str(a.data/row['human_video'])))>0
        all_action.append(action);all_state.append(state)
        print(f'{i+1}/{len(rows)} verified {row["pair_id"]}',flush=True)
    stats={}
    for key,parts in [('action',all_action),('state',all_state)]:
        value=np.concatenate(parts).astype(np.float64)
        stats[key]={'mean':value.mean(0).tolist(),'std':np.maximum(value.std(0),.01).tolist()}
    write(a.output,{'schema':'act_robotwin_43to6_v1','dataset_revision':REVISION,'dino_revision':DINO_REVISION,
                    'data_amendment':{'excluded_pairs':excluded,'original_pairs':original_count,'reason':'frozen released RGB decoding failures'},
                    'smoke_only':bool(a.smoke_pairs),'rows':rows,'stats':stats,'files':files,
                    'train_manifest_sha256':sha(a.data/'manifests/train.jsonl'),'train_tasks':43,
                    'eval_tasks':list(EVAL_TASKS),'excluded_eval_task':'open_microwave',
                    'normalization':'train manifest frames only; per-channel mean/std floor0.01',
                    'action':'absolute native EEF16; source quaternion order preserved; 50Hz source grid'})
if __name__=='__main__':main()
