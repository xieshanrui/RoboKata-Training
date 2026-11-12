"""Freeze the existing comparison's six human RGB Demos, without evaluation."""
import argparse
from pathlib import Path
from .protocol import read,write,sha,EVAL_TASKS,EXCLUDED

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--demo-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    entries=read(a.source_manifest).values()
    selected={}
    for row in entries:
        if row['task'] in EVAL_TASKS:
            if row['task'] in selected:raise ValueError('Multiple Demos for one task')
            assert sha(a.demo_root/row['file'])==row['sha256'],row['file']
            selected[row['task']]=row
    assert set(selected)==set(EVAL_TASKS)
    a.output.mkdir(parents=True)
    write(a.output/'demo_map.json',{t:selected[t]['file'] for t in EVAL_TASKS})
    write(a.output/'provenance.json',{'source_manifest':str(a.source_manifest.resolve()),
          'source_sha256':sha(a.source_manifest),'demo_root':str(a.demo_root.resolve()),
          'demos':selected,'excluded_eval_task':EXCLUDED,'training_tasks_unchanged':43,
          'rollouts_launched':0})
    print('Frozen six existing RGB Demos; no policy execution')

if __name__=='__main__':main()
