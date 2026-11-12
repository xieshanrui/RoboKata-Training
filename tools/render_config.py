"""Resolve explicit environment paths into a new run config; no training."""
import argparse,hashlib,json,os
from pathlib import Path
from string import Template
p=argparse.ArgumentParser();p.add_argument('template',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
env=dict(os.environ,ROOT=str(Path(__file__).resolve().parents[1]))
if env.get('PI_DATA'):
 path=Path(env['PI_DATA'])/'normalization.json'
 if path.is_file():env['NORMALIZATION_SHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
def expand(x):
 if isinstance(x,str):return Template(x).substitute(env)
 if isinstance(x,list):return [expand(v) for v in x]
 if isinstance(x,dict):return {k:expand(v) for k,v in x.items()}
 return x
cfg=expand(json.loads(a.template.read_text(encoding='utf-8')))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x',encoding='utf-8') as f:json.dump(cfg,f,indent=2)
print(a.output)
