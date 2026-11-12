"""Cache frozen pre-adapter DINO CLS features, keyed by actual Demo bytes."""
import argparse
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from .protocol import read,write,sha,DINO_REVISION,verify_local_dino
from .data import demo_rgb
from .model import demo_encoder

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--ig-root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--hf-cache',type=Path);p.add_argument('--dino-path',type=Path)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.dino_path:
        verify_local_dino(a.dino_path)
        local=str(a.dino_path)
    else:
        local=snapshot_download('facebook/dinov2-large',revision=DINO_REVISION,cache_dir=a.hf_cache,
                            allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])
        write(Path(local)/'revision.json',{'revision':DINO_REVISION,'files':{
            n:sha(Path(local)/n) for n in ['config.json','preprocessor_config.json','model.safetensors']}})
    enc=demo_encoder(a.ig_root).to(a.device).eval();enc._meta=dict(enc._meta,hf_model=local)
    enc._load_backbone();enc._backbone.to(a.device);a.cache.mkdir(parents=True,exist_ok=True)
    for name in sorted({r['human_video'] for r in read(a.prepared)['rows']}):
        path=a.data/name;key=sha(path);out=a.cache/(key+'.pt')
        if out.exists():
            v=torch.load(out,weights_only=True);assert v['encoder_revision']==DINO_REVISION and v['demo_sha256']==key
            continue
        with torch.no_grad():raw,_=enc._encode_dino(demo_rgb(path).unsqueeze(0))
        assert raw.shape==(1,1024) and torch.isfinite(raw).all()
        torch.save({'raw_cls':raw.cpu(),'demo_sha256':key,'encoder_revision':DINO_REVISION},out)
        print(name,flush=True)
if __name__=='__main__':main()
