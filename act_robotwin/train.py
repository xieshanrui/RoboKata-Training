"""Single-GPU ACT training. Smoke mode cannot produce a formal checkpoint."""
import argparse
from pathlib import Path
import json,time,random,math
import numpy as np
import torch
from torch.utils.data import DataLoader
from .protocol import read,write,sha
from .data import RobotwinDataset
from .model import ACT

def main():
    p=argparse.ArgumentParser()
    for k in ['data','prepared','ig-root','feature-cache','run']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cuda')
    p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--resume',type=Path)
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    a=p.parse_args();prepared=read(a.prepared)
    if a.batch_size<1 or a.epochs<1 or not 0<=a.warmup_epochs<a.epochs:raise ValueError('Invalid batch/epoch/warmup budget')
    if bool(a.smoke_steps)!=prepared['smoke_only']:raise ValueError('Smoke/full data and execution modes must match')
    if a.smoke_steps and not 1<=a.smoke_steps<=20:raise ValueError('Smoke budget is 1–20 updates')
    if a.resume and a.smoke_steps:raise ValueError('Smoke checkpoint cannot resume formal training')
    if a.run.exists() and not a.resume:raise FileExistsError(a.run)
    a.run.mkdir(parents=True,exist_ok=True)
    for n,h in prepared['files'].items():
        if sha(a.data/n)!=h:raise ValueError('Changed prepared data: '+n)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    dataset=RobotwinDataset(a.data,a.prepared,a.feature_cache)
    model=ACT(a.ig_root).to(a.device)
    groups=[{'params':[v for k,v in model.named_parameters() if not k.startswith('core.backbones.')]},
            {'params':[v for k,v in model.named_parameters() if k.startswith('core.backbones.')],'lr':1e-5}]
    optimizer=torch.optim.AdamW(groups,lr=a.lr,weight_decay=1e-4)
    from diffusers.optimization import get_scheduler
    batches=math.ceil(len(dataset)/a.batch_size)
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=a.warmup_epochs*batches,
                            num_training_steps=a.epochs*batches)
    source_hashes={str(f.relative_to(a.ig_root)):sha(f) for rel in ['examples/baselines/act/act','examples/baselines/encoders/task_encoder/video_backbone.py'] for f in ([a.ig_root/rel] if (a.ig_root/rel).is_file() else (a.ig_root/rel).rglob('*.py'))}
    local_hashes={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}
    start_epoch=0;start_batch=0;step=0
    if a.resume:
        c=torch.load(a.resume,map_location=a.device,weights_only=True)
        assert not c['smoke_only'] and c['prepared_sha256']==sha(a.prepared)
        assert c['config']==model.config and c['batch_size']==a.batch_size and c['seed']==a.seed and c['lr']==a.lr
        assert c['epochs']==a.epochs and c['warmup_epochs']==a.warmup_epochs
        assert c.get('precision','fp32')==a.precision
        if c.get('upstream_files')!=source_hashes or c.get('local_files')!=local_hashes:
            raise ValueError('Checkpoint source identity is missing or changed; resume legacy checkpoints only with their frozen source')
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        step=c['step'];start_epoch=c['next_epoch'];start_batch=c.get('next_batch',0);torch.set_rng_state(c['rng'].cpu())
        if str(a.device).startswith('cuda'):torch.cuda.set_rng_state(c['cuda_rng'].cpu())
    if start_epoch>=a.epochs:raise ValueError('Checkpoint already reached the frozen epoch budget')
    write(a.run/'config.json',{'model':model.config,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'prepared_sha256':sha(a.prepared),'upstream_files':source_hashes,'local_files':local_hashes,'train_frames':len(dataset),'smoke_only':prepared['smoke_only'],'selection':'final epoch, no held-out task checkpoint selection'})
    if str(a.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();metrics=[];last_batch=None
    def save(epoch,next_batch):
        c={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':model.config,
           'stats':prepared['stats'],'prepared_sha256':sha(a.prepared),'step':step,
           'next_epoch':epoch,'next_batch':next_batch,'smoke_only':prepared['smoke_only'],
           'batch_size':a.batch_size,'seed':a.seed,'lr':a.lr,'precision':a.precision,
           'scheduler':scheduler.state_dict(),'epochs':a.epochs,'warmup_epochs':a.warmup_epochs,
           'upstream_files':source_hashes,'local_files':local_hashes,
           'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if str(a.device).startswith('cuda') else None}
        temp=a.run/'latest.tmp';torch.save(c,temp);temp.replace(a.run/'latest.pt')
        if next_batch==0 and epoch%2==0:
            import os
            stage=a.run/f'epoch_{epoch:03d}.pt'
            if not stage.exists():os.link(a.run/'latest.pt',stage)
        write(a.run/'checkpoint_status.json',{'step':step,'next_epoch':epoch,'next_batch':next_batch,'time':time.time()})
    for epoch in range(start_epoch,a.epochs):
        order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
        offset=start_batch if epoch==start_epoch else 0
        loader=DataLoader(torch.utils.data.Subset(dataset,order[offset*a.batch_size:]),batch_size=a.batch_size,shuffle=False,drop_last=False,num_workers=a.workers,
                          generator=torch.Generator().manual_seed(a.seed+epoch),pin_memory=str(a.device).startswith('cuda'))
        model.train()
        batch_ready=time.monotonic()
        for batch_index,batch in enumerate(loader,start=offset):
            tick=time.monotonic();data_seconds=tick-batch_ready
            batch={k:v.to(a.device,non_blocking=True) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
                loss,parts=model.loss(batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            used_lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();step+=1
            if str(a.device).startswith('cuda'):torch.cuda.synchronize()
            m={'step':step,'epoch':epoch,'batch':len(batch['state']),'loss':loss.item(),
               'grad_norm':float(grad),'lr':used_lr,'data_seconds':data_seconds,
               'seconds':time.monotonic()-tick,**{k:float(v) for k,v in parts.items()}}
            metrics.append(m)
            with (a.run/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(m)+'\n')
            print(json.dumps(m),flush=True);last_batch={k:v[:1] for k,v in batch.items()}
            write(a.run/'latest_metrics.json',{**m,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                  'total_updates':a.epochs*batches,'time':time.time()})
            if step==20 or step%5000==0:save(epoch,batch_index+1)
            batch_ready=time.monotonic()
            if a.smoke_steps and step>=a.smoke_steps:break
        save(epoch+1,0)
        if a.smoke_steps:break
    assert metrics and last_batch is not None
    # Exact checkpoint readback and deterministic prior-mode inference on real input.
    model.eval()
    with torch.no_grad():before=model(last_batch)[0].clone()
    saved=torch.load(a.run/'latest.pt',map_location=a.device,weights_only=True)
    model.load_state_dict(saved['model'],strict=True)
    with torch.no_grad():after=model(last_batch)[0]
    assert torch.isfinite(after).all() and torch.equal(before,after) and after.shape==(1,24,16)
    for k in ['demo_adapter.0.weight','core.action_head.weight']:assert torch.isfinite(saved['model'][k]).all()
    np.save(a.run/'inference_normalized.npy',after.cpu().numpy(),allow_pickle=False)
    result={'completed':True,'smoke_only':prepared['smoke_only'],'updates':step,'checkpoint_readback_exact':True,
            'inference_shape':list(after.shape),'wall_seconds':time.monotonic()-started,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if str(a.device).startswith('cuda') else None,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if str(a.device).startswith('cuda') else None,
            'median_update_seconds':float(np.median([m['seconds'] for m in metrics])),
            'median_data_and_update_seconds':float(np.median([m['data_seconds']+m['seconds'] for m in metrics])),
            'batch_size':a.batch_size}
    write(a.run/'complete.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
