"""Bounded single-GPU RoboTwin adaptation, final-checkpoint selection."""
import argparse,json,time,random,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from act_robotwin.protocol import write,sha
from .data import PairedDataset
from .model import build,trajectory_loss,REVISION


def main():
    p=argparse.ArgumentParser()
    for key in ['data','prepared','official','run']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--updates',type=int,default=10000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',type=Path)
    p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--dataset',choices=['robotwin','ig10k'],default='robotwin')
    a=p.parse_args()
    assert a.updates>=10 and a.batch_size>0 and 0<=a.smoke_steps<=10
    if a.run.exists() and not a.resume:raise FileExistsError(a.run)
    a.run.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.dataset=='ig10k':
        from osvi_ig.data import PairedDataset as IGDataset
        dataset=IGDataset(a.data,a.prepared)
    else:dataset=PairedDataset(a.data,a.prepared)
    assert dataset.smoke_only==bool(a.smoke_steps)
    print('Caching verified paired clips on CPU',flush=True)
    for i in range(len(dataset)):
        dataset[i]
        if (i+1)%100==0:print('cached',i+1,len(dataset),flush=True)
    dataset.photometric=True
    from torchvision.transforms import ColorJitter
    dataset.jitter=ColorJitter(.4,.4,.4)
    model=build(a.official).cuda().train()
    optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=2e-4,total_steps=a.updates,
        pct_start=0.,final_div_factor=1000,cycle_momentum=False)
    # Freeze schedules by update count; ten short epochs are not a trained baseline.
    config=dict(batch_per_gpu=a.batch_size,world_size=1,updates=a.updates,seed=a.seed,
        precision='bf16',activation_checkpointing=False,gradient_accumulation=False,
        photometric='ColorJitter(0.4,0.4,0.4), independent clips; temporal-consistent',
        geometry='no image flips/crops: native robot/world pose targets',
        pairing='fixed release manifest; no task-label model input',
        selection='final preselected update; no held-out checkpoint selection')
    config.update(getattr(dataset,'adaptation',{}))
    config['supervision']='native recorded action column; no IK/FK relabeling'
    source={str(f.relative_to(a.official)):sha(f) for f in a.official.rglob('*.py')}
    local={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}
    write(a.run/'config.json',dict(config=config,official_revision=REVISION,
        official_files=source,local_files=local,prepared_sha256=sha(a.prepared),smoke_only=dataset.smoke_only))
    step=0;start_epoch=0;start_batch=0
    if a.resume:
        c=torch.load(a.resume,map_location='cpu',weights_only=True)
        assert not c['smoke_only'] and not a.smoke_steps and c['config']==config
        assert c['prepared_sha256']==sha(a.prepared) and c['source_hashes']==source
        model.load_state_dict(c['model']);optim.load_state_dict(c['optimizer']);schedule.load_state_dict(c['scheduler'])
        step=c['step'];start_epoch=c['next_epoch'];start_batch=c['next_batch']
        torch.set_rng_state(c['rng']);torch.cuda.set_rng_state(c['cuda_rng'])
        ns=c['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),*ns[2:]))
        random.setstate(c['python_rng'])
    assert step<a.updates
    torch.cuda.reset_peak_memory_stats()
    def save(epoch,next_batch):
        ns=np.random.get_state()
        c=dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=schedule.state_dict(),
            smoke_only=dataset.smoke_only,official_revision=REVISION,source_hashes=source,step=step,
            config=config,prepared_sha256=sha(a.prepared),next_epoch=epoch,next_batch=next_batch,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),
            numpy_rng=(ns[0],ns[1].tolist(),*ns[2:]))
        temp=a.run/'latest.tmp';torch.save(c,temp);temp.replace(a.run/'latest.pt')
        if step%5000==0 or step==a.updates:
            import os
            dest=a.run/f'step_{step:06d}.pt'
            if not dest.exists():os.link(a.run/'latest.pt',dest)
        write(a.run/'checkpoint_status.json',dict(step=step,next_epoch=epoch,next_batch=next_batch,time=time.time()))
    started=time.monotonic();epoch=start_epoch;last=None
    while step<(a.smoke_steps or a.updates):
        order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
        if a.smoke_steps:order=(order*math.ceil(a.batch_size*a.smoke_steps/len(order)))[:a.batch_size*a.smoke_steps]
        offset=start_batch if epoch==start_epoch else 0
        # Augmentation seeds derive from (epoch, row) in __getitem__, so worker order
        # and the resume cursor cannot change the transformed training samples.
        dataset.epoch_seed=a.seed+epoch
        loader=DataLoader(Subset(dataset,order[offset*a.batch_size:]),batch_size=a.batch_size,
            num_workers=a.workers,shuffle=False,drop_last=False,pin_memory=True,
            generator=torch.Generator().manual_seed(a.seed+epoch),
            **({'prefetch_factor':1} if a.workers else {}))
        ready=time.monotonic()
        for bi,batch in enumerate(loader,start=offset):
            tick=time.monotonic();data_seconds=tick-ready
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            images,context=batch['images'],batch['context'];n=len(images)
            rate=torch.tensor(np.random.uniform(.3,1,n),device='cuda',dtype=torch.float32)[:,None,None,None,None]
            images=images*rate+images.roll(-1,0)[:,:1]*(1-rate)
            context=context*rate+context.roll(-1,0)[:,:1]*(1-rate)
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(images,context)
                latent=torch.nn.functional.l1_loss(out['forward_states'],out['enc_features'][:,11:].detach())
            path=trajectory_loss(out['waypoints'],batch['trajectory'])
            alpha=.95*np.exp(-step/(a.updates/10))+.05;loss=path+alpha*latent
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            lr=optim.param_groups[0]['lr'];optim.step();schedule.step()
            with torch.no_grad():
                m=.998+step/a.updates*.002
                for q,k in zip(model._embed.encoder.parameters(),model._embed.target_encoder.parameters()):
                    k.mul_(m).add_(q,alpha=1-m)
            step+=1;torch.cuda.synchronize()
            row=dict(step=step,epoch=epoch,batch=n,seconds=time.monotonic()-tick,data_seconds=data_seconds,
                loss=loss.item(),trajectory_loss=path.item(),world_model_loss=latent.item(),
                gradient_norm=grad.item(),learning_rate=lr,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                total_updates=a.updates,time=time.time())
            with (a.run/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            write(a.run/'latest_metrics.json',row);print(json.dumps(row),flush=True)
            last=(images[:1].detach(),context[:1].detach())
            if step==20 or step%1000==0 or step>=(a.smoke_steps or a.updates):save(epoch,bi+1)
            ready=time.monotonic()
            if step>=(a.smoke_steps or a.updates):break
        epoch+=1
    saved=torch.load(a.run/'latest.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v.detach().cpu(),saved['model'][k]) for k,v in model.state_dict().items())
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(*last)['waypoints'].float()
    assert torch.isfinite(pred).all() and pred.shape==(1,15,16)
    write(a.run/'complete.json',dict(completed=True,smoke_only=dataset.smoke_only,updates=step,
        checkpoint_readback_exact=True,checkpoint_sha256=sha(a.run/'latest.pt'),
        wall_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

if __name__=='__main__':main()
