"""Finite real-pair training/readback/inference check; never a benchmark model."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from act_robotwin.protocol import write,sha
from act_robotwin.data import rgb_at
from .data import PairedDataset
from .model import build,trajectory_loss,REVISION,encode_pose,decode_pose
from .policy import Policy

def main():
    p=argparse.ArgumentParser()
    for key in ['data','prepared','official','run']:p.add_argument('--'+key,required=True)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--steps',type=int,default=3)
    a=p.parse_args();root=Path(a.run);root.mkdir(parents=True,exist_ok=True)
    assert 2<=a.steps<=10
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    print('Loading paired RGB and measured trajectories',flush=True)
    dataset=PairedDataset(a.data,a.prepared);assert dataset.smoke_only
    samples=[dataset[i] for i in range(len(dataset))]
    for i,sample in enumerate(samples):
        pose=sample['initial_pose'].numpy()
        reconstructed=decode_pose(encode_pose(pose[None]))[0]
        assert np.allclose(reconstructed[:7],pose[:7],atol=1e-5) or np.allclose(reconstructed[3:7],-pose[3:7],atol=1e-5)
    print('Building official model and dual-arm head',flush=True)
    model=build(a.official).cuda().train()
    print('Model ready',sum(p.numel() for p in model.parameters()),flush=True)
    optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    # Keep the official ten-epoch pair-based schedule, stop this smoke after three updates.
    total_updates=10*int(np.ceil(2149/a.batch_size))
    schedule=torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=2e-4,total_steps=total_updates,
        pct_start=0.,final_div_factor=1000,cycle_momentum=False)
    tracked={name:dict(model.named_parameters())[name].detach().clone() for name in
        ['waypoint_head.2.weight','forward_model.transformer.wte.weight']
        if name in dict(model.named_parameters())}
    # Always verify one world-model parameter even if upstream renames its projection.
    wm_name=next(n for n,p in model.named_parameters() if n.startswith('forward_model.') and p.requires_grad)
    tracked[wm_name]=dict(model.named_parameters())[wm_name].detach().clone()
    metrics=[];started=time.perf_counter()
    for step in range(a.steps):
        torch.cuda.synchronize();tick=time.perf_counter()
        batch={k:v.cuda() for k,v in default_collate([samples[i%len(samples)] for i in range(a.batch_size)]).items()}
        images,context=batch['images'],batch['context']
        rate=torch.tensor(np.random.uniform(.3,1,a.batch_size),device='cuda',dtype=torch.float32)[:,None,None,None,None]
        images=images*rate+images.roll(-1,0)[:,:1]*(1-rate)
        context=context*rate+context.roll(-1,0)[:,:1]*(1-rate)
        optim.zero_grad(set_to_none=True)
        print('Forward',step+1,flush=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(images,context)
            latent_loss=torch.nn.functional.l1_loss(out['forward_states'],out['enc_features'][:,11:].detach())
        print('SoftDTW',step+1,flush=True)
        path_loss=trajectory_loss(out['waypoints'],batch['trajectory'])
        alpha=.95*np.exp(-step/(total_updates/10))+.05
        loss=path_loss+alpha*latent_loss
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'))
        assert torch.isfinite(grad)
        optim.step();schedule.step()
        with torch.no_grad():
            m=.998+step/total_updates*.002
            for q,k in zip(model._embed.encoder.parameters(),model._embed.target_encoder.parameters()):
                k.mul_(m).add_(q,alpha=1-m)
        torch.cuda.synchronize()
        row=dict(step=step+1,seconds=time.perf_counter()-tick,loss=loss.item(),
            trajectory_loss=path_loss.item(),world_model_loss=latent_loss.item(),gradient_norm=grad.item(),
            learning_rate=schedule.get_last_lr()[0],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        metrics.append(row);write(root/'metrics.json',metrics);print(json.dumps(row),flush=True)
    changed={name:float((dict(model.named_parameters())[name]-old).abs().max()) for name,old in tracked.items()}
    assert all(x>0 for x in changed.values()),changed
    checkpoint=root/'smoke.pt'
    source={str(p.relative_to(a.official)):sha(p) for p in Path(a.official).rglob('*.py')}
    torch.save(dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=schedule.state_dict(),
        smoke_only=True,official_revision=REVISION,source_hashes=source,step=a.steps,
        config=dict(batch_per_gpu=a.batch_size,world_size=1,activation_checkpointing=False,
            output='15x16 dual xyz/wxyz/closed_gripper; native robot coordinates',
            image_encoder='official ImageNet ResNet18 initialization',pair_count=2149),
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),prepared=dataset.prepared),checkpoint)
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert all(torch.equal(v.detach().cpu(),loaded['model'][k]) for k,v in model.state_dict().items())
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        inputs=samples[0]['images'][:1].repeat(6,1,1,1)[None].cuda()
        contexts=samples[0]['context'][None].cuda()
        reference=model(inputs,contexts)['waypoints'].float().cpu()
    del loaded,optim,model,tracked,batch,out,loss,path_loss,latent_loss
    torch.cuda.empty_cache()
    policy=Policy(checkpoint,a.official,allow_smoke=True)
    policy.reset(Path(a.data)/dataset.rows[0]['human_video'])
    frame=rgb_at(Path(a.data)/dataset.rows[0]['robot_video'],[0])[0]
    tick=time.perf_counter();pred=policy.predict(frame);torch.cuda.synchronize();infer_seconds=time.perf_counter()-tick
    assert np.allclose(pred,decode_pose(reference[0,-5:].numpy()),atol=1e-5)
    assert pred.shape==(5,16) and np.isfinite(pred).all()
    np.save(root/'recorded_prediction.npy',pred)
    write(root/'complete.json',dict(passed=True,smoke_only=True,updates=a.steps,batch_per_gpu=a.batch_size,gpus=1,
        unique_real_pairs=len(samples),metrics=metrics,parameter_changes=changed,checkpoint_sha256=sha(checkpoint),
        exact_checkpoint_readback=True,recorded_inference_shape=list(pred.shape),inference_seconds=infer_seconds,
        peak_allocated_gib=max(x['peak_allocated_gib'] for x in metrics),total_seconds=time.perf_counter()-started,
        no_physical_success_claim=True,formal_training=False))

if __name__=='__main__':main()
