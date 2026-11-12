"""Reuse the pinned IG DETRVAE and frozen DINO adapter; no video generator."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch import nn

DEFAULTS=dict(hidden_dim=1024,dim_feedforward=4096,nheads=16,enc_layers=12,
              dec_layers=12,dropout=.1,pre_norm=False,backbone='resnet18',
              position_embedding='sine',dilation=False,masks=False,
              include_depth=False,lr_backbone=1e-5,horizon=24,kl_weight=10.)

def upstream(path):
    path=Path(path).resolve()
    if not (path/'examples/baselines/act/act/detr_video/detr_vae.py').is_file():
        raise ValueError('Expected the frozen imitator_game repository')
    sys.path.insert(0,str(path))

def demo_encoder(ig_root):
    upstream(ig_root)
    from examples.baselines.encoders.task_encoder.video_backbone import FrozenVideoBackbone
    return FrozenVideoBackbone(backbone_type='dinov2_vitl14',latent_dim=256,
                               max_seq_patches=32,adapter_layers=1,num_sampled_frames=10)

class ACT(nn.Module):
    def __init__(self,ig_root,config=None):
        super().__init__(); upstream(ig_root)
        from examples.baselines.act.act.detr_video.backbone import build_backbone
        from examples.baselines.act.act.detr_video.transformer import build_transformer
        from examples.baselines.act.act.detr_video.detr_vae import DETRVAE,build_encoder
        self.config=dict(DEFAULTS,**(config or {}));args=SimpleNamespace(**self.config)
        # IG ACT consumes only raw_cls. Copy its exact trainable CLS adapter.
        self.demo_adapter=demo_encoder(ig_root).cls_adapter
        self.task_norm=nn.LayerNorm(256)
        self.core=DETRVAE([build_backbone(args)],build_transformer(args),build_encoder(args),
                         state_dim=16,action_dim=16,num_queries=args.horizon,video_feature_dim=256)

    def forward(self,batch,training=False):
        z=self.task_norm(self.demo_adapter(batch['demo_cls']))
        return self.core({'rgb':batch['rgb'],'state':batch['state']},
                         batch['actions'] if training else None,video_feature=z)

    def loss(self,batch):
        pred,(mu,logvar)=self(batch,training=True)
        l1=torch.nn.functional.l1_loss(pred.float(),batch['actions'].float())
        kl=(-.5*(1+logvar.float()-mu.float().square()-logvar.float().exp())).sum(1).mean()
        return l1+self.config['kl_weight']*kl,{'l1':l1.detach(),'kl':kl.detach()}
