"""RoboICL Query cameras: independent causal VAE streams, spatial width packing."""

import torch


class QueryViewsMixin:
    query_views = 1

    def _latest_rgb(self, rgb):
        if self.query_views == 1:
            return super()._latest_rgb(rgb)
        if rgb.ndim == 5 and rgb.shape[1] == 2:
            return rgb[:, None]
        if rgb.ndim == 6 and rgb.shape[2] == 2:
            return rgb[:, -1:]
        raise ValueError(f"Expected separate Query time and camera axes: {rgb.shape}")

    def _encode_stream_observation(self, video, *, initial=False):
        if self.query_views == 1:
            return super()._encode_stream_observation(video, initial=initial)
        if video.ndim != 6 or video.shape[2] != 2:
            raise ValueError(f"Expected Query [B,T,2,H,W,3], got {video.shape}")
        from examples.baselines.zero_wam.zero_wam.wan_va.modules.utils import WanVAEStreamingWrapper

        if initial:
            self.query_encoders = [WanVAEStreamingWrapper(self.vae) for _ in range(2)]
        if not hasattr(self, "query_encoders"):
            raise RuntimeError("Query stream must start with an initial observation")
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device)[None, :, None, None, None]
        std = torch.tensor(self.vae.config.latents_std, device=self.device)[None, :, None, None, None]
        views = []
        for view, encoder in enumerate(self.query_encoders):
            pixels = self._pixels(video[:, :, view])
            chunks = [pixels] if initial else pixels.split(4, dim=2)
            moments = torch.cat([encoder.encode_chunk(chunk) for chunk in chunks], dim=2)
            views.append(((moments.chunk(2, dim=1)[0].float() - mean) / std).to(self.dtype))
        return torch.cat(views, dim=-1)
