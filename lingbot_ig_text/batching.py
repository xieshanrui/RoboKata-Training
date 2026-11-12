"""Mixed complete episodes: mask padding and preserve equal episode losses."""
import torch
from torch.nn import functional as F


def collate(samples):
    lengths = torch.tensor([s['latents'].shape[1] for s in samples], dtype=torch.long)
    maximum = int(lengths.max())
    batch = {'ig_lengths': lengths}
    for key in ['latents', 'actions', 'actions_mask']:
        batch[key] = torch.stack([
            F.pad(s[key], (0, 0, 0, 0, 0, maximum - s[key].shape[1]))
            for s in samples
        ])
    batch['text_emb'] = torch.stack([s['text_emb'] for s in samples])
    return batch


def install_trainer(native):
    """Keep the native step/noise/optimizer; adapt only padding and reduction."""
    original_prepare = native.Trainer._prepare_input_dict

    def prepare(self, batch):
        result = original_prepare(self, batch)
        result['video_lengths'] = batch['ig_lengths']
        return result

    def compute_loss(self, inputs, prediction):
        latent, action = prediction
        target = inputs['latent_dict']['targets']
        b, _, frames, h, w = target.shape
        latent = native.data_seq_to_patch(self.patch_size, latent, frames, h, w, batch_size=b)
        action = native.rearrange(action, 'b (f n) c -> b c f n 1', f=frames)
        lengths = inputs['video_lengths']
        valid = torch.arange(frames, device=lengths.device)[None] < lengths[:, None]
        video_weights = self.train_scheduler_latent.training_weight(
            inputs['latent_dict']['timesteps'].flatten()).reshape(b, frames)
        action_weights = self.train_scheduler_action.training_weight(
            inputs['action_dict']['timesteps'].flatten()).reshape(b, frames)
        video_frame = (latent.float() - target.float().detach()).square().mean(dim=(1, 3, 4))
        error = (action.float() - inputs['action_dict']['targets'].float().detach()).square()
        mask = inputs['action_dict']['actions_mask'].float()
        action_frame = (error * mask).sum(dim=(1, 3, 4)) / (mask.sum(dim=(1, 3, 4)) + 1e-6)
        video_loss = ((video_frame * video_weights).masked_fill(~valid, 0).sum(1) / lengths).mean()
        action_loss = ((action_frame * action_weights).masked_fill(~valid, 0).sum(1) / lengths).mean()
        return video_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    native.Trainer._prepare_input_dict = prepare
    native.Trainer.compute_loss = compute_loss
