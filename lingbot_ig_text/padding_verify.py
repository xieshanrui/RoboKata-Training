"""One bounded real-batch padding perturbation during the runtime probe."""
import copy
import torch


@torch.no_grad()
def verify_padding(trainer, dataset, indices, collate):
    batch = trainer.convert_input_format(collate([dataset[i] for i in indices]))
    inputs = trainer._prepare_input_dict(batch)
    changed = copy.deepcopy(inputs)
    lengths = inputs['video_lengths']
    frames = batch['latents'].shape[2]
    valid = torch.arange(frames, device=lengths.device)[None] < lengths[:, None]
    assert (~valid).any(), 'Verification requires mixed lengths'
    for name in ['latent_dict', 'action_dict']:
        for key in ['noisy_latents', 'latent', 'targets']:
            changed[name][key].masked_fill_(~valid[:, None, :, None, None], 13.0)
    first = trainer.transformer(inputs, train_mode=True)
    second = trainer.transformer(changed, train_mode=True)
    differences = []
    for prediction, perturbed in zip(first, second):
        b, total, channels = prediction.shape
        tokens_per_frame = total // frames
        keep = valid[:, :, None, None].expand(b, frames, tokens_per_frame, channels)
        a = prediction.reshape(b, frames, tokens_per_frame, channels)[keep]
        c = perturbed.reshape(b, frames, tokens_per_frame, channels)[keep]
        assert torch.isfinite(a).all() and torch.isfinite(c).all()
        differences.append((a-c).abs().max().item())
    loss_a = trainer.compute_loss(inputs, first)
    loss_b = trainer.compute_loss(changed, second)
    loss_diff = max(abs(float(a)-float(b)) for a,b in zip(loss_a, loss_b))
    assert max(differences) == 0 and loss_diff == 0, (differences, loss_diff)
    return dict(real_region_prediction_max_diff=differences, loss_max_diff=loss_diff,
                lengths=lengths.cpu().tolist(), padded_frames=int((~valid).sum()), passed=True)
