from types import SimpleNamespace

import torch

import wan_va.train as train_module
from wan_va.configs.va_robotwin_train_cfg import va_robotwin_train_cfg
from wan_va.train import Trainer


class _Scheduler:
    num_train_timesteps = 1000
    timesteps = torch.arange(1000, dtype=torch.float32)

    def __init__(self, weights=(2.0, 4.0)):
        self.weights = torch.tensor(weights)

    def add_noise(self, latent, noise, timesteps, t_dim):
        return latent + noise * 0.0

    def training_target(self, latent, noise, timesteps):
        return latent + 1.0

    def training_weight(self, timesteps):
        return self.weights.to(timesteps.device)


def _loss_trainer():
    trainer = Trainer.__new__(Trainer)
    trainer.enable_mcp = False
    trainer.patch_size = (1, 1, 1)
    trainer.gradient_accumulation_steps = 1
    trainer.config = SimpleNamespace(
        num_mcp_modules=4,
        video_loss_reweight=True,
        action_loss_reweight=False,
    )
    trainer.train_scheduler_latent = _Scheduler((2.0, 4.0))
    trainer.train_scheduler_action = _Scheduler((11.0, 13.0))
    return trainer


def _loss_input():
    return {
        "latent_dict": {
            "targets": torch.zeros(1, 1, 2, 1, 1),
            "timesteps": torch.tensor([[0.0, 1.0]]),
        },
        "action_dict": {
            "targets": torch.zeros(1, 3, 2, 1, 1),
            "timesteps": torch.tensor([[0.0, 1.0]]),
            "actions_mask": torch.tensor(
                [[[[[1.0]], [[1.0]]],
                  [[[1.0]], [[1.0]]],
                  [[[0.0]], [[0.0]]]]]
            ),
        },
    }


def test_training_loss_uses_global_masked_mean():
    trainer = _loss_trainer()
    latent_pred = torch.ones(1, 2, 1)
    action_pred = torch.ones(1, 2, 3)

    latent_loss, action_loss, _ = trainer.compute_loss(
        _loss_input(), (latent_pred, action_pred)
    )

    torch.testing.assert_close(latent_loss, torch.tensor(3.0))
    torch.testing.assert_close(action_loss, torch.tensor(2.0 / 3.0))

def test_action_noise_input_is_not_zeroed_by_loss_mask(monkeypatch):
    calls = []

    def fixed_timestep_ids(**kwargs):
        calls.append(kwargs)
        return torch.zeros(kwargs["batch_size"], dtype=torch.long)

    monkeypatch.setattr(train_module, "sample_timestep_id", fixed_timestep_ids)
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.patch_size = (1, 1, 1)
    trainer.enable_mcp = False
    trainer.train_scheduler_latent = _Scheduler()
    trainer.train_scheduler_action = _Scheduler()
    trainer.config = SimpleNamespace(
        frame_chunk_size=1,
        max_frame_chunk_size=4,
        noisy_img_prob=1.0,
        noisy_cond_min_timestep_bd=0.0,
        noisy_cond_max_timestep_bd=1.0,
        drop_icl=1.0,
        icl_rope_h=24,
        attn_window=4,
        max_attn_window=64,
    )
    batch = {
        "latents": torch.ones(1, 1, 2, 1, 1),
        "actions": torch.ones(1, 3, 2, 1, 1),
        "actions_mask": torch.zeros(1, 3, 2, 1, 1),
        "text_emb": torch.ones(1, 2, 4),
    }

    prepared = trainer._prepare_input_dict(batch)

    expected_action_grid = train_module.get_mesh_id(
        2, 1, 1, t=1, action=False
    )
    assert torch.equal(prepared["action_dict"]["grid_id"][0], expected_action_grid)
    assert torch.equal(prepared["action_dict"]["latent"], batch["actions"])
    assert torch.equal(
        prepared["action_dict"]["targets"], batch["actions"] + 1.0
    )
    assert calls[1]["min_timestep_bd"] == 0.0
    assert calls[1]["max_timestep_bd"] == 1.0


def test_robotwin_posttraining_defaults_are_stable():
    assert va_robotwin_train_cfg.model_path.endswith('zero-wam-pretrain')
    assert va_robotwin_train_cfg.noisy_img_prob == 0.5
    assert va_robotwin_train_cfg.noisy_cond_min_timestep_bd == 0.0
    assert va_robotwin_train_cfg.video_loss_reweight is True
    assert va_robotwin_train_cfg.action_loss_reweight is False
    assert va_robotwin_train_cfg.learning_rate == 1e-4
    assert va_robotwin_train_cfg.weight_decay == 0.01
    assert va_robotwin_train_cfg.warmup_steps == 200
    assert va_robotwin_train_cfg.max_norm == 1.0
    assert va_robotwin_train_cfg.skip_step_grad_norm_multiplier == 20.0
