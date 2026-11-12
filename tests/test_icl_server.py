from types import SimpleNamespace

import torch

from wan_va.configs.va_robotwin_cfg import va_robotwin_cfg
from wan_va.modules.icl_model import PREDICTION_CACHE_TYPE
from wan_va.wan_va_server import VA_Server


class _Scheduler:
    def __init__(self):
        self.timesteps = torch.tensor([100.0])
        self.step_calls = 0

    def set_timesteps(self, _steps):
        self.timesteps = torch.tensor([100.0])

    def step(self, model_output, timestep, sample, return_dict=False):
        del model_output, timestep, return_dict
        self.step_calls += 1
        return sample + 1


class _Transformer:
    patch_size = (1, 1, 1)

    def __init__(self):
        self.calls = []

    def __call__(self, input_dict, update_cache, cache_name, mode, **kwargs):
        del cache_name, kwargs
        self.calls.append((mode, update_cache, input_dict))
        if mode == "forward_latent_only":
            data = input_dict["latent_res_lst"]["noisy_latents"]
            return torch.zeros(
                1, data.shape[2] * data.shape[3] * data.shape[4], data.shape[1]
            )
        data = input_dict["action_res_lst"]["noisy_latents"]
        return torch.zeros(1, data.shape[2] * data.shape[3], data.shape[1])

    def cache_counts(self):
        return {}


def _mock_server():
    server = VA_Server.__new__(VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.cache_name = "pos"
    server.chunk_idx = 0
    server.init_latent = None
    server.latent_height = 1
    server.latent_width = 1
    server.action_per_frame = 2
    server.action_mask = torch.tensor([True, False, True, False])
    server.use_icl_cfg = False
    server.target_text_cfg_active = False
    server.target_prompt_embeds = torch.zeros(1, 2, 8)
    server.transformer = _Transformer()
    server.scheduler = _Scheduler()
    server.action_scheduler = _Scheduler()
    server.job_config = SimpleNamespace(
        frame_chunk_size=2,
        action_dim=4,
        num_inference_steps=1,
        action_num_inference_steps=1,
        attn_window=4,
    )
    server._encode_initial_obs = lambda _obs: torch.zeros(1, 48, 1, 1, 1)
    server._encode_obs = lambda _obs: torch.zeros(1, 48, 1, 1, 1)
    server.postprocess_action = lambda action: action
    return server


def test_icl_action_denoising_matches_robotwin_sync_rollout():
    server = _mock_server()

    result = server._infer_icl({"obs": []})

    action_calls = [
        call for call in server.transformer.calls if call[0] == "forward_action_only"
    ]
    assert len(action_calls) == 1
    assert [call[1] for call in action_calls] == [0]
    assert server.action_scheduler.step_calls == 1

    for _, _, input_dict in action_calls:
        stream = input_dict["action_res_lst"]
        assert torch.count_nonzero(stream["noisy_latents"][:, ~server.action_mask]) == 0
        assert torch.all(stream["timesteps"] == 100)
        assert torch.all(stream["cache_type_ids"] == PREDICTION_CACHE_TYPE)

    assert torch.count_nonzero(result[:, ~server.action_mask]) == 0
    assert torch.count_nonzero(result[:, :, :1]) > 0
    assert torch.count_nonzero(server.last_predicted_actions[:, :, :1]) == 0
    assert server.last_predicted_actions.data_ptr() != result.data_ptr()


def test_robotwin_action_quantiles_match_released_eval_dataset():
    expected_q01 = [
        -0.054687286317348476, -9.834766387939453e-07,
        -0.0746749371290207, -1, -1, -1, -1, -0.3589721029996872,
        -1.1026859283447266e-06, -0.11628214418888091, -1, -1, -1, -1,
    ]
    expected_q99 = [
        0.35284559458494186, 0.3960446125268936, 0.1493294847011566,
        1, 1, 1, 1, 0.02997654676437378, 0.38793458312749857,
        0.18672722578048706, 1, 1, 1, 1,
    ]

    assert va_robotwin_cfg.norm_stat["q01"][:14] == expected_q01
    assert va_robotwin_cfg.norm_stat["q99"][:14] == expected_q99


def test_icl_observation_cache_reuses_exact_predicted_action():
    server = _mock_server()
    server.init_latent = torch.zeros(1, 48, 1, 1, 1)
    expected = torch.randn(1, 4, 2, 2, 1)
    server.last_predicted_actions = expected
    server.transformer.clear_prediction_cache = lambda *_args: None
    server.preprocess_action = lambda _action: (_ for _ in ()).throw(
        AssertionError("client action must not be re-normalized")
    )

    server._compute_icl_kv_cache({"obs": [], "state": "not-used"})

    action_calls = [
        call for call in server.transformer.calls if call[0] == "forward_action_only"
    ]
    cached = action_calls[-1][2]["action_res_lst"]["noisy_latents"]
    assert cached.data_ptr() == expected.data_ptr()


def test_icl_cfg_does_not_duplicate_observed_action_branch():
    server = _mock_server()
    server.use_icl_cfg = True
    server.init_latent = torch.zeros(1, 48, 1, 1, 1)
    server.last_predicted_actions = torch.randn(1, 4, 2, 2, 1)
    server.transformer.clear_prediction_cache = lambda *_args: None

    server._compute_icl_kv_cache({"obs": []})

    video_call, action_call = server.transformer.calls[-2:]
    assert video_call[2]["latent_res_lst"]["noisy_latents"].shape[2] == 4
    assert action_call[2]["action_res_lst"]["noisy_latents"].shape[2] == 2
    assert torch.all(action_call[2]["current_seq_ids"] == 0)
