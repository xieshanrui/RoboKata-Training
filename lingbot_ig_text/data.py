"""Text-only IG C15 adapter for the unmodified LingBot-VA trainer.

Reuse the verified causal Query VAE cache and IG action coordinate contract.
No human video, Demo state, action, or target-task robot data is returned.
"""
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class IGTextDataset(Dataset):
    def __init__(self, manifest, statistics, text_cache, action_space_source, cfg_prob=0.1):
        self.manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.records = self.manifest["sim_episodes"]
        self.text = torch.load(text_cache, map_location="cpu", weights_only=False)
        spec = importlib.util.spec_from_file_location("ig_episode_action_space", action_space_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.space = module.EpisodeActionSpace(json.loads(Path(statistics).read_text(encoding="utf-8")))
        self.channels = list(module.channel_ids(14))
        self.cfg_prob = cfg_prob
        assert len(self.records) == 3000 and self.space.statistics["episodes"] == 3000
        assert self.space.statistics["records_sha256"] == self.manifest["records_sha256"]
        assert all(r["human_repo_id"] in self.text["embeddings"] for r in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False, mmap=True)
        latent = payload["latent"].clone()
        frames = latent.shape[1]
        interval = 16
        assert payload["action_chunk_size"] == interval
        assert torch.equal(torch.as_tensor(payload["latent_nominal_anchor_indices"]), torch.arange(frames) * interval)
        raw = torch.as_tensor(payload["actions"]).float()
        states = torch.as_tensor(payload["states"]).float()
        assert raw.shape[1] == 16 and states.shape[1] == 18
        normalized, _ = self.space.normalize_episode(raw, states)
        indices = (torch.arange(frames)[:, None] - 1) * interval + torch.arange(interval)[None]
        valid = (indices >= 0) & (indices < len(raw))
        values = normalized[indices.clamp(0, len(raw) - 1)].clone()
        values[0] = 0
        valid[0] = True  # Match IG's supervised normalized-zero boundary group.
        actions = torch.zeros(30, frames, interval, 1)
        mask = torch.zeros_like(actions, dtype=torch.bool)
        actions[self.channels, :, :, 0] = values.permute(2, 0, 1)
        mask[self.channels, :, :, 0] = valid[None]
        text = self.text["embeddings"][record["human_repo_id"]]
        if torch.rand(()).item() < self.cfg_prob:
            text = self.text["empty"]
        assert torch.isfinite(latent).all() and torch.isfinite(actions).all()
        return dict(latents=latent, actions=actions, actions_mask=mask, text_emb=text)
