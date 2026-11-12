"""Segmented RGB Demo prefix with the current world-TCP10 action head."""

import dataclasses
import math

import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch.frame_demo_encoder import DemoAttentionPool
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch

class RoboICLPytorch(PI0Pytorch):
    IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")

    def __init__(self, config):
        # Native projections and checkpoint parameter names remain unchanged.
        super().__init__(dataclasses.replace(config, action_dim=32))
        self.config = config
        self.IMAGE_KEYS = tuple(config.query_image_keys)
        if config.checkpoint_layers is not None and max(config.checkpoint_layers) >= self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers:
            raise ValueError("Checkpoint index exceeds the actual decoder depth")
        self.paligemma_with_expert.checkpoint_layers = config.checkpoint_layers
        if config.demo_compression_layer is not None:
            # Preserve the reviewed stage-specific additive attention masks.
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        vision_width = self.paligemma_with_expert.paligemma.config.vision_config.hidden_size
        prefix_width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.demo_pool = DemoAttentionPool(
            vision_width, config.demo_frames, config.demo_views, config.demo_tokens_per_view, config.demo_pool_heads,
            valid_only=config.pool_valid_only,
        )
        self.demo_time_embedding = nn.Parameter(torch.randn(config.demo_frames, prefix_width) * 0.01)
        self.demo_view_embedding = nn.Parameter(torch.randn(config.demo_views, prefix_width) * 0.01)
        self.source_embedding = nn.Embedding(max(2, len(config.action_contracts)), prefix_width)
        nn.init.normal_(self.source_embedding.weight, std=0.01)
        self.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(requires_grad=False)
        self.register_buffer("phase_code", torch.tensor(0, dtype=torch.long))
        self.phase = "A"
        self.register_load_state_dict_post_hook(self._restore_phase)
        self.set_phase("A")

    def gradient_checkpointing_enable(self):
        if self.config.checkpoint_strategy == "none":
            super().gradient_checkpointing_disable()
        else:
            super().gradient_checkpointing_enable()
            if self.config.checkpoint_strategy in ("layer", "selective"):
                # Keep the joint decoder's per-layer checkpoints, without an
                # enclosing checkpoint that recomputes the entire decoder.
                self.gradient_checkpointing_enabled = False

    def _restore_phase(self, module, incompatible_keys):
        self.set_phase("ABC"[int(self.phase_code)])

    def load_base_checkpoint(self, path):
        """Initialize new modules once; resume must instead use a strict load."""
        missing, unexpected = safetensors.torch.load_model(self, str(path), strict=False)
        added = (
            "demo_pool.",
            "demo_time_embedding",
            "demo_view_embedding",
            "source_embedding.",
            "phase_code",
        )
        if unexpected or any(not key.startswith(added) for key in missing):
            raise ValueError(f"Incompatible pi0 checkpoint: missing={missing}, unexpected={unexpected}")
        if not missing:
            raise ValueError("Expected a native pi0 initialization checkpoint, not a RoboICL resume")

    def set_phase(self, phase):
        if phase != "A":
            raise ValueError("The current segmented model keeps SigLIP frozen")
        self.phase = phase
        self.phase_code.zero_()
        vision = self.paligemma_with_expert.paligemma.model.vision_tower
        vision.requires_grad_(False)
        vision.eval()

    def train(self, mode=True):  # noqa: FBT002 - PyTorch Module interface
        super().train(mode)
        self.paligemma_with_expert.paligemma.model.vision_tower.eval()
        return self

    def _prefix_blocks(self, mask, demo_length):
        blocks = torch.zeros_like(mask)
        if self.config.demo_memory_mode == "icrt_causal":
            # Original RGB-only adaptation of ICRT's causal trajectory context.
            # Keep the baseline's eight spatial/view tokens within each time block.
            frame_tokens = self.config.demo_views * self.config.demo_tokens_per_view
            blocks[:, :demo_length:frame_tokens] = True
            if mask.shape[1] > demo_length:
                blocks[:, demo_length] = True  # Source and current Query share a later block.
        return blocks

    def _source(self, observation):
        source = observation.source_id
        if source is None or source.dtype != torch.long or source.shape != observation.state.shape[:1]:
            raise ValueError("source_id must be int64 [batch]; indices follow the configured action encoding and contracts")
        if torch.any(source != 0):
            raise ValueError("Unknown action contract")
        return source

    def _build_prefix_compression_plan(self, observation, prefix_valid, prefix_blocks):
        if self.config.demo_compression_layer is None:
            return None
        from openpi.models_pytorch.demo_compression import build_compression_plan
        frames = observation.demo_valid.shape[1]
        windows = math.ceil(frames / self.config.demo_sample_hz / self.config.demo_window_seconds)
        return build_compression_plan(
            observation.demo_timestamps, prefix_valid, prefix_blocks,
            frames=frames, views=self.config.demo_views,
            tokens_per_view=self.config.demo_tokens_per_view, window_count=windows,
            window_seconds=self.config.demo_window_seconds, compression_layer=self.config.demo_compression_layer)

    def _project_state(self, state, source_id=None):
        return self.state_proj(state.to(self.state_proj.weight.dtype))

    def _project_action_input(self, actions, source_id=None):
        return self.action_in_proj(actions.to(self.action_in_proj.weight.dtype))

    def _project_actions(self, hidden, observation=None):
        self._source(observation)
        return self.action_out_proj(hidden.to(self.action_out_proj.weight.dtype))

    def _embed_observation_suffix(self, observation, noisy_actions, time):
        return self.embed_suffix(observation.state, noisy_actions, time, self._source(observation))

    def _demo_tokens(self, observation, *, train):
        valid = observation.demo_valid
        if valid is None or valid.ndim != 3 or valid.shape[1] < 1:
            raise ValueError("Expected a nonempty Demo frame/view mask")
        expected = (observation.state.shape[0], valid.shape[1], self.config.demo_views)
        if valid is None or valid.shape != expected or valid.dtype != torch.bool:
            raise ValueError("Expected boolean Demo frame/view mask")
        if (observation.demo_rgb is None) == (observation.demo_patches is None):
            raise ValueError("Provide exactly one of Demo RGB or raw SigLIP patches")
        if observation.demo_patches is not None:
            if self.phase != "A":
                raise ValueError("Feature caches are forbidden after vision unfreezing")
            patches = observation.demo_patches
        else:
            rgb = observation.demo_rgb
            if rgb.shape != (*expected, 3, 224, 224) or not rgb.is_floating_point():
                raise ValueError("Demo RGB must be normalized float [B,T,V,3,224,224]")
            rgb = torch.where(valid[..., None, None, None], rgb, 0)
            pieces = [self._apply_checkpoint(self.paligemma_with_expert.embed_image_patches, chunk)
                      for chunk in rgb.flatten(0, 2).split(self.config.demo_encode_batch)]
            patches = torch.cat(pieces).reshape(*expected, -1, self.demo_pool.width)
        # A deliberately uses the same BF16 cache boundary in online and cached
        # execution, including when trainable weights use FP32 masters with AMP.
        if self.phase == "A":
            patches = patches.to(torch.bfloat16)
        tokens, mask = self.demo_pool(patches, valid)
        tokens = self.paligemma_with_expert.project_image_features(tokens)
        # Retain all trained positions; later frames reuse the terminal learned
        # embedding while causal positions and timestamp windows keep advancing.
        time_ids = torch.arange(valid.shape[1], device=self.demo_time_embedding.device)
        time_ids = time_ids.clamp_max(self.demo_time_embedding.shape[0] - 1)
        positions = self.demo_time_embedding[time_ids, None] + self.demo_view_embedding[None]
        positions = positions[:, :, None].expand(-1, -1, self.config.demo_tokens_per_view, -1).flatten(0, 2)
        tokens = tokens + positions[None].to(tokens.dtype)
        return torch.where(mask[..., None], tokens, 0), mask

    def _query_patch_tokens(self, observation, patches):
        if self.phase != "A":
            raise ValueError("Query feature caches are forbidden after vision unfreezing")
        vision = self.paligemma_with_expert.paligemma.config.vision_config
        patch_count = (vision.image_size // vision.patch_size) ** 2
        expected = (observation.state.shape[0], len(self.IMAGE_KEYS), patch_count, vision.hidden_size)
        if patches.shape != expected or patches.dtype != torch.bfloat16:
            raise ValueError("Query patches must be BF16 [batch, camera, patch, vision_width]")
        masks = [observation.image_masks.get(key, torch.ones(
            expected[0], dtype=torch.bool, device=patches.device)) for key in self.IMAGE_KEYS]
        if any(mask.shape != expected[:1] or mask.dtype != torch.bool for mask in masks):
            raise ValueError("Expected boolean per-camera Query masks")
        # Cache only raw SigLIP patches. The projector remains trainable, and
        # language scaling is identical to PI0Pytorch.embed_prefix.
        images = []
        image_masks = []
        for view, mask in enumerate(masks):
            features = torch.where(mask[:, None, None], patches[:, view], 0)
            images.append(self._apply_checkpoint(self.paligemma_with_expert.project_image_features, features))
            image_masks.append(mask[:, None].expand(-1, patch_count))
        language = self._apply_checkpoint(
            self.paligemma_with_expert.embed_language_tokens, observation.tokenized_prompt)
        language = language * math.sqrt(language.shape[-1])
        return (torch.cat((*images, language), dim=1),
                torch.cat((*image_masks, observation.tokenized_prompt_mask), dim=1))

    def _query_tokens(self, observation, *, train):
        patches = getattr(observation, "query_patches", None)
        if patches is not None:
            return self._query_patch_tokens(observation, patches)

        train = train and self.training
        processed = preprocess_observation_pytorch(
            observation, train=train and (self.phase != "A" or self.config.frozen_query_augmentation),
            image_keys=self.IMAGE_KEYS
        )
        if self.phase == "A" and self.config.query_encode_batch is not None:
            rgb = torch.stack([processed.images[key] for key in self.IMAGE_KEYS], dim=1)
            pieces = [self._apply_checkpoint(self.paligemma_with_expert.embed_image_patches, chunk)
                      for chunk in rgb.flatten(0, 1).split(self.config.query_encode_batch)]
            patches = torch.cat(pieces).reshape(*rgb.shape[:2], -1, self.demo_pool.width).to(torch.bfloat16)
            return self._query_patch_tokens(processed, patches)
        query, query_mask, _ = self.embed_prefix(
            [processed.images[key] for key in self.IMAGE_KEYS],
            [processed.image_masks[key] for key in self.IMAGE_KEYS],
            processed.tokenized_prompt,
            processed.tokenized_prompt_mask,
        )
        return query, query_mask

    def _embed_observation_prefix(self, observation, *, train):
        train = train and self.training
        query, query_mask = self._query_tokens(observation, train=train)
        demo, demo_mask = self._demo_tokens(observation, train=train)
        source = self.source_embedding(self._source(observation))[:, None].to(query.dtype)
        source_mask = torch.ones_like(query_mask[:, :1])
        mask = torch.cat((demo_mask, source_mask, query_mask), dim=1)
        return torch.cat((demo.to(query.dtype), source, query), dim=1), mask, self._prefix_blocks(mask, demo.shape[1])

    def forward(self, observation, actions, noise=None, time=None, *, action_valid=None):
        if action_valid is None or action_valid.shape != actions.shape or action_valid.dtype != torch.bool:
            raise ValueError("Explicit boolean action supervision mask required")
        counts = action_valid.sum(dim=(1, 2))
        if torch.any(counts == 0):
            raise ValueError("Every sample requires a valid action label")
        # Invalid labels do not enter x_t; validity is never an attention/input feature.
        actions = torch.where(action_valid, actions, 0)
        losses = super().forward(observation, actions, noise=noise, time=time).float()
        return torch.where(action_valid, losses, 0).sum(dim=(1, 2)) / counts
