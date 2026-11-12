"""Opt-in PyTorch pi0 with RGB-only Demo conditioning."""

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_frame_demo_pytorch import RoboICLPytorch


@dataclasses.dataclass(frozen=True)
class RoboICLConfig(Pi0Config):
    action_dim: int = 32
    action_encoding: Literal["world_tcp10", "robotwin_eef16", "ig_joint16"] = "world_tcp10"
    demo_frames: int = 60
    demo_views: int = 2
    query_image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    demo_tokens_per_view: int = 4
    demo_pool_heads: int = 8
    demo_encode_batch: int = 4
    query_encode_batch: int | None = None
    checkpoint_strategy: Literal["nested", "layer", "selective", "none"] = "nested"
    checkpoint_layers: tuple[int, ...] | None = None
    pool_valid_only: bool = False
    demo_memory_mode: Literal["icrt_causal"] = "icrt_causal"
    demo_conditioning: Literal["rgb"] = "rgb"
    demo_compression_layer: int = 2
    demo_sample_hz: float = 5.0
    demo_window_seconds: float = 1.0
    known_zero_padding_bridge: bool = True
    frozen_query_augmentation: bool = True
    # Retained in checkpoint metadata; only the current self world-TCP contract is supported.
    action_contracts: tuple[str, ...] = ("self",)
    pytorch_compile_mode: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.checkpoint_strategy not in ("nested", "layer", "selective", "none"):
            raise ValueError("Unknown segmented execution strategy")
        if self.checkpoint_strategy == "selective":
            if not self.checkpoint_layers or len(set(self.checkpoint_layers)) != len(self.checkpoint_layers) or min(self.checkpoint_layers) < 0:
                raise ValueError("Selective checkpointing requires unique nonnegative layer indices")
        elif self.checkpoint_layers is not None:
            raise ValueError("Layer indices require the selective checkpoint strategy")
        if self.pi05 or self.demo_views not in (1, 2) or self.demo_encode_batch < 1:
            raise ValueError("Segmented Demo requires pi0 and one or two Demo cameras")
        cameras = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        if not self.query_image_keys or len(set(self.query_image_keys)) != len(self.query_image_keys) or any(
                key not in cameras for key in self.query_image_keys):
            raise ValueError("Query cameras must be a nonempty ordered subset of native pi0 cameras")
        if (self.action_encoding not in ("world_tcp10", "robotwin_eef16", "ig_joint16") or self.action_dim != 32
                or self.action_horizon != 50 or tuple(self.action_contracts) != ("self",)):
            raise ValueError("Use an explicit native action contract padded to32, with horizon50")
        if (self.demo_memory_mode != "icrt_causal" or self.demo_conditioning != "rgb"
                or self.demo_compression_layer != 2 or self.demo_window_seconds != 1.0
                or self.demo_sample_hz != 5.0 or not self.known_zero_padding_bridge
                or not self.frozen_query_augmentation):
            raise ValueError("Use the current 5Hz RGB Demo / layer-2 one-second compression configuration")
        if self.pytorch_compile_mode is not None:
            raise ValueError("Segmented Demo uses eager execution")
        if self.demo_frames < 1 or (self.query_encode_batch is not None and self.query_encode_batch < 1):
            raise ValueError("Frame count and vision batch must be positive")

    def create(self, rng):
        raise NotImplementedError("RoboICL is a PyTorch model; use create_pytorch")

    @property
    def native_action_dim(self):
        return 10 if self.action_encoding == "world_tcp10" else 16

    def create_pytorch(self):
        return RoboICLPytorch(self)

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        keys = self.query_image_keys
        return dataclasses.replace(observation,
            state=jax.ShapeDtypeStruct((batch_size, 32), jnp.float32),
            images={key: observation.images[key] for key in keys},
            image_masks={key: observation.image_masks[key] for key in keys},
            demo_rgb=jax.ShapeDtypeStruct((batch_size, self.demo_frames, self.demo_views, 3, 224, 224), jnp.float32),
            demo_valid=jax.ShapeDtypeStruct((batch_size, self.demo_frames, self.demo_views), jnp.bool_),
            demo_timestamps=jax.ShapeDtypeStruct((batch_size, self.demo_frames), jnp.float32),
            source_id=jax.ShapeDtypeStruct((batch_size,), jnp.int64)), actions
