from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch.demo_compression import PrefixKVBundle, compress_demo_hidden


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.project_image_features(self.embed_image_patches(image))

    def embed_image_patches(self, image: torch.Tensor):
        """Shared SigLIP output; the only valid frozen-vision cache boundary."""
        return self.paligemma.model.vision_tower(image).last_hidden_state

    def project_image_features(self, patches: torch.Tensor):
        projector = self.paligemma.model.multi_modal_projector
        return projector(patches.to(projector.linear.weight.dtype))

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def _compressed_prefix(self, hidden, plan, masks, positions, use_cache):
        from transformers.cache_utils import DynamicCache
        if not use_cache:
            raise ValueError("Compressed prefix-only execution requires a fresh cache")
        model = self.paligemma.language_model
        if not 0 < plan.compression_layer < len(model.layers):
            raise ValueError("Compression boundary must fall between decoder layers")
        cache = DynamicCache()
        if model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        for index, layer in enumerate(model.layers):
            if index == plan.compression_layer:
                hidden = compress_demo_hidden(hidden, plan)
            stage = plan.stage(index)
            rope = model.rotary_emb(hidden, positions[stage])
            hidden = layer(
                hidden, attention_mask=masks[stage], position_ids=positions[stage],
                past_key_value=cache, use_cache=True,
                cache_position=torch.arange(hidden.shape[1], device=hidden.device),
                position_embeddings=rope, adarms_cond=None,
            )[0]
            if cache.get_seq_length(index) != hidden.shape[1]:
                raise RuntimeError("Prefix KV length differs from this layer's physical slots")
        hidden, _ = model.norm(hidden, None)
        return hidden, PrefixKVBundle(cache, plan.compression_layer, plan.early_valid, plan.late_valid)

    def _compressed_suffix(self, hidden, bundle, masks, positions, condition):
        model = self.gemma_expert.model
        if model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        for index, layer in enumerate(model.layers):
            stage = int(index >= bundle.compression_layer)
            physical_prefix = bundle.valid_at(index).shape[1]
            if bundle.cache.get_seq_length(index) != physical_prefix:
                raise RuntimeError("Cached prefix length differs from this layer's layout")
            rope = model.rotary_emb(hidden, positions[stage])
            hidden = layer(
                hidden, attention_mask=masks[stage], position_ids=positions[stage],
                past_key_value=bundle.cache, use_cache=False,
                cache_position=physical_prefix + torch.arange(hidden.shape[1], device=hidden.device),
                position_embeddings=rope, adarms_cond=condition,
            )[0]
        return model.norm(hidden, condition)[0]

    def _compressed_joint_layer(self, index, streams, mask, positions, conditions):
        """Use the same per-stream decoder kernels during training and caching.

        Prefix cannot read suffix, so a fresh layer-local KV object preserves
        the joint graph while matching BF16 inference matrix shapes. The cache
        is created inside the checkpointed function and never escapes it.
        """
        from transformers.cache_utils import DynamicCache
        prefix, suffix = streams
        length = prefix.shape[1]
        language = self.paligemma.language_model
        expert = self.gemma_expert.model
        cache = DynamicCache()
        prefix_positions, suffix_positions = positions[:, :length], positions[:, length:]
        prefix_out = language.layers[index](
            prefix, attention_mask=mask[:, :, :length, :length], position_ids=prefix_positions,
            past_key_value=cache, use_cache=True,
            cache_position=torch.arange(length, device=prefix.device),
            position_embeddings=language.rotary_emb(prefix, prefix_positions), adarms_cond=conditions[0],
        )[0]
        if cache.get_seq_length(index) != length:
            raise RuntimeError("Layer-local prefix KV differs from the current layout")
        suffix_out = expert.layers[index](
            suffix, attention_mask=mask[:, :, length:, :], position_ids=suffix_positions,
            past_key_value=cache, use_cache=False,
            cache_position=length + torch.arange(suffix.shape[1], device=suffix.device),
            position_embeddings=expert.rotary_emb(suffix, suffix_positions), adarms_cond=conditions[1],
        )[0]
        return [prefix_out, suffix_out]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        prefix_compression_plan=None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if prefix_compression_plan is not None and inputs_embeds[1] is None:
            if past_key_values is not None:
                raise ValueError("Compressed prefill must start from an empty prefix cache")
            prefix_output, prefix_past_key_values = self._compressed_prefix(
                inputs_embeds[0], prefix_compression_plan, attention_mask, position_ids, use_cache)
            suffix_output = None
        elif isinstance(past_key_values, PrefixKVBundle):
            if inputs_embeds[0] is not None or use_cache:
                raise ValueError("Compressed suffix execution only reads the immutable prefix cache")
            suffix_output = self._compressed_suffix(
                inputs_embeds[1], past_key_values, attention_mask, position_ids, adarms_cond[1])
            prefix_output = prefix_past_key_values = None
        elif inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if prefix_compression_plan is None and self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                if prefix_compression_plan is not None:
                    return self._compressed_joint_layer(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond)
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                layer_attention_mask, layer_positions = attention_mask, position_ids
                if prefix_compression_plan is not None:
                    if not 0 < prefix_compression_plan.compression_layer < num_layers:
                        raise ValueError("Compression boundary must fall between decoder layers")
                    if layer_idx == prefix_compression_plan.compression_layer:
                        inputs_embeds = [compress_demo_hidden(inputs_embeds[0], prefix_compression_plan), inputs_embeds[1]]
                    stage = prefix_compression_plan.stage(layer_idx)
                    layer_attention_mask, layer_positions = attention_mask[stage], position_ids[stage]
                selected_layers = getattr(self, "checkpoint_layers", None)
                if use_gradient_checkpointing and (selected_layers is None or layer_idx in selected_layers):
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        layer_attention_mask,
                        layer_positions,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, layer_attention_mask, layer_positions, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
