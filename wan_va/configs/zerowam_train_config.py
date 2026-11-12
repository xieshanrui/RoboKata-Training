# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict


zerowam_train_cfg = EasyDict()

# Zero-WAM always trains the video, action, ICL, and MCP branches together.
zerowam_train_cfg.enable_mcp = True
zerowam_train_cfg.num_mcp_modules = 4
zerowam_train_cfg.mcp_blocks_per_group = 1
zerowam_train_cfg.mcp_hidden_collect_layers = [3, 11, 19, 29]
zerowam_train_cfg.mcp_snr_shift = 10.0
zerowam_train_cfg.mcp_loss_weights = [0.5, 0.25, 0.15, 0.1]
zerowam_train_cfg.future_chunk_stride = 2

zerowam_train_cfg.noisy_img_prob = 0.5
zerowam_train_cfg.noisy_cond_min_timestep_bd = 0.0
zerowam_train_cfg.noisy_cond_max_timestep_bd = 1.0
zerowam_train_cfg.video_loss_reweight = True
zerowam_train_cfg.action_loss_reweight = False

zerowam_train_cfg.drop_icl = 0.1
zerowam_train_cfg.droptext_target = 0.4
zerowam_train_cfg.text_encoder_type = 'umt_dense'
zerowam_train_cfg.icl_rope_h = 24
zerowam_train_cfg.frame_chunk_size = 0
zerowam_train_cfg.max_frame_chunk_size = 4
zerowam_train_cfg.attn_window = 0
zerowam_train_cfg.max_attn_window = 64
