# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from copy import deepcopy
from pathlib import Path

from .va_agibot_train_cfg import va_agibot_train_cfg


va_interna1_train_cfg = deepcopy(va_agibot_train_cfg)
va_interna1_train_cfg.__name__ = 'Config: VA InternData-A1 ICL train'
_repo_root = Path(__file__).resolve().parents[2]
_human_gen_root = _repo_root / 'data' / 'HumanGen'
va_interna1_train_cfg.dataset_path = str(_human_gen_root / 'interna1_data')
va_interna1_train_cfg.icl_manifest_path = str(
    _human_gen_root / 'icl_configs' / 'ICL_config_interna1.json'
)
va_interna1_train_cfg.human_latent_path = str(
    _human_gen_root / 'human_latents' / 'interna1'
)
va_interna1_train_cfg.obs_cam_keys = ['images.rgb.head']
