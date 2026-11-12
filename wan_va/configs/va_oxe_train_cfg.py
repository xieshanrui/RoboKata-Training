# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from copy import deepcopy
from pathlib import Path

from .va_agibot_train_cfg import va_agibot_train_cfg


va_oxe_train_cfg = deepcopy(va_agibot_train_cfg)
va_oxe_train_cfg.__name__ = 'Config: VA Bridge OXE ICL train'
_repo_root = Path(__file__).resolve().parents[2]
_human_gen_root = _repo_root / 'data' / 'HumanGen'
va_oxe_train_cfg.dataset_path = str(_human_gen_root / 'oxe_data')
va_oxe_train_cfg.icl_manifest_path = str(
    _human_gen_root / 'icl_configs' / 'ICL_config_oxe.json'
)
va_oxe_train_cfg.human_latent_path = str(
    _human_gen_root / 'human_latents' / 'oxe'
)
va_oxe_train_cfg.obs_cam_keys = ['observation.images.image']
