# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from pathlib import Path

from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_agibot_train_cfg = EasyDict(__name__='Config: VA AgiBot ICL train')
va_agibot_train_cfg.update(va_robotwin_train_cfg)

_repo_root = Path(__file__).resolve().parents[2]
_human_gen_root = _repo_root / 'data' / 'HumanGen'
va_agibot_train_cfg.dataset_path = str(_human_gen_root / 'agibot_data')
va_agibot_train_cfg.load_robotwin_stats_from_dataset = False
va_agibot_train_cfg.excluded_task_names = []
va_agibot_train_cfg.expected_num_train_tasks = None
# Robot latents live inside each LeRobot task, matching the Robotwin layout.
va_agibot_train_cfg.robot_latent_path = ''
va_agibot_train_cfg.icl_manifest_path = str(
    _human_gen_root / 'icl_configs' / 'ICL_config_agibot.json'
)
va_agibot_train_cfg.human_latent_path = str(
    _human_gen_root / 'human_latents' / 'agibot'
)
va_agibot_train_cfg.obs_cam_keys = [
    'observation.images.head',
    'observation.images.hand_left',
    'observation.images.hand_right',
]
