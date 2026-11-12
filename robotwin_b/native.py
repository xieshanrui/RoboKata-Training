"""Expose only the upstream RoboTwin config and action helpers.

The upstream package initializer eagerly imports FlashAttention. The isolated
A+B1 backend supplies its own attention and does not need that dependency.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

spec = importlib.util.find_spec("wan_va")
if spec is None or spec.origin is None:
    raise ImportError("Add the upstream Zero-WAM checkout to PYTHONPATH")
root = Path(spec.origin).parent
for name, path in (("_robotwin_native", root),
                   ("_robotwin_native.configs", root / "configs"),
                   ("_robotwin_native.dataset", root / "dataset")):
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

from _robotwin_native.configs.va_robotwin_cfg import load_robotwin_norm_stat, va_robotwin_cfg
from _robotwin_native.configs.va_robotwin_train_cfg import va_robotwin_train_cfg
from _robotwin_native.dataset.robotwin_action import preprocess_robotwin_actions
