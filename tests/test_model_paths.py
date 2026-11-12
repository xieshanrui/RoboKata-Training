from pathlib import Path

import pytest

from wan_va.modules.utils import resolve_model_component
from wan_va.train import Trainer


def test_model_components_resolve_from_model_root(tmp_path):
    for component in ("transformer", "vae", "text_encoder", "tokenizer"):
        (tmp_path / component).mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        "{}", encoding="utf-8"
    )

    assert resolve_model_component(tmp_path, "vae") == str(tmp_path / "vae")
    assert Trainer._resolve_transformer_path(tmp_path) == str(
        tmp_path / "transformer"
    )


def test_transformer_path_requires_model_root(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text("{}", encoding="utf-8")

    assert Trainer._resolve_transformer_path(tmp_path) == str(transformer)
    with pytest.raises(FileNotFoundError, match="Missing model component"):
        Trainer._resolve_transformer_path(transformer)


def test_missing_model_component_reports_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="text_encoder"):
        resolve_model_component(Path(tmp_path), "text_encoder")
