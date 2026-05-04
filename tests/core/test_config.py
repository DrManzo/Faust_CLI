"""Tests for the top-level config loader (faust.config.load_config)."""

from pathlib import Path

from faust.config import load_config
from faust.core.models import AppConfig


def test_default_config_yaml_loads_to_appconfig():
    """configs/default.yaml loads into AppConfig with expected fields."""
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), "configs/default.yaml must exist for this test"

    config = load_config(config_path)
    assert isinstance(config, AppConfig)

    # Core fields
    assert config.backend in ("ollama", "openai_compat")
    assert isinstance(config.model, str)
    assert isinstance(config.system_prompt, str)
    assert config.system_prompt.strip() != ""

    # Temperature in a sane range
    assert 0.0 <= config.temperature <= 2.0

    # Nested Ollama config
    assert config.ollama.base_url.startswith("http://") or config.ollama.base_url.startswith("https://")
    assert config.ollama.request_timeout > 0