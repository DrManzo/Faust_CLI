"""Configuration loader for Faust."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from faust.core.models import AppConfig
from faust.exceptions import ConfigError

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read a YAML config file and return a validated AppConfig.

    Raises ConfigError if the file is missing or invalid.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        return AppConfig(**raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
