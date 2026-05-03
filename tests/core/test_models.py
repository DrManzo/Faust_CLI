"""Tests for core/models.py — data model validation."""

from faust.core.models import AppConfig, Message, Role


def test_message_to_dict():
    msg = Message(role=Role.USER, content="Hello")
    assert msg.to_dict() == {"role": "user", "content": "Hello"}


def test_app_config_defaults():
    config = AppConfig()
    assert config.model == "llama3.3:8b"
    assert config.backend == "ollama"
    assert 0.0 <= config.temperature <= 2.0
