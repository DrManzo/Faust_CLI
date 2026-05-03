"""Tests for core/session.py — session lifecycle."""

from pathlib import Path

from faust.core.models import AppConfig
from faust.core.session import append_turn, create_session, export_session, reset_session


def test_create_session():
    config = AppConfig()
    session = create_session(config)
    assert session.model == config.model
    assert len(session.turns) == 0


def test_append_and_reset_session(tmp_path: Path):
    config = AppConfig()
    session = create_session(config)
    session = append_turn(session, "Hello", "Hi there")
    assert len(session.turns) == 1

    # Export and reset
    export_path = tmp_path / "session.json"
    export_session(session, export_path)
    assert export_path.exists()

    session = reset_session(session)
    assert len(session.turns) == 0
