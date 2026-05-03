"""Prompt assembly logic."""

from __future__ import annotations

from faust.core.context import get_context_messages
from faust.core.models import AppConfig, Message, Role, Session


def build_prompt(session: Session, user_input: str, config: AppConfig) -> list[Message]:
    """Assemble the full prompt for the current turn."""

    history = get_context_messages(session, config.context_window)
    user_msg = Message(role=Role.USER, content=user_input)
    return [*history, user_msg]
