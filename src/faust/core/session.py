"""Session lifecycle management."""

from __future__ import annotations

import uuid
from pathlib import Path

from faust.core.models import AppConfig, Message, Role, Session, Turn


def create_session(config: AppConfig) -> Session:
    """Create a new Session from the active AppConfig."""

    return Session(
        id=str(uuid.uuid4()),
        model=config.model,
        system_prompt=config.system_prompt,
    )


def append_turn(session: Session, user_content: str, assistant_content: str) -> Session:
    """Append a completed user/assistant exchange to the session."""

    turn = Turn(
        user_message=Message(role=Role.USER, content=user_content),
        assistant_message=Message(role=Role.ASSISTANT, content=assistant_content),
    )
    return session.model_copy(update={"turns": [*session.turns, turn]})


def reset_session(session: Session) -> Session:
    """Clear all turns but preserve model and system prompt."""

    return session.model_copy(update={"turns": []})


def export_session(session: Session, path: Path) -> None:
    """Write session history to a JSON file at the given path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session.model_dump_json(indent=2))
