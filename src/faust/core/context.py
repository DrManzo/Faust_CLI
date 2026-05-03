"""Context window management for Faust sessions."""

from __future__ import annotations

from faust.core.models import Message, Role, Session


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate based on character count (~4 chars per token)."""

    return sum(len(m.content) // 4 for m in messages)


def get_context_messages(session: Session, max_tokens: int) -> list[Message]:
    """Return a chronologically ordered list of Messages within max_tokens.

    Strategy:
    1. Always include the system prompt.
    2. Walk turns newest to oldest until the budget is exhausted.
    3. Return messages in chronological order.
    """

    system_msg = Message(role=Role.SYSTEM, content=session.system_prompt)
    budget = max_tokens - estimate_tokens([system_msg])

    selected: list[Message] = []
    for turn in reversed(session.turns):
        candidates: list[Message] = []
        if turn.user_message:
            candidates.append(turn.user_message)
        if turn.assistant_message:
            candidates.append(turn.assistant_message)
        cost = estimate_tokens(candidates)
        if cost > budget:
            break
        selected = candidates + selected
        budget -= cost

    return [system_msg, *selected]
