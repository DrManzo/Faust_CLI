"""Tests for LangGraph wiring in faust.adapters.graph."""

from __future__ import annotations

from faust.adapters.graph import build_prompt, llm_node, build_graph
from faust.core.models import AppConfig, Message, Role, Session


class FakeAdapter:
    """Small fake adapter that yields deterministic chunks."""

    def __init__(self, chunks=None, should_fail: bool = False):
        self.chunks = chunks or ["Hello ", "world"]
        self.should_fail = should_fail
        self.last_messages = None

    def generate(self, messages, stream: bool = True):
        self.last_messages = messages
        if self.should_fail:
            raise RuntimeError("fake adapter failure")
        for chunk in self.chunks:
            yield chunk


def make_state(messages=None):
    """Minimal valid FaustState payload for graph tests."""
    config = AppConfig()
    session = Session(id="test-session", model=config.model)

    return {
        "session": session,
        "config": config,
        "user_input": "Hello",
        "messages": messages or [],
        "response": "",
        "error": None,
    }


def test_build_prompt_injects_system_message_when_missing():
    """build_prompt should prepend a system message when one is absent."""
    state = make_state(
        messages=[
            Message(role=Role.USER, content="Hi"),
        ]
    )

    result = build_prompt(state)
    messages = result["messages"]

    assert len(messages) == 2
    assert messages[0].role == Role.SYSTEM
    assert messages[0].content == state["config"].system_prompt
    assert messages[1].role == Role.USER
    assert messages[1].content == "Hi"


def test_build_prompt_does_not_duplicate_existing_system_message():
    """build_prompt should leave messages unchanged if a system message already exists."""
    state = make_state(
        messages=[
            Message(role=Role.SYSTEM, content="Existing system prompt"),
            Message(role=Role.USER, content="Hi"),
        ]
    )

    result = build_prompt(state)
    messages = result["messages"]

    assert len(messages) == 2
    assert messages[0].role == Role.SYSTEM
    assert messages[0].content == "Existing system prompt"


def test_llm_node_concatenates_streamed_chunks():
    """llm_node should join adapter chunks into one response string."""
    adapter = FakeAdapter(chunks=["Fa", "ust"])
    state = make_state(
        messages=[
            Message(role=Role.SYSTEM, content="You are Faust."),
            Message(role=Role.USER, content="Say your name"),
        ]
    )

    result = llm_node(state, adapter=adapter)

    assert result["response"] == "Faust"
    assert result["error"] is None
    assert adapter.last_messages == [
        {"role": "system", "content": "You are Faust."},
        {"role": "user", "content": "Say your name"},
    ]


def test_llm_node_returns_error_on_adapter_failure():
    """llm_node should return an error string instead of raising."""
    adapter = FakeAdapter(should_fail=True)
    state = make_state(
        messages=[
            Message(role=Role.USER, content="Hi"),
        ]
    )

    result = llm_node(state, adapter=adapter)

    assert result["response"] == ""
    assert "fake adapter failure" in result["error"]


def test_build_graph_runs_end_to_end():
    """Compiled graph should inject prompt and produce a full response."""
    adapter = FakeAdapter(chunks=["Hello ", "from graph"])
    graph = build_graph(adapter)

    state = make_state(
        messages=[
            Message(role=Role.USER, content="Test graph"),
        ]
    )

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "thread_id": "test-thread",
            }
        },
    )

    assert result["response"] == "Hello from graph"
    assert result["error"] is None
    assert result["messages"][0].role == Role.SYSTEM
    assert result["messages"][1].role == Role.USER