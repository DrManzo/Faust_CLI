"""LangGraph state graph for Faust."""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from faust.core.models import FaustState, Message, Role


def build_prompt(state: FaustState) -> dict:
    """Ensure system prompt is the first message in the conversation."""
    current = list(state.get("messages", []))
    has_system = any(m.role == Role.SYSTEM for m in current)
    if not has_system and state.get("config"):
        system_msg = Message(
            role=Role.SYSTEM,
            content=state["config"].system_prompt,
        )
        current = [system_msg] + current
    return {"messages": current}


def llm_node(state: FaustState, adapter) -> dict:
    """Call the LLM adapter and collect the full streaming response."""
    message_dicts = [m.to_dict() for m in state["messages"]]
    full_response = ""
    try:
        for chunk in adapter.generate(message_dicts, stream=True):
            full_response += chunk
        return {"response": full_response, "error": None}
    except Exception as exc:
        return {"response": "", "error": str(exc)}


def build_graph(adapter) -> CompiledStateGraph:
    """Build and compile the LangGraph state graph."""
    workflow = StateGraph(FaustState)
    workflow.add_node("build_prompt", build_prompt)
    workflow.add_node("llm", partial(llm_node, adapter=adapter))
    workflow.set_entry_point("build_prompt")
    workflow.add_edge("build_prompt", "llm")
    workflow.set_finish_point("llm")

    # InMemorySaver keeps conversation context alive for the session
    # SQLite persistence will be added in a later step
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)
