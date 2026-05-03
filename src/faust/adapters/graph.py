"""LangGraph StateGraph definition for Faust."""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from faust.adapters.base import LLMAdapter
from faust.core.models import FaustState
from faust.core.prompt import build_prompt
from faust.exceptions import BackendError, GraphError


def _get_db_path() -> str:
    """Return the filesystem path for the SQLite checkpoint database.

    The database is stored in the project-local data/faust.db file.
    """

    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "faust.db")


def build_graph(adapter: LLMAdapter):
    """Compile and return the Faust base StateGraph with SQLite checkpointing."""

    graph = StateGraph(FaustState)

    def build_prompt_node(state: FaustState) -> dict:
        """Assemble the ordered message list for the current turn."""

        messages = build_prompt(state["session"], state["user_input"], state["config"])
        return {"messages": messages}

    async def llm_node(state: FaustState) -> dict:
        """Stream a response from the backend and collect it into state."""

        tokens: list[str] = []
        try:
            async for token in adapter.stream(state["messages"]):
                tokens.append(token)
        except BackendError as exc:
            return {"error": str(exc), "response": ""}
        return {"response": "".join(tokens), "error": None}

    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("build_prompt")
    graph.add_edge("build_prompt", "llm")
    graph.add_edge("llm", END)

    try:
        db_path = _get_db_path()
        checkpointer = SqliteSaver.from_conn_string(db_path)
        return graph.compile(checkpointer=checkpointer)
    except Exception as exc:
        raise GraphError(f"Graph compilation failed: {exc}") from exc
