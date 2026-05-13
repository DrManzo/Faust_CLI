"""faust run — single-shot prompt command.

Persistence guarantee
---------------------
This command passes ``thread_id`` and ``user_id`` through the
``configurable`` dict to the graph, which means the LangGraph
MemorySaver checkpointer *will* persist state across multiple
``faust run`` calls that share the same ``--thread`` value.

Callers that want a truly stateless single-shot call should pass
a unique ``--thread`` value each time (e.g. a timestamp or UUID).
"""

from __future__ import annotations

import typer
from rich.console import Console
from typer.models import OptionInfo

from faust.core.models import Message, Role, Session

console = Console()


def _resolve_option(value, fallback: str) -> str:
    """Convert Typer OptionInfo defaults into plain strings."""
    if isinstance(value, OptionInfo):
        return fallback
    return str(value)


def run(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to send to Faust."),
    thread_id: str = typer.Option(
        "default",
        "--thread",
        "-t",
        help=(
            "Thread ID for checkpointing.  Reuse the same value across "
            "calls to accumulate context in the MemorySaver store."
        ),
    ),
    user_id: str = typer.Option(
        "default",
        "--user",
        "-u",
        help="Stable user ID for long-term memory.",
    ),
) -> None:
    """Send a single prompt and print the response."""
    obj = ctx.obj or {}
    config = obj.get("config")
    graph = obj.get("graph")

    if not config or not graph:
        typer.echo("Error: config or graph not initialized.", err=True)
        raise typer.Exit(1)

    thread_id = _resolve_option(thread_id, "default")
    user_id = _resolve_option(user_id, "default")

    session = Session(id=thread_id, model=config.model)

    # State initialisation kept in sync with chat.py.
    # NOTE: run.py is single-shot so approval-gate fields start inert.
    # TODO(cleanup): chat.py and run.py share this init block — extract
    # into a shared _build_initial_state() helper in a dedicated Phase 3
    # refactor step once Phase 2 has been exercised.
    state: dict = {
        "session": session,
        "config": config,
        "user_id": user_id,
        "user_input": prompt,
        "intent": None,
        "active_agent": None,
        "requested_role": None,
        "task_type": None,
        "requested_tests": [],
        "test_approved": False,
        "test_proposal": None,
        "test_report_path": None,
        "execution_notes": None,
        "messages": [Message(role=Role.USER, content=prompt)],
        "recalled_memories": [],
        "artifacts": [],
        "response": "",
        "error": None,
    }

    try:
        # Both thread_id and user_id are forwarded in `configurable` so the
        # LangGraph MemorySaver checkpointer writes state under the correct
        # composite key.  This is what makes `faust run --thread <id>`
        # persist across multiple invocations.
        result = graph.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            },
        )

        state.update(result)
        response = state.get("response", "")
        error = state.get("error")

        if error:
            console.print(f"[red]Error:[/red] {error}")
            raise typer.Exit(1)
        else:
            console.print(response)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {exc}")
        raise typer.Exit(1)
