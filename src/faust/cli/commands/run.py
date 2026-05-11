"""faust run — single-shot prompt command."""

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
        help="Thread ID for checkpointing.",
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

    state: dict = {
        "session": session,
        "config": config,
        "user_id": user_id,
        "user_input": prompt,
        "intent": None,
        "active_agent": None,
        "messages": [Message(role=Role.USER, content=prompt)],
        "recalled_memories": [],
        "artifacts": [],
        "response": "",
        "error": None,
    }

    try:
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

    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {exc}")
        raise typer.Exit(1)