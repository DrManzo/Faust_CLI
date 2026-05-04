"""faust run — single-shot prompt command."""

from __future__ import annotations

import typer
from rich.console import Console

from faust.core.models import Message, Role

console = Console()


def run(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to send to Faust."),
    thread_id: str = typer.Option(
        "default",
        "--thread",
        "-t",
        help="Thread ID for checkpointing.",
    ),
) -> None:
    """Send a single prompt and print the response."""
    obj = ctx.obj or {}
    config = obj.get("config")
    graph = obj.get("graph")

    if not config or not graph:
        typer.echo("Error: config or graph not initialized.", err=True)
        raise typer.Exit(1)

    state: dict = {
        "session": None,
        "config": config,
        "user_input": prompt,
        "messages": [Message(role=Role.USER, content=prompt)],
        "response": "",
        "error": None,
    }

    try:
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
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