"""faust chat — interactive conversation loop."""

from __future__ import annotations

import typer
from rich.console import Console

from faust.core.models import Message, Role

console = Console()


def chat(
    ctx: typer.Context,
    thread_id: str = typer.Option(
        "default", "--thread", "-t", help="Conversation thread ID."
    ),
) -> None:
    """Start an interactive chat session with Faust using llama3:8b."""
    obj = ctx.obj or {}
    config = obj.get("config")
    graph = obj.get("graph")

    if not config or not graph:
        typer.echo("Error: config or graph not initialized.", err=True)
        raise typer.Exit(1)

    state: dict = {
        "session": None,
        "config": config,
        "user_input": "",
        "messages": [],
        "response": "",
        "error": None,
    }

    console.print(
        f"[bold green]Faust[/bold green] — chatting with "
        f"[cyan]{config.model}[/cyan]"
    )
    console.print(
        "Type [bold yellow]exit[/bold yellow] or "
        "[bold yellow]quit[/bold yellow] to end.\n"
    )

    while True:
        try:
            user_input = typer.prompt("")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        stripped = user_input.strip()

        if stripped.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if not stripped:
            continue

        state["messages"].append(Message(role=Role.USER, content=stripped))
        state["user_input"] = stripped

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
            elif response:
                console.print(f"\n[bold]Faust:[/bold] {response}\n")
                state["messages"].append(
                    Message(role=Role.ASSISTANT, content=response)
                )
            else:
                console.print("[dim]No response received.[/dim]")

        except Exception as exc:
            console.print(f"[red]Unexpected error:[/red] {exc}")