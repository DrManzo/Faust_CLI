"""faust chat — interactive conversation loop."""

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


def chat(
    ctx: typer.Context,
    thread_id: str = typer.Option(
        "default", "--thread", "-t", help="Conversation thread ID."
    ),
    user_id: str = typer.Option(
        "default", "--user", "-u", help="Stable user ID for long-term memory."
    ),
) -> None:
    """Start an interactive chat session with Faust."""
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
        "user_input": "",
        "intent": None,
        "active_agent": None,
        "messages": [],
        "recalled_memories": [],
        "artifacts": [],
        "response": "",
        "error": None,
    }

    console.print(
        f"[bold green]Faust[/bold green] — chatting with "
        f"[cyan]{config.model}[/cyan]"
    )
    console.print(
        f"[dim]Thread:[/dim] {thread_id}    [dim]User:[/dim] {user_id}"
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
        state["user_id"] = user_id
        state["intent"] = None
        state["active_agent"] = None

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
            elif response:
                console.print(f"\n[bold]Faust:[/bold] {response}\n")
                state["messages"].append(
                    Message(role=Role.ASSISTANT, content=response)
                )
            else:
                console.print("[dim]No response received.[/dim]")

        except Exception as exc:
            console.print(f"[red]Unexpected error:[/red] {exc}")