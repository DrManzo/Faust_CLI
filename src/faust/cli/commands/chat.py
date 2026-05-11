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


def _print_debug_state(state: dict) -> None:
    """Render bounded internal execution state for debugging."""
    active_agent = state.get("active_agent")
    requested_role = state.get("requested_role")
    task_type = state.get("task_type")
    requested_tests = state.get("requested_tests") or []
    execution_notes = state.get("execution_notes")
    error = state.get("error")

    console.print("[dim]--- debug ---[/dim]")
    if active_agent:
        console.print(f"[dim]active_agent:[/dim] {active_agent}")
    if requested_role:
        console.print(f"[dim]requested_role:[/dim] {requested_role}")
    if task_type:
        console.print(f"[dim]task_type:[/dim] {task_type}")
    if requested_tests:
        console.print(
            "[dim]requested_tests:[/dim] " + ", ".join(requested_tests)
        )
    if execution_notes:
        console.print(f"[dim]execution_notes:[/dim] {execution_notes}")
    if error:
        console.print(f"[dim]error:[/dim] {error}")
    console.print("[dim]-------------[/dim]")


def chat(
    ctx: typer.Context,
    thread_id: str = typer.Option(
        "default", "--thread", "-t", help="Conversation thread ID."
    ),
    user_id: str = typer.Option(
        "default", "--user", "-u", help="Stable user ID for long-term memory."
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show bounded internal routing and execution state.",
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
        "requested_role": None,
        "task_type": None,
        "requested_tests": [],
        "execution_notes": None,
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
    if debug:
        console.print("[dim]Debug mode enabled.[/dim]")
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
        state["requested_role"] = None
        state["task_type"] = None
        state["requested_tests"] = []
        state["execution_notes"] = None
        state["error"] = None
        state["response"] = ""

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

            if debug:
                _print_debug_state(state)
                console.print()

        except Exception as exc:
            console.print(f"[red]Unexpected error:[/red] {exc}")