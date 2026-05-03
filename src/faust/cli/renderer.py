"""All Rich-based terminal output for Faust."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from faust.core.models import Session

console = Console()


def print_banner() -> None:
    console.print(Panel(
        Text("FAUST", justify="center", style="bold white"),
        subtitle="[dim]local LLM CLI[/dim]",
        border_style="cyan",
    ))


def print_response(text: str) -> None:
    console.print(Markdown(text))


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_session_info(session: Session) -> None:
    console.print(
        f"[dim]Session:[/dim] {session.id[:8]}  "
        f"[dim]Model:[/dim] {session.model}  "
        f"[dim]Turns:[/dim] {len(session.turns)}"
    )
