"""faust config — show current configuration."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def config(ctx: typer.Context) -> None:
    """Show the currently loaded Faust configuration."""
    obj = ctx.obj or {}
    cfg = obj.get("config")

    if cfg is None:
        typer.echo("Error: config not initialized.", err=True)
        raise typer.Exit(1)

    table = Table(
        title="Faust Configuration",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("backend", cfg.backend)
    table.add_row("model", cfg.model)
    table.add_row("temperature", str(cfg.temperature))
    table.add_row("context_window", str(cfg.context_window))
    table.add_row("ollama.base_url", cfg.ollama.base_url)
    table.add_row("ollama.request_timeout", str(cfg.ollama.request_timeout))
    table.add_row(
        "system_prompt",
        cfg.system_prompt[:60] + "..." if len(cfg.system_prompt) > 60 else cfg.system_prompt,
    )

    console.print(table)