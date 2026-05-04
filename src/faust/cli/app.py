"""Faust CLI entry point."""

from __future__ import annotations

import typer

from faust.cli.commands.chat import chat as chat_cmd
from faust.cli.commands.run import run as run_cmd

app = typer.Typer(
    help="Faust — local AI assistant powered by Ollama.",
    no_args_is_help=True,
)

app.command("chat")(chat_cmd)
app.command("run")(run_cmd)
