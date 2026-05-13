"""Faust CLI entry point."""

from __future__ import annotations

import typer

from faust.cli.commands.chat import chat as chat_cmd
from faust.cli.commands.run import run as run_cmd

app = typer.Typer(
    help="Faust — local AI assistant powered by Ollama.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback()
def default(ctx: typer.Context) -> None:
    """
    Default behavior when you run `faust` without a subcommand.

    - `faust` -> start interactive chat
    """
    if ctx.invoked_subcommand is not None:
        return

    chat_cmd(ctx)


app.command("chat")(chat_cmd)
app.command("run")(run_cmd)