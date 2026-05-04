"""Faust CLI entry point."""

from __future__ import annotations

from typing import List, Optional, Annotated

import typer

from faust.cli.commands.chat import chat as chat_cmd
from faust.cli.commands.run import run as run_cmd

app = typer.Typer(
    help="Faust — local AI assistant powered by Ollama.",
    no_args_is_help=False,          # don't show help when no args
    invoke_without_command=True,    # run callback when no subcommand
)


@app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    # Capture any positional args passed after `faust` (e.g. `faust what is 2+2?`)
    prompt_parts: Annotated[
        Optional[List[str]],
        typer.Argument(
            metavar="[PROMPT]...",
            help="Optional prompt to send directly to Faust.",
        ),
    ] = None,
) -> None:
    """
    Default behavior when you run `faust` without an explicit subcommand.

    - `faust`             → start interactive chat (same as `faust chat`)
    - `faust <prompt...>` → single-shot run (same as `faust run "<prompt>"`)
    """
    # If user explicitly invoked a subcommand (chat/run), do nothing here
    if ctx.invoked_subcommand:
        return

    obj = ctx.obj or {}
    config = obj.get("config")
    graph = obj.get("graph")

    if config is None or graph is None:
        typer.echo("Error: config or graph not initialized.", err=True)
        raise typer.Exit(1)

    # No prompt given: drop into the interactive chat loop
    if not prompt_parts:
        return chat_cmd(ctx)

    # Prompt provided: treat it as a single-shot run
    prompt = " ".join(prompt_parts)
    return run_cmd(ctx, prompt=prompt)


# Explicit subcommands still available
app.command("chat")(chat_cmd)
app.command("run")(run_cmd)