"""Faust CLI entry point."""

from __future__ import annotations

from typing import Optional

import typer

from faust.cli.commands.chat import chat as chat_cmd
from faust.cli.commands.run import run as run_cmd

app = typer.Typer(
    help="Faust — local AI assistant powered by Ollama.",
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)


@app.callback()
def default(
    ctx: typer.Context,
    prompt_parts: list[str] | None = typer.Argument(None),
) -> None:
    """
    Default behavior when you run `faust` without a subcommand.

    - `faust` -> start interactive chat
    - `faust <prompt>` -> run single-shot prompt
    """
    if ctx.invoked_subcommand is not None:
        return

    if prompt_parts:
        prompt = " ".join(prompt_parts).strip()
        if prompt:
            ctx.invoke(run_cmd, ctx=ctx, prompt=prompt)
            return

    chat_cmd(ctx)


app.command("chat")(chat_cmd)
app.command("run")(run_cmd)