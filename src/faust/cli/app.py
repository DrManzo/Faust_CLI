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
def default(
    ctx: typer.Context,
    user_id: str = typer.Option(
        "default",
        "--user",
        "-u",
        help="Stable user ID / profile (e.g. DrManzo).",
    ),
    thread_id: str = typer.Option(
        "default",
        "--thread",
        "-t",
        help="Conversation thread ID.",
    ),
) -> None:
    """
    Default behavior when you run `faust` without a subcommand.

    Examples:
        faust                              # interactive chat, default profile
        faust --user DrManzo               # interactive chat as DrManzo
        faust --user DrManzo --thread dev  # named thread
    """
    if ctx.invoked_subcommand is not None:
        return

    # Store resolved values in ctx.obj for downstream access.
    obj = ctx.ensure_object(dict)
    obj["_default_user_id"] = user_id
    obj["_default_thread_id"] = thread_id

    # Call chat_cmd directly with ctx so it receives its required first arg.
    chat_cmd(ctx, user_id=user_id, thread_id=thread_id)


app.command("chat")(chat_cmd)
app.command("run")(run_cmd)
