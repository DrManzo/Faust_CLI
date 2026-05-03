"""Typer application root. Registers all Faust CLI commands."""

import typer

from faust.cli.commands import chat, config, run

app = typer.Typer(
    name="faust",
    help="Faust — local LLM CLI",
    no_args_is_help=True,
)

app.add_typer(chat.app, name="chat")
app.add_typer(run.app, name="run")
app.add_typer(config.app, name="config")
