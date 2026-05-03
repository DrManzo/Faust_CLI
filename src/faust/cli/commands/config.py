"""`faust config` — view and edit runtime configuration."""

from __future__ import annotations

import typer
import yaml

from faust.config import DEFAULT_CONFIG_PATH

app = typer.Typer()


@app.command("show")
def show() -> None:
    """Display the current configuration."""

    typer.echo(DEFAULT_CONFIG_PATH.read_text())


@app.command("set")
def set_value(
    key: str = typer.Argument(..., help="Config key to set"),
    value: str = typer.Argument(..., help="Value to assign"),
) -> None:
    """Set a top-level configuration key."""

    raw: dict = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    raw[key] = value
    DEFAULT_CONFIG_PATH.write_text(yaml.dump(raw, default_flow_style=False))
    typer.echo(f"Set {key} = {value}")
