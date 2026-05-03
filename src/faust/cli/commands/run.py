"""`faust run "<prompt>"` — single-shot inference."""

from __future__ import annotations

import asyncio

import typer

from faust.cli.renderer import print_error
from faust.core.session import create_session

app = typer.Typer()


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="The prompt to send"),
) -> None:
    """Run a single prompt and print the response."""

    asyncio.run(_run_once(ctx.obj["config"], ctx.obj["graph"], prompt))


async def _run_once(config, graph, user_input: str) -> None:
    session = create_session(config)
    state = {
        "session": session,
        "config": config,
        "user_input": user_input,
        "messages": [],
        "response": "",
        "error": None,
    }
    result = await graph.ainvoke(state)
    if result.get("error"):
        print_error(result["error"])
        raise typer.Exit(code=1)
    typer.echo(result["response"])
