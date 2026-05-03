"""`faust chat` — interactive multi-turn session."""

from __future__ import annotations

import asyncio

import typer

from faust.cli.renderer import print_banner, print_error, print_session_info
from faust.core.session import append_turn, create_session

app = typer.Typer()


@app.callback(invoke_without_command=True)
def chat(ctx: typer.Context) -> None:
    """Start an interactive chat session with Faust."""

    asyncio.run(_chat_loop(ctx.obj["config"], ctx.obj["graph"]))


async def _chat_loop(config, graph) -> None:
    session = create_session(config)
    print_banner()
    print_session_info(session)

    while True:
        try:
            user_input = input("
[you] ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("
Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "exit", "quit"}:
            typer.echo("Goodbye.")
            break

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
            continue

        response = result["response"]
        typer.echo(f"
[faust] {response}")
        session = append_turn(session, user_input, response)
