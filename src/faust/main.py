"""Faust entry point: wire config, adapter, graph, and CLI."""

from __future__ import annotations

import typer

from faust.adapters.graph import build_graph
from faust.adapters.ollama import OllamaAdapter
from faust.adapters.openai_compat import OpenAICompatAdapter
from faust.cli.app import app
from faust.config import load_config
from faust.exceptions import ConfigError


def main() -> None:
    config = load_config()

    match config.backend:
        case "ollama":
            adapter = OllamaAdapter(config)
        case "openai_compat":
            adapter = OpenAICompatAdapter(config)
        case _:
            raise ConfigError(
                f"Unknown backend: '{config.backend}'. "
                "Valid options: 'ollama', 'openai_compat'"
            )

    graph = build_graph(adapter)
    app(obj={"adapter": adapter, "config": config, "graph": graph})


if __name__ == "__main__":  # pragma: no cover
    main()
