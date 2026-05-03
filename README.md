# Faust — Local CLI LLM

A local, offline-first CLI for interacting with LLMs via Ollama or any
OpenAI-compatible server, with a LangGraph-powered agent layer.

## Quickstart

```bash
# 1. Install dependencies (requires uv)
make install

# 2. Pull a model via Ollama
ollama pull llama3.3:8b

# 3. Start a chat session
make run

# 4. Or run a single prompt
faust run "Explain recursion in one sentence"
```

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com) running locally (default backend)

## Recommended Models

| Model | Role | Command |
|---|---|---|
| `llama3.3:8b` | General brain (default) | `ollama pull llama3.3:8b` |
| `deepseek-r1:8b` | Reasoner / planner | `ollama pull deepseek-r1:8b` |
| `qwen2.5-coder:14b` | Code specialist | `ollama pull qwen2.5-coder:14b` |

## Data Storage

Faust stores LangGraph checkpoints in a local SQLite database:

- `data/faust.db` — LangGraph SqliteSaver database
- `data/sessions/` — optional JSON session exports

The `data/` folder is ignored by version control and is created automatically at runtime.

## Adding a New Component

```bash
python scaffolding.py add-component <name>
```
