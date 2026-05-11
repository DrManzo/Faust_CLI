# Faust — Local CLI LLM

A local, offline-first CLI for interacting with LLMs via Ollama or any OpenAI-compatible server, with a LangGraph-powered agent layer and configurable checkpoint persistence.[1][2]

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

# 5. Run the test suite
python -m pytest -q
faust-tests
```

## Requirements

- Python 3.12+[1]
- [uv](https://github.com/astral-sh/uv) package manager[1]
- [Ollama](https://ollama.com) running locally for the default backend[1]

## Recommended Models

| Model | Role | Command |
|---|---|---|
| `llama3.3:8b` | General brain (default) | `ollama pull llama3.3:8b` |
| `deepseek-r1:8b` | Reasoner / planner | `ollama pull deepseek-r1:8b` |
| `qwen2.5-coder:14b` | Code specialist | `ollama pull qwen2.5-coder:14b` |

## Data Storage

Faust supports local LangGraph checkpoint persistence and project-local runtime data.[1][2]

- `data/faust.db` — local LangGraph SQLite checkpoint database when SQLite persistence is enabled.[1]
- `data/sessions/` — optional JSON session exports.[1]
- `data/` — created automatically at runtime and ignored by version control.[1]

## Testing

The current automated baseline is 17 passing tests, and the recommended local validation commands are shown below.[3]

```bash
python -m pytest -q
python -m pytest -vv
faust-tests
```

Using `python -m pytest` is the safest option in virtual environments because it avoids shell PATH issues with the `pytest` executable.[3]

## Adding a New Component

```bash
python scaffolding.py add-component <name>
```
