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

Using `python -m pytest` is the safest option in virtual environments because it avoids shell PATH issues with the `pytest` executable.[3]report.md`.[file:191]

### Changed
- Updated the default backend configuration to use Ollama as the primary runtime backend.[3]
- Standardized the default model around the local Ollama workflow used in the current implementation.[3]
- Updated test and developer workflow so validation can be run from the repository root with:
- `python -m pytest -q`,[3]
- `python -m pytest -vv`,[3]
- `faust-tests`.[3]
- Updated `pyproject.toml` so pytest works correctly with the `src/` layout and console scripts install cleanly.[3]
- Updated graph construction and tests so `build_graph` receives the required config object during end-to-end execution.[3]

### Fixed
- Fixed duplicate `pyproject.toml` script and pytest configuration issues that caused parsing and install failures.[3]
- Fixed graph test execution by passing the required LangGraph checkpoint `thread_id` configuration.[3]
- Fixed the end-to-end graph test mismatch after the `build_graph(adapter, config)` signature change.[3]
- Fixed CLI tests to align with actual Rich/Typer output behavior during automated runs.[3]
- Fixed the test report workflow so the report is written reliably to `reports/test-report.md`.[3]

### Testing
- Current automated test baseline: 17 passing tests.[3]
- Verified local commands:
- `python -m pytest -q`,[3]
- `python -m pytest -vv`,[3]
- `faust-tests`.[3]

## [0.1.0] — Initial scaffold
- Component-based file structure: `cli/`, `core/`, `adapters/`, `agents/`.[3]
- Pydantic v2 data models: `Message`, `Turn`, `Session`, `AppConfig`, `FaustState`.[3]
- Initial LangGraph graph structure with `build_prompt` and `llm` nodes.[3]
- Ollama and OpenAI-compatible backend adapter scaffolding.[3]
- Typer CLI scaffold: `faust chat`, `faust run`, `faust config`.[3]
- Rich terminal renderer.[3]
- Early SQLite checkpointing design planning.[3]
