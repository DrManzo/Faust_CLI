# Faust — Local CLI LLM

Faust is a local, offline-first CLI assistant for interacting with language models through Ollama or any OpenAI-compatible server. It uses Typer for the CLI, LangGraph for agent orchestration, and configurable local checkpoint persistence for conversation state.

Faust now includes an initial long-term memory layer with deterministic recall for a small set of supported user facts, plus natural memory writes for simple self-fact statements such as name, birthdate, and favorite editor.

## Features

- Local-first CLI workflow.
- Ollama as the default backend.
- OpenAI-compatible adapter support.
- LangGraph-powered state and routing.
- Configurable checkpoint persistence.
- Long-term memory foundation with user-scoped namespaces.
- Deterministic recall for supported facts:
- `profile.name`
- `profile.birthdate`
- `preference.favorite_editor`
- Implicit memory writes for supported self-fact statements such as:
- `I am Javier`
- `my birthday is April 4th 1994`
- `my favorite editor is Vim`

## Quickstart

### 1. Install dependencies

Faust uses `uv` for dependency management.

```bash
make install
```

### 2. Pull a local model

```bash
ollama pull llama3.3:8b
```

### 3. Start a chat session

```bash
make run
```

You can also start Faust directly:

```bash
faust
```

### 4. Run a single prompt

```bash
faust run "Explain recursion in one sentence"
```

### 5. Run the test suite

```bash
python -m pytest -q
faust-tests
```

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) for the default local backend

## Recommended models

| Model | Role | Command |
|---|---|---|
| `llama3.3:8b` | General default model | `ollama pull llama3.3:8b` |
| `deepseek-r1:8b` | Reasoning and planning | `ollama pull deepseek-r1:8b` |
| `qwen2.5-coder:14b` | Code-focused tasks | `ollama pull qwen2.5-coder:14b` |

## CLI usage

### Interactive chat

```bash
faust
```

Example:

```text
: I am Javier
Faust: Okay — I'll remember that your name is Javier.

: What is my name?
Faust: Your name is Javier.
```

### One-shot prompts

```bash
faust run "Summarize the CAP theorem in two sentences"
```

### Other commands

```bash
faust --help
faust config
```

## Memory behavior

Faust currently uses three practical layers of state:

- **Checkpoint memory** for conversation/thread continuity through LangGraph persistence.
- **Long-term memory** for durable user facts and preferences.
- **In-memory state** during graph execution for short-lived routing and response flow.

Today, the supported durable long-term memory slots are:

- `profile.name`
- `profile.birthdate`
- `preference.favorite_editor`

### Explicit memory writes

Faust supports explicit durable memory instructions such as:

```text
remember that my favorite editor is Vim
remember that my name is Javier
```

### Implicit memory writes

Faust also supports a narrow set of natural self-fact statements:

```text
I am Javier
my name is Javier
my birthday is April 4th 1994
I was born on April 4th 1994
my favorite editor is Vim
my favorite editor is actually Neovim
```

### Deterministic recall

For simple supported recall questions, Faust answers directly from memory before using the normal LLM path.

Examples:

```text
Who am I?
What is my name?
When was I born?
What is my birthday?
What is my favorite editor?
```

Open-ended or derived questions still use the normal model path when appropriate.

## Data storage

Faust stores runtime data locally.

- `data/faust.db` — local LangGraph SQLite checkpoint database when SQLite persistence is enabled.
- `data/faust_memory.db` — local SQLite long-term memory database when SQLite long-term memory is enabled.
- `data/sessions/` — optional JSON session exports.
- `data/` — created automatically at runtime and ignored by version control.

This project is configured so local runtime data and personal memory artifacts are not committed to Git.

## Testing

The current automated baseline is **47 passing tests**.

Recommended local validation commands:

```bash
python -m pytest -q
python -m pytest -vv
faust-tests
```

Using `python -m pytest` is the safest option inside virtual environments because it avoids shell PATH issues with the standalone `pytest` executable.

## Project structure

```text
src/faust/
├── adapters/
├── agents/
├── cli/
└── core/

tests/
├── adapters/
├── cli/
└── core/
```

## Development notes

- The default local workflow is Ollama-first.
- The graph currently supports deterministic routing for a small supported set of memory facts.
- The current memory system is intentionally narrow and durable rather than broad and fuzzy.
- The next architectural step is to formalize memory buckets more explicitly across checkpoint memory, long-term memory, and ephemeral execution state.

## Adding a new component

```bash
python scaffolding.py add-component <name>
```

## Status

Faust currently has:
- a working Typer CLI,
- LangGraph graph wiring,
- checkpoint persistence,
- Ollama and OpenAI-compatible adapters,
- a first long-term memory foundation,
- deterministic recall for supported user facts,
- and tested implicit memory writes for simple self-fact statements.

This is an actively evolving local assistant project, with the current focus on durable architecture and reliable offline-first behavior.
