# Faust — Local CLI LLM

Faust is a local, offline-first CLI assistant for interacting with language models through Ollama or any OpenAI-compatible server. It uses Typer for the CLI, LangGraph for agent orchestration, and configurable local checkpoint persistence for conversation state.

Faust includes a long-term memory layer with deterministic recall for supported user facts, natural memory writes for simple self-fact statements, and minimal multi-role routing for assistant, reasoner, and coder workflows.

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
- Multi-role graph routing:
- `assistant` for general conversation
- `reasoner` for planning and decomposition
- `coder` for code-oriented tasks with scoped test execution

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

: Plan this: add support for favorite shell
Faust: [reasoner output with decomposition steps]

: Write code to add a regression test for location correction
Faust: [coder output with scoped pytest execution]
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

Faust uses three practical layers of state:

- **Checkpoint memory** for conversation/thread continuity through LangGraph persistence.
- **Long-term memory** for durable user facts and preferences.
- **Ephemeral state** during graph execution for short-lived routing and response flow.

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

## Role routing

Faust uses minimal task classification to route turns through appropriate graph paths:

- **assistant** — general conversation and Q&A
- **reasoner** — planning, decomposition, and multi-step thinking
- **coder** — code generation, refactoring, and scoped test execution

The classifier prioritizes coding and reasoning markers before memory-slot keyword matches to prevent false positives during planning workflows.

## Data storage

Faust stores runtime data locally.

- `data/faust.db` — local LangGraph SQLite checkpoint database when SQLite persistence is enabled.
- `data/faust_memory.db` — local SQLite long-term memory database when SQLite long-term memory is enabled.
- `data/sessions/` — optional JSON session exports.
- `data/` — created automatically at runtime and ignored by version control.

This project is configured so local runtime data and personal memory artifacts are not committed to Git.

## Testing

The current automated baseline is **47+ passing tests**.

Recommended local validation commands:

```bash
python -m pytest -q
python -m pytest -vv
faust-tests
```

Using `python -m pytest` is the safest option inside virtual environments because it avoids shell PATH issues with the standalone `pytest` executable.

### Scoped test execution

The coder role supports explicit scoped test requests:

```text
Write code to add a regression test for location correction and run tests/adapters/test_graph.py::test_location_correction
```

Faust will execute the specified pytest target and return results with exit codes and output.

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
- The graph supports deterministic routing for a small supported set of memory facts.
- The current memory system is intentionally narrow and durable rather than broad and fuzzy.
- Role routing uses minimal shared-state fields and normalized execution outputs.
- Scoped test execution is constrained to explicit pytest targets under `tests/` for security.

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
- a long-term memory foundation,
- deterministic recall for supported user facts,
- tested implicit memory writes for simple self-fact statements,
- minimal multi-role routing for assistant, reasoner, and coder flows,
- and scoped pytest execution for code-oriented workflows.

This is an actively evolving local assistant project, with the current focus on durable architecture, reliable offline-first behavior, and practical multi-role graph routing.
