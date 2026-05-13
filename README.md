# Faust

> A local-first, multi-role AI CLI built on LangGraph — not a chatbot wrapper, a reasoning engine you can run on your own hardware.

Faust runs entirely offline. No API keys. No cloud dependency. You own the model, the memory, and the graph.

---

## What Faust actually does

Most "AI CLIs" are a prompt piped to an API. Faust is different.

Every turn runs through a compiled **LangGraph state machine** that classifies your intent, routes to the right agent, manages short- and long-term memory, and can propose and execute scoped pytest runs — all without touching the internet.

```text
you: plan out the architecture for a plugin system

Faust: [reasoner]
  Step 1 — Define plugin interface contract
  Step 2 — Add a plugin registry to AppConfig
  Step 3 — Lazy-load plugins at startup
  Step 4 — Write integration tests under tests/plugins/
```

```text
you: write the registry and run tests/plugins/test_registry.py

Faust: [coder]
  def PluginRegistry: ...
  → Scoped pytest run: 1 passed in 0.04s ✓
```

```text
you: remember that my preferred shell is zsh

Faust: Got it — I'll remember that your preferred shell is zsh.

you: what shell do I use?

Faust: [memory — deterministic, no LLM call]
  Your preferred shell is zsh.
```

---

## Architecture

```
Input
  └─► classify_task
        └─► determine_role
              └─► route_role ──► assistant ──► build_prompt ──► LLM stream
                            ├──► reasoner  ──► build_prompt ──► LLM stream
                            ├──► coder     ──► build_prompt ──► LLM stream ──► [scoped pytest]
                            ├──► test_proposer ──► proposal (no execution)
                            └──► memory_router ──► save / deterministic_recall / LLM
```

The graph is compiled once at startup and reused for all turns. State is scoped per thread and user. Every node writes to a shared `FaustState` dict — no hidden side channels.

---

## Features

### Multi-role graph routing
Faust classifies every input and routes it to the best agent path.

| Role | Triggered by | Does |
|---|---|---|
| `assistant` | General Q&A | Conversational response |
| `reasoner` | Planning, decomposition, "think through" | Step-by-step analysis |
| `coder` | Code, refactor, "write code for" | Code generation + optional scoped test run |
| `test_proposer` | "draft a test", "propose a test for" | Proposes test code — never executes without approval |
| `memory_writer` | "remember that...", implicit self-facts | Writes durable user fact to memory store |
| `memory_answer` | Supported slot recall questions | Answers directly from memory — zero LLM latency |

### Deterministic memory recall
For supported user facts, Faust answers **without calling the model at all**. No hallucination risk on your own data.

Supported memory slots:

| Slot | Written by | Example |
|---|---|---|
| `profile.name` | `I am Javier` / `my name is Javier` | `Who am I?` → `Your name is Javier.` |
| `profile.birthdate` | `my birthday is April 4th 1994` | `When was I born?` → `Your birthdate is April 4th 1994.` |
| `profile.location` | `I live in San Bernardino` | `Where do I live?` → `Your location is San Bernardino.` |
| `preference.favorite_editor` | `my favorite editor is Vim` | `What editor do I use?` → `Your favorite editor is Vim.` |
| `preference.favorite_shell` | `my favorite shell is zsh` | `Which shell do I prefer?` → `Your preferred shell is zsh.` |

Slot corrections with "actually" overwrite the previous value cleanly.

### Safe test proposal and execution
The coder and test_proposer roles separate **proposal** from **execution**:

- `test_proposer` drafts test code and returns it for human review — `subprocess.run` is never called.
- `coder` can execute scoped pytest targets when `test_approved=True` in state.
- All targets must live under `tests/`. Shell injection tokens (`;`, `&&`, `|`, `` ` ``) are rejected at the gate.
- `_write_test_report` writes only to `reports/` — never to `src/`.

### Three memory layers

| Layer | Scope | Backend |
|---|---|---|
| Ephemeral state | Single graph turn | In-memory dict |
| Checkpoint memory | Thread / conversation | SQLite or in-memory |
| Long-term memory | Cross-session user facts | SQLite or in-memory |

### Local-first, offline by design
- Default backend: [Ollama](https://ollama.com)
- OpenAI-compatible adapter included for any compatible local server
- No telemetry. No external calls during inference.

---

## Quickstart

### 1. Install

```bash
make install
```

Faust uses `uv` for fast, reproducible installs.

### 2. Pull models

```bash
ollama pull llama3:8b          # assistant
ollama pull deepseek-r1:8b     # reasoner
ollama pull qwen2.5-coder:14b  # coder
```

### 3. Run

```bash
faust                          # interactive session
faust run "explain LangGraph"  # single-shot prompt
```

---

## Recommended models

| Model | Role | Pull |
|---|---|---|
| `llama3:8b` | General assistant | `ollama pull llama3:8b` |
| `deepseek-r1:8b` | Reasoning and planning | `ollama pull deepseek-r1:8b` |
| `qwen2.5-coder:14b` | Code-oriented tasks | `ollama pull qwen2.5-coder:14b` |

---

## CLI reference

```bash
faust                             # Open interactive session (default)
faust run "<prompt>"              # Single-shot prompt, no session
faust config                      # Show resolved config
faust --help                      # Full command reference
```

### Keyboard shortcuts in interactive mode

| Input | Action |
|---|---|
| `quit` or `exit` | End session cleanly |
| `/exit` | Force exit |

---

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) running locally

---

## Test suite

```bash
pytest -q                        # fast baseline
pytest -v --tb=short             # verbose with tracebacks
pytest tests/adapters/test_graph.py -v   # graph layer only
```

Current automated baseline: **90 passing tests in < 0.6s** (graph suite, all in-memory).

---

## Data storage

All runtime data stays local and is excluded from version control.

```text
data/faust.db          ← LangGraph checkpoint database (SQLite)
data/faust_memory.db   ← Long-term memory store (SQLite)
data/sessions/         ← Optional JSON session exports
reports/               ← Scoped pytest reports written by coder node
```

---

## Project structure

```text
src/faust/
├── adapters/     ← Ollama + OpenAI adapters, graph.py
├── agents/       ← Agent node definitions
├── cli/          ← Typer CLI commands
└── core/         ← Models, config, memory, testing helpers

tests/
├── adapters/     ← Graph-level integration tests (90 tests)
├── cli/          ← CLI command tests
└── core/         ← Unit tests for core modules
```

---

## Development

```bash
# Add a new component via scaffolding
python scaffolding.py add-component <name>

# Run the full suite
make test

# Run only the graph suite
pytest tests/adapters/test_graph.py -v
```

---

## Design principles

- **Graph over glue.** Every routing decision is an explicit node in a compiled state machine, not an if-else chain in a callback.
- **Determinism over probability** for anything you already know. Supported memory recall never uses the LLM.
- **Proposal before execution.** Tests are drafted and shown to you before any subprocess fires.
- **Narrow and durable over broad and fuzzy.** Memory slots are explicit contracts, not embeddings or fuzzy search.
- **Local always.** Ollama-first. No outbound calls during inference.

---

## Status

Faust is an actively evolving local assistant project. The current foundation is intentionally stable — graph wiring, memory system, role routing, and test safety boundaries are all locked behind tests before new capabilities are added.

**Current focus:** CLI UX polish, expanded memory slots, and plugin architecture groundwork.
