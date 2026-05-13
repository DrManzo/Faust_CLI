# Faust

> A local-first, multi-role AI CLI built on LangGraph — not a chatbot wrapper, a reasoning engine you can run on your own hardware.

Faust runs entirely offline. No API keys. No cloud dependency. You own the model, the memory, and the graph.

---

## What Faust actually does

Most “AI CLIs” are a prompt piped to an API. Faust is different.

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

Faust: Got it — I’ll remember that your preferred shell is zsh.

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
| `reasoner` | Planning, decomposition, “think through” | Step-by-step analysis |
| `coder` | Code, refactor, “write code for” | Code generation + optional scoped test run |
| `test_proposer` | “draft a test”, “propose a test for” | Proposes test code — never executes without approval |
| `memory_writer` | “remember that…”, implicit self-facts | Writes durable user fact to memory store |
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

Slot corrections with “actually” overwrite the previous value cleanly.

### Supervised self-coding loop

Faust can work on its own codebase through a controlled loop with a hard approval gate before anything runs.

```bash
faust loop --task "Fix the piped stdin exit bug"
```

```text
Faust: [loop]
  Proposal: Catch typer.Abort on exhausted stdin in chat.py
  Diff: ...
  Test targets: tests/cli/test_exit_behavior.py::test_chat_piped_stdin_exhaustion_exits_cleanly

  Type approve / yes / confirm to run, or anything else to skip.

you: approve

Faust: Running approved targets: tests/cli/test_exit_behavior.py::...
  Tests: PASSED
  Report written to: reports/loop_20260513_202518.txt
```

Nothing executes until you explicitly approve. No targets outside `tests/` are accepted.

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
ollama pull llama3.3:8b        # assistant (current default)
ollama pull deepseek-r1:8b     # reasoner
ollama pull qwen2.5-coder:14b  # coder
```

### 3. Run

```bash
faust                          # interactive session
faust run "explain LangGraph"  # single-shot prompt
faust loop                     # supervised self-coding loop
```

---

## Recommended models

| Model | Role | Pull |
|---|---|---|
| `llama3.3:8b` | General assistant | `ollama pull llama3.3:8b` |
| `deepseek-r1:8b` | Reasoning and planning | `ollama pull deepseek-r1:8b` |
| `qwen2.5-coder:14b` | Code-oriented tasks | `ollama pull qwen2.5-coder:14b` |

---

## CLI reference

```bash
faust                             # Open interactive session (default)
faust run "<prompt>"              # Single-shot prompt, persisted to --thread
faust loop                        # Supervised self-coding control loop
faust loop --task "<task>"        # Seed the loop with an initial task
faust loop --thread step11        # Named thread for loop continuity
faust config                      # Show resolved config
faust --help                      # Full command reference
```

### Keyboard shortcuts in interactive mode

| Input | Action |
|---|---|
| `quit`, `exit`, or `q` | End session cleanly |
| `/exit` | Force exit (also works in loop) |

---

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) running locally

---

## Test suite

```bash
pytest tests/ -v --tb=short      # full suite (186 tests)
pytest tests/adapters/test_graph.py -v   # graph layer only (90 tests)
pytest tests/cli/ -v             # CLI and loop tests
```

Current automated baseline: **186 passing tests in < 0.65s**, all in-memory.

---

## Data storage

All runtime data stays local and is excluded from version control.

```text
data/faust.db          ← LangGraph checkpoint database (SQLite)
data/faust_memory.db   ← Long-term memory store (SQLite)
data/sessions/         ← Optional JSON session exports
reports/               ← Scoped pytest reports written by coder and loop nodes
```

---

## Project structure

```text
src/faust/
├── adapters/     ← Ollama + OpenAI adapters, graph.py
├── agents/       ← Agent node definitions
├── cli/          ← Typer CLI commands (chat, loop, run, config)
│   ├── commands/   ← chat.py, loop.py, run.py
│   └── constants.py ← Shared exit tokens, approval regex, option resolver
└── core/         ← Models, config, memory, testing helpers

tests/
├── adapters/     ← Graph + Ollama adapter tests (92 tests)
├── agents/       ← Agent-level tests (Step 11+)
├── cli/          ← CLI, loop, renderer tests (36 tests)
├── core/         ← Config, models, session, memory, testing (58 tests)
└── test_results/ ← Test run artifacts
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

# Run the loop in supervised mode
faust loop --task "<describe what you want Faust to work on>"
```

---

## Design principles

- **Graph over glue.** Every routing decision is an explicit node in a compiled state machine, not an if-else chain in a callback.
- **Determinism over probability** for anything you already know. Supported memory recall never uses the LLM.
- **Proposal before execution.** Tests are drafted and shown to you before any subprocess fires.
- **Narrow and durable over broad and fuzzy.** Memory slots are explicit contracts, not embeddings or fuzzy search.
- **Approval gates, not trust.** The loop never runs code without explicit human confirmation. Every production change goes through you.
- **Local always.** Ollama-first. No outbound calls during inference.

---

## Status

Faust is an actively evolving local assistant project. The current foundation is intentionally stable — graph wiring, memory system, role routing, and test safety boundaries are all locked behind tests before new capabilities are added.

**Step 10 complete.** CLI exit behavior is clean across all modes (interactive, piped, single-shot). The supervised self-coding loop (`faust loop`) is live with a hard approval gate. All 186 tests pass. The repo is clean and ready for Step 11.
