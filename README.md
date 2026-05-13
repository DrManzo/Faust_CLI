# Faust

> A local-first, multi-role AI CLI built on LangGraph — not a chatbot wrapper, a reasoning engine you can run on your own hardware.

Faust runs entirely offline. No API keys. No cloud dependency. You own the model, the memory, and the graph.

---

## What Faust actually does

Most "AI CLIs" are a prompt piped to an API. Faust is different.

Every turn runs through a compiled **LangGraph state machine** that classifies your intent, routes to the right agent, manages short- and long-term memory, proposes and writes scoped code changes, and can invoke typed tools — all without touching the internet.

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
you: read src/faust/cli/constants.py and summarize it

Faust: [tool_call → read_file]
  constants.py defines CLI_VERSION, approval regex, exit tokens,
  and the _resolve_option helper. No approval required.
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
              └─► route_role ──► assistant    ──► build_prompt ──► LLM stream
                            ├──► reasoner     ──► build_prompt ──► LLM stream
                            ├──► coder        ──► build_prompt ──► LLM stream ──► [scoped pytest]
                            ├──► test_proposer ──► proposal (no execution)
                            ├──► memory_router ──► save / deterministic_recall / LLM
                            └──► tool_call    ──► PluginRegistry.dispatch
                                                    ├──► read_file   (no approval)
                                                    └──► run_pytest  (approval required)
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
| `memory_writer` | "remember that…", implicit self-facts | Writes durable user fact to memory store |
| `memory_answer` | Supported slot recall questions | Answers directly from memory — zero LLM latency |
| `tool_call` | "read …", "run pytest …" | Dispatches to a registered plugin via PluginRegistry |

### Plugin / tool registry

Faust now supports first-class tool invocations through a typed `PluginRegistry`.

```python
from faust.core.plugins import get_registry

registry = get_registry()
result = registry.dispatch("read_file", {"path": "src/faust/cli/constants.py"}, approval_override=False)
```

Built-in tools in Step 11:

| Tool | Requires approval | Description |
|---|---|---|
| `read_file` | No | Reads a file under `src/`. Path traversal outside `src/` is rejected. |
| `run_pytest` | **Yes** | Runs scoped pytest targets. Must pass through the same approval gate as test execution. |

Tools that require approval are blocked by `PermissionError` in `dispatch` unless `approval_override=True` — which is only set after the operator explicitly approves. No speculative tools are built.

### `faust loop` — edit-and-propose cycle

Faust can propose real code changes, show you a unified diff, and write the file **only after you approve**.

```bash
faust loop --task "add a docstring to one function"
```

```text
Faust: [loop — edit proposal]
  File: src/faust/cli/commands/loop.py
  --- a/src/faust/cli/commands/loop.py
  +++ b/src/faust/cli/commands/loop.py
  @@ -42,6 +42,11 @@
   def _run_tests(targets):
  +    """Execute scoped pytest targets and return the CompletedProcess result."""

  Test targets: tests/cli/test_loop_editor.py

  Type approve / yes / confirm to write and run, or anything else to discard.

you: approve

Faust: File written: src/faust/cli/commands/loop.py
  Running: tests/cli/test_loop_editor.py
  Tests: PASSED
  Report: reports/loop_20260513_214502.txt
```

On rejection: nothing is written, nothing is run. If tests fail after the write, the failure is printed clearly — Faust never auto-reverts. The human decides.

**Safety boundaries (unchanged from Step 9/10):**
- No writes outside `src/`
- No test targets outside `tests/`
- No execution without explicit approval
- No shell injection acceptance (`;`, `&&`, `|`, `` ` ``, `$()`)
- Approval vocabulary: `approve` / `yes` / `confirm`

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
- All targets must live under `tests/`. Shell injection tokens are rejected at the gate.
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
pytest tests/ -v --tb=short                          # full suite (261 tests)
pytest tests/agents/ -v                              # agent node isolation (20 tests)
pytest tests/adapters/test_graph.py -v               # graph layer (82 tests)
pytest tests/cli/ -v                                 # CLI, loop, and editor tests (48 tests)
pytest tests/core/ -v                                # core + plugin registry tests
pytest tests/adapters/test_tool_call_node.py -v      # tool_call node (13 tests)
```

Current automated baseline: **261 passing tests in < 0.60s**, all in-memory.

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
├── adapters/     ← Ollama + OpenAI adapters, graph.py (incl. tool_call node)
├── agents/       ← Agent node definitions
├── cli/          ← Typer CLI commands (chat, loop, run, config)
│   ├── commands/   ← chat.py, loop.py, run.py
│   └── constants.py ← Shared exit tokens, approval regex, option resolver
└── core/         ← Models, config, memory, testing helpers, plugins.py
    └── plugins.py  ← PluginRegistry, PluginEntry, read_file, run_pytest

tests/
├── adapters/     ← Graph + Ollama + tool_call node tests
├── agents/       ← Agent-level isolation tests (Step 11 — 20 tests)
├── cli/          ← CLI, loop, editor, renderer tests (48 tests)
├── core/         ← Config, models, session, memory, testing, plugins
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

# Run agent isolation tests
pytest tests/agents/ -v

# Run the loop in supervised mode
faust loop --task "<describe what you want Faust to work on>"
```

---

## Design principles

- **Graph over glue.** Every routing decision is an explicit node in a compiled state machine, not an if-else chain in a callback.
- **Determinism over probability** for anything you already know. Supported memory recall never uses the LLM.
- **Proposal before execution.** Code changes are shown as diffs and tests are drafted for review before any subprocess fires.
- **Narrow and durable over broad and fuzzy.** Memory slots are explicit contracts, not embeddings or fuzzy search.
- **Approval gates, not trust.** The loop never writes or runs code without explicit human confirmation. Every production change goes through you.
- **Typed tools, not magic.** The PluginRegistry enforces input schemas and approval requirements — no tool executes as a side effect.
- **Local always.** Ollama-first. No outbound calls during inference.

---

## Status

Faust is an actively evolving local assistant project. The current foundation is intentionally stable — graph wiring, memory system, role routing, plugin registry, and test safety boundaries are all locked behind tests before new capabilities are added.

**Step 11 complete.** Agent-level unit tests, the `faust loop` edit-and-propose cycle (unified diff → approval → write → test run), and the `PluginRegistry` with `tool_call` graph node are all live. **261 tests pass, 0 failures.** All Step 9 and Step 10 safety boundaries remain in force.
