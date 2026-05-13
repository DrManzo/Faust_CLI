# Changelog

All notable changes to Faust are documented here, step by step.

---

## [Step 11] — 2026-05-13 — Agent Tests, Edit-and-Propose Cycle, Plugin Registry

Step 11 delivered in three strictly ordered phases. Phase 2 did not start until Phase 1 tests were passing. Phase 3 did not start until Phase 2 was exercised. All safety boundaries from Step 9 and Step 10 remain in force.

### Phase 1 — Agent-level unit tests

**`test(agents): add isolated unit tests for all five agent nodes`**

Filled `tests/agents/` with five new test modules — one per agent node. Each test stubs the LLM adapter, asserts on state fields written by the node (not just the response string), and is fully self-contained with no shared state between tests.

| File | Node under test | Tests |
|---|---|---|
| `test_assistant_node.py` | `assistant_node` | 4 |
| `test_reasoner_node.py` | `reasoner_node` | 4 |
| `test_coder_node.py` | `coder_node` | 4 |
| `test_test_proposer_node.py` | `test_proposal_node` | 4 |
| `test_memory_writer_node.py` | `save_memory` | 4 |

Key assertions per node:
- `assistant` — `response` content, `active_agent == "assistant"`, `execution_notes`, error capture
- `reasoner` — `response`, `active_agent == "reasoner"`, `intent == "reasoning"`, error capture
- `coder` — `response`, `active_agent == "coder"`, `intent == "coding"`, error capture
- `test_proposer` — `test_proposal == response`, `test_approved is False`, `subprocess.run` never called
- `memory_writer` — slot written, `category == "preference"`, slot overwrite semantics, `active_agent == "memory_write"`

Validation: `pytest tests/agents/ -v --tb=short` → **20 passed**

### Phase 2 — `faust loop` edit-and-propose cycle

**`feat(loop): add unified-diff proposal, write-on-approval, and scoped test runner`**

Extended `faust loop` so it can propose a real code change, render a unified diff in the proposal panel, write the file only after explicit approval, and then run the scoped test targets.

- `src/faust/loop_editor.py` — pure dataclass `EditProposal` + `build_unified_diff` + `write_approved_change`
  - `write_approved_change` rejects any target path outside `src/` — hard boundary, no exceptions
  - `build_unified_diff` uses stdlib `difflib.unified_diff` — no external dependency
- `src/faust/cli/commands/loop.py` — `_handle_edit_proposal`
  - Renders the unified diff in the proposal panel before anything is written
  - Gates on `_APPROVE_RE` (unchanged vocabulary: `approve` / `yes` / `confirm`)
  - On approval: calls `write_approved_change`, then `_run_tests`, then prints file written + pass/fail
  - On rejection: discards the proposal, nothing written, nothing run
  - If tests fail after the write: prints the failure clearly, does **not** auto-revert — human decides

**Safety invariants — unchanged:**
- No writes outside `src/`
- No test targets outside `tests/`
- No execution without explicit approval
- No shell injection acceptance
- Approval vocabulary unchanged

Validation:
- Manual smoke test: `faust loop --task "add a docstring to one function"` ✓
- `pytest tests/cli/test_exit_behavior.py -v --tb=short` → **27 passed**
- `pytest tests/cli/test_loop_editor.py -v --tb=short` → **21 passed**

### Phase 3 — Plugin / tool registry

**`feat(core): introduce PluginRegistry and built-in read_file / run_pytest tools`**

Introduced a `PluginRegistry` in `src/faust/core/plugins.py` so Faust can invoke defined tools as a first-class graph operation.

- `PluginEntry` dataclass — `name`, `fn`, `input_schema`, `requires_approval`, `description`
- `PluginRegistry` — dict-backed store with `register`, `unregister`, `get`, `list_names`, `dispatch`
  - `dispatch` raises `PermissionError` for tools with `requires_approval=True` unless `approval_override=True`
  - `dispatch` raises `KeyError` for unknown tool names — no silent fallback
- `_builtin_read_file` — reads a file under `src/`. Traversal outside `src/` raises `ValueError`.
- `_builtin_run_pytest` — runs scoped pytest targets through the existing `_safe_test_target` gate. `requires_approval=True`.
- `_safe_test_target` — validates that a target is a safe `tests/` path. Rejects traversal, shell tokens, and empty strings.
- `get_registry()` — module-level singleton pre-populated with `read_file` and `run_pytest`.

**`feat(graph): add tool_call node and route tool_call intent through classify_task / route_role`**

- `tool_call_node` added to `adapters/graph.py` — dispatches to `get_registry()`, resets `test_approved` after use, routes `PermissionError` to the approval prompt without opening the gate
- `classify_task` extended to classify `tool_call` intent
- `route_role` extended to dispatch `tool_call` to `tool_call_node`

**No speculative tools built.** Only `read_file` and `run_pytest` are implemented in this step.

Validation:
- `pytest tests/core/ -v --tb=short` → **21 plugin tests + all prior core tests passed**
- `pytest tests/adapters/test_graph.py -v --tb=short` → **82 passed**
- `pytest tests/adapters/test_tool_call_node.py -v --tb=short` → **13 passed**
- Manual smoke test: `faust loop --task "read src/faust/cli/constants.py and summarize it"` ✓

### Safety guarantees — unchanged from Step 9/10
- No execution without explicit operator approval
- No pytest targets outside `tests/`
- No shell metacharacter acceptance
- No writes outside `src/` (loop editor) or `reports/` (test runner)
- No silent production code writes outside the approved workflow

### Baseline after Step 11
- **261 passing tests** across all suites — < 0.60s, fully in-memory
- Breakdown: 82 graph, 13 tool_call, 20 agent isolation, 48 CLI/loop/editor, 21 plugin registry, remaining core

---

## [Step 10] — 2026-05-13 — CLI Polish, Controlled Loop, and Structural Cleanup

Step 10 delivered in three ordered phases: CLI reliability first, supervised self-coding loop second, structural refactor third. No phase started until the previous one was validated.

### Phase 1 — CLI Polish and Persistence Cleanup

**`fix(cli): handle piped stdin exit cleanly`**
- Added `_EXIT_TOKENS` frozen set (`exit`, `quit`, `q`, `/exit`) checked before any graph invocation.
- Added `typer.Abort` to the `except` clause on `typer.prompt` alongside `EOFError` and `KeyboardInterrupt` — this is the exact exception Typer raises when a pipe closes, eliminating the trailing `Aborted.` message.
- Exit tokens now short-circuit before a model turn is ever attempted.

**`fix(cli): persist faust run turns to checkpoint memory`**
- Documented the existing persistence mechanism clearly: same `--thread` = same checkpoint key = accumulated context across `faust run` calls.
- Clarified how to get stateless behavior (unique `--thread` per call).
- Added `re-raise typer.Exit` so piped callers receive correct exit codes on failure.
- Closes the known `faust run` persistence gap tracked since Step 9.

### Phase 2 — Controlled Loop

**`feat(loop): add approval-gated self-coding control loop`**
- New `src/faust/cli/commands/loop.py` — the supervised self-improvement loop.
- Faust drafts proposals (description + diff + test targets) and displays them for human review.
- Nothing executes until the operator types `approve`, `yes`, or `confirm`.
- Approved targets are passed to `pytest -v --tb=short` under the existing safety gate.
- Timestamped reports written to `reports/loop_YYYYMMDD_HHMMSS.txt`.
- Last 20 lines of output printed inline.

**`feat(cli): register loop command`**
- `loop_cmd` imported and registered as `faust loop` in `app.py`.

**`test(cli): add targeted exit-behavior tests`**
- New `tests/cli/test_exit_behavior.py` — 27 tests covering:
  - All exit token paths (interactive and loop).
  - Piped stdin exhaustion exits cleanly.
  - Post-turn `/exit` terminates with exactly one graph call.
  - `_is_safe_target` safety gate matrix.

### Phase 3 — Structural Refactor and Housekeeping

**`refactor(loop): extract shared constants + ground node-id proposals`**
- New `src/faust/cli/constants.py` — single source of truth for `_APPROVE_RE`, `_EXIT_TOKENS`, and `_resolve_option`. Both `chat.py` and `loop.py` now import from it.
- `_APPROVE_RE` extended to also match `confirm`.
- New `_extract_node_ids()` in `loop.py` — when the model returns sloppy output (e.g. `test_foo::test_bar` without a `tests/` prefix), the function scans for an embedded valid node-id, emits a visible warning, and uses it. Pure garbage is rejected with a clear message before the approval prompt. Eliminates the silent `not found` errors from pytest on malformed targets.
- `_is_safe_target` regex extended to accept full pytest node-id forms: `tests/path.py::test_func` and `tests/path.py::Class::method`.

**`test(loop): extend safety-gate tests to cover node-id normalisation`**
- `_is_safe_target` parametrize table extended with full node-id forms and malformed cases.
- Five new `_extract_node_ids` tests: valid pass-through, missing prefix rejection, embedded extraction, pure garbage rejection, empty/blank inputs.

**`fix(loop): allow bare tests/ directory sweep in _SAFE_TARGET_RE`**
- Changed `[a-zA-Z0-9_/\-]+` to `*` so `tests/` (full directory sweep) is accepted.
- Previously rejected due to `+` (one-or-more) requiring at least one character after the slash.

**`cleanup: resolve three pre-Step-11 housekeeping gaps`**
- `tests/test results/` renamed to `tests/test_results/` (space removed from directory name).
- `tests/agents/.gitkeep` and `tests/core/.gitkeep` added so both directories are git-tracked.
- `tests/cli/test_renderer.py` added — four tests covering all public functions in `renderer.py`.

**`fix(tests): align stale test stubs to current AppConfig shape`**
- `FakeOllamaConfig` in `test_ollama.py` updated to expose `.models` `SimpleNamespace` matching the multi-model `ModelConfig` shape.
- `test_app_config_defaults` updated from `llama3:8b` to `llama3.3:8b` to match the current `AppConfig` default.

### Safety guarantees — unchanged
- No execution without explicit operator approval.
- No pytest targets outside `tests/`.
- No shell metacharacter acceptance.
- No silent production code writes outside the approved workflow.

### Baseline after Step 10
- **186 passing tests** across all suites — 0.65s, fully in-memory.

---

## [Step 9] — 2026-05-13 — Test Safety System

Faust can now **propose** and **execute** scoped pytest runs as a first-class graph operation — with a hard approval gate that keeps humans in control.

### Added

**`src/faust/core/testing.py`** — testing helper module
- `detect_test_request` — classifies user input as `test_draft`, `test_run`, or `none`
- `format_test_proposal` — wraps LLM output into a reviewable proposal dict, always `approved=False`
- `validate_pytest_targets` — accepts only safe `tests/` paths, rejects shell injection tokens
- `build_approval_prompt` — human-readable approval prompt before any execution
- `record_execution_note` — compact state note after a test run

**Graph nodes and routing (`adapters/graph.py`)**
- `test_proposal_node` — drafts test code, never calls `subprocess.run`
- `run_requested_tests` — executes scoped pytest targets, writes reports to `reports/`
- `should_run_requested_tests` — approval gate: `test_approved=True` required to execute
- `testproposer` role added to `classify_task`, `determine_role`, and `route_role`

**`core/models.py`**
- `FaustState` extended: `test_proposal`, `test_approved`, `test_report_path`

**Tests**
- `tests/core/test_testing.py` — 30+ tests covering all helper functions
- `tests/adapters/test_graph.py` — graph-level Step 9 coverage
- `TestProductionCodeSafety` class: guarantees no `src/` path can be accepted as a test target

### Safety guarantees
- Test proposals and test execution are **separate graph nodes** — one never implies the other
- All execution targets must start with `tests/`
- Shell injection tokens (`;`, `&&`, `|`, `` ` ``, `$()`) are rejected at validation
- `_write_test_report` writes only to `reports/`, never touches `src/`
- Production code implementation is always human-controlled

### Baseline after Step 9
- **90 passing tests** in `tests/adapters/test_graph.py` — 0.48s, fully in-memory

---

## [Step 8] — Memory Retrieval Pipeline

Relevant long-term memories are now injected into the prompt **before** the LLM sees the input, not after.

### Added
- `retrieve_memories` node — runs before `build_prompt`, populates `memory_hits` and `recalled_memories`
- `_filter_relevant_memories` — scores and ranks candidates by slot match and keyword overlap
- `memory_query` and `memory_hits` added to `FaustState` for this-turn memory context
- `build_prompt` updated to inject recalled memories into the system message

---

## [Step 7] — Multi-Role Graph Routing + Favorite Shell Slot

Formal multi-agent routing pipeline with three agent roles and the shell preference memory slot.

### Added
- `FaustState` formalized with bucket documentation (checkpoint / long-term / ephemeral fields)
- `classify_task` — classifies input as `general`, `coding`, `reasoning`, `memory`, or `test_draft`
- `determine_role` — maps task type to agent role
- `route_role` — conditional edge dispatcher for the compiled graph
- `assistant`, `reasoner`, `coder` nodes fully wired with execution notes
- `should_run_requested_tests` approval gate introduced (first version)
- `preference.favorite_shell` memory slot: write, recall, and correction

---

## [Step 6] — Deterministic Slot Recall

For supported user facts, Faust answers directly from memory — no LLM call, no hallucination risk.

### Added
- `SlotSpec` dataclass and `SLOT_SPECS` registry
- `detect_recall_slot` — maps recall phrasings to slot keys
- `has_recalled_slot` — checks ephemeral state for a loaded slot value
- `route_memory` — dispatches to `memory_write`, `memory_recall`, or `role_router`
- `memory_answer_node` — short-circuit path for deterministic recall
- Deterministic slots: `preference.favorite_editor`, `preference.favorite_shell`, `profile.name`, `profile.birthdate`, `profile.location`

---

## [Step 5] — Natural Language Memory Writes

Faust persists user facts from natural self-fact statements without requiring explicit `remember that` phrasing.

### Added
- `_detect_slot` and `_extract_slot_value` — pattern-based slot detection
- `_is_memory_write` / `_extract_memory_candidate` — implicit write classifiers
- `save_memory` node with slot-based overwrite semantics
- Support for natural phrasings:
  - `I am Javier` / `my name is Javier`
  - `my birthday is April 4th 1994`
  - `my favorite editor is Vim` / `my favorite editor is actually Neovim`
  - `my favorite shell is zsh`
  - `I live in San Bernardino` / `Actually, I live in Los Angeles`

---

## [Step 4] — Durable Long-Term Memory Foundations

Persistent, user-scoped memory store with a clean interface over two backends.

### Added
- `MemoryRecord` model — typed record with key, slot, text, category, source, timestamps
- `MemorySettings` — configurable backend selection
- `make_memory_store` — factory returning SQLite or in-memory store
- `_memory_namespace` — user-scoped namespace helper
- SQLite backend for durable cross-session persistence
- In-memory backend for fast test execution (no disk I/O)

---

## [Step 3] — Checkpoint Persistence

Conversation state is now resumable across turns via LangGraph checkpointing.

### Added
- `make_checkpointer` — factory returning SQLite or in-memory LangGraph checkpointer
- Thread-scoped conversation state: same `thread_id` resumes where it left off
- SQLite backend (`data/faust.db`) for local durable sessions
- In-memory backend for test isolation (no bleed between test threads)
- `uuid4()`-based thread IDs in tests to prevent checkpoint contamination

---

## [Step 2] — LangGraph Graph Wiring

The core state machine is compiled and wired. All subsequent steps build on this foundation.

### Added
- `build_graph` — compiles a `StateGraph(FaustState)` with checkpointer injected
- Entry point, edges, and `END` termination wired
- `FakeAdapter` introduced for fast in-memory graph tests
- `FaustState` initial definition

---

## [Step 1] — CLI and Ollama Adapter

Faust boots from the command line and talks to a local model.

### Added
- Typer CLI with `faust` (interactive) and `faust run` (single-shot) commands
- Default entry to `chat_cmd` via callback context (`ctx.invoke`) — no subcommand required
- `OllamaAdapter` with streaming `generate` method
- `AppConfig` loaded from `configs/default.yaml`
- Interactive REPL loop with `quit`/`exit` termination
- Debug mode output showing active agent, role, and task type per turn
