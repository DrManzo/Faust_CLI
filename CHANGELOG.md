# Changelog

All notable changes to Faust are documented here, step by step.

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

---

## Known issues

| Issue | Severity | Status |
|---|---|---|
| Piped stdin (`printf ... \| faust`) exits with a cosmetic `Aborted.` message after the last turn | Low | Tracked — targeted for CLI polish pass |
| `faust run` single-shot command does not yet persist to checkpoint | Low | Tracked |
