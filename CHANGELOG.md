# Changelog

## [Step 9] — 2026-05-13

### Added
- `src/faust/core/testing.py` — Step 9 tests-only helper module
  - `detect_test_request` — classify user input as `test_draft`, `test_run`, or `none`
  - `format_test_proposal` — wrap LLM output into a reviewable proposal dict (always `approved=False`)
  - `validate_pytest_targets` — accept only safe `tests/` paths, reject shell injections
  - `build_approval_prompt` — human-readable approval prompt shown before any execution
  - `record_execution_note` — compact terminal note for state injection after a test run
- `tests/core/test_testing.py` — full coverage for the testing helper module
  - 30+ tests covering detection, proposal formatting, target validation, approval prompt, execution notes
  - `TestProductionCodeSafety` class guaranteeing no `src/` path can be accepted as a valid target
  - subprocess.run monkey-patch guard on `format_test_proposal`

### Already present from Step 9 graph-level work
- `adapters/graph.py` — `test_proposal_node`, `run_requested_tests`, `_write_test_report`,
  `should_run_requested_tests` (approval gate), `test_proposer` role routing
- `core/models.py` — `FaustState.test_proposal`, `.test_approved`, `.test_report_path`
- `tests/adapters/test_graph.py` — graph-level Step 9 coverage (50+ tests)

### Step 9 safety boundaries
- Faust proposes tests; humans approve.
- Pytest execution only fires when `test_approved=True` in state.
- All targets must start with `tests/`.
- Shell injection tokens (`;`, `&&`, `|`, `` ` ``, `$(`) are rejected.
- `test_proposal_node` never calls `subprocess.run`.
- `_write_test_report` writes only to `reports/`, never to `src/`.
- Production code implementation remains fully human-controlled.

---

## [Step 8] — Memory retrieval before prompt generation

- Saved-fact retrieval injected into ephemeral state before `build_prompt`.
- `retrieve_memories` populates `memory_hits` and `recalled_memories`.
- `_filter_relevant_memories` scores and ranks candidates by slot match and keyword overlap.

## [Step 7] — Shared-state formalization and multi-role routing

- `FaustState` formalized with bucket guide (checkpoint / long-term / ephemeral).
- `classify_task` → `determine_role` → `route_role` pipeline.
- `assistant`, `reasoner`, `coder` nodes wired.
- `should_run_requested_tests` approval gate first introduced.

## [Step 6] — Deterministic slot recall

- `SlotSpec` dataclass and `SLOT_SPECS` registry.
- Deterministic recall for `preference.favorite_editor`, `preference.favorite_shell`,
  `profile.name`, `profile.birthdate`, `profile.location`.
- `route_memory` / `memory_answer_node` short-circuit path.

## [Step 5] — Natural memory writes

- Implicit slot detection via `_detect_slot` + `_extract_slot_value`.
- `_is_memory_write` / `_extract_memory_candidate` helpers.
- `save_memory` node with slot-based overwrite semantics.

## [Step 4] — Durable long-term memory foundations

- `MemoryRecord` model, `MemorySettings`, SQLite and in-memory backends.
- `make_memory_store`, `_memory_namespace`, user-scoped namespaces.

## [Step 3] — Checkpoint persistence

- `make_checkpointer` with SQLite and in-memory backends.
- Thread-scoped resumable conversation state via LangGraph checkpointer.

## [Step 2] — LangGraph graph wiring

- `build_graph` with `StateGraph(FaustState)` and compiled checkpointer.
- Entry point, edges, and `END` wiring.

## [Step 1] — CLI and Ollama adapter

- Typer CLI with `faust` interactive chat and `faust run` single-shot command.
- `OllamaAdapter` with streaming `generate`.
- `AppConfig` loaded from `configs/default.yaml`.
