# Faust Test Report — Step 9 Checkpoint

**Status:** BASELINE  
**Step:** 9 — tests-only helper workflow  
**Date:** 2026-05-13  

## What was added

### New files
- `src/faust/core/testing.py` — narrow Step 9 helper module (pure functions, no subprocess, no production-code writes)
- `tests/core/test_testing.py` — full test coverage for the helper module

### Already present in the repo (Step 9 graph-level work)
- `src/faust/adapters/graph.py` — `test_proposal_node`, `run_requested_tests`, `should_run_requested_tests`, `_write_test_report`, `test_proposer` role routing, approval gate (`test_approved`)
- `src/faust/core/models.py` — `FaustState` extended with `test_proposal`, `test_approved`, `test_report_path`
- `tests/adapters/test_graph.py` — full graph-level Step 9 coverage including approval gate, report writing, scoped target validation, and no-subprocess guarantees

## Step 9 boundaries enforced

| Boundary | Location enforced |
|---|---|
| `test_proposal_node` never calls `subprocess.run` | `tests/adapters/test_graph.py::test_test_proposal_node_does_not_run_pytest` |
| `run_requested_tests` requires `test_approved=True` | `graph.py::should_run_requested_tests` + test `test_should_run_requested_tests_blocks_without_approval` |
| All pytest targets must start with `tests/` | `graph.py::run_requested_tests` + `core/testing.py::validate_pytest_targets` |
| Shell injection tokens are rejected | `core/testing.py::validate_pytest_targets` + `tests/core/test_testing.py::TestValidatePytestTargets` |
| `format_test_proposal` always sets `approved=False` | `tests/core/test_testing.py::TestFormatTestProposal::test_returns_approved_false` |
| `validate_pytest_targets` cannot accept `src/` paths | `tests/core/test_testing.py::TestProductionCodeSafety` |
| Report files are written only to `reports/` | `tests/adapters/test_graph.py::test_write_test_report_does_not_modify_production_code` |

## Memory preservation

All existing Step 1-8 memory tests remain in place:
- deterministic slot recall (favorite editor, shell, name, birthdate, location)
- saved-fact retrieval across threads for the same user
- user-scoped memory namespaces
- recalled memories injected into ephemeral state before prompt generation
- natural self-fact writes (implicit slot detection)

## Non-regression coverage

- `bare faust` interactive chat path: untouched
- `faust run` single-shot path: untouched
- `test_proposer` routes to `END`, not through `save_memory` (graph wiring verified)
- `test_approved` defaults to `False` in `make_state` test fixture

---

*Generated at Step 9 checkpoint — human-controlled production code, Faust handles tests and reports.*
