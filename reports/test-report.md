# Faust Test Report — Step 9 Full Validation

**Status:** BASELINE  
**Step:** 9 — 15-test live validation suite  
**Date:** 2026-05-13  

## Test suite: tests/core/test_step9_validation.py

| # | Name | Difficulty | What it validates |
|---|---|---|---|
| 1 | `test_detect_recall_slot_name` | Easy | `_detect_recall_slot` → `profile.name` |
| 2 | `test_detect_recall_slot_editor` | Easy | `_detect_recall_slot` → `preference.favorite_editor` |
| 3 | `test_is_memory_write_name` | Easy | `_is_memory_write` True for slot write |
| 4 | `test_is_memory_write_returns_false_for_question` | Easy | `_is_memory_write` False for recall |
| 5 | `test_classify_task_test_draft` | Easy | `classify_task` → `test_draft` |
| 6 | `test_classify_task_does_not_confuse_draft_with_coding` | Medium | Draft ≠ coding routing |
| 7 | `test_determine_role_selects_test_proposer` | Medium | `determine_role` → `test_proposer` |
| 8 | `test_memory_answer_node_returns_name_from_recall` | Medium | `memory_answer_node` grounded recall |
| 9 | `test_save_memory_persists_name_slot` | Medium | `save_memory` InMemoryStore write |
| 10 | `test_should_run_requested_tests_blocks_without_approval` | Medium | Approval gate → `save_memory` |
| 11 | `test_write_test_report_creates_file` | Hard | Report file created on pass |
| 12 | `test_write_test_report_with_fake_error` | Hard | Report contains fake AssertionError |
| 13 | `test_write_test_report_with_rejected_targets` | Hard | Rejected targets recorded in report |
| 14 | `test_test_proposal_node_does_not_run_subprocess` | Hard | `test_proposal_node` never runs pytest |
| 15 | `test_run_requested_tests_rejects_src_and_writes_report` | Hard | Full execution path with fake failure |

## Step 9 boundaries verified by this suite

- `test_proposal_node` never calls `subprocess.run` (Test 14)
- `run_requested_tests` approval gate blocks execution without `test_approved=True` (Test 10)
- `src/` targets are rejected by target validation (Test 15)
- Report files are written to `reports/` only, never to `src/` (Tests 11–13, 15)
- Fake error/failure is recorded faithfully in the report (Test 12, 15)
- Memory regression: slot recall, slot write, and grounded answer still work (Tests 1–4, 8–9)

---

*Run with: `pytest tests/core/test_step9_validation.py -v`*
