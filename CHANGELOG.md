# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and follows a human-readable release structure with grouped sections such as Added, Changed, Fixed, and Security.

## [Unreleased]

### Added
- Added explicit memory bucket documentation to `FaustState` for checkpoint memory, long-term memory, and ephemeral state.
- Added minimal shared-state routing fields:
- `requested_role` for role selection
- `task_type` for task classification
- `ephemeral_context` for execution-local state
- `requested_tests` for scoped test targeting
- `execution_notes` for normalized output tracking
- Added minimal multi-role graph routing:
- `assistant` for general conversation
- `reasoner` for planning and decomposition
- `coder` for code generation and testing
- Added scoped pytest execution for code-oriented flows using explicit `tests/...` targets.
- Added live CLI smoke validation coverage for assistant conversation, deterministic memory write/recall, reasoner routing, and coder routing with scoped test execution.

### Changed
- Clarified the shared LangGraph state contract so all model roles read from the same `FaustState` structure and write back normalized outputs.
- Updated graph behavior so coding flows can attach scoped test requests and execute them before completion.
- Updated task classification so planning prompts route to the `reasoner` path before slot-keyword memory matches can incorrectly capture them.
- Preserved the Step 6 deterministic memory foundation while extending the graph with minimal role routing instead of introducing a broader agent framework.

### Fixed
- Fixed incorrect routing where planning prompts about memory-slot work could be classified as memory turns instead of reasoning turns.
- Fixed `run_requested_tests()` output normalization so scoped test execution returns consistent state data across success, invalid-target, timeout, and error branches.
- Fixed location-correction overwrite handling so lightweight correction phrasing like `actually` is stripped before slot extraction for supported profile memory writes.
- Fixed confirmation responses for supported slot writes so acknowledgments come from slot-specific templates rather than generic memory text.

### Security
- Kept scoped test execution constrained to explicit pytest targets under `tests/` and rejected unsafe or invalid command-like inputs.
- Preserved the narrow slot-based long-term memory model instead of expanding durable extraction into broader fuzzy memory capture.

### Testing
- Verified targeted graph tests for scoped pytest execution behavior.
- Verified graph module tests after Step 7 state-contract and routing updates.
- Verified full automated test suite after Step 7 changes.
- Verified live CLI smoke behavior for assistant responses, name memory write and recall, reasoner routing for planning prompts, and coder routing with real scoped pytest execution.

## [0.1.3] - 2026-05-11

### Added
- Added deterministic memory recall routing for supported user-fact questions before the normal LLM path.
- Added slot-backed long-term memory handling for `profile.name`, `profile.birthdate`, and `preference.favorite_editor`.
- Added user-scoped long-term memory namespaces.
- Added implicit durable memory writes for natural self-fact inputs such as `I am Javier`, `my birthday is April 4th 1994`, and `my favorite editor is Vim`.
- Added correction-aware overwrite handling for supported slot writes such as `my favorite editor is actually Vim`.
- Added expanded graph test coverage for deterministic recall routing, implicit self-fact writes, slot overwrites, and end-to-end graph recall behavior.

### Changed
- Strengthened graph routing so simple supported memory recall questions resolve through a deterministic memory-answer path instead of falling through to the normal assistant path.
- Updated memory write detection to support both explicit `remember ...` phrasing and narrow implicit self-fact statements for supported slots.
- Improved live CLI behavior so supported personal fact recall is more consistent and less likely to produce hallucinated answers.
- Preserved the Step 5 long-term memory foundation while tightening the routing and surface behavior layer rather than replacing the architecture.

### Fixed
- Fixed inconsistent routing where some simple recall turns could leak into the normal LLM path instead of using direct memory answers.
- Fixed slot overwrite behavior for supported preference/profile facts by normalizing lightweight correction phrasing like `actually`.
- Fixed graph/test alignment so memory-routing behavior is verified through a larger automated suite.

### Security
- Ignored local runtime data and stopped tracking personal memory/session artifacts in version control.
- Kept project-local runtime data under `data/` as a local-only, Git-ignored storage area.

### Testing
- Verified full automated test suite with 47 passing tests.
- Verified deterministic CLI memory behavior for name recall, birthdate recall, favorite editor recall, and open-ended follow-up prompts that still route through the normal model path when appropriate.

## [0.1.2] - 2026-05-11

### Added
- Added a configurable long-term memory store layer.
- Added user-scoped memory namespaces for durable memory separation.
- Added slot-based memory handling for `preference.favorite_editor`, `profile.name`, and `profile.birthdate`.
- Added explicit memory write handling for `remember that ...` style inputs.
- Added memory retrieval logic inside the LangGraph flow.
- Added deterministic memory-answer routing before the normal LLM path for the initial supported memory cases.

### Changed
- Extended the LangGraph architecture to include long-term memory retrieval and write support.
- Established the architectural foundation for memory-aware CLI behavior and future routing improvements.

### Fixed
- Updated and expanded tests to cover the initial long-term memory behavior and graph wiring.

### Testing
- Verified the Step 5 checkpoint and pushed the long-term memory foundation to GitHub.

## [0.1.1] - 2026-05-11

### Changed
- Updated the default backend configuration to use Ollama as the primary runtime backend.
- Standardized the default model around the local Ollama workflow used in the current implementation.
- Updated the test and developer workflow so validation can be run from the repository root with `python -m pytest -q`, `python -m pytest -vv`, and `faust-tests`.
- Updated `pyproject.toml` so pytest works correctly with the `src/` layout and console scripts install cleanly.
- Updated graph construction and tests so `build_graph(adapter, config)` receives the required config object during end-to-end execution.

### Fixed
- Fixed duplicate `pyproject.toml` script and pytest configuration issues that caused parsing and install failures.
- Fixed graph test execution by passing the required LangGraph checkpoint `thread_id` configuration.
- Fixed the end-to-end graph test mismatch after the `build_graph(adapter, config)` signature change.
- Fixed CLI tests to align with actual Rich/Typer output behavior during automated runs.
- Fixed the test report workflow so the report is written reliably to `reports/test-report.md`.

### Testing
- Verified local validation commands: `python -m pytest -q`, `python -m pytest -vv`, and `faust-tests`.

## [0.1.0] - Initial scaffold

### Added
- Component-based file structure: `cli/`, `core/`, `adapters/`, and `agents/`.
- Pydantic v2 data models: `Message`, `Turn`, `Session`, `AppConfig`, and `FaustState`.
- Initial LangGraph graph structure with `build_prompt` and `llm` nodes.
- Ollama and OpenAI-compatible backend adapter scaffolding.
- Typer CLI scaffold: `faust chat`, `faust run`, and `faust config`.
- Rich terminal renderer.
- Early SQLite checkpointing design planning.
