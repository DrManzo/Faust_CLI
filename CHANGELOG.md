# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), using human-readable sections so contributors can quickly understand what changed.

## [Unreleased]

### Added
- Added a working local CLI assistant flow backed by Ollama and LangGraph.
- Added configuration loading from `configs/default.yaml` into a validated `AppConfig`.
- Added `OllamaAdapter` support for local streaming chat completions via Ollama `/api/chat`.
- Added LangGraph graph wiring with:
  - `build_prompt` node to ensure a system prompt is injected,
  - `llm` node to collect streamed model output into a final response.
- Added in-memory LangGraph checkpointing using `InMemorySaver`.
- Added safe LangGraph serializer configuration using `JsonPlusSerializer` allow-list entries for Faust model types.
- Added Typer CLI support for:
  - `faust` → interactive chat by default,
  - `faust chat` → explicit interactive chat,
  - `faust run "<prompt>"` → single-shot prompt execution,
  - `faust <prompt...>` → root callback routing to single-shot execution.
- Added automated pytest coverage for:
  - core models,
  - config loading,
  - Ollama adapter behavior,
  - LangGraph graph wiring,
  - CLI command behavior,
  - session helpers.
- Added a `faust-tests` helper command to run pytest and generate a markdown test report.
- Added markdown test reporting to `reports/test-report.md`.

### Changed
- Updated the default backend configuration to use Ollama as the primary runtime backend.
- Standardized the default model around the local Ollama workflow used in the current implementation.
- Updated test and developer workflow so validation can be run from the repository root with:
  - `pytest -q`
  - `pytest -vv`
  - `faust-tests`
- Updated `pyproject.toml` so pytest works correctly with the `src/` layout and console scripts install cleanly.

### Fixed
- Fixed duplicate `pyproject.toml` script and pytest configuration issues that caused parsing and install failures.
- Fixed graph test execution by passing the required LangGraph checkpoint `thread_id` configuration.
- Fixed CLI tests to align with actual Rich/Typer output behavior during automated runs.
- Fixed the test report workflow so the report is written reliably to `reports/test-report.md`.

### Testing
- Current automated test baseline: 17 passing tests.
- Verified local commands:
  - `pytest -q`
  - `pytest -vv`
  - `faust-tests`

## [0.1.0] — Initial scaffold
- Component-based file structure: `cli/`, `core/`, `adapters/`, `agents/`
- Pydantic v2 data models: `Message`, `Turn`, `Session`, `AppConfig`, `FaustState`
- Initial LangGraph graph structure with `build_prompt` and `llm` nodes
- Ollama and OpenAI-compatible backend adapter scaffolding
- Typer CLI scaffold: `faust chat`, `faust run`, `faust config`
- Rich terminal renderer
- Early SQLite checkpointing design planning