# Faust — Architecture Reference

## Components

| Component | Folder | Changes When |
|---|---|---|
| Interface & Rendering | `cli/` | Commands, flags, or output style change |
| Business Logic & Models | `core/` | Session behavior, prompt logic, or data models change |
| LLM Backend Bridges | `adapters/` | A new backend is added or the graph is modified |
| AI Agents | `agents/` | A new agent is added (Phase 2) |
| Data & Checkpoints | `data/` | Local SQLite DB and session exports |

## Import Rule — Never Break This

cli/      → may import from: core/, adapters/
adapters/ → may import from: core/ only (no cli/)
core/     → imports NOTHING from cli/, adapters/, or agents/
agents/   → may import from: core/, adapters/ (Phase 2)

## LangGraph State and Node Contract

Faust uses LangGraph with a typed state object (FaustState). Every node:

1. Accepts the current FaustState.
2. Returns a partial state (only the keys to update).
3. Lets LangGraph merge that partial state into the existing state.

Nodes never replace the full state, they only update their own fields.

## Checkpointing and Storage

Faust uses LangGraph's SqliteSaver to store graph checkpoints in a
local SQLite database located at `data/faust.db`. This enables:

- Pause/resume of conversations
- Replay and inspection of state transitions
- Persistent history across runs

Sessions can optionally be exported as JSON files under `data/sessions/`.
