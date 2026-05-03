# Changelog

## [0.1.0] — Initial scaffold
- Component-based file structure: cli/, core/, adapters/, agents/
- Pydantic v2 data models: Message, Turn, Session, AppConfig, FaustState
- LangGraph base graph with build_prompt and llm nodes
- SQLite checkpointing design using langgraph-checkpoint-sqlite
- Ollama and OpenAI-compatible backend adapters
- Typer CLI: faust chat, faust run, faust config
- Rich terminal renderer
