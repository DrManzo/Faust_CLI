# Configuration Reference

File: `configs/default.yaml`

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | string | `llama3.3:8b` | Model name as registered in Ollama |
| `backend` | string | `ollama` | `ollama` or `openai_compat` |
| `temperature` | float | `0.7` | Sampling temperature (0.0-2.0) |
| `context_window` | int | `8192` | Max tokens kept in context history |
| `system_prompt` | string | see default.yaml | System prompt prepended to every session |
| `openai_compat.base_url` | string | `http://localhost:1234/v1` | Base URL for OpenAI-compatible server |
| `openai_compat.api_key` | string | `local` | API key (dummy value for local servers) |
