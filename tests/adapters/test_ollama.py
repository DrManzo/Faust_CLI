"""Integration test stub for OllamaAdapter — requires Ollama running locally."""

import pytest


@pytest.mark.asyncio
async def test_health_check_requires_ollama():
    """Requires a local Ollama instance. Run manually when available."""

    pytest.skip("Requires local Ollama — run manually with: pytest tests/adapters/")
