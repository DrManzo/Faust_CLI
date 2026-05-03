# How to Add a New LLM Adapter

## 1. Create the file

`src/faust/adapters/my_backend.py`

Or use the scaffolding tool:

```bash
python scaffolding.py add-adapter my_backend
```

## 2. Subclass LLMAdapter

```python
from faust.adapters.base import LLMAdapter
from faust.core.models import AppConfig, Message
from faust.exceptions import BackendError

class MyBackendAdapter(LLMAdapter):

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        # initialize your client here

    async def stream(self, messages: list[Message]):
        # yield response tokens one at a time
        # raise BackendError on failure
        ...

    async def health_check(self) -> bool:
        # return True if backend is reachable
        ...
```

## 3. Register in main.py

Add a case to the match config.backend block:

```python
case "my_backend":
    adapter = MyBackendAdapter(config)
```

## 4. Add config support

Set in configs/default.yaml:

```yaml
backend: "my_backend"
```
