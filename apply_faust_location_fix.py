from pathlib import Path

GRAPH = Path("src/faust/adapters/graph.py")
TESTS = Path("tests/adapters/test_graph.py")

text = GRAPH.read_text()

text = text.replace(
    '""""LangGraph state graph for Faust."""',
    '"""LangGraph state graph for Faust."""',
)

if "from dataclasses import dataclass" not in text:
    old = "from faust.core.models import AppConfig, FaustState, MemoryRecord, Message, Role\n"
    new = old + "\nfrom dataclasses import dataclass\n"
    if old not in text:
        raise SystemExit("Could not find import anchor.")
    text = text.replace(old, new, 1)

old_slot_spec = """@dataclass(frozen=True)
class SlotSpec:
    slot: str
    category: str
    write_patterns: tuple[tuple[str, str], ...]
    recall_phrases: tuple[str, ...]
    stored_text_template: str
    response_template: str
    score_phrases: tuple[str, ...] = ()
    priority: int = 200
"""

new_slot_spec = """@dataclass(frozen=True)
class SlotSpec:
    slot: str
    category: str
    write_patterns: tuple[tuple[str, str | None], ...]
    recall_phrases: tuple[str, ...]
    stored_text_template: str
    response_template: str
    confirm_template: str
    score_phrases: tuple[str, ...] = ()
    priority: int = 200
"""

if old_slot_spec in text:
    text = text.replace(old_slot_spec, new_slot_spec, 1)

slot_updates = [
    (
        '''        stored_text_template="The user's favorite editor is {value}.",
        response_template="Your favorite editor is {value}.",
        score_phrases=("favorite editor", "preferred editor"),''',
        '''        stored_text_template="The user's favorite editor is {value}.",
        response_template="Your favorite editor is {value}.",
        confirm_template="Okay — I'll remember that your favorite editor is {value}.",
        score_phrases=("favorite editor", "preferred editor"),''',
    ),
    (
        '''        stored_text_template="The user's favorite shell is {value}.",
        response_template="Your favorite shell is {value}.",
        score_phrases=("favorite shell", "preferred shell"),''',
        '''        stored_text_template="The user's favorite shell is {value}.",
        response_template="Your favorite shell is {value}.",
        confirm_template="Okay — I'll remember that your favorite shell is {value}.",
        score_phrases=("favorite shell", "preferred shell"),''',
    ),
    (
        '''        stored_text_template="The user's name is {value}.",
        response_template="Your name is {value}.",
        score_phrases=("my name", "who am i"),''',
        '''        stored_text_template="The user's name is {value}.",
        response_template="Your name is {value}.",
        confirm_template="Okay — I'll remember that your name is {value}.",
        score_phrases=("my name", "who am i"),''',
    ),
    (
        '''        stored_text_template="The user's birthdate is {value}.",
        response_template="Your birthdate is {value}.",
        score_phrases=("birthdate", "birthday", "born"),
        priority=220,''',
        '''        stored_text_template="The user's birthdate is {value}.",
        response_template="Your birthdate is {value}.",
        confirm_template="Okay — I'll remember that your birthdate is {value}.",
        score_phrases=("birthdate", "birthday", "born"),
        priority=220,''',
    ),
    (
        '''        stored_text_template="The user's location is {value}.",
        response_template="Your location is {value}.",
        score_phrases=("location", "live", "from"),''',
        '''        stored_text_template="The user's location is {value}.",
        response_template="Your location is {value}.",
        confirm_template="Okay — I'll remember that your location is {value}.",
        score_phrases=("location", "live", "from"),''',
    ),
]

for old, new in slot_updates:
    if old in text and "confirm_template" not in old:
        text = text.replace(old, new, 1)

old_extract = """def _extract_slot_value(spec: SlotSpec, original: str, normalized: str) -> str | None:
    for pattern, required_prefix in spec.write_patterns:
        match = re.match(pattern, original, flags=re.IGNORECASE)
        if not match:
            continue
        if required_prefix and not normalized.startswith(required_prefix):
            continue

        candidate = _strip_correction_prefixes(match.group(1))

        if spec.slot == "preference.favorite_editor" and "editor" not in normalized:
            if pattern.startswith(r"i prefer"):
                continue

        if spec.slot == "preference.favorite_shell" and "shell" not in normalized:
            if pattern.startswith(r"i prefer"):
                continue

        return candidate
    return None
"""

new_extract = """def _extract_slot_value(spec: SlotSpec, original: str, normalized: str) -> str | None:
    cleaned_original = re.sub(
        r"^(actually|no[, ]+|nope[, ]+|it's|it is)[,\\s]+",
        "",
        original,
        flags=re.IGNORECASE,
    ).strip()
    cleaned_normalized = _normalize_text(cleaned_original)

    for pattern, required_prefix in spec.write_patterns:
        match = re.match(pattern, cleaned_original, flags=re.IGNORECASE)
        if not match:
            continue
        if required_prefix and not cleaned_normalized.startswith(required_prefix):
            continue

        candidate = _strip_correction_prefixes(match.group(1))

        if spec.slot == "preference.favorite_editor" and "editor" not in cleaned_normalized:
            if pattern.startswith(r"i prefer"):
                continue

        if spec.slot == "preference.favorite_shell" and "shell" not in cleaned_normalized:
            if pattern.startswith(r"i prefer"):
                continue

        return candidate
    return None
"""

if old_extract not in text:
    raise SystemExit("Could not find _extract_slot_value block.")
text = text.replace(old_extract, new_extract, 1)

old_save = """def save_memory(state: FaustState, *, store) -> dict:
    \"\"\"Persist explicit or implicit durable memory requests.\"\"\"
    config = state["config"]
    user_input = state.get("user_input", "").strip()

    if not config.memory.enabled or store is None or not user_input:
        return {}

    slot, extracted_value = _extract_memory_candidate(user_input)
    if slot is None or not extracted_value:
        return {}

    namespace = _memory_namespace(state)
    now = datetime.now(timezone.utc)

    if slot == "preference.favorite_editor":
        stored_text = f"The user's favorite editor is {extracted_value}."
        key = slot
        category = "preference"
    elif slot == "preference.favorite_shell":
        stored_text = f"The user's favorite shell is {extracted_value}."
        key = slot
        category = "preference"
    elif slot == "profile.name":
        stored_text = f"The user's name is {extracted_value}."
        key = slot
        category = "profile"
    elif slot == "profile.birthdate":
        stored_text = f"The user's birthdate is {extracted_value}."
        key = slot
        category = "profile"
    else:
        normalized_text, normalized_category = _normalize_memory_text(user_input)
        stored_text = normalized_text
        key = str(uuid4())
        category = normalized_category

    record = MemoryRecord(
        key=key,
        slot=slot,
        text=stored_text,
        category=category,
        source="explicit",
        created_at=now,
        updated_at=now,
    )

    try:
        store.put(namespace, key, record.model_dump(mode="json"))
    except Exception:
        return {}

    recalled = [
        m
        for m in state.get("recalled_memories", [])
        if not (slot is not None and m.slot == slot)
    ]
    recalled.append(record)
    recalled = _dedupe_memories(recalled)

    if slot == "preference.favorite_editor":
        response = f"Okay — I'll remember that your favorite editor is {extracted_value}."
    elif slot == "preference.favorite_shell":
        response = f"Okay — I'll remember that your favorite shell is {extracted_value}."
    elif slot == "profile.name":
        response = f"Okay — I'll remember that your name is {extracted_value}."
    elif slot == "profile.birthdate":
        response = f"Okay — I'll remember that your birthdate is {extracted_value}."
    else:
        response = "Okay — I'll remember that."

    return {
        "recalled_memories": recalled,
        "response": response,
        "error": None,
        "intent": "memory_write",
        "active_agent": "memory_write",
        "task_type": "memory",
        "execution_notes": "Persisted durable memory record.",
        "requested_tests": [],
    }
"""

new_save = """def save_memory(state: FaustState, *, store) -> dict:
    \"\"\"Persist explicit or implicit durable memory requests.\"\"\"
    config = state["config"]
    user_input = state.get("user_input", "").strip()

    if not config.memory.enabled or store is None or not user_input:
        return {}

    slot, extracted_value = _extract_memory_candidate(user_input)
    if slot is None or not extracted_value:
        return {}

    namespace = _memory_namespace(state)
    now = datetime.now(timezone.utc)
    spec = _find_slot_spec(slot) if slot else None

    if spec is not None:
        stored_text = spec.stored_text_template.format(value=extracted_value)
        key = slot
        category = spec.category
    else:
        normalized_text, normalized_category = _normalize_memory_text(user_input)
        stored_text = normalized_text
        key = str(uuid4())
        category = normalized_category

    record = MemoryRecord(
        key=key,
        slot=slot,
        text=stored_text,
        category=category,
        source="explicit",
        created_at=now,
        updated_at=now,
    )

    try:
        store.put(namespace, key, record.model_dump(mode="json"))
    except Exception:
        return {}

    recalled = [
        m
        for m in state.get("recalled_memories", [])
        if not (slot is not None and m.slot == slot)
    ]
    recalled.append(record)
    recalled = _dedupe_memories(recalled)

    if spec is not None:
        response = spec.confirm_template.format(value=extracted_value)
    else:
        response = "Okay — I'll remember that."

    return {
        "recalled_memories": recalled,
        "response": response,
        "error": None,
        "intent": "memory_write",
        "active_agent": "memory_write",
        "task_type": "memory",
        "execution_notes": "Persisted durable memory record.",
        "requested_tests": [],
    }
"""

if old_save not in text:
    raise SystemExit("Could not find old save_memory block.")
text = text.replace(old_save, new_save, 1)

GRAPH.write_text(text)

if TESTS.exists():
    tests = TESTS.read_text()
    if "def test_location_correction_actually(" not in tests:
        tests += """

def test_location_correction_actually(store, mock_adapter, config):
    \"\"\"Correction with 'actually' should overwrite location slot.\"\"\"
    state = create_state(config, "I live in San Bernardino")
    result = save_memory(state, store=store)
    assert "San Bernardino" in result["response"]

    stored = store.get(("faust_memory", "default"), "profile.location")
    assert stored.value["text"] == "The user's location is San Bernardino."

    state2 = create_state(config, "Actually, I live in Los Angeles")
    result2 = save_memory(state2, store=store)
    assert "Los Angeles" in result2["response"]

    updated = store.get(("faust_memory", "default"), "profile.location")
    assert updated.value["text"] == "The user's location is Los Angeles."
"""
        TESTS.write_text(tests)

print("Patched graph.py and tests.")
