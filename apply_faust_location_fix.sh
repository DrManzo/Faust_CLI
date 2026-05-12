#!/usr/bin/env bash
set -euo pipefail

FILE="src/faust/adapters/graph.py"
TEST_FILE="tests/adapters/test_graph.py"
BACKUP_DIR=".faust_patch_backup_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"
cp "$FILE" "$BACKUP_DIR/graph.py.bak"
if [ -f "$TEST_FILE" ]; then
  cp "$TEST_FILE" "$BACKUP_DIR/test_graph.py.bak"
fi

python <<'PY'
from pathlib import Path
import re
import sys

file_path = Path("src/faust/adapters/graph.py")
test_path = Path("tests/adapters/test_graph.py")

text = file_path.read_text()

text = text.replace(
    '""""LangGraph state graph for Faust."""',
    '"""LangGraph state graph for Faust."""',
)

if "from dataclasses import dataclass" not in text:
    anchor = "from faust.core.models import AppConfig, FaustState, MemoryRecord, Message, Role\n"
    if anchor not in text:
        sys.exit("Could not find import anchor for dataclass insertion.")
    text = text.replace(anchor, anchor + "\nfrom dataclasses import dataclass\n")

slot_class_pattern = re.compile(
    r"@dataclass\(frozen=True\)\nclass SlotSpec:\n(?:    .*\n)+?(?=\n\nSLOT_SPECS:)",
    re.MULTILINE,
)
slot_class_repl = """@dataclass(frozen=True)
class SlotSpec:
    slot: str
    category: str
    write_patterns: tuple[tuple[str, str | None], ...]
    recall_phrases: tuple[str, ...]
    stored_text_template: str
    response_template: str
    confirm_template: str
    score_phrases: tuple[str, ...] = ()
    priority: int = 200"""
text, count = slot_class_pattern.subn(slot_class_repl, text)
if count != 1:
    sys.exit("Failed to replace SlotSpec definition exactly once.")

slot_replacements = {
    """response_template="Your favorite editor is {value}.",
        score_phrases=("favorite editor", "preferred editor"),""":
    """response_template="Your favorite editor is {value}.",
        confirm_template="Okay — I'll remember that your favorite editor is {value}.",
        score_phrases=("favorite editor", "preferred editor"),""",

    """response_template="Your favorite shell is {value}.",
        score_phrases=("favorite shell", "preferred shell"),""":
    """response_template="Your favorite shell is {value}.",
        confirm_template="Okay — I'll remember that your favorite shell is {value}.",
        score_phrases=("favorite shell", "preferred shell"),""",

    """response_template="Your name is {value}.",
        score_phrases=("my name", "who am i"),""":
    """response_template="Your name is {value}.",
        confirm_template="Okay — I'll remember that your name is {value}.",
        score_phrases=("my name", "who am i"),""",

    """response_template="Your birthdate is {value}.",
        score_phrases=("birthdate", "birthday", "born"),""":
    """response_template="Your birthdate is {value}.",
        confirm_template="Okay — I'll remember that your birthdate is {value}.",
        score_phrases=("birthdate", "birthday", "born"),""",

    """response_template="Your location is {value}.",
        score_phrases=("location", "live", "from"),""":
    """response_template="Your location is {value}.",
        confirm_template="Okay — I'll remember that your location is {value}.",
        score_phrases=("location", "live", "from"),""",
}
for old, new in slot_replacements.items():
    text = text.replace(old, new)

extract_pattern = re.compile(
    r"def _extract_slot_value\(spec: SlotSpec, original: str, normalized: str\) -> str \| None:\n(?:    .*\n)+?(?=\n\ndef _detect_slot)",
    re.MULTILINE,
)
extract_repl = """def _extract_slot_value(spec: SlotSpec, original: str, normalized: str) -> str | None:
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
    return None"""
text, count = extract_pattern.subn(extract_repl, text)
if count != 1:
    sys.exit("Failed to replace _extract_slot_value exactly once.")

save_pattern = re.compile(
    r"def save_memory\(state: FaustState, \*, store\) -> dict:\n(?:    .*\n)+?(?=\n\ndef make_checkpointer)",
    re.MULTILINE,
)
save_repl = """def save_memory(state: FaustState, *, store) -> dict:
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
    }"""
text, count = save_pattern.subn(save_repl, text)
if count != 1:
    sys.exit("Failed to replace save_memory exactly once.")

file_path.write_text(text)

if test_path.exists():
    test_text = test_path.read_text()
    if "def test_location_correction_actually(" not in test_text:
        addition = """

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
        test_path.write_text(test_text + addition)

print("Patched src/faust/adapters/graph.py")
print("Patched tests/adapters/test_graph.py (if needed)")
PY

echo
echo "Backups saved in: $BACKUP_DIR"
echo "Now run:"
echo "  pytest tests/adapters/test_graph.py::test_location_correction_actually -q"
echo "  pytest tests/adapters/test_graph.py -q"
echo "  pytest -q"
