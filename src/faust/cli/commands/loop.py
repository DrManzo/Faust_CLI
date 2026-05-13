"""faust loop — approval-gated self-coding control loop.

This command implements the supervised self-improvement loop described
in Step 10 Phase 2.  Faust can draft narrow code changes for its own
repo, present them for human approval, and run only approved, scoped
pytest targets.

Safety constraints (NEVER relaxed):
  - No execution without explicit approval from Javier / Marcus.
  - No pytest targets outside the ``tests/`` directory.
  - No shell injection: targets validated against an allow-pattern
    before being handed to subprocess.
  - No silent production code writes outside the approved workflow.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from faust.cli.constants import _APPROVE_RE, _EXIT_TOKENS, _resolve_option
from faust.core.models import Message, Role, Session

console = Console()

# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

# Valid pytest node-id forms accepted by _is_safe_target:
#   tests/                            (full directory sweep)
#   tests/path/file.py                (file target)
#   tests/path/file.py::test_func     (single test)
#   tests/path/file.py::Class::method (class method)
# Rejects: absolute paths, ../traversal, shell metacharacters.
#
# Note: [a-zA-Z0-9_/\-]* uses * (zero-or-more) so bare 'tests/' is valid.
_SAFE_TARGET_RE = re.compile(
    r"^tests/[a-zA-Z0-9_/\-]*(?:\.py(?:::[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)?)?)?/?$"
)


def _is_safe_target(target: str) -> bool:
    """Return True only if *target* is a safe, scoped pytest path.

    Accepts:
      - tests/                               (directory sweep)
      - tests/path/file.py                   (file target)
      - tests/path/file.py::test_func        (single function)
      - tests/path/file.py::Class::method    (class method)

    Rejects:
      - Absolute paths
      - ../traversal
      - Shell metacharacters (; & | ` $ < > ! \\)
      - Anything not rooted at tests/
    """
    t = target.strip()
    if not t:
        return False
    if not t.startswith("tests/"):
        return False
    if re.search(r"[;&|`$<>\\!]", t):
        return False
    try:
        file_part = t.split("::")[0]
        resolved = Path(file_part).resolve()
        cwd = Path.cwd().resolve()
        resolved.relative_to(cwd)
    except ValueError:
        return False
    return bool(_SAFE_TARGET_RE.match(t))


# ---------------------------------------------------------------------------
# Node-id normalisation
# ---------------------------------------------------------------------------

_NODE_ID_RE = re.compile(
    r"(tests/[a-zA-Z0-9_/\-]+\.py(?:::[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)?)?)"
)


def _extract_node_ids(raw: list[str]) -> list[str]:
    """Extract valid pytest node-ids from a list of raw model-generated strings.

    For each entry:
    - If it is already a safe target, keep it as-is.
    - Otherwise try to extract an embedded node-id via regex scan.
    - If nothing valid is found, the entry is dropped and a warning is
      printed so the operator can see what was rejected and why.
    """
    result: list[str] = []
    for raw_target in raw:
        t = raw_target.strip()
        if _is_safe_target(t):
            result.append(t)
            continue
        m = _NODE_ID_RE.search(t)
        if m and _is_safe_target(m.group(1)):
            console.print(
                f"  [yellow]⚠[/yellow]  Extracted node-id from sloppy target: "
                f"[cyan]{m.group(1)}[/cyan]  [dim](original: {t!r})[/dim]"
            )
            result.append(m.group(1))
        else:
            console.print(
                f"  [red]✗[/red]  Rejected malformed target: [dim]{t!r}[/dim]\n"
                f"     [dim]Expected: tests/<path>.py[::test_name]  "
                f"Got: {t!r}[/dim]"
            )
    return result


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _make_report_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    return reports_dir / f"loop_{ts}.txt"


def _run_tests(targets: list[str]) -> tuple[int, str, Path]:
    report_path = _make_report_path()
    cmd = [sys.executable, "-m", "pytest"] + targets + ["-v", "--tb=short"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = result.stdout + result.stderr
    report_path.write_text(combined, encoding="utf-8")
    return result.returncode, combined, report_path


# ---------------------------------------------------------------------------
# Proposal rendering
# ---------------------------------------------------------------------------

def _display_proposal(proposal: dict) -> None:
    """Pretty-print a code or test proposal for human review.

    Normalises test_targets through _extract_node_ids() so that malformed
    model output is caught and reported BEFORE the operator sees the
    approval prompt — not after pytest fails with 'not found'.
    """
    description = proposal.get("description", "(no description)")
    code_diff = proposal.get("code_diff", "")
    raw_targets = proposal.get("test_targets") or []

    safe_targets = _extract_node_ids(raw_targets)

    console.print()
    console.print(
        Panel(
            f"[bold]Description:[/bold]\n{description}",
            title="🔧 Faust Proposal",
            border_style="yellow",
        )
    )

    if code_diff:
        console.print(
            Panel(
                Syntax(code_diff, "diff", theme="monokai", line_numbers=False),
                title="Proposed Code Change",
                border_style="cyan",
            )
        )

    if safe_targets:
        console.print(f"\n[bold]Test targets (approved scope):[/bold]")
        for t in safe_targets:
            console.print(f"  [green]✓[/green] {t}")
    elif raw_targets:
        console.print(
            "\n[yellow]No safe targets remain after normalisation. "
            "Nothing will run on approval.[/yellow]"
        )
    else:
        console.print("\n[dim]No test targets in this proposal.[/dim]")

    proposal["test_targets"] = safe_targets

    console.print()
    console.print(
        "[yellow]Type [bold]approve[/bold] / [bold]yes[/bold] / [bold]confirm[/bold] "
        "to run the approved test targets, or any other input to skip.[/yellow]"
    )
    console.print()


# ---------------------------------------------------------------------------
# Graph interaction helpers
# ---------------------------------------------------------------------------

def _ask_faust(
    graph,
    state: dict,
    user_input: str,
    thread_id: str,
    user_id: str,
) -> dict:
    state["messages"].append(Message(role=Role.USER, content=user_input))
    state["user_input"] = user_input
    state["intent"] = None
    state["active_agent"] = None
    state["requested_role"] = None
    state["task_type"] = None
    state["execution_notes"] = None
    state["error"] = None
    state["response"] = ""
    state["test_report_path"] = None
    state["test_approved"] = False
    state["requested_tests"] = []
    state["test_proposal"] = None

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        },
    )
    state.update(result)
    return state


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

def loop(
    ctx: typer.Context,
    thread_id: str = typer.Option(
        "loop", "--thread", "-t",
        help="Thread ID (defaults to 'loop' for continuity across sessions).",
    ),
    user_id: str = typer.Option(
        "default", "--user", "-u",
        help="Stable user ID for long-term memory.",
    ),
    task: str = typer.Option(
        "", "--task",
        help="Optional initial task description to seed the loop.",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show internal routing state after each turn."
    ),
) -> None:
    """Run the supervised self-coding control loop.

    Faust drafts code changes and test runs for human approval.
    Nothing executes until you explicitly approve it.

    Examples:
        faust loop
        faust loop --task "Fix the piped stdin exit bug"
        faust loop --user DrManzo --thread step10
    """
    obj = ctx.obj or {}
    config = obj.get("config")
    graph = obj.get("graph")

    if not config or not graph:
        typer.echo("Error: config or graph not initialized.", err=True)
        raise typer.Exit(1)

    thread_id = _resolve_option(thread_id, "loop")
    user_id = _resolve_option(user_id, "default")
    task = _resolve_option(task, "")

    session = Session(id=thread_id, model=config.model)

    state: dict = {
        "session": session,
        "config": config,
        "user_id": user_id,
        "user_input": "",
        "intent": None,
        "active_agent": None,
        "requested_role": None,
        "task_type": None,
        "requested_tests": [],
        "test_approved": False,
        "test_proposal": None,
        "test_report_path": None,
        "execution_notes": None,
        "messages": [],
        "recalled_memories": [],
        "artifacts": [],
        "response": "",
        "error": None,
    }

    models = config.models
    console.print(
        f"[bold green]Faust Loop[/bold green] — supervised self-coding mode\n"
        f"[dim]assistant:[/dim] [cyan]{models.default}[/cyan]  "
        f"[dim]coder:[/dim] [cyan]{models.coder}[/cyan]  "
        f"[dim]reasoner:[/dim] [cyan]{models.planner}[/cyan]"
    )
    console.print(f"[dim]Thread:[/dim] {thread_id}    [dim]User:[/dim] {user_id}")
    console.print("[dim]Nothing executes without your explicit approval.[/dim]\n")
    console.print(
        "Type [bold yellow]exit[/bold yellow] or [bold yellow]quit[/bold yellow] "
        "to end the loop.\n"
    )

    prompt_label = user_id if user_id and user_id != "default" else "you"
    _pending_proposal: dict | None = None

    if task.strip():
        console.print(f"[dim]Seeding loop with task:[/dim] {task.strip()}\n")
        try:
            state = _ask_faust(graph, state, task.strip(), thread_id, user_id)
            response = state.get("response", "")
            if response:
                console.print(f"\n[bold]Faust:[/bold] {response}\n")
                state["messages"].append(Message(role=Role.ASSISTANT, content=response))
            raw_proposal = state.get("test_proposal")
            if raw_proposal:
                _pending_proposal = (
                    raw_proposal if isinstance(raw_proposal, dict)
                    else {
                        "description": str(raw_proposal),
                        "test_targets": state.get("requested_tests") or [],
                    }
                )
                _display_proposal(_pending_proposal)
        except Exception as exc:
            console.print(f"[red]Error during task seed:[/red] {exc}")

    while True:
        try:
            user_input = typer.prompt(prompt_label)
        except (EOFError, KeyboardInterrupt, typer.Abort):
            console.print("\n[dim]Loop ended.[/dim]")
            break

        stripped = user_input.strip()

        if stripped.lower() in _EXIT_TOKENS:
            console.print("[dim]Goodbye.[/dim]")
            break

        if not stripped:
            continue

        if _pending_proposal and _APPROVE_RE.match(stripped):
            targets = _extract_node_ids(_pending_proposal.get("test_targets") or [])
            if not targets:
                console.print(
                    "[yellow]No safe test targets in this proposal — nothing to run.[/yellow]"
                )
                _pending_proposal = None
                continue

            console.print(f"[green]Running approved targets:[/green] {', '.join(targets)}")
            returncode, output, report_path = _run_tests(targets)

            status = "[green]PASSED[/green]" if returncode == 0 else "[red]FAILED[/red]"
            console.print(f"\n[bold]Tests:[/bold] {status}")
            console.print(f"[dim]Report written to:[/dim] {report_path}")
            tail_lines = output.strip().splitlines()[-20:]
            if tail_lines:
                console.print()
                console.print("[dim]--- last 20 lines ---[/dim]")
                for line in tail_lines:
                    console.print(f"[dim]{line}[/dim]")
                console.print("[dim]----------------------[/dim]\n")

            _pending_proposal = None
            continue

        try:
            state = _ask_faust(graph, state, stripped, thread_id, user_id)
        except Exception as exc:
            console.print(f"[red]Unexpected error:[/red] {exc}")
            continue

        response = state.get("response", "")
        error = state.get("error")

        if error:
            console.print(f"[red]Error:[/red] {error}")
        elif response:
            console.print(f"\n[bold]Faust:[/bold] {response}\n")
            state["messages"].append(Message(role=Role.ASSISTANT, content=response))
        else:
            console.print("[dim]No response received.[/dim]")

        raw_proposal = state.get("test_proposal")
        if raw_proposal:
            _pending_proposal = (
                raw_proposal if isinstance(raw_proposal, dict)
                else {
                    "description": str(raw_proposal),
                    "test_targets": state.get("requested_tests") or [],
                }
            )
            _display_proposal(_pending_proposal)
        else:
            _pending_proposal = None

        if debug:
            from faust.cli.commands.chat import _print_debug_state
            _print_debug_state(state)
            console.print()
