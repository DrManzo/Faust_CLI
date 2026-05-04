"""Run pytest and write a simple Markdown test report to reports/test-report.md."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import pytest


def run_tests_and_write_report(output_path: Path) -> int:
    """Run pytest programmatically and write a markdown report.

    Returns pytest's exit code.
    """
    # Run pytest and capture results via TestReportCollector
    # We'll use pytest's own reporting hooks by running it as a subprocess-like call.
    # For a lightweight approach, just run pytest -q and parse summary line.
    args: List[str] = ["-q"]
    print(f"[test-report] Running pytest {' '.join(args)}")

    # Run tests
    exit_code = pytest.main(args)

    # NOTE: pytest.main doesn't give us structured per-test results without
    # a plugin; keep this first version simple: we only record exit code.
    # You can expand this later with pytest plugins for richer detail.

    now = dt.datetime.now().isoformat(timespec="seconds")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Test Report",
        "",
        f"*Generated on: {now}*",
        "",
        "## Summary",
        "",
        f"- Exit code: `{exit_code}`",
        "",
        "A non-zero exit code means at least one test failed. "
        "Run `pytest -vv` for detailed output.",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[test-report] Wrote report to {output_path}")
    return int(exit_code)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    report_path = project_root / "reports" / "test-report.md"
    exit_code = run_tests_and_write_report(report_path)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
