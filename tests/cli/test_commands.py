"""Tests for CLI command registration."""

from typer.testing import CliRunner

from faust.cli.app import app

runner = CliRunner()


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "faust" in result.output.lower()
