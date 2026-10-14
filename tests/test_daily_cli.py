from __future__ import annotations

import json

from typer.testing import CliRunner

import cli.daily as daily
import cli.main as main


def test_daily_dry_run_and_report_do_not_run_interactive_or_paid_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "run_analysis", lambda **kw: (_ for _ in ()).throw(AssertionError("interactive")))
    monkeypatch.setattr(daily.DailyRunner, "tick", lambda *a: (_ for _ in ()).throw(AssertionError("paid")))
    runner = CliRunner()
    database = str(tmp_path / "daily.sqlite3")
    result = runner.invoke(main.app, ["daily", "tick", "nvda", "--database", database, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ticker"] == "NVDA"
    result = runner.invoke(main.app, ["daily", "report", "NVDA", "--database", database, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["all"]["scored"] == 0 and data["all"]["absolute_error"] is None


def test_daily_errors_are_nonzero_and_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(daily.DailyRunner, "tick", lambda *a: ([], ["missing close"]))
    result = CliRunner().invoke(main.app, ["daily", "tick", "--database", str(tmp_path / "d.sqlite3")])
    assert result.exit_code == 1 and "ERROR (will retry): missing close" in result.output


def test_unsupported_ticker_and_invalid_delay_are_rejected(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main.app, ["daily", "tick", "0700.HK", "--database", str(tmp_path / "d.sqlite3")])
    assert result.exit_code == 1 and "US stock" in result.output
    result = runner.invoke(main.app, ["daily", "tick", "--delay-minutes", "0"])
    assert result.exit_code == 2


def test_help_exposes_daily_commands():
    runner = CliRunner()
    assert "daily" in runner.invoke(main.app, ["--help"]).output
    result = runner.invoke(main.app, ["daily", "--help"])
    assert result.exit_code == 0 and "tick" in result.output and "report" in result.output
