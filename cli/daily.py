"""Non-interactive daily commands; importing them never starts jobs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from filelock import Timeout

from tradingagents.daily.market import TradingCalendar, us_ticker
from tradingagents.daily.reporting import export_reports, render_summary, summarize
from tradingagents.daily.runner import DailyRunner
from tradingagents.daily.store import DailyStore
from tradingagents.default_config import DEFAULT_CONFIG

app = typer.Typer(help="Live US premarket forecasts, postmarket reflection, and evaluation.")
DEFAULT_DATABASE = Path.home() / ".tradingagents" / "daily" / "daily.sqlite3"
logger = logging.getLogger(__name__)


@app.command()
def tick(
    ticker: str = typer.Argument("NVDA", help="US stock/ETF, Yahoo symbol (e.g. NVDA, BRK-B)"),
    database: Annotated[Path, typer.Option("--database")] = DEFAULT_DATABASE,
    calendar: str = typer.Option("NASDAQ", "--calendar", help="NASDAQ or NYSE"),
    lead_minutes: int = typer.Option(60, "--lead-minutes", min=5, max=240),
    delay_minutes: int = typer.Option(30, "--delay-minutes", min=15, max=240),
    control: bool = typer.Option(False, "--control", help="Also run a separate no-memory graph; adds LLM cost"),
    language: str = typer.Option("Simplified Chinese", "--language"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show schedule; no market-data or LLM calls"),
):
    """Run only due work, safely repeatable by a five-minute system timer."""
    try:
        ticker = us_ticker(ticker)
        store = DailyStore(database)
        schedule = TradingCalendar(calendar)
        runner = DailyRunner(
            {**DEFAULT_CONFIG, "output_language": language}, store, schedule,
            lead_minutes=lead_minutes, delay_minutes=delay_minutes, control=control,
        )
        if dry_run:
            typer.echo(json.dumps(runner.plan(ticker), indent=2, ensure_ascii=False))
            return
        messages, failures = runner.tick(ticker)
        directory = export_reports(store, ticker, schedule)
        for message in messages:
            typer.echo(message)
        typer.echo(f"Reports: {directory}")
        for failure in failures:
            typer.echo(f"ERROR (will retry): {failure}", err=True)
        if failures:
            raise typer.Exit(code=1)
    except Timeout:
        typer.echo("BUSY: another daily worker owns this ticker; retry on the next tick.", err=True)
        raise typer.Exit(code=2) from None
    except typer.Exit:
        raise
    except Exception as exc:
        logger.exception("Daily tick failed")
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def report(
    ticker: str = typer.Argument("NVDA"),
    database: Annotated[Path, typer.Option("--database")] = DEFAULT_DATABASE,
    calendar: str = typer.Option("NASDAQ", "--calendar"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable metrics and sample counts"),
):
    """Read local scores and lessons without market-data or model calls."""
    try:
        ticker = us_ticker(ticker)
        store, schedule = DailyStore(database), TradingCalendar(calendar)
        summary = summarize(store, ticker, schedule)
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False) if as_json else render_summary(summary))
        export_reports(store, ticker, schedule)
    except Exception as exc:
        logger.exception("Daily report failed")
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from None
