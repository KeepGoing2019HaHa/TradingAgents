"""Scheduled, bounded MSFT strategy evolution."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from filelock import Timeout

from tradingagents.daily.store import DailyStore
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evolution.batch import CANDIDATE_COUNT, AllHistoryRunner
from tradingagents.evolution.store import EvolutionStore

app = typer.Typer(help="One MSFT candidate per day, evaluated on all frozen labelled history.")
DATABASE = Path.home() / ".tradingagents" / "daily" / "daily.sqlite3"


def progress_logging():
    logger = logging.getLogger("tradingagents.evolution")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())


@app.command()
def tick(
    database: Annotated[Path, typer.Option("--database")] = DATABASE,
    models: str = typer.Option("", "--models", help="Comma-separated approved model IDs on the configured provider"),
    repeats: int = typer.Option(1, "--repeats", min=1, max=5),
    dry_run: bool = typer.Option(False, "--dry-run"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Resume today's pending batch without replacing candidates or successes"),
):
    """Try one candidate daily; reuse fixed predictions for unchanged history."""
    path = database.expanduser().with_name("evolution.sqlite3")
    if dry_run:
        typer.echo(json.dumps({"ticker": "MSFT", "database": str(path), "minimum_samples": 1,
                           "candidate_count": CANDIDATE_COUNT, "dataset_mode": "all_history", "repeats": repeats,
                               "prediction_cache": "immutable-first-success-v1",
                               "schedule": "21:15 America/New_York daily",
                               "branch": "rsi-msft/YYYY-MM-DD", "no_provider_calls": True}, indent=2))
        return
    try:
        progress_logging()
        config = {**DEFAULT_CONFIG, "output_language": "Simplified Chinese"}
        if models:
            config["rsi_models"] = ",".join(item.strip() for item in models.split(",") if item.strip())
        runner = AllHistoryRunner(config, EvolutionStore(path), Path(__file__).resolve().parents[1], repeats=repeats)
        if database.expanduser().is_file():
            runner.sync_outcomes(DailyStore(database, read_only=True))
        record = runner.run(retry_failed=retry_failed)
        typer.echo(json.dumps({key: value for key, value in record.items()
                       if key not in {"pairs", "attempts", "retry_history", "execution_settings", "candidates", "baseline_attempts"}}, ensure_ascii=False, indent=2))
        if record["status"] in {"failed", "running"}:
            raise typer.Exit(code=1)
    except Timeout:
        typer.echo("BUSY: MSFT evolution/capture is already running", err=True)
        raise typer.Exit(code=2) from None
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Evolution failed: {type(exc).__name__}; previous champion retained", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def recover(
    database: Annotated[Path, typer.Option("--database")] = DATABASE,
    day: str | None = typer.Option(None, "--day", help="Original New York run day; defaults to the most complete pending run"),
):
    """Resume the saved candidate and dataset, skipping successful analyses."""
    try:
        progress_logging()
        if day:
            from datetime import date

            if date.fromisoformat(day).isoformat() != day:
                raise ValueError("Invalid recovery day")
        config = {**DEFAULT_CONFIG, "output_language": "Simplified Chinese"}
        store = EvolutionStore(database.expanduser().with_name("evolution.sqlite3"))
        runner = AllHistoryRunner(config, store, Path(__file__).resolve().parents[1])
        record = runner.recover(day=day)
        if record is None:
            typer.echo("No recoverable evaluation due.")
            return
        public = {key: record.get(key) for key in (
            "day", "status", "error", "next_retry_at", "failed_analysis", "gate_reason", "backtest", "branch",
        )}
        public["completed_analyses"] = len(record.get("attempts", []))
        typer.echo(json.dumps(public, ensure_ascii=False, indent=2))
        if record["status"] == "failed":
            raise typer.Exit(code=1)
    except Timeout:
        typer.echo("BUSY: another MSFT worker owns the evolution lock; will retry later.")
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Recovery failed: {type(exc).__name__}; saved successes retained", err=True)
        raise typer.Exit(code=1) from None
