"""Bounded, public research data for a read-only dashboard (no provider calls)."""

from __future__ import annotations

from datetime import date

from tradingagents.daily.market import TradingCalendar, us_ticker
from tradingagents.daily.models import utc_now
from tradingagents.daily.reporting import summarize
from tradingagents.daily.store import DailyStore

HISTORY_PAGE_SIZE = 5
LESSON_PAGE_SIZE = 10


def _history_item(record: dict) -> dict:
    outcome = record["outcome"]
    if not record["eligible"]:
        status = "late"
    elif outcome and outcome["excluded"]:
        status = "excluded"
    elif outcome is None:
        status = "pending"
    elif record["reflection"] is None:
        status = "awaiting_reflection"
    else:
        status = "reviewed"
    reflection = record["reflection"]
    return {
        "id": record["id"], "day": record["day"], "status": status,
        "created_at": record["created_at"], "data_as_of": record["data_as_of"],
        "reference_close": record["reference"]["close"],
        "reference_date": record["reference"]["date"],
        "forecast": record["forecast"],
        "actual_close": outcome["bar"]["close"] if outcome else None,
        "scores": outcome["scores"] if outcome else None,
        "reflection": reflection["analysis"] if reflection else None,
        "candidate": reflection["candidate"] if reflection else None,
    }


def snapshot(
    store: DailyStore, ticker: str, *, before: str | None = None,
    lesson_before: int | None = None, now=None,
) -> dict:
    """Whitelist public fields; never export paths, provider settings or error logs."""
    ticker = us_ticker(ticker)
    if before is not None and date.fromisoformat(before).isoformat() != before:
        raise ValueError("Expected a canonical history date")
    if lesson_before is not None and lesson_before < 1:
        raise ValueError("Expected a positive lesson cursor")
    now = now or utc_now()
    summary = summarize(store, ticker, TradingCalendar(), now)
    history = store.history(ticker, before=before, limit=HISTORY_PAGE_SIZE + 1)
    lessons = store.lesson_history(
        ticker, now, before_id=lesson_before, limit=LESSON_PAGE_SIZE + 1,
    )
    return {
        "ticker": ticker, "generated_at": summary["generated_at"], "currency": "USD",
        "market_timezone": "America/New_York",
        "summary": {key: summary[key] for key in ("all", "last_20_sessions", "last_60_sessions")},
        "history": {
            "items": [_history_item(item) for item in history[:HISTORY_PAGE_SIZE]],
            "next_before": history[HISTORY_PAGE_SIZE - 1]["day"] if len(history) > HISTORY_PAGE_SIZE else None,
        },
        "lessons": {
            "items": [{
                key: item[key] for key in (
                    "id", "role", "condition", "advice", "status", "source_day",
                    "available_at", "supports", "contradicts", "last_tested",
                )
            } for item in lessons[:LESSON_PAGE_SIZE]],
            "next_before": lessons[LESSON_PAGE_SIZE - 1]["id"] if len(lessons) > LESSON_PAGE_SIZE else None,
        },
    }
