import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.daily.dashboard import snapshot
from tradingagents.daily.models import DailyForecast, DailyReflection, score
from tradingagents.daily.reporting import statistics
from tradingagents.daily.store import DailyStore


def seed(store, day, *, eligible=True, settled=True):
    forecast = DailyForecast(close=102, low=98, high=105, up_probability=0.7,
                             rationale="Daily evidence", risks="Uncertainty")
    record = store.freeze({
        "ticker": "NVDA", "day": day, "variant": "memory", "eligible": eligible,
        "created_at": day + "T12:40:00+00:00", "data_as_of": day + "T12:39:00+00:00",
        "forecast": forecast.model_dump(), "reference": {"date": day, "close": 100},
        "signature": "v1", "reports": {"market": "PRIVATE internal report" * 10000},
        "settings": {"backend_url": "PRIVATE URL"}, "lessons": [],
    })
    now = datetime.fromisoformat(day + "T21:00:00+00:00")
    if settled and eligible:
        store.save_outcome(record["id"], {
            "bar": {"close": 101}, "scores": score(forecast, 100, 101), "excluded": None,
        }, now)
        store.save_reflection(record, DailyReflection(
            analysis="Direction alone is not causal evidence.",
            candidate={"role": "market", "condition": "High volatility", "advice": "Check calibration"},
            evaluations=[],
        ), now)
    return record


def test_read_only_snapshot_metrics_and_bounded_pages(tmp_path):
    path = tmp_path / "daily.sqlite3"
    store = DailyStore(path)
    for day in range(1, 13):
        seed(store, f"2026-09-{day:02}")
    seed(store, "2026-09-13", eligible=False)
    seed(store, "2026-09-14", settled=False)
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    store.event("NVDA", "error", "PRIVATE token and local path", now)
    before = path.read_bytes()
    readonly = DailyStore(path, read_only=True)
    result = snapshot(readonly, "NVDA", now=now)
    assert result["summary"]["all"]["scored"] == 12
    assert result["summary"]["all"]["pending"] == 1
    assert result["summary"]["all"]["late"] == 1
    assert result["summary"]["all"]["absolute_error"] == 1
    assert result["summary"]["all"]["direction_hit"] == 1
    assert [r["status"] for r in result["history"]["items"][:2]] == ["pending", "late"]
    assert len(result["history"]["items"]) == 5 and len(result["lessons"]["items"]) == 10
    assert "PRIVATE" not in json.dumps(result)
    assert statistics(store.metric_records("NVDA")) == statistics(store.records("NVDA"))

    second = snapshot(readonly, "NVDA", now=now, before=result["history"]["next_before"],
                      lesson_before=result["lessons"]["next_before"])
    assert max(r["day"] for r in second["history"]["items"]) < result["history"]["next_before"]
    assert len(second["lessons"]["items"]) == 2 and second["lessons"]["next_before"] is None
    assert path.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        readonly.event("NVDA", "test", "no writes", now)


def test_empty_is_not_zero_accuracy_and_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing" / "daily.sqlite3"
    readonly = DailyStore(path, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        snapshot(readonly, "NVDA")
    assert not path.parent.exists()
    store = DailyStore(path)
    result = snapshot(store, "MSFT")
    assert result["history"]["items"] == [] and result["lessons"]["items"] == []
    assert result["summary"]["all"]["direction_hit"] is None
    assert result["summary"]["all"]["scored"] == 0


def test_dashboard_retains_expired_lessons_but_analysis_does_not(tmp_path):
    store = DailyStore(tmp_path / "daily.sqlite3")
    seed(store, "2026-01-05")
    now = datetime(2026, 1, 5, 21, tzinfo=timezone.utc) + timedelta(days=90)
    assert not store.lessons("NVDA", now)
    result = snapshot(store, "NVDA", now=now)
    assert result["lessons"]["items"][0]["status"] == "expired"
    assert result["lessons"]["items"][0]["source_day"] == "2026-01-05"
    assert snapshot(store, "AAPL", now=now)["history"]["items"] == []


@pytest.mark.parametrize("kwargs", [{"before": "2026-9-1"}, {"before": "invalid"}, {"lesson_before": 0}])
def test_dashboard_rejects_invalid_cursors(tmp_path, kwargs):
    with pytest.raises(ValueError):
        snapshot(DailyStore(tmp_path / "daily.sqlite3"), "NVDA", **kwargs)
