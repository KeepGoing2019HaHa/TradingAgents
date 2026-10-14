from __future__ import annotations

import copy
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from filelock import Timeout
from pydantic import ValidationError

from tradingagents.agents.context import get_instrument_context_from_state
from tradingagents.daily import market, runner
from tradingagents.daily.market import DataNotReady, TradingCalendar
from tradingagents.daily.models import DailyForecast, DailyReflection, score, timestamp
from tradingagents.daily.reporting import export_reports, statistics, summarize
from tradingagents.daily.store import DailyStore
from tradingagents.default_config import DEFAULT_CONFIG

UTC = timezone.utc
FORECAST = DailyForecast(close=102, low=98, high=105, up_probability=0.7,
                         rationale="Evidence-based forecast", risks="Uncertain event response")


def at(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def bar(day, close=100, split=0):
    return {
        "date": day, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000, "split": split, "source": "fake",
    }


def payload(day="2026-10-01", variant="memory", lessons=None, eligible=True):
    return {
        "ticker": "NVDA", "day": day, "variant": variant,
        "started_at": f"{day}T12:30:00.000000+00:00",
        "data_as_of": f"{day}T12:40:00.000000+00:00",
        "created_at": f"{day}T12:41:00.000000+00:00", "eligible": eligible,
        "settle_after": f"{day}T20:30:00.000000+00:00",
        "reference": bar("2026-09-30"), "forecast": FORECAST.model_dump(),
        "lessons": lessons or [], "reports": {}, "signature": "test-version",
    }


def reflection(candidate=True, evaluations=None):
    return DailyReflection(
        analysis="Price direction alone does not establish causality.",
        candidate={"role": "market", "condition": "When volatility is elevated",
                   "advice": "Check interval calibration before using a narrow interval"} if candidate else None,
        evaluations=evaluations or [],
    )


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    now = [at("2026-10-01T12:30:00")]
    prompts, graphs = [], []
    bars = {"2026-09-30": bar("2026-09-30"), "2026-10-01": bar("2026-10-01", 101)}
    failures = {"reflection": False, "graph": False}

    class Model:
        def invoke(self, prompt):
            raise RuntimeError("reflection provider unavailable")

        def with_structured_output(self, schema):
            def invoke(prompt):
                prompts.append((schema, prompt))
                if schema is DailyReflection:
                    if failures["reflection"]:
                        raise RuntimeError("reflection provider unavailable")
                    return reflection()
                return FORECAST
            return SimpleNamespace(invoke=invoke)

    class Graph:
        def __init__(self, config, analysis_context):
            self.config = config
            self.context = analysis_context
            self.quick_thinking_llm = Model()
            graphs.append(self)

        def propagate(self, ticker, day):
            if failures["graph"]:
                raise RuntimeError("analysis provider unavailable")
            return {"market_report": "Grounded market report", "final_trade_decision": "Rating: Buy"}, "Buy"

    monkeypatch.setattr(runner, "TradingAgentsGraph", Graph)
    monkeypatch.setattr(runner, "create_llm_client", lambda **kw: SimpleNamespace(get_llm=Model))
    monkeypatch.setattr(runner, "fetch_bars", lambda ticker, start, end: {
        day: value for day, value in bars.items() if start <= day <= end
    })
    store = DailyStore(tmp_path / "daily.sqlite3")
    config = {**copy.deepcopy(DEFAULT_CONFIG), "output_language": "Simplified Chinese"}
    service = runner.DailyRunner(config, store, TradingCalendar(), clock=lambda: now[0])
    return SimpleNamespace(now=now, prompts=prompts, graphs=graphs, bars=bars, failures=failures,
                           store=store, service=service, graph_type=Graph)


def test_complete_daily_cycle_is_frozen_scored_and_read_before_next_analysis(workflow):
    w = workflow
    messages, failures = w.service.tick(" nvda ")
    assert failures == [] and "frozen" in messages[0]
    first = w.store.records("NVDA")[0]
    assert first["eligible"] and first["reference"]["date"] == "2026-09-30"
    assert first["forecast"] == FORECAST.model_dump()
    assert first["lessons"] == [] and first["outcome"] is None
    assert w.graphs[0].config["memory_log_path"] is None
    assert w.graphs[0].config["checkpoint_enabled"] is False
    assert "regular-session close" in w.graphs[0].context["all"]

    w.service.tick("NVDA")
    assert len(w.graphs) == 1
    assert w.store.records("NVDA")[0]["created_at"] == first["created_at"]
    w.now[0] = at("2026-10-01T20:29:59")
    w.service.tick("NVDA")
    assert w.store.records("NVDA")[0]["outcome"] is None

    w.now[0] = at("2026-10-01T20:30:00")
    assert w.service.tick("NVDA")[1] == []
    settled = w.store.records("NVDA")[0]
    assert settled["outcome"]["scores"]["absolute_error"] == 1
    assert settled["outcome"]["scores"]["direction_hit"] is True
    assert settled["reflection"]["candidate"]["role"] == "market"
    assert settled["forecast"] == first["forecast"]
    calls = len(w.prompts)
    w.service.tick("NVDA")
    assert len(w.prompts) == calls

    w.now[0] = at("2026-10-02T12:30:00")
    w.service.tick("NVDA")
    assert len(w.graphs) == 2
    assert "volatility" in w.graphs[1].context["market"]
    assert "volatility" in w.graphs[1].context["decision"]
    assert "news" not in w.graphs[1].context
    assert len(w.store.records("NVDA")[1]["lessons"]) == 1


def test_late_analysis_is_persisted_but_never_scored_or_learned(workflow, monkeypatch):
    w = workflow
    original = w.graph_type.propagate

    def finish_at_open(self, ticker, day):
        w.now[0] = at("2026-10-01T13:30:00")
        return original(self, ticker, day)

    monkeypatch.setattr(w.graph_type, "propagate", finish_at_open)
    messages, failures = w.service.tick("NVDA")
    assert failures == [] and "LATE" in messages[0]
    w.now[0] = at("2026-10-01T21:00:00")
    w.service.tick("NVDA")
    record = w.store.records("NVDA")[0]
    assert record["eligible"] is False and record["outcome"] is None
    assert not w.store.lessons("NVDA", w.now[0])
    assert statistics([record])["scored"] == 0


def test_after_open_never_backfills_forecast(workflow):
    w = workflow
    w.now[0] = at("2026-10-01T13:30:00")
    messages, failures = w.service.tick("NVDA")
    assert not failures and "will not backfill" in messages[0]
    assert w.graphs == [] and w.store.records("NVDA") == []


def test_missing_exact_date_stays_pending_and_returns_failure(workflow):
    w = workflow
    w.service.tick("NVDA")
    del w.bars["2026-10-01"]
    w.now[0] = at("2026-10-01T21:00:00")
    failures = w.service.tick("NVDA")[1]
    assert "Missing NVDA bar for 2026-10-01" in failures[0]
    assert w.store.records("NVDA")[0]["outcome"] is None
    assert w.store.recent_events("NVDA")
    w.bars["2026-10-01"] = bar("2026-10-01", 101)
    assert not w.service.tick("NVDA")[1]


def test_reflection_retry_keeps_original_outcome_and_prediction(workflow):
    w = workflow
    w.service.tick("NVDA")
    w.now[0] = at("2026-10-01T21:00:00")
    w.failures["reflection"] = True
    assert w.service.tick("NVDA")[1]
    record = w.store.records("NVDA")[0]
    assert record["outcome"]["bar"]["close"] == 101 and record["reflection"] is None
    w.failures["reflection"] = False
    w.bars["2026-10-01"] = bar("2026-10-01", 999)
    assert not w.service.tick("NVDA")[1]
    assert w.store.records("NVDA")[0]["outcome"]["bar"]["close"] == 101
    assert w.store.records("NVDA")[0]["reflection"]


def test_failed_analysis_can_retry_before_open_only(workflow):
    w = workflow
    w.failures["graph"] = True
    assert w.service.tick("NVDA")[1]
    assert not w.store.records("NVDA")
    w.failures["graph"] = False
    assert not w.service.tick("NVDA")[1]
    assert len(w.store.records("NVDA")) == 1


def test_split_excluded_and_unexplained_reference_revision_pending(workflow):
    w = workflow
    w.service.tick("NVDA")
    w.now[0] = at("2026-10-01T21:00:00")
    w.bars["2026-09-30"] = bar("2026-09-30", 50)
    assert "revised" in w.service.tick("NVDA")[1][0]
    assert w.store.records("NVDA")[0]["outcome"] is None
    w.bars["2026-10-01"] = bar("2026-10-01", 50.5, split=2)
    assert not w.service.tick("NVDA")[1]
    record = w.store.records("NVDA")[0]
    assert record["outcome"]["excluded"] and record["outcome"]["scores"] is None
    assert record["reflection"] is None and not w.store.lessons("NVDA", w.now[0])


def test_control_is_separate_has_no_memory_and_does_not_create_lessons(workflow):
    w = workflow
    prior = w.store.freeze(payload("2026-09-29"))
    w.store.save_reflection(prior, reflection(), at("2026-09-29T21:00:00"))
    w.service.control = True
    assert not w.service.tick("NVDA")[1]
    assert "market" in w.graphs[0].context
    assert list(w.graphs[1].context) == ["all"]
    assert w.graphs[0].config["results_dir"] != w.graphs[1].config["results_dir"]
    w.now[0] = at("2026-10-01T21:00:00")
    w.service.tick("NVDA")
    today = [r for r in w.store.records("NVDA") if r["day"] == "2026-10-01"]
    assert statistics(today)["control_pairs"] == 1
    assert next(r for r in today if r["variant"] == "control")["reflection"] is None


@pytest.mark.parametrize("day", ["2026-07-03", "2026-12-25", "2026-10-03"])
def test_holidays_and_weekends(day):
    assert TradingCalendar().on(date.fromisoformat(day)) is None


def test_dst_early_close_and_previous_session():
    calendar = TradingCalendar()
    assert calendar.on(date(2026, 3, 6)).open.hour == 14
    assert calendar.on(date(2026, 3, 9)).open.hour == 13
    early = calendar.on(date(2026, 11, 27))
    assert early.close == at("2026-11-27T18:00:00")
    assert calendar.previous(early).day == "2026-11-25"


def test_early_close_settlement_is_not_fixed_to_1600(workflow):
    w = workflow
    w.now[0] = at("2026-11-27T13:30:00")
    w.bars.update({"2026-11-25": bar("2026-11-25"), "2026-11-27": bar("2026-11-27", 101)})
    w.service.tick("NVDA")
    record = w.store.records("NVDA")[0]
    assert record["settle_after"] == timestamp(at("2026-11-27T18:30:00"))
    w.now[0] = at("2026-11-27T18:30:00")
    assert not w.service.tick("NVDA")[1]
    assert w.store.records("NVDA")[0]["reflection"]


def test_forecast_validation_and_exact_score_formulas():
    scores = score(FORECAST, 100, 101)
    assert scores["absolute_error"] == 1
    assert scores["absolute_percentage_error"] == pytest.approx(1 / 101)
    assert scores["brier"] == pytest.approx(0.09)
    assert scores["interval_width"] == 7 and scores["interval_score"] == 7
    assert score(FORECAST, 100, 110)["interval_score"] == 57
    assert score(FORECAST, 100, 95)["interval_score"] == 37
    assert score(FORECAST, 100, 100)["direction_hit"] is False
    for change in ({"close": float("nan")}, {"low": 103}, {"high": 101},
                   {"up_probability": 1.1}, {"close": -1}, {"close": True}):
        with pytest.raises(ValidationError):
            DailyForecast.model_validate({**FORECAST.model_dump(), **change})
    with pytest.raises(ValueError):
        score(FORECAST, float("nan"), 100)
    with pytest.raises(ValueError):
        timestamp(datetime(2026, 10, 1))


def test_duplicate_freeze_and_concurrent_worker_rejected(tmp_path):
    store = DailyStore(tmp_path / "daily.sqlite3")
    store.freeze(payload())
    with pytest.raises(sqlite3.IntegrityError):
        store.freeze({**payload(), "forecast": {**FORECAST.model_dump(), "close": 999}})
    assert store.records("NVDA")[0]["forecast"] == FORECAST.model_dump()
    second = DailyStore(store.path)
    with store.lock("NVDA"), pytest.raises(Timeout), second.lock("nvda"):
        pass
    with second.lock("NVDA"), store.lock("MSFT"):
        assert len(second.records("NVDA")) == 1


def test_lesson_availability_and_evidence_never_leak_backwards(tmp_path):
    store = DailyStore(tmp_path / "daily.sqlite3")
    seed = store.freeze(payload("2026-09-01"))
    known = at("2026-09-01T21:00:00")
    store.save_reflection(seed, reflection(), known)
    assert not store.lessons("NVDA", known)
    lesson = store.lessons("NVDA", known + timedelta(seconds=1))[0]
    assert lesson["status"] == "candidate"
    assert not store.lessons("AAPL", known + timedelta(days=1))

    for i in range(2, 7):
        record = store.freeze(payload(f"2026-09-{i:02}", lessons=[lesson]))
        evaluation = {"lesson_id": lesson["id"], "verdict": "supports", "evidence": "Condition verified"}
        store.save_reflection(record, reflection(False, [evaluation]), known + timedelta(days=i))
    assert store.lessons("NVDA", known + timedelta(days=1))[0]["supports"] == 0
    assert store.lessons("NVDA", known + timedelta(days=7))[0]["status"] == "supported"
    assert not store.lessons("NVDA", known + timedelta(days=100))


def test_contradictions_retire_lesson_and_invalid_evaluation_is_atomic(tmp_path):
    store = DailyStore(tmp_path / "daily.sqlite3")
    known = at("2026-09-01T21:00:00")
    store.save_reflection(store.freeze(payload("2026-09-01")), reflection(), known)
    lesson = store.lessons("NVDA", known + timedelta(seconds=1))[0]
    for i in range(2, 5):
        record = store.freeze(payload(f"2026-09-{i:02}", lessons=[lesson]))
        evaluation = {"lesson_id": lesson["id"], "verdict": "contradicts", "evidence": "Contrary evidence"}
        store.save_reflection(record, reflection(False, [evaluation]), known + timedelta(days=i))
    assert store.lessons("NVDA", known + timedelta(days=2))[0]["status"] == "candidate"
    assert not store.lessons("NVDA", known + timedelta(days=5))
    record = store.freeze(payload("2026-09-10"))
    with pytest.raises(ValueError, match="unprovided"):
        store.save_reflection(record, reflection(True, [evaluation]), known + timedelta(days=10))
    assert store.records("NVDA")[-1]["reflection"] is None


def test_role_context_reaches_analysts_and_decision_roles_without_changing_normal_runs():
    state = {"company_of_interest": "NVDA", "instrument_context": "Identity"}
    assert get_instrument_context_from_state(state) == "Identity"
    state["analysis_context"] = {"all": "Daily horizon", "market": "Technical lesson", "decision": "All lessons"}
    assert get_instrument_context_from_state(state, "market") == "Identity\n\nDaily horizon\n\nTechnical lesson"
    assert get_instrument_context_from_state(state, "news") == "Identity\n\nDaily horizon"
    assert get_instrument_context_from_state(state) == "Identity\n\nDaily horizon\n\nAll lessons"


def test_structured_fallback_must_still_be_valid_json():
    class Model:
        text = FORECAST.model_dump_json()

        def with_structured_output(self, schema):
            raise NotImplementedError

        def invoke(self, prompt):
            assert "JSON object" in prompt
            return SimpleNamespace(content=self.text)

    model = Model()
    assert runner.invoke_validated(model, DailyForecast, "forecast") == FORECAST
    model.text = "Probably buy, price unknown."
    with pytest.raises(ValidationError):
        runner.invoke_validated(model, DailyForecast, "forecast")


def test_provider_rejecting_native_schema_can_return_validated_json(caplog):
    class Model:
        def with_structured_output(self, schema):
            def reject(prompt):
                raise ValueError("Native JSON schema is not supported by this endpoint")
            return SimpleNamespace(invoke=reject)

        def invoke(self, prompt):
            assert "JSON object" in prompt
            return SimpleNamespace(content=FORECAST.model_dump_json())

    assert runner.invoke_validated(Model(), DailyForecast, "forecast") == FORECAST
    assert "structured-output invocation failed" in caplog.text


def test_reports_sample_counts_calibration_and_session_windows(workflow):
    w = workflow
    w.service.control = True
    w.service.tick("NVDA")
    w.now[0] = at("2026-10-01T21:00:00")
    w.service.tick("NVDA")
    data = summarize(w.store, "NVDA", w.service.calendar, w.now[0])
    assert data["all"]["scored"] == 1 and data["last_20_sessions"]["scored"] == 1
    assert data["all"]["absolute_error"] == 1
    assert data["all"]["control_pairs"] == 1
    assert sum(bucket["count"] for bucket in data["all"]["calibration"]) == 1
    assert data["all"]["calibration"][3]["up_frequency"] == 1
    later = summarize(w.store, "NVDA", w.service.calendar, at("2027-02-01T22:00:00"))
    assert later["all"]["scored"] == 1 and later["last_60_sessions"]["scored"] == 0
    directory = export_reports(w.store, "NVDA", w.service.calendar, w.now[0])
    assert "Reflection" in (directory / "2026-10-01.md").read_text()
    assert json.loads((directory / "summary.json").read_text())["all"]["scored"] == 1
    assert not list(directory.glob("tmp*"))


@pytest.mark.parametrize("change", [
    {"Close": float("nan")}, {"Volume": 0}, {"High": 90}, {"Low": 110}, {"Stock Splits": float("nan")},
])
def test_market_bars_reject_incomplete_data(monkeypatch, change):
    values = {"Open": 100, "High": 102, "Low": 98, "Close": 101, "Volume": 1000, "Stock Splits": 0}
    values.update(change)
    frame = pd.DataFrame([values], index=pd.DatetimeIndex(["2026-10-01"], tz="America/New_York"))
    instrument = SimpleNamespace(
        history=lambda **kw: frame,
        history_metadata={"currency": "USD", "exchangeTimezoneName": "America/New_York", "instrumentType": "EQUITY"},
    )
    monkeypatch.setattr(market.yf, "Ticker", lambda ticker: instrument)
    with pytest.raises(DataNotReady):
        market.fetch_bars("NVDA", "2026-10-01", "2026-10-01")


def test_market_fetch_is_fresh_unadjusted_regular_session_and_us_only(monkeypatch):
    calls = []
    frame = pd.DataFrame(
        [{"Open": 100, "High": 102, "Low": 98, "Close": 101, "Volume": 1000}],
        index=pd.DatetimeIndex(["2026-10-01"], tz="America/New_York"),
    )
    instrument = SimpleNamespace(
        history=lambda **kw: calls.append(kw) or frame,
        history_metadata={"currency": "USD", "exchangeTimezoneName": "America/New_York", "instrumentType": "EQUITY"},
    )
    monkeypatch.setattr(market.yf, "Ticker", lambda ticker: instrument)
    assert market.fetch_bars("NVDA", "2026-10-01", "2026-10-01")["2026-10-01"]["close"] == 101
    assert calls[0]["end"] == "2026-10-02"
    assert calls[0]["auto_adjust"] is False and calls[0]["prepost"] is False
    instrument.history_metadata["currency"] = "CAD"
    with pytest.raises(ValueError, match="verified USD"):
        market.fetch_bars("NVDA", "2026-10-01", "2026-10-01")
    for ticker in ("0700.HK", "BTCUSD", "XAUUSD", "../../NVDA", ""):
        with pytest.raises(ValueError, match="US stock"):
            market.us_ticker(ticker)


def test_schedule_validation_and_dry_run_do_not_call_providers(workflow, monkeypatch):
    w = workflow
    monkeypatch.setattr(runner, "fetch_bars", lambda *args: pytest.fail("dry run fetched prices"))
    plan = w.service.plan("NVDA")
    assert plan["predict_from"] == timestamp(at("2026-10-01T12:30:00"))
    assert plan["settle_after"] == timestamp(at("2026-10-01T20:30:00"))
    assert w.graphs == [] and w.store.records("NVDA") == []
    with pytest.raises(ValueError):
        runner.DailyRunner({}, w.store, w.service.calendar, delay_minutes=0)
    with pytest.raises(ValueError):
        TradingCalendar("HKEX")


def test_multiple_stocks_share_database_without_locking_each_other(workflow):
    w = workflow
    with w.store.lock("NVDA"):
        assert not w.service.tick("MSFT")[1]
        assert not w.service.tick("AAPL")[1]
        with pytest.raises(Timeout):
            w.service.tick("NVDA")
    assert not w.service.tick("NVDA")[1]
    assert all(len(w.store.records(ticker)) == 1 for ticker in ("NVDA", "MSFT", "AAPL"))
    assert w.store.lessons("MSFT", w.now[0]) == []


def test_control_cannot_start_when_primary_finishes_after_open(workflow, monkeypatch):
    w = workflow
    original = w.graph_type.propagate
    w.service.control = True

    def slow(self, ticker, day):
        w.now[0] = at("2026-10-01T13:31:00")
        return original(self, ticker, day)

    monkeypatch.setattr(w.graph_type, "propagate", slow)
    failures = w.service.tick("NVDA")[1]
    assert len(w.graphs) == 1
    assert "refusing a retrospective forecast" in failures[0]
    assert w.store.prediction("NVDA", "2026-10-01", "control") is None


@pytest.mark.parametrize(("supports", "contradicts", "expected"), [
    (4, 0, "candidate"), (5, 0, "supported"), (7, 3, "supported"), (6, 3, "candidate"),
])
def test_exact_lesson_support_thresholds(tmp_path, supports, contradicts, expected):
    store = DailyStore(tmp_path / "daily.sqlite3")
    known = at("2026-09-01T21:00:00")
    store.save_reflection(store.freeze(payload("2026-09-01")), reflection(), known)
    lesson = store.lessons("NVDA", known + timedelta(seconds=1))[0]
    votes = ["supports"] * supports + ["contradicts"] * contradicts
    for day, vote in enumerate(votes, start=2):
        record = store.freeze(payload(f"2026-09-{day:02}", lessons=[lesson]))
        evaluation = {"lesson_id": lesson["id"], "verdict": vote, "evidence": "Documented evidence"}
        store.save_reflection(record, reflection(False, [evaluation]), known + timedelta(days=day))
    assert store.lessons("NVDA", known + timedelta(days=15))[0]["status"] == expected


def test_lessons_expire_at_exactly_ninety_days(tmp_path):
    store = DailyStore(tmp_path / "daily.sqlite3")
    known = at("2026-09-01T21:00:00")
    store.save_reflection(store.freeze(payload("2026-09-01")), reflection(), known)
    assert store.lessons("NVDA", known + timedelta(days=90, microseconds=-1))
    assert not store.lessons("NVDA", known + timedelta(days=90))
