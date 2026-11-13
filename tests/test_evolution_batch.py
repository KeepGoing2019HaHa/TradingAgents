from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tradingagents.daily.models import DailyForecast
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evolution import batch, runner
from tradingagents.evolution.engine import BASE_PROMPT, Proposal, Strategy
from tradingagents.evolution.replay import TOOL_NAMES
from tradingagents.evolution.store import EvolutionStore


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    now = [datetime(2026, 11, 13, 3, tzinfo=timezone.utc)]
    for day in ("2026-11-09", "2026-11-10", "2026-11-11"):
        store.freeze({"schema": 1, "ticker": "MSFT", "day": day,
                      "captured_at": day + "T12:00:00+00:00", "cutoff": day + "T14:30:00+00:00",
                      "reference": {"close": 100},
                      "tools": {name: {"output": "frozen"} for name in TOOL_NAMES}})
        store.label(day, {"close": 105, "available_at": day + "T21:00:00+00:00"})
    store.reserve_test(["2026-11-09"], "2026-11-09")
    config = {**deepcopy(DEFAULT_CONFIG), "quick_think_llm": "baseline", "deep_think_llm": "baseline"}
    service = batch.AllHistoryRunner(config, store, tmp_path, clock=lambda: now[0])
    calls, proposals = [], []

    def propose(champion, samples):
        proposals.append(samples)
        return batch.ProposalBatch(proposals=[Proposal(
            strategy=Strategy(quick_model=f"candidate-{number}", deep_model="baseline",
                              prompt=BASE_PROMPT + f" Alternative {number}.", tools=list(TOOL_NAMES)),
            summary=f"Candidate {number}", rationale="Distinct hypothesis",
        ) for number in range(1, 6)])

    def forecast(strategy, inputs, settings, output):
        assert "target" not in inputs and "lessons" not in inputs
        calls.append((strategy.quick_model, inputs["day"]))
        close = {"baseline": 102, "candidate-1": 101, "candidate-2": 103,
                 "candidate-3": 104, "candidate-4": 99, "candidate-5": 102}[strategy.quick_model]
        return DailyForecast(close=close, low=95, high=110, up_probability=0.7,
                             rationale="frozen", risks="uncertain"), {}

    monkeypatch.setattr(service, "propose_batch", propose)
    monkeypatch.setattr(runner, "forecast", forecast)
    monkeypatch.setattr(batch, "archive_branch", lambda *args, **kwargs: {"branch": "test-branch", "commit": "test"})
    return SimpleNamespace(service=service, store=store, now=now, calls=calls, proposals=proposals,
                           forecast=forecast)


def test_five_candidates_use_all_history_and_share_baseline(workflow):
    flow = workflow
    result = flow.service.run()
    assert result["status"] == "promoted" and result["best_candidate"] == 3
    assert result["dataset"]["mode"] == "all_history"
    assert len(result["dataset"]["test_days"]) == 3
    assert len(result["candidates"]) == 5
    assert len(flow.calls) == 18
    assert sum(model == "baseline" for model, _ in flow.calls) == 3
    for number in range(1, 6):
        assert sum(model == f"candidate-{number}" for model, _ in flow.calls) == 3
    assert flow.store.samples()[0]["used_by"] == "2026-11-09"
    assert flow.store.samples()[1]["used_by"] is None
    assert flow.store.champion()["id"] == result["candidate_version"]
    assert flow.service.run() == result and len(flow.calls) == 18


def test_batch_retry_skips_successes_and_does_not_regenerate_candidates(workflow, monkeypatch):
    flow = workflow

    class RateLimitError(Exception):
        status_code = 429

    def limited(strategy, *args):
        if strategy.quick_model == "candidate-2":
            raise RateLimitError("limited")
        return flow.forecast(strategy, *args)

    monkeypatch.setattr(runner, "forecast", limited)
    initial = flow.service.run()
    assert initial["status"] == "retry_wait"
    assert len(flow.calls) == 6
    assert flow.service.recover()["status"] == "retry_wait" and len(flow.calls) == 6
    flow.now[0] += timedelta(minutes=15)
    monkeypatch.setattr(runner, "forecast", flow.forecast)
    result = flow.service.recover()
    assert result["status"] == "promoted"
    assert len(flow.calls) == 18 and len(flow.proposals) == 1
    assert len(set(flow.calls)) == 18


def test_batch_proposals_must_be_five_and_distinct(workflow):
    proposals = workflow.service.propose_batch({}, []).proposals
    with pytest.raises(ValueError):
        batch.ProposalBatch(proposals=proposals[:4])
    with pytest.raises(ValueError):
        batch.ProposalBatch(proposals=[proposals[0]] * 5)


def test_worse_candidates_never_promote(workflow, monkeypatch):
    flow = workflow

    def forecast(strategy, inputs, settings, output):
        flow.calls.append((strategy.quick_model, inputs["day"]))
        return DailyForecast(close=99, low=95, high=110, up_probability=0.1,
                             rationale="test", risks="test"), {}

    monkeypatch.setattr(runner, "forecast", forecast)
    result = flow.service.run()
    assert result["status"] == "rejected"
    assert flow.store.champion()["id"] == result["parent_version"]
    assert len(result["candidates"]) == 5


def test_direction_beats_smaller_mae_across_candidates(workflow, monkeypatch):
    flow = workflow

    def forecast(strategy, inputs, settings, output):
        flow.calls.append((strategy.quick_model, inputs["day"]))
        close = 130 if strategy.quick_model == "candidate-4" else 99
        return DailyForecast(close=close, low=90, high=140, up_probability=0.7,
                             rationale="test", risks="test"), {}

    monkeypatch.setattr(runner, "forecast", forecast)
    result = flow.service.run()
    assert result["status"] == "promoted" and result["best_candidate"] == 4
    assert result["backtest"]["candidate"]["mae"] > result["backtest"]["current"]["mae"]


def test_new_sample_is_in_next_day_full_history_not_a_holdout(workflow):
    flow = workflow
    first = flow.service.run()
    flow.now[0] += timedelta(days=1)
    sample = deepcopy(flow.store.samples()[-1]["inputs"])
    sample["day"] = "2026-11-12"
    flow.store.freeze(sample)
    flow.store.label(sample["day"], {"close": 105, "available_at": "2026-11-13T21:00:00+00:00"})
    result = flow.service.run()
    assert len(result["dataset"]["test_days"]) == 4
    assert result["dataset"]["mode"] == "all_history"
    assert result["parent_version"] == first["candidate_version"]
    assert result["expected_analyses"] == 24


def test_migration_archives_old_run_and_preserves_samples(workflow):
    flow = workflow
    old = {"day": "2026-11-12", "status": "running", "old_result": "retained"}
    flow.store.save_run(old)
    flow.service.run()
    with flow.store.connect() as database:
        archived = database.execute("SELECT payload FROM archived_runs WHERE day=?", (old["day"],)).fetchone()
    assert archived and "retained" in archived["payload"]
    assert flow.store.samples()[0]["used_by"] == "2026-11-09"


def test_dashboard_exposes_all_five_candidate_results_without_predictions(workflow):
    from tradingagents.evolution.dashboard import snapshot

    flow = workflow
    flow.service.run()
    result = snapshot(flow.store)
    record = result["history"]["items"][0]
    assert len(record["candidates"]) == 5
    assert record["expected_analyses"] == 18
    assert record["completed_analyses"] == 18
    assert "attempts" not in record["candidates"][0]
    assert record["candidates"][2]["status"] == "better"


def test_unlabelled_or_future_outcomes_are_excluded_from_all_history(workflow):
    flow = workflow
    sample = deepcopy(flow.store.samples()[-1]["inputs"])
    sample["day"] = "2026-11-12"
    flow.store.freeze(sample)
    flow.store.label(sample["day"], {"close": 105, "available_at": "2026-11-13T04:00:00+00:00"})
    sample["day"] = "2026-11-13"
    flow.store.freeze(sample)
    result = flow.service.run()
    assert len(result["dataset"]["test_days"]) == 3
    assert "2026-11-12" not in result["dataset"]["test_days"]
    assert "2026-11-13" not in result["dataset"]["test_days"]


def test_one_failed_candidate_does_not_repeat_or_discard_other_candidates(workflow, monkeypatch):
    flow = workflow

    class InvalidRequestError(Exception):
        status_code = 400

    def failed(strategy, *args):
        if strategy.quick_model == "candidate-3":
            raise InvalidRequestError("unavailable model")
        return flow.forecast(strategy, *args)

    monkeypatch.setattr(runner, "forecast", failed)
    result = flow.service.run()
    assert result["status"] == "promoted" and result["best_candidate"] == 2
    assert result["candidates"][2]["status"] == "failed"
    assert len(flow.calls) == len(set(flow.calls)) == 15


def test_pending_batch_does_not_expand_when_new_label_arrives(workflow, monkeypatch):
    flow = workflow

    def failed(*args):
        raise TimeoutError("temporary")

    monkeypatch.setattr(runner, "forecast", failed)
    initial = flow.service.run()
    sample = deepcopy(flow.store.samples()[-1]["inputs"])
    sample["day"] = "2026-11-12"
    flow.store.freeze(sample)
    flow.store.label(sample["day"], {"close": 105, "available_at": "2026-11-13T03:01:00+00:00"})
    flow.now[0] += timedelta(minutes=2)
    monkeypatch.setattr(runner, "forecast", flow.forecast)
    result = flow.service.recover()
    assert result["dataset"] == initial["dataset"]
    assert len(flow.calls) == 18


def test_candidate_previews_fit_api_limit_and_leave_full_audit_intact(workflow):
    import json

    from tradingagents.evolution.dashboard import snapshot

    flow = workflow
    record = flow.service.run()
    for candidate in record["candidates"]:
        candidate["strategy"]["prompt"] = "研" * 6000
        candidate["rationale"] = "由" * 4000
        candidate["summary"] = "摘" * 2000
    record["strategy"]["prompt"] = "研" * 6000
    record["previous_strategy"]["prompt"] = "旧" * 6000
    record["rationale"] = "由" * 4000
    record["summary"] = "摘" * 2000
    for number in range(13, 18):
        flow.store.save_run({**record, "day": f"2026-11-{number:02d}"})
    response = snapshot(flow.store)
    assert len(json.dumps(response, ensure_ascii=False).encode()) < 600_000
    assert response["history"]["items"][0]["candidates"][0]["text_truncated"]
    assert len(flow.store.runs()[0]["candidates"][0]["strategy"]["prompt"]) == 6000
