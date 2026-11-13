from copy import deepcopy

import pytest

from tradingagents.evolution.store import EvolutionStore


def evidence(day="2026-10-01"):
    return {"schema": 1, "ticker": "MSFT", "day": day,
            "captured_at": day + "T12:30:00+00:00", "cutoff": day + "T13:30:00+00:00",
            "reference": {"close": 100}, "tools": {"prices": "previous closes"}}


def test_dataset_is_immutable_and_separates_future_labels(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    inputs = evidence()
    checksum = store.freeze(inputs)
    assert store.freeze(inputs) == checksum
    changed = deepcopy(inputs)
    changed["tools"]["prices"] = "revised prices"
    with pytest.raises(ValueError, match="cannot be replaced"):
        store.freeze(changed)
    store.label(inputs["day"], {"close": 103})
    sample = store.samples()[0]
    assert sample["inputs"] == inputs and sample["target"] == {"close": 103}
    assert "target" not in sample["inputs"]
    with pytest.raises(ValueError, match="cannot be replaced"):
        store.label(inputs["day"], {"close": 104})


def test_dataset_rejects_memory_other_tickers_and_late_inputs(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    for change in ({"memory": "future lesson"}, {"ticker": "NVDA"},
                   {"captured_at": "2026-10-01T13:30:00+00:00"}):
        with pytest.raises(ValueError):
            store.freeze({**evidence(), **change})


def test_proposal_input_has_global_byte_budget_preserves_days_and_does_not_mutate_sources():
    import json

    from tradingagents.evolution.replay import TOOL_NAMES
    from tradingagents.evolution.runner import PROPOSAL_INPUT_BYTES, proposal_input

    samples = []
    for number in range(1, 11):
        inputs = evidence(f"2026-10-{number:02d}")
        inputs["tools"] = {name: {"output": "start " + "研究" * 10_000 + " latest row", "available": True}
                           for name in TOOL_NAMES}
        samples.append({"inputs": inputs, "target": {"close": 105}})
    original = deepcopy(samples)
    prompt = proposal_input("Development only; no held-out outcomes.", samples)
    assert len(prompt.encode("utf-8")) <= PROPOSAL_INPUT_BYTES
    training = json.loads(prompt.split("(omissions are not missing source data): ", 1)[1])
    assert len(training) == 10
    assert [item["day"] for item in training] == [item["inputs"]["day"] for item in samples]
    assert all(item["target"] == {"close": 105} for item in training)
    assert all(entry["truncated"] for item in training for entry in item["evidence"].values())
    assert training[0]["evidence"]["get_news"]["excerpt"].endswith("latest row")
    assert "\\u7814" not in prompt
    assert samples == original


def test_proposal_metadata_overflow_fails_before_provider_call():
    from tradingagents.evolution.runner import PROPOSAL_INPUT_BYTES, proposal_input

    with pytest.raises(ValueError, match="metadata exceeds"):
        proposal_input("x" * (PROPOSAL_INPUT_BYTES + 1), [])


def test_holdouts_are_consumed_once_and_reservation_is_atomic(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    for day in ("2026-10-01", "2026-10-02"):
        store.freeze(evidence(day))
        store.label(day, {"close": 102})
    store.reserve_test(["2026-10-02"], "2026-10-03")
    with pytest.raises(ValueError):
        store.reserve_test(["2026-10-01", "2026-10-02"], "2026-10-04")
    assert store.samples()[0]["used_by"] is None


def test_only_one_champion_and_read_only_dashboard(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    first = store.version({"model": "first"}, "2026-10-01")
    second = store.version({"model": "second"}, "2026-10-02", first)
    store.promote(first)
    store.promote(second)
    assert store.champion()["id"] == second
    store.save_run({"day": "2026-10-02", "status": "promoted"})
    reader = EvolutionStore(store.path, read_only=True)
    assert reader.runs()[0]["status"] == "promoted"


def test_frozen_tools_never_call_live_provider_or_fall_back(monkeypatch):
    from tradingagents.agents import tools
    from tradingagents.dataflows.config import run_config

    def forbidden(*args, **kwargs):
        raise AssertionError("Live data access during replay")

    monkeypatch.setattr(tools, "route_to_vendor", forbidden)
    inputs = evidence()
    inputs["tools"] = {"get_stock_data": {"output": "archived data", "args": {}}}
    config = {"evolution_inputs": inputs, "evolution_policy": {"tools": ["get_stock_data", "get_news"]}}
    with run_config(config):
        result = tools.get_stock_data.invoke({"symbol": "MSFT", "start_date": "2020-01-01", "end_date": "2030-01-01"})
        assert "archived data" in result
        with pytest.raises(ValueError, match="fallback is forbidden"):
            tools.get_news.invoke({"ticker": "MSFT", "start_date": "2020-01-01", "end_date": "2026-10-01"})
        with pytest.raises(ValueError, match="disabled"):
            tools.get_fundamentals.invoke({"ticker": "MSFT", "curr_date": "2026-10-01"})


def test_tool_policy_matches_analyst_execution_plan():
    from tradingagents.dataflows.config import run_config
    from tradingagents.evolution.replay import REQUIRED_TOOLS
    from tradingagents.graph.analyst_execution import build_analyst_execution_plan

    with run_config({"evolution_policy": {"tools": sorted(REQUIRED_TOOLS)}}):
        plan = build_analyst_execution_plan(("market", "news", "fundamentals"))
        assert {tool.name for spec in plan.specs for tool in spec.tools} == REQUIRED_TOOLS


def test_strategy_cannot_add_unknown_tools_or_drop_core_evidence():
    from tradingagents.evolution.engine import BASE_PROMPT, Strategy
    from tradingagents.evolution.replay import TOOL_NAMES

    values = {"quick_model": "model-a", "deep_model": "model-b", "prompt": BASE_PROMPT}
    assert Strategy(**values, tools=list(TOOL_NAMES))
    with pytest.raises(ValueError):
        Strategy(**values, tools=["arbitrary_network_request"])
    with pytest.raises(ValueError):
        Strategy(**values, tools=[*TOOL_NAMES, "get_news"])


def test_git_archive_creates_branch_without_switching_or_committing_worktree(tmp_path):
    import subprocess

    from tradingagents.evolution.branches import archive_branch

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q", "-b", "rsi")
    git("-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    original = git("rev-parse", "HEAD")
    (tmp_path / "unrelated.txt").write_text("user work")
    status = git("status", "--porcelain")
    result = archive_branch(tmp_path, "2026-10-01", {"status": "rejected", "summary": "Test"})
    assert result["branch"] == "rsi-msft/2026-10-01"
    assert '"summary":"Test"' in git("show", result["branch"] + ":rsi-msft.json")
    assert git("rev-parse", "HEAD") == original
    assert git("status", "--porcelain") == status


def test_gate_rejects_equal_or_worse_candidates_and_accepts_paired_improvement():
    from tradingagents.daily.models import DailyForecast, score
    from tradingagents.evolution.runner import acceptance

    def scored(close):
        return score(DailyForecast(close=close, low=95, high=110, up_probability=0.7,
                                   rationale="test", risks="test"), 100, 105)

    pairs = [{"day": f"2026-10-{number:02d}", "current_score": scored(102), "candidate_score": scored(104)}
             for number in range(1, 6)]
    assert acceptance(pairs)["promote"]
    for pair in pairs:
        pair["candidate_score"] = scored(102)
    assert not acceptance(pairs)["promote"]
    pairs[0]["candidate_score"] = scored(96)
    assert not acceptance(pairs)["promote"]


@pytest.mark.parametrize(("current_close", "candidate_close", "promote"), [
    (99, 130, True),
    (130, 99, False),
    (102, 104, True),
    (104, 102, False),
    (102, 102, False),
    (102, 102.01, True),
    (98, 99, True),
])
def test_direction_is_primary_and_mae_only_breaks_ties(current_close, candidate_close, promote):
    from tradingagents.daily.models import DailyForecast, score
    from tradingagents.evolution.runner import acceptance

    def scored(close, probability):
        return score(DailyForecast(close=close, low=90, high=max(110, close),
                                   up_probability=probability, rationale="test", risks="test"), 100, 105)

    pairs = [{"day": f"2026-10-{number:02d}",
              "current_score": scored(current_close, 0.9),
              "candidate_score": scored(candidate_close, 0.1)} for number in range(1, 6)]
    result = acceptance(pairs)
    assert result["promote"] is promote
    assert result["selection_rule"] == "direction_then_mae_v2"
    assert result["candidate"]["brier"] > result["current"]["brier"]
    if result["direction_improvement"] != 0:
        assert result["paired_gain_metric"] == "direction_accuracy"
    else:
        assert result["paired_gain_metric"] == "mae"


def test_direction_gain_is_not_vetoed_by_mae_or_bootstrap():
    from tradingagents.daily.models import DailyForecast, score
    from tradingagents.evolution.runner import acceptance

    def scored(close):
        return score(DailyForecast(close=close, low=90, high=140, up_probability=0.7,
                                   rationale="test", risks="test"), 100, 105)

    pairs = [{"day": f"2026-10-{number:02d}", "current_score": scored(102),
              "candidate_score": scored(102)} for number in range(1, 6)]
    pairs[0].update(current_score=scored(99), candidate_score=scored(130))
    result = acceptance(pairs)
    assert result["direction_improvement"] == pytest.approx(0.2)
    assert result["mae_improvement"] < 0
    assert result["paired_gain_lower_95"] == 0
    assert result["promote"]


@pytest.fixture
def evolution_workflow(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from tradingagents.daily.models import DailyForecast
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.evolution import runner
    from tradingagents.evolution.engine import BASE_PROMPT, Proposal, Strategy
    from tradingagents.evolution.replay import TOOL_NAMES

    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    for number in range(1, 16):
        day = f"2026-09-{number:02d}"
        inputs = evidence(day)
        inputs["tools"] = {name: {"output": "frozen", "args": {}} for name in TOOL_NAMES}
        store.freeze(inputs)
        store.label(day, {"close": 105, "available_at": day + "T21:00:00+00:00"})
    now = [datetime(2026, 10, 1, 23, tzinfo=timezone.utc)]
    config = {**deepcopy(DEFAULT_CONFIG), "quick_think_llm": "baseline", "deep_think_llm": "baseline"}
    service = runner.EvolutionRunner(config, store, tmp_path, clock=lambda: now[0])
    calls, branches, proposals = [], [], []

    def propose(champion, development):
        proposals.append(development)
        assert max(sample["inputs"]["day"] for sample in development) < "2026-09-11"
        return Proposal(strategy=Strategy(quick_model="candidate", deep_model="baseline",
                                          prompt=BASE_PROMPT + " Calibrate uncertainty." * len(proposals), tools=list(TOOL_NAMES)),
                        summary="Candidate model", rationale="Development-only hypothesis")

    def predict(strategy, inputs, config, output):
        assert "target" not in inputs and "memory" not in inputs and "lessons" not in inputs
        calls.append((strategy.quick_model, deepcopy(inputs)))
        close = 104 if strategy.quick_model == "candidate" else 102
        return DailyForecast(close=close, low=95, high=110, up_probability=0.7,
                             rationale="frozen", risks="uncertain"), {}

    def archive(repository, day, record, **kwargs):
        branches.append(day)
        return {"branch": "rsi-msft/" + day, "commit": "test-commit"}

    monkeypatch.setattr(service, "propose", propose)
    monkeypatch.setattr(runner, "forecast", predict)
    monkeypatch.setattr(runner, "archive_branch", archive)
    return SimpleNamespace(store=store, service=service, now=now, calls=calls, branches=branches,
                           proposals=proposals, module=runner)


def test_daily_evolution_pairs_identical_inputs_and_only_promotes_once(evolution_workflow):
    flow = evolution_workflow
    record = flow.service.run()
    assert record["status"] == "promoted" and record["backtest"]["sessions"] == 5
    assert len(record["pairs"]) == 10 and len(flow.calls) == 20
    for offset in range(0, len(flow.calls), 2):
        assert flow.calls[offset][1] == flow.calls[offset + 1][1]
    assert flow.store.champion()["id"] == record["candidate_version"]
    assert all(sample["used_by"] == "2026-10-01" for sample in flow.store.samples()[-5:])
    assert flow.service.run() == record
    assert len(flow.calls) == 20 and len(flow.proposals) == 1


def test_all_history_scoring_does_not_require_fresh_holdouts(evolution_workflow):
    flow = evolution_workflow
    result = flow.service.run()
    result["dataset"]["mode"] = "all_history"
    flow.service.score_completed(result)
    assert result["backtest"]["promote"]
    assert result["backtest"]["checks"]["complete_history"]
    assert "fresh_holdout" not in result["backtest"]["checks"]
    assert "全历史" in result["gate_reason"]


def test_all_history_scoring_still_requires_all_pairs(evolution_workflow):
    flow = evolution_workflow
    result = flow.service.run()
    result["dataset"]["mode"] = "all_history"
    result["pairs"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        flow.service.score_completed(result)


def test_failed_backtest_keeps_champion_and_consumes_holdout(evolution_workflow, monkeypatch):
    flow = evolution_workflow

    def failure(*args):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(flow.module, "forecast", failure)
    with pytest.raises(RuntimeError):
        flow.service.run()
    failed = flow.store.runs()[0]
    assert failed["status"] == "failed" and failed["branch"]
    assert flow.store.champion()["id"] == failed["parent_version"]
    assert all(sample["used_by"] for sample in flow.store.samples()[-5:])
    assert flow.service.run()["status"] == "failed"


def test_evaluate_resumes_only_missing_analysis_without_repeating_successes(evolution_workflow, monkeypatch):
    from tradingagents.evolution.engine import Strategy

    flow = evolution_workflow
    original = flow.module.forecast

    def fail_candidate(strategy, inputs, config, output):
        if strategy.quick_model == "candidate":
            raise RuntimeError("candidate interrupted")
        return original(strategy, inputs, config, output)

    monkeypatch.setattr(flow.module, "forecast", fail_candidate)
    with pytest.raises(RuntimeError):
        flow.service.run()
    record = flow.store.runs()[0]
    assert len(record["attempts"]) == 1
    saved = deepcopy(record["attempts"][0])
    champion = flow.store.champion()
    test = flow.store.samples()[-5:]
    monkeypatch.setattr(flow.module, "forecast", original)
    flow.service.evaluate(record, champion, Strategy.model_validate(record["strategy"]), test)
    assert record["attempts"][0] == saved
    assert len(flow.calls) == 20
    assert len(record["attempts"]) == 20 and len(record["pairs"]) == 10
    flow.service.evaluate(record, champion, Strategy.model_validate(record["strategy"]), test)
    assert len(flow.calls) == 20 and len(record["pairs"]) == 10


def test_cached_analysis_rejects_changed_inputs(evolution_workflow):
    from tradingagents.evolution.engine import Strategy

    flow = evolution_workflow
    flow.service.run()
    record = flow.store.runs()[0]
    record["attempts"][0]["input_hash"] = "changed"
    with pytest.raises(ValueError, match="input hash changed"):
        flow.service.evaluate(record, {"strategy": record["previous_strategy"]},
                              Strategy.model_validate(record["strategy"]), flow.store.samples()[-5:])


def test_rate_limit_saves_progress_and_schedules_cooldown(evolution_workflow, monkeypatch):
    from datetime import timedelta

    flow = evolution_workflow
    original = flow.module.forecast

    class RateLimitError(Exception):
        status_code = 429

    def limited(strategy, inputs, config, output):
        if strategy.quick_model == "candidate":
            raise RateLimitError("rate limited")
        return original(strategy, inputs, config, output)

    monkeypatch.setattr(flow.module, "forecast", limited)
    flow.service.run()
    record = flow.store.runs()[0]
    assert record["status"] == "retry_wait" and record["completed_at"] is None
    assert record["next_retry_at"] == (flow.now[0] + timedelta(minutes=15)).isoformat(timespec="microseconds")
    assert len(record["attempts"]) == 1
    assert record["failed_analysis"]["variant"] == "candidate"


def test_recovery_waits_then_resumes_missing_analysis_and_preserves_holdout(evolution_workflow, monkeypatch):
    from datetime import timedelta

    flow = evolution_workflow
    original = flow.module.forecast

    class RateLimitError(Exception):
        status_code = 429

    def limited(strategy, inputs, config, output):
        if strategy.quick_model == "candidate":
            raise RateLimitError("throttled")
        return original(strategy, inputs, config, output)

    monkeypatch.setattr(flow.module, "forecast", limited)
    flow.service.run()
    initial = flow.store.runs()[0]
    assert len(initial["attempts"]) == 1
    assert flow.service.recover()["status"] == "retry_wait"
    assert len(flow.calls) == 1
    flow.now[0] += timedelta(minutes=15)
    monkeypatch.setattr(flow.module, "forecast", original)
    monkeypatch.setattr(flow.service, "propose", lambda *args: pytest.fail("Resume must not create a proposal"))
    result = flow.service.recover()
    assert result["status"] == "promoted"
    assert result["attempts"][0] == initial["attempts"][0]
    assert len(flow.calls) == 20 and len(result["pairs"]) == 10
    assert result["candidate_version"] == initial["candidate_version"]
    assert result["dataset"] == initial["dataset"]
    assert all(item["used_by"] == initial["day"] for item in flow.store.samples()[-5:])


def test_legacy_recovery_completes_diagnostic_only_without_repeating_successes(evolution_workflow, monkeypatch):
    flow = evolution_workflow
    original = flow.module.forecast

    def fail_candidate(strategy, inputs, config, output):
        if strategy.quick_model == "candidate":
            raise RuntimeError("interrupted")
        return original(strategy, inputs, config, output)

    monkeypatch.setattr(flow.module, "forecast", fail_candidate)
    with pytest.raises(RuntimeError):
        flow.service.run()
    record = flow.store.runs()[0]
    record.update(status="running", error=None)
    record.pop("replay_signature")
    record.pop("execution_settings")
    flow.store.save_run(record)
    parent = flow.store.champion()["id"]
    monkeypatch.setattr(flow.module, "forecast", original)
    result = flow.service.recover(day=record["day"])
    assert result["status"] == "diagnostic" and result["promotion_blocked"]
    assert len(flow.calls) == 20
    assert flow.store.champion()["id"] == parent


def test_recovery_stops_after_five_failures_of_same_analysis(evolution_workflow, monkeypatch):
    from datetime import timedelta

    flow = evolution_workflow
    original = flow.module.forecast

    class RateLimitError(Exception):
        status_code = 429

    def limited(strategy, inputs, config, output):
        if strategy.quick_model == "candidate":
            raise RateLimitError("throttled")
        return original(strategy, inputs, config, output)

    monkeypatch.setattr(flow.module, "forecast", limited)
    flow.service.run()
    for _ in range(4):
        flow.now[0] += timedelta(hours=3)
        flow.service.recover()
    result = flow.store.runs()[0]
    assert result["status"] == "failed"
    assert result["failed_analysis"]["failure_count"] == 5
    assert len(flow.calls) == 1
    assert flow.service.recover() is None


def test_recovery_uses_original_settings_when_runtime_config_changes(evolution_workflow, monkeypatch):
    from datetime import timedelta

    flow = evolution_workflow
    original = flow.module.forecast
    original_rounds = flow.service.config["max_debate_rounds"]

    def failure(*args):
        raise TimeoutError("temporary")

    monkeypatch.setattr(flow.module, "forecast", failure)
    flow.service.run()
    flow.now[0] += timedelta(minutes=2)
    flow.service.config["max_debate_rounds"] = original_rounds + 10

    def predict(strategy, inputs, config, output):
        assert config["max_debate_rounds"] == original_rounds
        return original(strategy, inputs, config, output)

    monkeypatch.setattr(flow.module, "forecast", predict)
    assert flow.service.recover()["status"] == "promoted"


def test_retry_after_is_respected_and_bad_request_is_not_transient(evolution_workflow):
    from datetime import timedelta
    from types import SimpleNamespace

    from tradingagents.evolution.retries import retry_delay, transient_provider_error

    error = Exception("throttle")
    error.status_code = 429
    error.response = SimpleNamespace(headers={"retry-after": "3600"})
    assert retry_delay(error, 1, evolution_workflow.now[0]) == timedelta(hours=1)
    assert transient_provider_error(error)
    error.status_code = 400
    assert not transient_provider_error(error)


def test_permanent_authentication_failure_is_not_automatically_retried(evolution_workflow, monkeypatch):
    flow = evolution_workflow

    class AuthenticationError(Exception):
        status_code = 401

    def fail(*args):
        raise AuthenticationError("invalid credentials")

    monkeypatch.setattr(flow.module, "forecast", fail)
    with pytest.raises(AuthenticationError):
        flow.service.run()
    result = flow.store.runs()[0]
    assert result["status"] == "failed" and not result.get("next_retry_at")
    assert flow.service.recover() is None


@pytest.mark.parametrize("change", ["candidate", "input_hash", "manifest", "executor"])
def test_recovery_rejects_changed_candidate_data_or_executor(evolution_workflow, monkeypatch, change):
    flow = evolution_workflow

    def failure(*args):
        raise TimeoutError("temporary")

    monkeypatch.setattr(flow.module, "forecast", failure)
    flow.service.run()
    record = flow.store.runs()[0]
    record.pop("next_retry_at")
    if change == "candidate":
        record["strategy"]["prompt"] += " changed"
    elif change == "input_hash":
        record["dataset"]["hashes"][0] = "changed"
    elif change == "manifest":
        record["dataset"]["manifest_hash"] = "changed"
    else:
        record["replay_signature"] = "changed"
    flow.store.save_run(record)
    result = flow.service.recover()
    assert result["status"] == "failed" and result["error"] == "ValueError"
    assert not flow.calls


def test_transient_provider_errors_do_not_trigger_immediate_freetext_fallback():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tradingagents.agents.structured import invoke_structured_or_freetext

    error = Exception("provider throttled")
    error.status_code = 429
    structured = SimpleNamespace(invoke=Mock(side_effect=error))
    plain = SimpleNamespace(invoke=Mock())
    with pytest.raises(Exception, match="throttled"):
        invoke_structured_or_freetext(structured, plain, "prompt", str, "test")
    plain.invoke.assert_not_called()


def test_explicit_retry_preserves_failed_proposal_audit(evolution_workflow, monkeypatch):
    flow = evolution_workflow
    original = flow.service.propose

    def fail(*args):
        raise RuntimeError("proposal input too large")

    monkeypatch.setattr(flow.service, "propose", fail)
    with pytest.raises(RuntimeError):
        flow.service.run()
    failed = flow.store.runs()[0]
    assert not any(sample["used_by"] for sample in flow.store.samples())
    monkeypatch.setattr(flow.service, "propose", original)
    assert flow.service.run()["status"] == "failed"
    assert not flow.calls
    result = flow.service.run(retry_failed=True)
    assert result["status"] == "promoted"
    assert result["retry_history"] == [failed]


def test_retry_rejects_already_consumed_holdout(evolution_workflow, monkeypatch):
    flow = evolution_workflow

    def fail(*args):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(flow.module, "forecast", fail)
    with pytest.raises(RuntimeError):
        flow.service.run()
    with pytest.raises(ValueError, match="holdout consumption"):
        flow.service.run(retry_failed=True)


def test_retry_checks_consumption_even_if_candidate_audit_is_missing(evolution_workflow):
    flow = evolution_workflow
    flow.store.reserve_test([flow.store.samples()[-1]["inputs"]["day"]], "2026-10-01")
    flow.store.save_run({"day": "2026-10-01", "status": "failed"})
    with pytest.raises(ValueError, match="holdout consumption"):
        flow.service.run(retry_failed=True)


def test_rejected_candidate_never_replaces_champion(evolution_workflow, monkeypatch):
    flow = evolution_workflow
    original = flow.module.forecast

    def predict(strategy, inputs, config, output):
        prediction, reports = original(strategy, inputs, config, output)
        if strategy.quick_model == "candidate":
            prediction = prediction.model_copy(update={"close": 101.0})
        return prediction, reports

    monkeypatch.setattr(flow.module, "forecast", predict)
    result = flow.service.run()
    assert result["status"] == "rejected"
    assert result["backtest"]["candidate"]["mae"] > result["backtest"]["current"]["mae"]
    assert flow.store.champion()["id"] == result["parent_version"]


@pytest.mark.parametrize(("current_close", "candidate_close", "status"), [
    (99.0, 130.0, "promoted"),
    (130.0, 99.0, "rejected"),
])
def test_champion_selection_prioritizes_direction_over_mae(
    evolution_workflow, monkeypatch, current_close, candidate_close, status,
):
    flow = evolution_workflow
    original = flow.module.forecast

    def predict(strategy, inputs, config, output):
        prediction, reports = original(strategy, inputs, config, output)
        close = candidate_close if strategy.quick_model == "candidate" else current_close
        return prediction.model_copy(update={"close": close, "high": 140.0}), reports

    monkeypatch.setattr(flow.module, "forecast", predict)
    result = flow.service.run()
    assert result["status"] == status
    assert result["backtest"]["paired_gain_metric"] == "direction_accuracy"
    assert "方向准确率" in result["gate_reason"]
    expected = result["candidate_version"] if status == "promoted" else result["parent_version"]
    assert flow.store.champion()["id"] == expected


def test_daily_diagnostic_backtest_does_not_promote_or_consume_fresh_samples(evolution_workflow):
    from datetime import timedelta

    flow = evolution_workflow
    flow.service.run()
    original_champion = flow.store.champion()["id"]
    flow.now[0] += timedelta(days=1)
    sample = deepcopy(flow.store.samples()[-1]["inputs"])
    sample["day"] = "2026-09-16"
    flow.store.freeze(sample)
    flow.store.label(sample["day"], {"close": 105, "available_at": "2026-09-16T21:00:00+00:00"})
    record = flow.service.run()
    assert record["status"] == "diagnostic"
    assert record["dataset"]["mode"] == "development_replay"
    assert not record["backtest"]["promote"]
    assert not record["backtest"]["checks"]["fresh_holdout"]
    assert flow.store.champion()["id"] == original_champion
    assert flow.store.samples()[-1]["used_by"] is None
    assert sample["day"] not in record["dataset"]["test_days"]
    assert record["branch"] == "rsi-msft/2026-10-02"
    assert len(flow.calls) == 40


def test_initial_data_shortage_never_calls_models(tmp_path, monkeypatch):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.evolution.runner import EvolutionRunner

    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    service = EvolutionRunner(DEFAULT_CONFIG, store, tmp_path)
    monkeypatch.setattr("tradingagents.evolution.runner.archive_branch", lambda *args, **kwargs: {"branch": "test"})

    def forbidden(*args):
        raise AssertionError("No models before sufficient data exists")

    monkeypatch.setattr(service, "propose", forbidden)
    result = service.run()
    assert result["status"] == "waiting_data"
    assert result["backtest"] is None


def test_interrupted_run_is_not_retried_on_same_holdout(evolution_workflow):
    flow = evolution_workflow
    flow.store.save_run({"day": "2026-10-01", "status": "running"})
    result = flow.service.run()
    assert result["status"] == "failed" and result["error"] == "InterruptedRun"
    assert not flow.calls and not flow.proposals


def test_evolution_dashboard_is_read_only_paginated_and_does_not_leak_inputs(tmp_path):
    from tradingagents.evolution.dashboard import snapshot

    missing = EvolutionStore(tmp_path / "missing.sqlite3", read_only=True)
    assert snapshot(missing)["history"]["items"] == []
    assert not missing.path.exists()
    store = EvolutionStore(tmp_path / "evolution.sqlite3")
    for number in range(1, 8):
        store.save_run({"day": f"2026-10-{number:02d}", "status": "waiting_data", "secret": "private"})
    first = snapshot(EvolutionStore(store.path, read_only=True))
    assert len(first["history"]["items"]) == 5
    assert "secret" not in first["history"]["items"][0]
    assert len(snapshot(store, before=first["history"]["next_before"])["history"]["items"]) == 2
