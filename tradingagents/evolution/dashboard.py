from tradingagents.daily.models import timestamp, utc_now
from tradingagents.evolution.store import EvolutionStore


def public_candidate(candidate: dict, dataset: dict) -> dict:
    result = {key: candidate.get(key) for key in (
        "number", "status", "candidate_version", "strategy", "summary", "rationale",
        "backtest", "error", "next_retry_at", "gate_reason",
    )}
    result["strategy"] = dict(result["strategy"] or {})
    result["text_truncated"] = False
    for owner, key, limit in ((result, "summary", 500), (result, "rationale", 800),
                              (result["strategy"], "prompt", 1000)):
        text = owner.get(key) or ""
        if len(text) > limit:
            owner[key] = text[:limit] + "..."
            result["text_truncated"] = True
    result["completed_analyses"] = sum(attempt["variant"] == "candidate" for attempt in candidate.get("attempts", []))
    result["expected_analyses"] = len(dataset.get("test_days", [])) * dataset.get("repeats", 0)
    return result


def snapshot(store: EvolutionStore, *, before: str | None = None) -> dict:
    result = {"ticker": "MSFT", "generated_at": timestamp(utc_now()), "champion": None,
              "summary": {"samples": 0, "labelled": 0, "fresh": 0},
              "history": {"items": [], "next_before": None}}
    result["summary"]["candidate_count"] = 5
    if not store.path.is_file():
        return result
    champion = store.champion()
    if champion:
        result["champion"] = {key: champion[key] for key in ("id", "parent", "strategy", "created_at")}
    with store.connect() as db:
        counts = db.execute("SELECT COUNT(*) AS samples, COUNT(target) AS labelled, "
                            "COUNT(CASE WHEN target IS NOT NULL AND used_by IS NULL THEN 1 END) AS fresh FROM samples").fetchone()
    result["summary"] = dict(counts)
    result["summary"]["candidate_count"] = 5
    runs = store.runs(before=before, limit=6)
    public_keys = (
        "day", "ticker", "status", "started_at", "completed_at", "parent_version", "candidate_version",
        "strategy", "previous_strategy", "summary", "rationale", "gate_reason", "backtest", "dataset",
        "branch", "commit", "execution_signature", "error",
        "next_retry_at", "failed_analysis", "recovery_count",
        "policy", "best_candidate", "candidate_count",
    )
    for record in runs[:5]:
        item = {key: record.get(key) for key in public_keys}
        item["sample_results"] = [{"day": pair["day"], "repeat": pair["repeat"],
                                   "current_close": pair["current"]["close"],
                                   "candidate_close": pair["candidate"]["close"],
                                   "current_error": pair["current_score"]["absolute_error"],
                                   "candidate_error": pair["candidate_score"]["absolute_error"]}
                                  for pair in record.get("pairs", [])[-100:]]
        item["sample_results_total"] = len(record.get("pairs", []))
        item["completed_pairs"] = record.get("completed_pairs", len(record.get("pairs", [])))
        item["completed_analyses"] = len(record.get("attempts", []))
        dataset = record.get("dataset", {})
        item["dataset"] = {key: dataset.get(key) for key in (
            "mode", "total", "fresh", "minimum_train", "minimum_test", "repeats", "cutoff", "manifest_hash",
        )} if dataset else None
        item["expected_analyses"] = record.get("expected_analyses", len(dataset.get("test_days", [])) * dataset.get("repeats", 0) * 2)
        item["candidates"] = [public_candidate(candidate, dataset) for candidate in record.get("candidates", [])]
        result["history"]["items"].append(item)
    if len(runs) > 5:
        result["history"]["next_before"] = runs[4]["day"]
    return result
