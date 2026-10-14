from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile

from tradingagents.daily.market import MARKET_ZONE, TradingCalendar, us_ticker
from tradingagents.daily.models import timestamp, utc_now
from tradingagents.daily.store import DailyStore


def statistics(records: list[dict], days: set[str] | None = None) -> dict:
    selected = [r for r in records if days is None or r["day"] in days]
    primary = [r for r in selected if r["variant"] == "memory"]
    valid = [r for r in primary if r["eligible"] and r["outcome"] and r["outcome"]["scores"]]
    scores = [r["outcome"]["scores"] for r in valid]
    result = {
        "predictions": len(primary), "scored": len(valid),
        "late": sum(not r["eligible"] for r in primary),
        "pending": sum(r["eligible"] and not r["outcome"] for r in primary),
        "excluded": sum(bool(r["outcome"] and r["outcome"]["excluded"]) for r in primary),
        "awaiting_reflection": sum(not r["reflection"] for r in valid),
        "signatures": sorted({r["signature"] for r in valid}),
    }
    for field in (
        "absolute_error", "absolute_percentage_error", "direction_hit", "interval_hit",
        "interval_width", "interval_score", "brier",
        "naive_absolute_error", "naive_absolute_percentage_error",
    ):
        result[field] = mean(float(item[field]) for item in scores) if scores else None
    calibration = []
    for bucket in range(5):
        samples = [r for r in valid if min(int(r["forecast"]["up_probability"] * 5), 4) == bucket]
        calibration.append({
            "from": bucket / 5, "to": (bucket + 1) / 5, "count": len(samples),
            "mean_probability": mean(r["forecast"]["up_probability"] for r in samples) if samples else None,
            "up_frequency": mean(float(r["outcome"]["scores"]["actual_up"]) for r in samples) if samples else None,
        })
    result["calibration"] = calibration
    controls = {
        r["day"]: r for r in selected
        if r["variant"] == "control" and r["eligible"] and r["outcome"] and r["outcome"]["scores"]
    }
    pairs = [
        (r, controls[r["day"]]) for r in valid
        if r["day"] in controls and controls[r["day"]]["signature"] == r["signature"]
        and controls[r["day"]]["reference"]["close"] == r["reference"]["close"]
    ]
    result["control_pairs"] = len(pairs)
    result["paired_memory_mae"] = mean(r["outcome"]["scores"]["absolute_error"] for r, _ in pairs) if pairs else None
    result["paired_control_mae"] = mean(r["outcome"]["scores"]["absolute_error"] for _, r in pairs) if pairs else None
    return result


def summarize(store: DailyStore, ticker: str, calendar: TradingCalendar, now=None) -> dict:
    now = now or utc_now()
    records = store.metric_records(ticker)
    day = now.astimezone(MARKET_ZONE).date()
    closed = [
        session.day for session in calendar.sessions(day - timedelta(days=150), day)
        if session.close < now
    ]
    return {
        "ticker": ticker, "generated_at": timestamp(now), "calendar": calendar.exchange,
        "all": statistics(records),
        "last_20_sessions": statistics(records, set(closed[-20:])),
        "last_60_sessions": statistics(records, set(closed[-60:])),
        "active_lessons": store.lessons(ticker, now),
        "recent_errors": store.recent_events(ticker),
    }


def _number(value, percent=False) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}" if percent else f"{value:.4f}"


def render_summary(summary: dict) -> str:
    lines = [
        f"# {summary['ticker']} daily forecast evaluation",
        f"\nGenerated: {summary['generated_at']} | Calendar: {summary['calendar']}\n",
        "| Window | Scored | MAE ($) | Naive MAE ($) | Direction | 80% coverage | Width ($) | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("All", "all"), ("20 sessions", "last_20_sessions"), ("60 sessions", "last_60_sessions")):
        data = summary[key]
        lines.append(
            f"| {label} | {data['scored']} | {_number(data['absolute_error'])} | "
            f"{_number(data['naive_absolute_error'])} | {_number(data['direction_hit'], True)} | "
            f"{_number(data['interval_hit'], True)} | {_number(data['interval_width'])} | {_number(data['brier'])} |"
        )
    data = summary["all"]
    lines.extend([
        f"\nPending outcomes: {data['pending']}; pending reflections: {data['awaiting_reflection']}; "
        f"late: {data['late']}; corporate-action exclusions: {data['excluded']}.",
        f"\nPaired no-memory controls: {data['control_pairs']}; "
        f"memory MAE: {_number(data['paired_memory_mae'])}; "
        f"control MAE: {_number(data['paired_control_mae'])}.",
        "\nDirection means close > frozen previous close; flat is not up. "
        "The naive forecast is the frozen previous close. No trade execution, costs or P&L are simulated. "
        "Control runs fetch live inputs separately, so this is not a controlled causal experiment. "
        "Small samples, mixed model versions and changing markets cannot prove improvement.",
        f"\nScored configuration signatures: {', '.join(data['signatures']) or 'none'}",
        "\n## Probability calibration\n",
        "| Probability band | Samples | Mean forecast | Actual up frequency |",
        "|---|---:|---:|---:|",
    ])
    for bucket in data["calibration"]:
        lines.append(
            f"| {bucket['from']:.0%}-{bucket['to']:.0%} | {bucket['count']} | "
            f"{_number(bucket['mean_probability'], True)} | {_number(bucket['up_frequency'], True)} |"
        )
    lines.append("\n## Active lessons (provisional, not causal rules)\n")
    for lesson in summary["active_lessons"]:
        lines.append(
            f"- #{lesson['id']} [{lesson['role']}, {lesson['status']}, "
            f"support={lesson['supports']}, against={lesson['contradicts']}] "
            f"{lesson['condition']} -> {lesson['advice']}"
        )
    if summary["recent_errors"]:
        lines.append("\n## Recent errors (retained even after a successful retry)\n")
        lines.extend(f"- {item['at']} {item['phase']}: {item['message']}" for item in summary["recent_errors"])
    return "\n".join(lines) + "\n"


def render_day(records: list[dict]) -> str:
    lines = [f"# {records[0]['ticker']} / {records[0]['day']}\n"]
    for record in records:
        forecast = record["forecast"]
        lines.extend([
            f"## {record['variant']}",
            f"\nStatus: {'premarket' if record['eligible'] else 'LATE - excluded'}",
            f"\nStarted: {record['started_at']}; inputs collected by: {record['data_as_of']}; "
            f"frozen: {record['created_at']}",
            f"\nReference: ${record['reference']['close']} ({record['reference']['date']})",
            f"\nPredicted close: ${forecast['close']}; 80% interval: "
            f"${forecast['low']} - ${forecast['high']}; P(up): {forecast['up_probability']:.1%}",
            f"\n### Analysis\n{forecast['rationale']}\n\n### Risks\n{forecast['risks']}",
            f"\nLessons read: {', '.join(str(item['id']) for item in record['lessons']) or 'none'}",
            f"\nConfiguration signature: {record['signature']}",
        ])
        if outcome := record["outcome"]:
            lines.append(f"\n### Actual close\n${outcome['bar']['close']}")
            lines.append("\n```json\n" + json.dumps(outcome, indent=2, ensure_ascii=False) + "\n```")
        if reflection := record["reflection"]:
            lines.append(f"\n### Reflection\n{reflection['analysis']}")
            if candidate := reflection["candidate"]:
                lines.append(f"\nCandidate lesson: {candidate['condition']} -> {candidate['advice']}")
        elif record["eligible"] and record["variant"] == "memory":
            lines.append("\nReflection: pending or excluded; see outcome.")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(text)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def export_reports(store: DailyStore, ticker: str, calendar: TradingCalendar, now=None) -> Path:
    ticker = us_ticker(ticker)
    directory = store.path.parent / "reports" / ticker
    summary = summarize(store, ticker, calendar, now)
    _atomic_write(directory / "summary.md", render_summary(summary))
    _atomic_write(directory / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    by_day: dict[str, list[dict]] = {}
    for record in store.records(ticker):
        by_day.setdefault(record["day"], []).append(record)
    for day, records in by_day.items():
        _atomic_write(directory / f"{day}.md", render_day(records))
    return directory
