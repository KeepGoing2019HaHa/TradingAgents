from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingagents.daily.models import DailyForecast, timestamp, utc_now
from tradingagents.dataflows.config import run_config
from tradingagents.evolution.replay import REQUIRED_TOOLS, TOOL_NAMES
from tradingagents.evolution.store import EvolutionStore, canonical, digest

BASE_PROMPT = (
    "Forecast MSFT's regular-session closing price, not a multi-day rating. "
    "Use only supplied point-in-time evidence. Compare momentum with volatility and "
    "event risk; explicitly calibrate uncertainty and avoid unsupported precision."
)
REPORT_KEYS = (
    "market_report", "news_report", "fundamentals_report", "investment_plan",
    "trader_investment_plan", "final_trade_decision",
)


class Strategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    quick_model: str = Field(min_length=1, max_length=150)
    deep_model: str = Field(min_length=1, max_length=150)
    prompt: str = Field(min_length=20, max_length=6000)
    tools: list[str] = Field(min_length=4, max_length=len(TOOL_NAMES))

    @model_validator(mode="after")
    def valid_tools(self):
        if (len(set(self.tools)) != len(self.tools) or not set(self.tools) >= REQUIRED_TOOLS
                or not set(self.tools) <= set(TOOL_NAMES)):
            raise ValueError("Strategy must select unique allowed tools and retain core evidence")
        return self


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Strategy
    summary: str = Field(min_length=1, max_length=2000)
    rationale: str = Field(min_length=1, max_length=4000)


def baseline(config: dict, store: EvolutionStore, at: str) -> dict:
    if champion := store.champion():
        Strategy.model_validate(champion["strategy"])
        return champion
    strategy = Strategy(quick_model=config["quick_think_llm"], deep_model=config["deep_think_llm"],
                        prompt=BASE_PROMPT, tools=list(TOOL_NAMES))
    version_id = store.version(strategy.model_dump(), at)
    store.promote(version_id)
    return store.champion()


def capture(session, reference: dict, config: dict, clock=utc_now) -> dict:
    from tradingagents.agents import tools

    day = session.day
    previous = reference["date"]
    start = (date.fromisoformat(previous) - timedelta(days=365)).isoformat()
    news_start = (date.fromisoformat(day) - timedelta(days=7)).isoformat()
    calls = {
        "get_stock_data": {"symbol": "MSFT", "start_date": start, "end_date": previous},
        "get_verified_market_snapshot": {"symbol": "MSFT", "curr_date": previous, "look_back_days": 30},
        "get_indicators": {"symbol": "MSFT", "curr_date": previous, "look_back_days": 30,
                           "indicator": "close_50_sma,close_200_sma,close_10_ema,macd,macds,macdh,rsi,boll,boll_ub,boll_lb,atr,vwma"},
        "get_news": {"ticker": "MSFT", "start_date": news_start, "end_date": day},
        "get_global_news": {"curr_date": day, "look_back_days": 7, "limit": 10},
        "get_fundamentals": {"ticker": "MSFT", "curr_date": day},
        "get_balance_sheet": {"ticker": "MSFT", "curr_date": day},
        "get_cashflow": {"ticker": "MSFT", "curr_date": day},
        "get_income_statement": {"ticker": "MSFT", "curr_date": day},
        "get_insider_transactions": {"ticker": "MSFT"},
        "get_macro_indicators": {"indicator": "fed_funds_rate", "curr_date": day},
        "get_prediction_markets": {"topic": "Federal Reserve interest rate", "limit": 6},
    }
    entries = {}
    with run_config({**config, "evolution_inputs": None, "evolution_policy": None}):
        for name, arguments in calls.items():
            if clock() >= session.open:
                raise ValueError("Snapshot collection crossed market open")
            arguments = {**arguments, "trade_date": day}
            try:
                output = str(getattr(tools, name).invoke(arguments))
                if not output.strip():
                    raise ValueError("Empty tool response")
                available = True
            except Exception as exc:
                if name in REQUIRED_TOOLS:
                    raise
                output, available = f"Unavailable: {type(exc).__name__}", False
            entries[name] = {"args": arguments, "output": output, "available": available,
                             "captured_at": timestamp(clock())}
    return {"schema": 1, "ticker": "MSFT", "day": day, "reference": reference,
            "captured_at": timestamp(clock()), "cutoff": timestamp(session.open), "tools": entries}


def forecast(strategy: Strategy, inputs: dict, config: dict, output: Path) -> tuple[DailyForecast, dict]:
    from tradingagents.daily.runner import invoke_validated
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    allowed = {"ticker", "day", "captured_at", "cutoff", "reference", "tools", "schema"}
    if inputs["ticker"] != "MSFT" or set(inputs) != allowed:
        raise ValueError("Only frozen MSFT inputs may enter a replay")
    if not set(strategy.tools) <= set(inputs["tools"]):
        raise ValueError("Missing archived tool data; replay cannot fetch replacements")
    settings = copy.deepcopy(config)
    settings.update(
        quick_think_llm=strategy.quick_model, deep_think_llm=strategy.deep_model,
        memory_log_path=None, checkpoint_enabled=False, evolution_inputs=inputs,
        evolution_policy=strategy.model_dump(), results_dir=str(output),
        data_cache_dir=str(output / "cache"),
    )
    context = {
        "all": (
            f"MSFT premarket forecast for {inputs['day']}. Evidence cutoff: {inputs['captured_at']}. "
            f"Previous-session OHLCV: {canonical(inputs['reference'])}. "
            "Ignore all knowledge of events after this cutoff, including remembered historical prices. "
            "No historical memory, past reports or external tools are available. "
            "Tools serve fixed archived windows; never infer a fresh query was executed. "
            f"Only these tools are available: {', '.join(strategy.tools)}. "
            "Skip instructions asking for disabled tools.\nStrategy instructions:\n" + strategy.prompt
        )
    }
    with run_config(settings):
        graph = TradingAgentsGraph(selected_analysts=("market", "news", "fundamentals"),
                                   config=settings, analysis_context=context)
        state, _ = graph.propagate("MSFT", inputs["day"])
        reports = {key: state.get(key, "") for key in REPORT_KEYS}
        prompt = (
            context["all"] + f"\nWrite explanations in {settings.get('output_language', 'English')}. "
            "Return today's close in USD, an 80% predictive interval for that close, and "
            "probability of closing strictly above the previous-session close. "
            "Use only these newly generated reports:\n" + canonical(reports)
        )
        result = invoke_validated(graph.quick_thinking_llm, DailyForecast, prompt)
    return result, reports


def execution_settings(config: dict) -> dict:
    return {key: config.get(key) for key in (
        "llm_provider", "backend_url", "temperature", "max_tokens", "max_debate_rounds",
        "max_risk_discuss_rounds", "max_recur_limit", "output_language", "anthropic_effort",
        "openai_reasoning_effort", "google_thinking_level",
    )}


def execution_signature(config: dict, *, replay_only: bool = False) -> str:
    settings = execution_settings(config)
    root = Path(__file__).resolve().parents[1]
    settings["source"] = {str(path.relative_to(root)): digest(path.read_text())
                          for path in sorted(root.rglob("*.py")) if not replay_only
                          or path.relative_to(root).parts[0] != "evolution"
                          or path.name in {"engine.py", "replay.py"}}
    return digest(settings)
