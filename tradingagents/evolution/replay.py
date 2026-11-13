from functools import wraps

from tradingagents.dataflows.config import get_config

TOOL_NAMES = (
    "get_stock_data", "get_verified_market_snapshot", "get_indicators",
    "get_news", "get_global_news", "get_fundamentals", "get_balance_sheet",
    "get_cashflow", "get_income_statement", "get_insider_transactions",
    "get_macro_indicators", "get_prediction_markets",
)
REQUIRED_TOOLS = {"get_stock_data", "get_verified_market_snapshot", "get_news", "get_fundamentals"}


def selected_tools(tools):
    policy = get_config().get("evolution_policy")
    if policy is None:
        return tools
    return tuple(item for item in tools if item.name in policy["tools"])


def frozen_tool(function):
    @wraps(function)
    def invoke(*args, **kwargs):
        inputs = get_config().get("evolution_inputs")
        if inputs is None:
            return function(*args, **kwargs)
        trade_date = kwargs.get("trade_date")
        if trade_date and trade_date != inputs["day"]:
            raise ValueError("Replay trade date does not match the frozen sample")
        policy = get_config().get("evolution_policy")
        if not policy or function.__name__ not in policy["tools"]:
            raise ValueError("Tool is disabled by the frozen strategy")
        entry = inputs["tools"].get(function.__name__)
        if entry is None:
            raise ValueError("Tool is absent from the frozen dataset; live fallback is forbidden")
        from tradingagents.evolution.store import canonical

        return (
            "Frozen MSFT evidence only. This tool serves the archived canonical window, "
            "not a new query. Do not infer missing data from the requested arguments.\n"
            + canonical(entry)
        )
    return invoke
