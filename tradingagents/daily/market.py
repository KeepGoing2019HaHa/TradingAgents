from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal
import yfinance as yf

from tradingagents.daily.models import timestamp
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import yf_retry

MARKET_ZONE = ZoneInfo("America/New_York")


class DataNotReady(RuntimeError):
    """A session cannot be evaluated with the available vendor data."""


def us_ticker(value: str) -> str:
    value = normalize_symbol(value)
    if not value or not re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:-[A-Z])?", value):
        raise ValueError("Daily mode requires a US stock/ETF ticker, e.g. NVDA or BRK-B")
    return value


@dataclass(frozen=True)
class Session:
    day: str
    open: datetime
    close: datetime

    def payload(self) -> dict:
        return {"day": self.day, "open": timestamp(self.open), "close": timestamp(self.close)}


class TradingCalendar:
    def __init__(self, exchange: str = "NASDAQ"):
        if exchange not in {"NASDAQ", "NYSE"}:
            raise ValueError("Daily mode currently supports NASDAQ and NYSE calendars only")
        self.exchange = exchange
        self.calendar = mcal.get_calendar(exchange)

    def sessions(self, start: date, end: date) -> list[Session]:
        schedule = self.calendar.schedule(start_date=start, end_date=end)
        return [
            Session(
                day=index.strftime("%Y-%m-%d"),
                open=row["market_open"].to_pydatetime(),
                close=row["market_close"].to_pydatetime(),
            )
            for index, row in schedule.iterrows()
        ]

    def on(self, day: date) -> Session | None:
        sessions = self.sessions(day, day)
        return sessions[0] if sessions else None

    def previous(self, session: Session) -> Session:
        day = date.fromisoformat(session.day)
        sessions = self.sessions(day - timedelta(days=45), day - timedelta(days=1))
        if not sessions:
            raise ValueError(f"No previous trading session found for {session.day}")
        return sessions[-1]


def fetch_bars(ticker: str, start: str, end: str) -> dict[str, dict]:
    """Fresh regular-session, non-dividend-adjusted bars; never reuse intraday cache."""
    instrument = yf.Ticker(us_ticker(ticker))
    frame = yf_retry(lambda: instrument.history(
        start=start,
        end=(date.fromisoformat(end) + timedelta(days=1)).isoformat(),
        interval="1d", auto_adjust=False, back_adjust=False,
        actions=True, prepost=False, raise_errors=True,
    ))
    if frame.empty:
        raise DataNotReady(f"No fresh daily bars for {ticker}: {start} through {end}")
    metadata = instrument.history_metadata
    if (metadata.get("currency") != "USD"
            or metadata.get("exchangeTimezoneName") != "America/New_York"
            or metadata.get("instrumentType") not in {"EQUITY", "ETF"}):
        raise ValueError(f"{ticker} is not a verified USD US equity/ETF on the supported schedule")
    bars = {}
    for index, row in frame.iterrows():
        day = index.strftime("%Y-%m-%d")
        if not start <= day <= end:
            continue
        values = {key.lower(): float(row[key]) for key in ("Open", "High", "Low", "Close", "Volume")}
        if any(not math.isfinite(v) for v in values.values()):
            raise DataNotReady(f"Non-finite OHLCV for {ticker} on {day}")
        if (min(values[k] for k in ("open", "high", "low", "close")) <= 0
                or values["volume"] <= 0
                or not values["low"] <= min(values["open"], values["close"])
                or not values["high"] >= max(values["open"], values["close"])):
            raise DataNotReady(f"Incomplete or inconsistent OHLCV for {ticker} on {day}")
        split = float(row.get("Stock Splits", 0))
        if not math.isfinite(split):
            raise DataNotReady(f"Invalid split data for {ticker} on {day}")
        bars[day] = {**values, "split": split, "source": "yfinance", "date": day}
    return bars


def exact_bar(bars: dict[str, dict], ticker: str, day: str) -> dict:
    if day not in bars:
        raise DataNotReady(f"Missing {ticker} bar for {day}; refusing to substitute an older session")
    return bars[day]
