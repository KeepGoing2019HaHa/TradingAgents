from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from filelock import FileLock

from tradingagents.daily.models import DailyReflection, timestamp
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component


class DailyStore:
    """SQLite is authoritative; forecasts and outcomes are insert-only snapshots."""

    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path).expanduser().resolve()
        self.read_only = read_only
        if read_only:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    day TEXT NOT NULL,
                    variant TEXT NOT NULL CHECK (variant IN ('memory', 'control')),
                    created_at TEXT NOT NULL,
                    eligible INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(ticker, day, variant)
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
                    available_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reflections (
                    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
                    available_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY,
                    prediction_id INTEGER NOT NULL UNIQUE REFERENCES predictions(id),
                    ticker TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    role TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    advice TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lesson_evidence (
                    lesson_id INTEGER NOT NULL REFERENCES lessons(id),
                    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
                    available_at TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    PRIMARY KEY(lesson_id, prediction_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(
            self.path.as_uri() + "?mode=ro" if self.read_only else str(self.path),
            uri=self.read_only, timeout=15,
        )
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def lock(self, ticker: str):
        # OS-backed lock is released even when a worker is killed mid-provider call.
        component = safe_ticker_component(normalize_symbol(ticker))
        return FileLock(f"{self.path}.{component}.lock", timeout=0)

    def event(self, ticker: str, phase: str, message: str, now: datetime):
        with self.connect() as db:
            db.execute("INSERT INTO events(at,ticker,phase,message) VALUES (?,?,?,?)",
                       (timestamp(now), ticker, phase, message))

    def recent_events(self, ticker: str) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM events WHERE ticker=? ORDER BY id DESC LIMIT 10", (ticker,)
            )]

    def prediction(self, ticker: str, day: str, variant: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT id,payload FROM predictions WHERE ticker=? AND day=? AND variant=?",
                (ticker, day, variant),
            ).fetchone()
        return {"id": row["id"], **json.loads(row["payload"])} if row else None

    def freeze(self, payload: dict) -> dict:
        with self.connect() as db:
            cursor = db.execute(
                """INSERT INTO predictions(ticker,day,variant,created_at,eligible,payload)
                   VALUES (?,?,?,?,?,?)""",
                (payload["ticker"], payload["day"], payload["variant"], payload["created_at"],
                 int(payload["eligible"]), json.dumps(payload, ensure_ascii=False, allow_nan=False)),
            )
            return {"id": cursor.lastrowid, **payload}

    def records(self, ticker: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT p.id,p.payload,o.payload AS outcome,r.payload AS reflection,
                       r.available_at AS reflected_at
                FROM predictions p
                LEFT JOIN outcomes o ON o.prediction_id=p.id
                LEFT JOIN reflections r ON r.prediction_id=p.id
                WHERE p.ticker=? ORDER BY p.day,p.variant
            """, (ticker,)).fetchall()
        return [
            {"id": row["id"], **json.loads(row["payload"]),
             "outcome": json.loads(row["outcome"]) if row["outcome"] else None,
             "reflection": json.loads(row["reflection"]) if row["reflection"] else None,
             "reflected_at": row["reflected_at"]}
            for row in rows
        ]

    def save_outcome(self, prediction_id: int, payload: dict, now: datetime):
        with self.connect() as db:
            db.execute(
                "INSERT INTO outcomes(prediction_id,available_at,payload) VALUES (?,?,?)",
                (prediction_id, timestamp(now), json.dumps(payload, ensure_ascii=False, allow_nan=False)),
            )

    def metric_records(self, ticker: str) -> list[dict]:
        """Read only fields needed for metrics, not accumulated analyst reports."""
        with self.connect() as db:
            rows = db.execute("""
                SELECT p.day,p.variant,p.eligible,
                       json_extract(p.payload,'$.signature') AS signature,
                       json_extract(p.payload,'$.reference.close') AS reference_close,
                       json_extract(p.payload,'$.forecast.up_probability') AS up_probability,
                       o.payload AS outcome, r.prediction_id AS reflection_id
                FROM predictions p
                LEFT JOIN outcomes o ON o.prediction_id=p.id
                LEFT JOIN reflections r ON r.prediction_id=p.id
                WHERE p.ticker=? ORDER BY p.day,p.variant
            """, (ticker,)).fetchall()
        return [
            {"day": row["day"], "variant": row["variant"], "eligible": bool(row["eligible"]),
             "signature": row["signature"], "reference": {"close": row["reference_close"]},
             "forecast": {"up_probability": row["up_probability"]},
             "outcome": json.loads(row["outcome"]) if row["outcome"] else None,
             "reflection": row["reflection_id"] is not None}
            for row in rows
        ]

    def history(self, ticker: str, *, before: str | None = None, limit: int = 6) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT p.id,
                       json_remove(p.payload,'$.reports','$.settings','$.lessons') AS payload,
                       o.payload AS outcome,r.payload AS reflection
                FROM predictions p
                LEFT JOIN outcomes o ON o.prediction_id=p.id
                LEFT JOIN reflections r ON r.prediction_id=p.id
                WHERE p.ticker=? AND p.variant='memory' AND (? IS NULL OR p.day<?)
                ORDER BY p.day DESC LIMIT ?
            """, (ticker, before, before, limit)).fetchall()
        return [
            {"id": row["id"], **json.loads(row["payload"]),
             "outcome": json.loads(row["outcome"]) if row["outcome"] else None,
             "reflection": json.loads(row["reflection"]) if row["reflection"] else None}
            for row in rows
        ]

    def save_reflection(self, record: dict, reflection: DailyReflection, now: datetime):
        allowed = {lesson["id"] for lesson in record["lessons"]}
        ids = [item.lesson_id for item in reflection.evaluations]
        if len(set(ids)) != len(ids) or not set(ids) <= allowed:
            raise ValueError("Reflection evaluated duplicate or unprovided lesson IDs")
        if not record["eligible"] or record["variant"] != "memory":
            raise ValueError("Only eligible memory forecasts may generate lessons")
        at = timestamp(now)
        with self.connect() as db:
            db.execute("INSERT INTO reflections VALUES (?,?,?)",
                       (record["id"], at, reflection.model_dump_json()))
            if candidate := reflection.candidate:
                db.execute("""
                    INSERT INTO lessons(prediction_id,ticker,available_at,role,condition,advice)
                    VALUES (?,?,?,?,?,?)
                """, (record["id"], record["ticker"], at, candidate.role,
                      candidate.condition, candidate.advice))
            db.executemany("INSERT INTO lesson_evidence VALUES (?,?,?,?,?)", [
                (item.lesson_id, record["id"], at, item.verdict, item.evidence)
                for item in reflection.evaluations
            ])

    def _lesson_states(self, ticker: str, as_of: datetime) -> list[dict]:
        """Status is derived only from evidence available before this forecast."""
        cutoff = timestamp(as_of)
        expiry = timestamp(as_of - timedelta(days=90))
        with self.connect() as db:
            rows = db.execute("""
                SELECT l.*, p.day AS source_day,
                       COUNT(CASE WHEN e.verdict='supports' THEN 1 END) AS supports,
                       COUNT(CASE WHEN e.verdict='contradicts' THEN 1 END) AS contradicts,
                       MAX(e.available_at) AS last_tested
                FROM lessons l
                JOIN predictions p ON p.id=l.prediction_id
                LEFT JOIN lesson_evidence e ON e.lesson_id=l.id AND e.available_at<?
                WHERE l.ticker=? AND l.available_at<?
                GROUP BY l.id
            """, (cutoff, ticker, cutoff)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            support, against = item["supports"], item["contradicts"]
            total = support + against
            last_tested = item["last_tested"] or item["available_at"]
            if against >= 3 and against >= support:
                item["status"] = "retired"
            elif last_tested <= expiry:
                item["status"] = "expired"
            else:
                item["status"] = (
                    "supported" if support >= 5 and support / total >= 0.7 else "candidate"
                )
            result.append(item)
        return result

    def lessons(self, ticker: str, as_of: datetime, limit: int = 8) -> list[dict]:
        result = [
            item for item in self._lesson_states(ticker, as_of)
            if item["status"] in {"candidate", "supported"}
        ]
        result.sort(key=lambda item: (item["status"] == "supported", item["available_at"]), reverse=True)
        return result[:limit]

    def lesson_history(
        self, ticker: str, as_of: datetime, *, before_id: int | None = None, limit: int = 11,
    ) -> list[dict]:
        items = [
            item for item in self._lesson_states(ticker, as_of)
            if before_id is None or item["id"] < before_id
        ]
        items.sort(key=lambda item: item["id"], reverse=True)
        return items[:limit]
