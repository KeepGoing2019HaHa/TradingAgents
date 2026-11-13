from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from filelock import FileLock


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class EvolutionStore:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path).expanduser().resolve()
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS samples (
                        day TEXT PRIMARY KEY, inputs TEXT NOT NULL,
                        input_hash TEXT NOT NULL, target TEXT, used_by TEXT
                    );
                    CREATE TABLE IF NOT EXISTS versions (
                        id TEXT PRIMARY KEY, parent TEXT, strategy TEXT NOT NULL,
                        created_at TEXT NOT NULL, promoted INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS runs (
                        day TEXT PRIMARY KEY, payload TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS archived_runs (
                        id TEXT PRIMARY KEY, day TEXT NOT NULL, payload TEXT NOT NULL
                    );
                """)

    @contextmanager
    def connect(self):
        uri = self.path.as_uri() + ("?mode=ro" if self.read_only else "?mode=rwc")
        db = sqlite3.connect(uri, uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def lock(self):
        return FileLock(str(self.path) + ".lock", timeout=0)

    def freeze(self, inputs: dict) -> str:
        if inputs.get("ticker") != "MSFT":
            raise ValueError("Evolution only supports MSFT")
        allowed = {"ticker", "day", "captured_at", "cutoff", "reference", "tools", "schema"}
        if set(inputs) != allowed:
            raise ValueError("Dataset inputs must contain only frozen premarket evidence")
        captured, cutoff = datetime.fromisoformat(inputs["captured_at"]), datetime.fromisoformat(inputs["cutoff"])
        if captured.tzinfo is None or cutoff.tzinfo is None or captured >= cutoff:
            raise ValueError("Evidence must be captured before the market opens")
        encoded, checksum = canonical(inputs), digest(inputs)
        with self.connect() as db:
            previous = db.execute("SELECT input_hash FROM samples WHERE day=?", (inputs["day"],)).fetchone()
            if previous and previous["input_hash"] != checksum:
                raise ValueError("A frozen dataset sample cannot be replaced")
            db.execute("INSERT OR IGNORE INTO samples(day,inputs,input_hash) VALUES (?,?,?)",
                       (inputs["day"], encoded, checksum))
        return checksum

    def label(self, day: str, target: dict):
        encoded = canonical(target)
        with self.connect() as db:
            previous = db.execute("SELECT target FROM samples WHERE day=?", (day,)).fetchone()
            if previous is None:
                return
            if previous["target"] is not None and previous["target"] != encoded:
                raise ValueError("A frozen outcome cannot be replaced")
            db.execute("UPDATE samples SET target=? WHERE day=?", (encoded, day))

    def samples(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM samples ORDER BY day").fetchall()
        result = []
        for row in rows:
            inputs = json.loads(row["inputs"])
            if digest(inputs) != row["input_hash"]:
                raise ValueError("Dataset checksum mismatch")
            result.append({"inputs": inputs, "input_hash": row["input_hash"],
                           "target": json.loads(row["target"]) if row["target"] else None,
                           "used_by": row["used_by"]})
        return result

    def version(self, strategy: dict, at: str, parent: str | None = None) -> str:
        version_id = digest({"strategy": strategy, "parent": parent})[:16]
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO versions(id,parent,strategy,created_at) VALUES (?,?,?,?)",
                       (version_id, parent, canonical(strategy), at))
        return version_id

    def champion(self) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM versions WHERE promoted=1").fetchone()
        return {**dict(row), "strategy": json.loads(row["strategy"])} if row else None

    def strategy(self, version_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT strategy FROM versions WHERE id=?", (version_id,)).fetchone()
        return json.loads(row["strategy"]) if row else None

    def promote(self, version_id: str):
        with self.connect() as db:
            if not db.execute("SELECT id FROM versions WHERE id=?", (version_id,)).fetchone():
                raise ValueError("Unknown strategy version")
            db.execute("UPDATE versions SET promoted=0")
            db.execute("UPDATE versions SET promoted=1 WHERE id=?", (version_id,))

    def reserve_test(self, days: list[str], run_day: str):
        with self.connect() as db:
            for day in days:
                updated = db.execute("UPDATE samples SET used_by=? WHERE day=? AND used_by IS NULL AND target IS NOT NULL",
                                     (run_day, day))
                if updated.rowcount != 1:
                    raise ValueError("Holdout sample already consumed or not labelled")

    def save_run(self, record: dict):
        with self.connect() as db:
            db.execute("INSERT INTO runs(day,payload) VALUES (?,?) ON CONFLICT(day) DO UPDATE SET payload=excluded.payload",
                       (record["day"], canonical(record)))

    def replace_policy_run(self, record: dict):
        with self.connect() as db:
            previous = db.execute("SELECT payload FROM runs WHERE day=?", (record["day"],)).fetchone()
            if previous:
                payload = previous["payload"]
                db.execute("INSERT OR IGNORE INTO archived_runs VALUES (?,?,?)",
                           (hashlib.sha256(payload.encode()).hexdigest(), record["day"], payload))
            db.execute("INSERT INTO runs(day,payload) VALUES (?,?) ON CONFLICT(day) DO UPDATE SET payload=excluded.payload",
                       (record["day"], canonical(record)))

    def supersede_legacy_runs(self, policy: str):
        with self.connect() as db:
            rows = db.execute("SELECT day,payload FROM runs WHERE json_extract(payload,'$.status') "
                              "IN ('running','retry_wait')").fetchall()
            for row in rows:
                record = json.loads(row["payload"])
                if record.get("policy") == policy:
                    continue
                db.execute("INSERT OR IGNORE INTO archived_runs VALUES (?,?,?)",
                           (hashlib.sha256(row["payload"].encode()).hexdigest(), row["day"], row["payload"]))
                record.update(status="superseded", gate_reason="已切换为每日五候选、全历史回测规则；旧任务结果保留，不再续跑。")
                db.execute("UPDATE runs SET payload=? WHERE day=?", (canonical(record), row["day"]))

    def finish(self, record: dict, *, promote: str | None = None):
        with self.connect() as db:
            if promote:
                row = db.execute("SELECT id FROM versions WHERE promoted=1").fetchone()
                if not row or row["id"] != record["parent_version"]:
                    raise ValueError("Champion changed during evaluation")
                if not db.execute("SELECT id FROM versions WHERE id=?", (promote,)).fetchone():
                    raise ValueError("Unknown candidate")
                db.execute("UPDATE versions SET promoted=0")
                db.execute("UPDATE versions SET promoted=1 WHERE id=?", (promote,))
            db.execute("UPDATE runs SET payload=? WHERE day=?", (canonical(record), record["day"]))

    def runs(self, *, before: str | None = None, limit: int = 20) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM runs WHERE (? IS NULL OR day<?) ORDER BY day DESC LIMIT ?",
                              (before, before, min(max(limit, 1), 100))).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def run_record(self, day: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE day=?", (day,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def pending_runs(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM runs WHERE json_extract(payload,'$.status') "
                              "IN ('running','retry_wait','failed') ORDER BY day").fetchall()
        return [json.loads(row["payload"]) for row in rows]
