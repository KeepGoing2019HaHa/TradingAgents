import os
import subprocess
from datetime import date
from pathlib import Path

from tradingagents.evolution.store import canonical


def archive_branch(repository: Path, day: str, payload: dict, *, parent_branch: str | None = None) -> dict:
    if date.fromisoformat(day).isoformat() != day:
        raise ValueError("Invalid evolution day")
    branch = f"rsi-msft/{day}"
    environment = {**os.environ, "GIT_AUTHOR_NAME": "TradingAgents RSI",
                   "GIT_AUTHOR_EMAIL": "rsi@localhost", "GIT_COMMITTER_NAME": "TradingAgents RSI",
                   "GIT_COMMITTER_EMAIL": "rsi@localhost"}

    def git(*arguments, data=None, check=True):
        return subprocess.run(["git", "-C", str(repository), *arguments], input=data,
                              capture_output=True, check=check,
                              env=environment, timeout=30)

    ref = f"refs/heads/{branch}"
    old = git("rev-parse", "--verify", ref, check=False)
    if old.returncode == 0:
        parent = old.stdout.decode().strip()
    else:
        ancestor = git("rev-parse", "--verify", f"refs/heads/{parent_branch}", check=False) if parent_branch else None
        parent = (ancestor.stdout if ancestor and ancestor.returncode == 0 else git("rev-parse", "HEAD").stdout).decode().strip()
    blob = git("hash-object", "-w", "--stdin", data=(canonical(payload) + "\n").encode()).stdout.strip()
    entries = git("ls-tree", "-z", parent).stdout.split(b"\0")
    entries = [entry for entry in entries if entry and entry.split(b"\t", 1)[1] != b"rsi-msft.json"]
    entries.append(b"100644 blob " + blob + b"\trsi-msft.json")
    tree = git("mktree", "-z", data=b"\0".join(entries) + b"\0").stdout.decode().strip()
    commit = git("commit-tree", tree, "-p", parent,
                 data=f"MSFT RSI {day}: {payload['status']}\n".encode()).stdout.decode().strip()
    git("update-ref", ref, commit, old.stdout.decode().strip() if old.returncode == 0 else "0" * 40)
    return {"branch": branch, "commit": commit}
