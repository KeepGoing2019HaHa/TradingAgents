# Daily forecasts and learning

## Scope

This is a live, research-only workflow for a configurable US equity or ETF
(NVDA by default). It predicts the **normal-session close in USD**, not a trade
execution price or a multi-day investment rating. It neither places orders nor
updates model weights. A rising accuracy curve is not guaranteed.

The existing analysis CLI and five-trading-day decision log remain unchanged.
Daily runs use a separate SQLite database, disable the legacy memory log and
checkpoints, and save their analysis reports under the database's sibling
`runs/<ticker>/<session>/<variant>/` tree. This separation prevents a partially
completed analysis from a different time from being resumed as a fresh
premarket prediction.

## Commands

Install/update using the project's normal `pip install -e .` (or `uv pip install -e .`).
The added dependencies are `pandas-market-calendars` and `filelock`.

```bash
tradingagents daily tick NVDA --dry-run
tradingagents daily tick NVDA
tradingagents daily report NVDA
tradingagents daily report NVDA --json
```

Every tick is a single process, not a daemon. It first retries earlier incomplete
outcomes/reflections, then performs today's forecast only inside the premarket
window. Outside that window it never manufactures a historical forecast.
No LLM or market-data calls are made by `--dry-run` or `report`.

Options for `tick`:

| Option | Default | Meaning |
|---|---|---|
| ticker argument | NVDA | US Yahoo ticker; use BRK-B, not BRK.B |
| `--database` | `~/.tradingagents/daily/daily.sqlite3` | Authoritative SQLite store |
| `--calendar` | NASDAQ | NASDAQ or NYSE |
| `--lead-minutes` | 60 | Start window before normal open; 5-240 |
| `--delay-minutes` | 30 | Wait after normal close; 15-240 |
| `--language` | Simplified Chinese | Language of generated analysis and reflection |
| `--control` | off | Additional full, independent no-memory graph; extra model cost |
| `--dry-run` | off | Print schedule in UTC, without provider calls |

LLM/provider settings and keys come from the existing environment and `.env`
mechanism. The working directory must be the project directory to load its
configuration. No keys are stored in the forecast's settings snapshot.
Use a verified US listing; the price provider checks USD currency, New York
timezone and EQUITY/ETF type. Other markets, crypto and futures are rejected.

## Reliable scheduling (Linux user systemd)

Templates assume the project is `~/TradingAgents` with interpreter
`~/TradingAgents/.venv/bin/python`. Edit the copied service if that differs, or
to change options. The instance name is the ticker. Do not put secrets in unit files.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/tradingagents-daily@.service ~/.config/systemd/user/
cp deploy/systemd/tradingagents-daily@.timer ~/.config/systemd/user/
systemctl --user daemon-reload

# Inspect the schedule first. The following enable command starts real paid work
# whenever jobs become due; run it only after configuring providers and budgets.
python -m cli.main daily tick NVDA --dry-run
systemctl --user enable --now tradingagents-daily@NVDA.timer tradingagents-daily@MSFT.timer tradingagents-daily@AAPL.timer

systemctl --user list-timers 'tradingagents-daily@*.timer'
journalctl --user -u tradingagents-daily@NVDA.service -n 100 --no-pager
```

For operation after logout and after reboot, an administrator may need to enable
lingering for the account (`loginctl enable-linger <account>`). This is a deployment
choice, not something the CLI enables automatically.

```bash
systemctl --user disable --now tradingagents-daily@NVDA.timer tradingagents-daily@MSFT.timer tradingagents-daily@AAPL.timer
```

The timer wakes every five minutes in the host's timezone; the application itself
uses `America/New_York` and the exchange calendar, handling DST, holidays and early
closes. Start with enough lead time for your selected models. An analysis finishing
at or after the open is marked late and excluded, even if started before opening.
If the primary run consumes the whole window, a control run is not started.

An OS lock per database and ticker prevents duplicate workers, including independent
timer/manual processes. Different tickers can analyze in parallel, with SQLite
serializing the short database writes. Each instance incurs its own provider usage.
The lock is automatically released on termination. A unique
`(ticker, session, variant)` key prevents replacement of a frozen prediction.
Prices and reflections commit separately: a failed reflection retries against
the same saved prices, not a newly revised outcome. Exit 1 signals failures,
exit 2 signals a concurrent worker. Consult the journal and retained error events;
there is no built-in email/push notification integration.

## What is frozen and scored

Each record contains the calendar session, previous-session price and timestamp,
analysis start, input-collection completion, forecast completion, model/configuration
signature, prompt version, reports, referenced lessons, and:

- predicted close;
- lower/upper endpoints of an **80% interval for the closing price**;
- probability of close strictly greater than the reference close;
- rationale and risks.

The collection timestamp is an upper bound for inputs gathered during that run,
not a claim that all live feeds share one timestamp. Eligible runs finish entirely
before opening. Reference OHLCV is archived, as are analysis reports; arbitrary
news/social feeds and every intermediate tool response are **not** fully archived.
This is not a point-in-time historical replay engine.

After the stored close plus delay, a fresh, exact-date regular-session bar is
required. Missing, nonfinite, zero-volume or inconsistent data stays pending; an
older bar is never substituted. Yahoo does not provide an immutable official
final-close certificate: delays and later corrections remain vendor limitations.
Prices use `auto_adjust=False` (no dividend adjustment). Splits affecting a delayed
evaluation exclude the sample; an unexplained revision of the frozen reference
blocks evaluation for investigation rather than manufacturing a return.

Deterministic metrics:

- MAE in dollars and mean absolute percentage error (denominator: actual close).
- Binary direction hit: predicted close > reference vs actual close > reference.
  Flat is "not up"; there is no post-hoc neutral threshold.
- 80% interval coverage, width, and interval score:
  width + 10 times the distance outside the interval (zero penalty inside).
- Brier score for upward probability; fixed 20-percentage-point calibration bins.
- Naive baseline: predict the frozen previous-session close.

Reports show cumulative and last 20/60 **exchange sessions**, not just the last
20/60 successful predictions, with sample counts and pending/late/excluded counts.
Unscored/late/split samples never enter aggregate accuracy or create lessons.
Markdown reports and JSON metrics are regenerable exports under
`reports/<ticker>/` next to the database; SQLite is authoritative. Back it up with
SQLite's backup API (or while no worker is running).

The stock-miniapp dashboard reads this same store with `DailyStore(..., read_only=True)`
and `tradingagents.daily.dashboard.snapshot`. It reuses the metric implementation,
returns five main forecasts and ten lessons per cursor page, and includes expired
and retired lessons for audit while keeping them out of future analysis prompts.
Only public research fields are returned, not provider settings, internal errors,
or full analyst reports. Reading the dashboard does not call data/model providers,
write files, run forecasts, or affect the existing manual-analysis quota.

## Experience lifecycle

Reflection receives only the frozen analysis, exact OHLCV outcome and computed
scores. It cannot infer a causal event from a price move and is explicitly told
that no postmarket news evidence was supplied. It may propose at most one
conditional lesson, or none, and evaluate only lesson IDs that were actually
provided to that prediction.

Lessons are same-ticker, role-filtered, and limited to eight per run:
market/news/social/fundamentals analysts receive relevant lessons; researchers,
managers, trader and risk debaters receive the combined shortlist. Conditions
must be checked against the current situation rather than automatically obeyed.
There is no semantic similarity model or cross-ticker transfer in this version.

New lessons are **candidate** hypotheses. Subsequent LLM evaluations record
support/contradiction/untested and evidence. Five supporting later samples with at
least 70% support give a **supported** label, not proof of causality. Three or more
contradictions at least as numerous as supports retire a lesson; lessons with no
evaluation for 90 days expire from retrieval. Original records remain in SQLite.
Availability timestamps and evidence filters are strictly before the next run's
start, so even a delayed retrospective reflection cannot leak into an earlier run.

Optional `--control` performs a separate no-memory analysis (no daily or legacy
lessons) using the same model settings and frozen reference. Only eligible pairs
with matching configuration signatures enter paired MAE. Live tools are fetched
separately and model generation is stochastic: this comparison is diagnostic,
not a clean causal ablation. Check versions, sample size, uncertainty and changing
market regimes before drawing conclusions. Forecast accuracy is not trading P&L.
