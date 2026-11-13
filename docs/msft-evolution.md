# MSFT strategy evolution

This workflow evolves the quick/deep model choice, a shared strategy prompt
overlay, and an allowlisted subset of existing tools. It reuses the market,
news, fundamentals, research, trader and risk graph. Social data is excluded
because its separate fetch path is not replayable. It does not change model
weights, arbitrary source code, evaluation rules or trading permissions.

Current policy: **five distinct candidates per day, backtested on all archived,
labelled history, with no development/test split** (`all-history-five-v1`).
In-sample improvement is sufficient for promotion; it does not prove future skill.

## Commands and schedule

Run from the TradingAgents project directory so the normal `.env` is loaded:

```sh
python -m cli.main daily tick MSFT --evolution --lead-minutes 120 --dry-run
python -m cli.main evolution tick --dry-run
python -m cli.main daily tick MSFT --evolution --lead-minutes 120
python -m cli.main evolution tick
```

The default authoritative stores are `~/.tradingagents/daily/daily.sqlite3`
and its sibling `evolution.sqlite3`. Both commands accept `--database` pointing
to the daily database. Do not run the same MSFT daily database from a second
scheduler without `--evolution`; the existing frozen-prediction uniqueness
constraint allows only one main forecast per day.

Models remain on the configured provider and endpoint. The default allowlist
contains the configured quick/deep models and the current champion's models.
Set `TRADINGAGENTS_RSI_MODELS=model-a,model-b` in the normal environment, or
use `evolution tick --models model-a,model-b`, to approve other available model
IDs. An unavailable model fails the candidate; there is no silent model fallback.

Install the existing five-minute daily timer, the MSFT-only override, and the
evolution service/timer:

```sh
mkdir -p ~/.config/systemd/user/tradingagents-daily@MSFT.service.d
cp deploy/systemd/tradingagents-daily@.service deploy/systemd/tradingagents-daily@.timer ~/.config/systemd/user/
cp deploy/systemd/tradingagents-daily@MSFT.service.d/evolution.conf ~/.config/systemd/user/tradingagents-daily@MSFT.service.d/
cp deploy/systemd/tradingagents-evolution.service deploy/systemd/tradingagents-evolution.timer ~/.config/systemd/user/
cp deploy/systemd/tradingagents-evolution-recovery.service deploy/systemd/tradingagents-evolution-recovery.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now tradingagents-daily@MSFT.timer tradingagents-evolution.timer tradingagents-evolution-recovery.timer
```

The evolution timer fires daily at **21:15 America/New_York** (including DST),
after the daily process normally settles the close. A missed timer fires after
restart. A database lock and a New York calendar-day key prevent duplicate
evolution. A separate recovery timer checks unfinished analyses every fifteen
minutes. Services have bounded execution time; successful analyses survive a
process exit and are not rerun. The nightly tick creates or resumes today's
five-candidate batch. The recovery service resumes older unfinished batches.
Work is serialized by the MSFT lock; provider limits or long evaluations can
make completion spill into later days.
Provider failures are recorded without exposing credentials in the dashboard.

Completed batches are idempotent for their New York date. Pending batches resume
without replacing generated candidates or successful predictions.

For a batch with existing candidates, use `evolution recover`, or
`evolution recover --day YYYY-MM-DD`, through the recovery service. The original
five candidates, baseline, models, complete session list and outcome manifest are
validated and retained. Completed analyses are keyed by session, repetition and
variant. Recovery rebuilds paired scores from those saved predictions and runs
only missing analyses. It never regenerates valid candidates or changes the
dataset while resuming. Historical `used_by` flags are ignored by this policy;
they remain only for auditing old records, not for gating new promotions.

429, timeout and connection/server errors pause the task as `retry_wait`.
Rate-limit cooldown starts at fifteen minutes, doubles to at most two hours,
and honors a longer provider Retry-After. Other transient or output-validation
errors start at one minute. Each analysis stops after five failures.
Authentication/configuration errors are not blindly
retried. A provider transient failure is not immediately repeated as a free-text
fallback. The miniapp reports saved progress and next-retry time. These controls
respect provider limits; they do not bypass account quotas.

Old-policy unfinished tasks are archived and marked superseded during migration,
not relabelled as successful all-history evaluations. Completed historical records
remain unchanged. New batches persist replay settings/signatures; altered replay
code, strategies or data stop recovery rather than silently mixing results.

Proposal generation uses deterministic, labelled head/tail excerpts with a
32,000 UTF-8-byte total input budget, a conservative upper bound on byte-level
tokenization that leaves room for schema and output. JSON in the proposal does
not escape non-ASCII text. Proposal generation receives bounded excerpts of the
latest ten labelled sessions; evaluation covers all available labelled history.
Excerpts shrink evenly to fit. Oversized metadata fails locally
before a provider call. The raw frozen inputs and their hashes are unchanged,
and backtests still use those full inputs, not the proposal excerpts.

## Frozen test data and leakage boundary

Before market open, a canonical bank of all allowed tool responses is captured
once, with tool arguments, retrieval timestamps, availability and reference
OHLCV. Price windows end at the previous session; news and current fundamentals
are captured before today's open. Optional missing sources are explicitly
recorded as unavailable; core-source failures stop collection. A retry reuses
the immutable bank rather than overwriting it with later information.

The forecast uses the same archived-tool path as backtests. Tool arguments
cannot cause a new vendor query: each enabled tool returns its named canonical
snapshot, with that fact stated explicitly. Changing tools changes which
snapshots the graph may access, not the archive available to each version.

Inputs are SHA-256 checked and stored separately from the eventual close and
its availability timestamp. Late forecasts and split-excluded outcomes do not
receive eligible test labels. Existing historical reports/memories are **not**
backfilled as point-in-time inputs. Incomplete past records cannot be made
causally valid by downloading today's version of their news or fundamentals.

Both versions receive identical inputs, fresh graph/model instances, disabled
legacy memory, no daily lessons, disabled checkpoints and separate report
directories. Candidate generation can use historical labels to propose improvements.
Each historical forecast still receives only its own archived premarket evidence,
not its outcome or another session's reports. The model's intrinsic
training knowledge cannot be erased; this is a prospective dataset, not a
claim to remove historical knowledge from an LLM's weights.

## Evaluation and promotion

At least **one complete, labelled archived session** is needed. A batch freezes
all eligible samples whose outcomes were available before its start. There is
no reserved test set, five-new-session wait, or exclusion of previously used
samples. A new session joins the next batch; running/recovering batches retain
their original manifest so all versions see exactly the same data.

Five distinct candidates are proposed in one structured call. Each may change
approved quick/deep models, a general prompt, and the allowed tool subset.
The baseline and each candidate are evaluated once per session by default
(`--repeats` accepts 1-5). Baseline predictions are shared across all five
comparisons, not rerun five times. Each candidate is ranked by direction accuracy
then MAE; the best complete candidate replaces the baseline only if strictly better.
A permanently failed candidate is recorded and excluded from ranking, not scored
as a success. The previous champion remains if no candidate improves.

For N historical sessions and R repetitions, a complete batch performs
**6 x N x R full graph analyses**, plus one proposal call. For example, 22 sessions
with the default single repetition require 132 analyses. Cost grows with history;
these are real paid provider calls. Progress is persisted after each success.

Version selection uses **direction accuracy first, then MAE only to break ties**
(`direction_then_mae_v2`):

- Higher direction accuracy wins, even if MAE is worse.
- Lower direction accuracy loses, even if MAE improves.
- Equal direction accuracy is decided by strictly lower MAE, without a 1% threshold.
- Equal direction accuracy and equal MAE keep the previous champion.

Direction means whether the predicted close is strictly above the frozen
previous-session close, compared with the actual direction; flat counts as
not up. The comparison uses unrounded accuracy over the identical paired runs.

Every pair for a ranked candidate must still complete with unchanged executor
source/settings. Promotion does not require independent holdouts. Brier,
interval scores and the naive baseline remain reported diagnostics, not vetoes
on the direction/MAE ordering. The one-sided 95% bootstrap lower bound is also
diagnostic: `paired_gain_metric` identifies direction-accuracy gain (fraction)
or, on a direction tie, MAE reduction (USD), clustered by day. Historical records
retain their original gate checks rather than being reclassified by the new rule.

This small-sample ranking is not proof of causality, trading profitability
or future superiority. A rejected, unchanged, incomplete or failed candidate
never replaces the champion. Promotion and the final run record commit in one
database transaction; the next MSFT forecast loads the new champion.

Each attempt creates `rsi-msft/YYYY-MM-DD` using Git plumbing without checking
out a branch or including uncommitted work. `rsi-msft.json` in that branch
contains versioned policy, summary, rationale, data hashes, predictions and
results, plus all five candidate definitions and their individual scores.
Raw vendor data remains outside Git in
SQLite. Commit source changes normally before production use; the executor hash
detects the actual Python sources but does not substitute for committing them.

The stock-miniapp **MSFT 进化** tab reads this store through an authenticated,
read-only API, with five attempts per page. It displays model/prompt/tool
differences for all five candidates, metrics, selected winner, branch/commit and
per-sample errors. Old records retain their original selection rules. Viewing
it neither launches jobs nor consumes manual-analysis quota. The mini-program
still needs its normal WeChat Developer Tools upload/release to reach users.

To stop future evolution without deleting history:

```sh
systemctl --user disable --now tradingagents-evolution.timer
systemctl --user disable --now tradingagents-evolution-recovery.timer
```

The daily MSFT process continues collecting evidence and using its last champion.
Back up both databases using SQLite's backup API; also retain the local Git refs.