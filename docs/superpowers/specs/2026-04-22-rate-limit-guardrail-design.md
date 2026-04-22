# Rate-Limit Guardrail for Anthropic API — Design

**Date:** 2026-04-22
**Status:** Approved (brainstorming phase complete)
**Scope:** Per-call-site retry wrapper around every `ChatAnthropic.invoke()` in the debate agent. Configurable via `.env`. Reactive only — no proactive token budgeting.

---

## Goal

Prevent the debate process from crashing when a single LLM call hits the company's 10k tokens/minute Anthropic rate limit. When a `429` comes back, the guardrail sleeps (honoring the server's `retry-after` header) and retries. The debate resumes transparently from where it paused.

Success criteria: a debate that previously died mid-round with a `RateLimitError` now completes, with each retry visible as its own span in LangSmith and Langfuse so the rate-limit event is never hidden.

## Non-goals

- **Proactive client-side throttling.** No token counting, no pre-call budgeting. The guardrail reacts to 429s, it doesn't predict them. (User explicitly chose this scope.)
- **Graceful degradation on exhaustion.** After `RATE_LIMIT_MAX_RETRIES` retries, the exception propagates and the debate dies. Matches the project's "surface failures, never hide them" philosophy.
- **Retry on non-rate-limit errors.** Network blips, 500s, auth failures propagate immediately. A 401 shouldn't trigger a 7-minute retry loop.
- **Shared state across processes.** Each Python process retries independently. If you run two debates in parallel, each has its own retry budget. No cross-process coordination.
- **Persistent state across crashes.** A debate that exhausts retries loses its transcript. Persistence is already an existing non-goal of this project.
- **Custom logging infrastructure.** Retry notifications print to stderr — matches the project's existing stderr-for-diagnostics pattern (`main.py:54-62`).

---

## Design decisions and their alternatives

### 1. Reactive vs proactive throttling
**Alternatives considered:** (a) token-bucket with tiktoken-style estimation before each call; (b) reactive 429 handling only; (c) both layered.
**Chosen:** (b) reactive only. **Why:** user confirmed crash-on-429 is the observed failure mode. Proactive throttling requires maintaining a sliding-window counter + accurate token estimates per call — nontrivial engineering for speculative benefit. Start simple; add (a) if bursts keep hitting the wall.

### 2. Wrap layer: per-call-site vs factory-level
**Alternatives considered:** (a) wrap `_model_for()` so all calls on the returned client retry transparently; (b) explicit `invoke_with_retry(llm, messages, config)` at each call site; (c) LangChain's `Runnable.with_retry()`.
**Chosen:** (b) per-call-site helper. **Why:** explicit at the call site — the retry behavior is visible in the node code, not hidden in a factory decorator. Matches the project's style (the ReAct loop is also explicit in-node, not abstracted away).

### 3. Retry-after header awareness
**Alternatives considered:** (a) pure exponential backoff; (b) respect `retry-after` header if present, fall back to exponential.
**Chosen:** (b) respect the header. **Why:** cheap precision. The Anthropic SDK exposes `e.response.headers.get("retry-after")` — parsing it is ~5 lines. Server-directed sleep is always at least as accurate as a guess.

### 4. Terminal behavior on retry exhaustion
**Alternatives considered:** (a) crash (propagate the exception); (b) degrade (return a sentinel string so the debate finishes with a visible gap).
**Chosen:** (a) crash. **Why:** this is a learning playground. Hiding a persistent rate-limit failure behind a placeholder would defeat the observability philosophy in the README.

### 5. Module placement
**Alternatives considered:** (a) inline into `debate/nodes.py`; (b) new module `debate/rate_limit.py`.
**Chosen:** (b) new module. **Why:** `nodes.py` already carries graph orchestration + model factory + ReAct loop. Rate-limit handling is a fourth concern. Isolating it keeps `nodes.py` focused and makes the guardrail independently testable (future-proofing — no tests today per project convention).

### 6. Configuration source
**Alternatives considered:** (a) Python constants in `nodes.py`; (b) `.env`-driven with defaults.
**Chosen:** (b) `.env`-driven. **Why:** user explicitly requested it. Rate-limit thresholds change as IT adjusts the tier — bolting them into code means a commit per adjustment.

---

## Architecture

### File changes

| File | Change | Responsibility |
|---|---|---|
| `debate/rate_limit.py` | **new** | `RateLimitConfig` dataclass + `invoke_with_retry` helper + `_compute_sleep_seconds` pure function. Single responsibility: retry on `RateLimitError`. |
| `debate/nodes.py` | modified | Replace 4 `llm.invoke(...)` call sites with `invoke_with_retry(llm, ...)`. No other logic changes. |
| `.env.example` | modified | Add 3-line `# --- Rate-limit guardrail ---` block with the new keys and their defaults as comments. |
| `main.py` | unchanged | The guardrail is self-contained. No new startup checks required — env values have defaults. |
| `debate/state.py` | unchanged | No state changes. |
| `debate/graph.py` | unchanged | Topology preserved. |
| `README.md` | light touch | One-line entry noting the new module in the "Project structure" section. |

### `debate/rate_limit.py` (new module)

```python
"""Reactive rate-limit guardrail for Anthropic API calls.

Wraps `ChatAnthropic.invoke()` with retry-on-429 behavior. Sleeps honor the
`retry-after` header when present; otherwise fall back to exponential backoff.
All sleeps are capped at RATE_LIMIT_MAX_SLEEP_SECONDS. On exhaustion, the
original exception propagates — we surface rate-limit failures, not hide them.

Config is read from .env at module import. Missing values use sensible defaults.
Non-numeric values raise ValueError at import time (fail fast, not on first 429).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable


@dataclass(frozen=True)
class RateLimitConfig:
    max_retries: int          # retries AFTER the initial attempt
    max_sleep_seconds: float  # ceiling for any single sleep
    base_sleep_seconds: float # starting backoff when no retry-after header

    @classmethod
    def load_from_env(cls) -> "RateLimitConfig":
        return cls(
            max_retries=int(os.getenv("RATE_LIMIT_MAX_RETRIES", "5")),
            max_sleep_seconds=float(os.getenv("RATE_LIMIT_MAX_SLEEP_SECONDS", "90")),
            base_sleep_seconds=float(os.getenv("RATE_LIMIT_BASE_SLEEP_SECONDS", "2")),
        )


_CONFIG = RateLimitConfig.load_from_env()


def _parse_retry_after(exc: anthropic.RateLimitError) -> float | None:
    """Extract retry-after header as seconds. None if missing/unparseable.

    Anthropic normally sends integer-seconds (e.g., "30"); HTTP spec also
    allows HTTP-date format, which we treat as unparseable and fall back.
    """
    try:
        raw = exc.response.headers.get("retry-after")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None  # HTTP-date format — caller falls back to exponential


def _compute_sleep_seconds(
    retry_after: float | None,
    attempt: int,
    cfg: RateLimitConfig,
) -> float:
    """Compute how long to sleep before the next retry.

    USER-WRITTEN. Three decisions encoded here:
    1. If `retry_after` is given, prefer it (but respect `max_sleep_seconds`).
    2. If `retry_after` is None, exponential backoff: base * 2**attempt.
    3. Always cap at `max_sleep_seconds`, floor at a small positive value
       (a retry-after of 0 is a server glitch — don't retry instantly).

    Args:
        retry_after: seconds from server's retry-after header, or None.
        attempt: 0-indexed attempt number (0 = first retry).
        cfg: loaded config.

    Returns:
        Seconds to sleep. Always > 0.
    """
    raise NotImplementedError("User to implement — see docstring.")


def invoke_with_retry(
    llm: Runnable,
    messages: list[BaseMessage],
    config: dict | None = None,
) -> Any:
    """Invoke `llm` with retry on anthropic.RateLimitError.

    Each retry is a fresh `.invoke()` call, so each attempt appears as its
    own span in LangSmith/Langfuse — retries are visible in traces, not hidden.

    Raises:
        anthropic.RateLimitError: after cfg.max_retries retries exhausted.
        Any other exception: propagated immediately (network, 500s, etc.).
    """
    cfg = _CONFIG
    for attempt in range(cfg.max_retries + 1):
        try:
            return llm.invoke(messages, config=config)
        except anthropic.RateLimitError as e:
            if attempt == cfg.max_retries:
                raise
            retry_after = _parse_retry_after(e)
            sleep_s = _compute_sleep_seconds(retry_after, attempt, cfg)
            print(
                f"[rate-limit] attempt {attempt + 1}/{cfg.max_retries}, "
                f"sleeping {sleep_s:.1f}s (retry-after={retry_after})",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
```

### `debate/nodes.py` changes

Four call sites change. Each `llm.invoke(...)` or `model.invoke(...)` becomes `invoke_with_retry(llm, ...)`:

| Line | Current | New |
|---|---|---|
| 60 | `response = llm.invoke(messages, config={...})` | `response = invoke_with_retry(llm, messages, config={...})` |
| 75 | `forced = _model_for(role).invoke(messages + [...], config={...})` | `forced = invoke_with_retry(_model_for(role), messages + [...], config={...})` |
| 118 | `response = _model_for("moderator").invoke([...], config=llm_config)` | `response = invoke_with_retry(_model_for("moderator"), [...], config=llm_config)` |
| 219 | `response = _model_for("judge").invoke([...], config=llm_config)` | `response = invoke_with_retry(_model_for("judge"), [...], config=llm_config)` |

New import at top:

```python
from debate.rate_limit import invoke_with_retry
```

No other logic changes in `nodes.py`.

### `.env.example` addition

Appended at the bottom of the existing file:

```
# --- Rate-limit guardrail for Anthropic API ---
# All three are optional; omitting them uses the defaults shown.

# How many retries AFTER the initial attempt before raising.
# Total attempts = RATE_LIMIT_MAX_RETRIES + 1.
RATE_LIMIT_MAX_RETRIES=5

# Upper bound on each individual sleep (seconds). Clamps both the
# retry-after header value and the exponential-backoff formula.
RATE_LIMIT_MAX_SLEEP_SECONDS=90

# Starting sleep for exponential backoff when the server doesn't send
# a retry-after header. Doubles each attempt until capped.
RATE_LIMIT_BASE_SLEEP_SECONDS=2
```

### `README.md` addition

One row added to the "Project structure" file list:

```
│   ├── rate_limit.py       # retry-on-429 guardrail around ChatAnthropic.invoke()
```

---

## The user contribution: `_compute_sleep_seconds`

The spec leaves `_compute_sleep_seconds` as a `NotImplementedError` stub. The user writes ~5-8 lines implementing it. Three real decisions hide in this function:

1. **Retry-after present.** Trust the server's suggested wait, but clamp at `cfg.max_sleep_seconds`. What if the server says 300s and your cap is 90s? The cap wins — you'll retry in 90s and likely get another 429, which is fine because you have more retries.
2. **Retry-after missing.** Exponential backoff: `cfg.base_sleep_seconds * (2 ** attempt)`. Could add jitter (small random fuzz) to avoid thundering-herd when multiple processes retry in sync — single-process debate today doesn't strictly need it, but idiomatic.
3. **Minimum floor.** A `retry-after: 0` response is a server glitch. Sleeping 0s wastes a retry attempt on what will almost certainly fail again. A ~1s floor is defensive.

This is the behavioral heart of the guardrail. Getting it wrong means either (a) retrying too fast and burning attempts, or (b) sleeping longer than needed and wasting wall time.

---

## Error handling

Philosophy: **visible failures, never silent ones** (matches the existing "rate-limit day stays visible" stance in README line 110).

| Scenario | Behavior |
|---|---|
| 429 from Anthropic | Caught. Sleep per `_compute_sleep_seconds`. Retry. |
| 429 persists past `max_retries` | `RateLimitError` propagates. Debate crashes. Transcript-to-date is lost (unchanged from current behavior). |
| Non-rate-limit exception (network, 500, auth) | Propagates immediately. Not our concern. |
| Invalid env value (e.g., `RATE_LIMIT_MAX_RETRIES=abc`) | `ValueError` at import time. `main.py` dies before graph is built — matches existing fail-fast pattern. |
| Missing env values | Use defaults. No error. |
| Tool-call failure inside ReAct loop | **Unchanged.** `_run_with_tools` catches tool errors independently (lines 68-71). No interaction with the rate-limit guardrail. |
| Rate-limit during `forced-close` call | Same guardrail applies. If retries exhaust, the exception propagates out of `_run_with_tools` → `proposer_node`/`critic_node` → `graph.invoke()`. |

### Tracing behavior

Each retry is a **separate LangSmith/Langfuse span** because we re-invoke the runnable. No custom span logic needed — LangChain's callbacks fire on each `.invoke()`. You'll see:

- `proposer-argument-r2-iter1` (first attempt, failed with 429)
- `[rate-limit] attempt 1/5, sleeping 30.0s (retry-after=30)` (stderr)
- `proposer-argument-r2-iter1` (retry attempt, succeeded or failed again)

Retries share the same `run_name` which is actually useful — you can filter by run name and see the cluster of attempts for a single logical call.

---

## Validation plan

Manual, via traces (matches project convention — no unit tests).

1. **Baseline (no rate limit):** run an existing topic. Expect zero retry messages on stderr, no behavioral change in the transcript, no new spans in traces. This proves the guardrail is transparent when it's not needed.

2. **Forced 429:** temporarily edit `invoke_with_retry` to raise a fake `anthropic.RateLimitError` on first attempt, second attempt succeeds. Run a debate. Expect:
   - `[rate-limit] attempt 1/5, sleeping ...` on stderr
   - Two spans in LangSmith with the same `run_name`
   - Debate completes successfully

3. **Forced exhaustion:** raise fake 429s on every attempt. Expect:
   - Six stderr messages (initial + 5 retries)
   - `RateLimitError` propagates and crashes the process
   - Sleeps follow the expected progression (2s, 4s, 8s, 16s, 32s, capped at 90s)

4. **Real rate limit (if reproducible):** run 10 debates back-to-back to trigger a genuine 10k/min bucket exhaustion. Verify the guardrail catches and recovers.

5. **Env-config round-trip:** set `RATE_LIMIT_MAX_RETRIES=1` in `.env`, force a 429, verify it gives up after 1 retry instead of 5.

---

## Open questions

None blocking. Deliberate deferrals:

- **Proactive throttling** — revisit if bursts keep exhausting retries. Would slot in alongside the reactive guardrail, not replace it.
- **Jitter in exponential backoff** — not needed for single-process today; add if we ever run parallel debates.
- **Persistence of partial transcripts on crash** — existing non-goal (README line 243). Rate-limit exhaustion inherits it.

---

## Summary

- One new module (`debate/rate_limit.py`), one modified module (`debate/nodes.py` — 4 call-site edits + 1 import), one config file addition (`.env.example`), one README touch-up.
- Graph topology, state shape, prompts, and tool config all unchanged.
- User writes `_compute_sleep_seconds` (~5-8 lines). Everything else is plumbing.
- Every retry is a visible span in LangSmith/Langfuse — the rate-limit event is never hidden.
- Three env knobs, all with defaults. IT can bump the tier without requiring a code change.
