# Anthropic Rate-Limit Guardrail — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project testing philosophy:** This project has no automated tests by design (README: "No unit tests (manual validation via traces)"). Verification steps below use **run-and-inspect** commands (Python REPL + end-to-end runs) instead of pytest. Each task still has an explicit verification gate — don't skip it.

**Goal:** Wrap every `ChatAnthropic.invoke()` call in the debate agent with a retry-on-429 guardrail so the process no longer crashes when the company's 10k tokens/minute Anthropic threshold is hit. Configurable via `.env`.

**Architecture:** A new self-contained module `debate/rate_limit.py` exposes `invoke_with_retry(llm, messages, config)`. It catches `anthropic.RateLimitError`, computes a sleep duration (honoring the server's `retry-after` header when present, exponential backoff otherwise, capped at a configurable max), retries up to a configurable number of attempts, and re-raises on exhaustion. The four LLM call sites in `debate/nodes.py` switch from `llm.invoke(...)` to `invoke_with_retry(llm, ...)`. No graph-topology, state-shape, or prompt changes.

**Tech Stack:** Python 3.13, LangGraph, LangChain (`langchain-core`, `langchain-anthropic`), `anthropic` (SDK, exceptions only — already pulled in transitively via `langchain-anthropic`).

**Spec:** `docs/superpowers/specs/2026-04-22-rate-limit-guardrail-design.md` (commit `3e6ec92`).

**⚠️ User-written section:** Task 2 contains ~5-8 lines the project owner (in learning mode) will write by hand. Agents executing this plan should **pause at Task 2 and hand back** rather than implement `_compute_sleep_seconds` themselves.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `debate/rate_limit.py` | NEW | `RateLimitConfig` + `invoke_with_retry` helper + `_compute_sleep_seconds` pure function + `_parse_retry_after` helper. Single responsibility: retry on `RateLimitError`. |
| `debate/nodes.py` | MODIFIED | Replace 4 `invoke(...)` call sites with `invoke_with_retry(...)`. Add one import. No other changes. |
| `.env.example` | MODIFIED | Append a `# --- Rate-limit guardrail ---` block with the three new keys and their defaults as comments. |
| `README.md` | MODIFIED | One-line entry in the "Project structure" file list. |
| `main.py` | unchanged | Guardrail is self-contained. Env values have defaults, so no new startup check. |
| `debate/state.py` | unchanged | No state changes. |
| `debate/graph.py` | unchanged | Topology preserved. |
| `requirements.txt` | unchanged | `anthropic` is already transitively installed via `langchain-anthropic`. |

---

## Task 1: Scaffold `debate/rate_limit.py` (everything except `_compute_sleep_seconds`)

**Files:**
- Create: `debate/rate_limit.py`

The module is created with `RateLimitConfig`, `_parse_retry_after`, and `invoke_with_retry` fully implemented. The only function left as a stub is `_compute_sleep_seconds`, which raises `NotImplementedError` — that's Task 2 (user-written).

After this task, importing the module will succeed; calling `invoke_with_retry` on a non-429 call will work; calling it on a 429 will raise `NotImplementedError` from inside the stub (until Task 2 fills it in).

- [ ] **Step 1: Verify the anthropic SDK exposes `RateLimitError`**

Run:
```bash
.venv/bin/python -c "import anthropic; print(anthropic.RateLimitError.__mro__)"
```

Expected: a tuple starting with `(<class 'anthropic.RateLimitError'>, <class 'anthropic.APIStatusError'>, ...)`. If you get `AttributeError: module 'anthropic' has no attribute 'RateLimitError'`, the anthropic SDK version has changed — check `pip show anthropic` and look up the new exception name before proceeding.

- [ ] **Step 2: Create `debate/rate_limit.py`**

Create the file with this exact content:

```python
"""Reactive rate-limit guardrail for Anthropic API calls.

Wraps `ChatAnthropic.invoke()` with retry-on-429 behaviour. Sleeps honour the
`retry-after` header when present; otherwise fall back to exponential backoff.
All sleeps are capped at RATE_LIMIT_MAX_SLEEP_SECONDS. On exhaustion, the
original exception propagates — we surface rate-limit failures, not hide them.

Config is read from .env at module import time. Missing values use sensible
defaults. Non-numeric values raise ValueError at import time (fail fast, not
on first 429). Because config is loaded at import, `debate/rate_limit.py`
MUST be imported after `load_dotenv()` in main.py — which it is, transitively
via debate/nodes.py.
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
    """Three knobs read from .env, all with defaults."""

    max_retries: int
    max_sleep_seconds: float
    base_sleep_seconds: float

    @classmethod
    def load_from_env(cls) -> "RateLimitConfig":
        """Read config from env vars. Non-numeric values raise ValueError."""
        return cls(
            max_retries=int(os.getenv("RATE_LIMIT_MAX_RETRIES", "5")),
            max_sleep_seconds=float(os.getenv("RATE_LIMIT_MAX_SLEEP_SECONDS", "90")),
            base_sleep_seconds=float(os.getenv("RATE_LIMIT_BASE_SLEEP_SECONDS", "2")),
        )


_CONFIG = RateLimitConfig.load_from_env()


def _parse_retry_after(exc: anthropic.RateLimitError) -> float | None:
    """Extract retry-after header as seconds. None if missing/unparseable.

    Anthropic normally sends integer-seconds (e.g., "30"). HTTP spec also
    allows HTTP-date format, which we treat as unparseable and fall back to
    exponential backoff. The `.response` attribute comes from APIStatusError.
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
        return None


def _compute_sleep_seconds(
    retry_after: float | None,
    attempt: int,
    cfg: RateLimitConfig,
) -> float:
    """Compute how long to sleep before the next retry.

    USER-WRITTEN (plan Task 2). Three decisions this function encodes:
    1. If `retry_after` is given, prefer it — but clamp at cfg.max_sleep_seconds.
    2. If `retry_after` is None, fall back to exponential backoff:
       cfg.base_sleep_seconds * (2 ** attempt), also clamped at max_sleep_seconds.
    3. Floor at a small positive value (e.g. 1s) — a retry-after of 0 is a
       server glitch, and retrying instantly wastes an attempt on what will
       almost certainly fail again.

    Args:
        retry_after: seconds from the server's retry-after header, or None.
        attempt: 0-indexed attempt number (0 = first retry, 1 = second, ...).
        cfg: loaded RateLimitConfig.

    Returns:
        Seconds to sleep. Always > 0.
    """
    raise NotImplementedError(
        "Plan Task 2 — project owner to implement. See docstring above."
    )


def invoke_with_retry(
    llm: Runnable,
    messages: list[BaseMessage],
    config: dict | None = None,
) -> Any:
    """Invoke `llm` with retry on anthropic.RateLimitError.

    Each retry is a fresh `.invoke()` call, so each attempt appears as its
    own span in LangSmith/Langfuse — retries are visible in traces, not hidden.
    Retries share the same `run_name` (from `config`) so you can filter a
    cluster of attempts together.

    Args:
        llm: any LangChain Runnable (typically a ChatAnthropic or a
             ChatAnthropic bound with `.bind_tools(...)`).
        messages: the message list passed to `.invoke()`.
        config: optional RunnableConfig dict (for tracing, callbacks, run_name).

    Returns:
        Whatever `llm.invoke(messages, config=config)` would return — usually
        an AIMessage.

    Raises:
        anthropic.RateLimitError: after cfg.max_retries retries exhausted.
        Any other exception: propagated immediately (network, 500s, auth, etc.).
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

- [ ] **Step 3: Verify the module imports cleanly**

Run:
```bash
.venv/bin/python -c "from debate.rate_limit import invoke_with_retry, RateLimitConfig; c = RateLimitConfig.load_from_env(); print(c)"
```

Expected output: something like
```
RateLimitConfig(max_retries=5, max_sleep_seconds=90.0, base_sleep_seconds=2.0)
```

If you get `ImportError` or `AttributeError`, the file has a typo — reread Step 2. If you get `ValueError: invalid literal for int()`, you have a non-numeric env var in your `.env` from an earlier Task 4 run — remove it for now.

- [ ] **Step 4: Verify the stub raises as expected**

Run:
```bash
.venv/bin/python -c "
from debate.rate_limit import _compute_sleep_seconds, RateLimitConfig
cfg = RateLimitConfig.load_from_env()
try:
    _compute_sleep_seconds(None, 0, cfg)
except NotImplementedError as e:
    print('stub raises correctly:', e)
"
```

Expected: `stub raises correctly: Plan Task 2 — project owner to implement. ...`

- [ ] **Step 5: Commit**

```bash
git add debate/rate_limit.py
git commit -m "$(cat <<'EOF'
feat: scaffold debate/rate_limit.py with retry-on-429 plumbing

Adds RateLimitConfig (env-driven, with defaults), _parse_retry_after
(reads the retry-after header defensively), and invoke_with_retry (the
loop itself). Leaves _compute_sleep_seconds as a NotImplementedError
stub — the sleep formula is the behavioural heart of the guardrail and
is intentionally left for the project owner to implement (learning
mode, plan Task 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: **[USER-WRITTEN]** Implement `_compute_sleep_seconds`

**Files:**
- Modify: `debate/rate_limit.py` (replace the `NotImplementedError` body of `_compute_sleep_seconds` with the actual implementation)

**⚠️ Agentic workers: stop here and hand back to the project owner.** This task is intentionally left for the human to write. ~5-8 lines of code. All plumbing around it is in place from Task 1.

### What you're implementing

Signature (already in the file):
```python
def _compute_sleep_seconds(
    retry_after: float | None,
    attempt: int,
    cfg: RateLimitConfig,
) -> float:
```

### The three decisions baked into this function

**1. Retry-after present (common case).**
Trust the server's suggested wait, but clamp at `cfg.max_sleep_seconds`. If Anthropic says "wait 300 seconds" and your cap is 90, use 90. You'll retry in 90s, likely get another 429, and retry again — that's fine, you have more attempts.

**2. Retry-after missing.**
Exponential backoff: `cfg.base_sleep_seconds * (2 ** attempt)`. With default config (base=2), the progression is 2s, 4s, 8s, 16s, 32s, 64s, 128s — the 128s value is then clamped down to `max_sleep_seconds=90`.

**3. Minimum floor.**
A `retry-after: 0` response is a server glitch. Sleeping 0s wastes an attempt on what will almost certainly fail again. Use ~1s as the floor.

### Hint (scaffolding, not the answer)

```python
# Pick your base: server-suggested or exponential fallback
# Clamp to cfg.max_sleep_seconds from above
# Floor at 1.0 from below
# Return
```

- [ ] **Step 1: Replace the stub body with your implementation**

Open `debate/rate_limit.py`, find the `raise NotImplementedError(...)` line inside `_compute_sleep_seconds`, and replace it with your logic. Keep the docstring — update only the function body.

- [ ] **Step 2: Verify the function works on the three key cases**

Run a quick sanity check in the REPL:

```bash
.venv/bin/python -c "
from debate.rate_limit import _compute_sleep_seconds, RateLimitConfig
cfg = RateLimitConfig(max_retries=5, max_sleep_seconds=90.0, base_sleep_seconds=2.0)

# Case 1: retry-after present and reasonable → honour it
print('case 1 (retry-after=30, attempt=0):', _compute_sleep_seconds(30.0, 0, cfg))
# Expected: 30.0

# Case 2: retry-after present but huge → clamp to max_sleep
print('case 2 (retry-after=500, attempt=0):', _compute_sleep_seconds(500.0, 0, cfg))
# Expected: 90.0

# Case 3: retry-after missing, first attempt → exponential base
print('case 3 (retry-after=None, attempt=0):', _compute_sleep_seconds(None, 0, cfg))
# Expected: 2.0 (= base_sleep_seconds * 2**0)

# Case 4: retry-after missing, later attempt → exponential grows
print('case 4 (retry-after=None, attempt=3):', _compute_sleep_seconds(None, 3, cfg))
# Expected: 16.0 (= 2.0 * 2**3)

# Case 5: retry-after missing, very late attempt → clamped at max
print('case 5 (retry-after=None, attempt=10):', _compute_sleep_seconds(None, 10, cfg))
# Expected: 90.0 (= min(2.0 * 2**10, 90.0))

# Case 6: retry-after=0 → floor kicks in
print('case 6 (retry-after=0, attempt=0):', _compute_sleep_seconds(0.0, 0, cfg))
# Expected: 1.0 (or whatever your floor is)
"
```

Expected: six lines printed, each matching the "Expected" comment. If any differ, reread the three decisions above.

- [ ] **Step 3: Commit**

```bash
git add debate/rate_limit.py
git commit -m "$(cat <<'EOF'
feat: implement _compute_sleep_seconds for rate-limit guardrail

Server-directed sleep when the retry-after header is present, exponential
backoff (base * 2**attempt) when it isn't. Both paths clamp at
max_sleep_seconds from above and floor at 1s from below — a retry-after
of 0 would otherwise waste an attempt.

Co-Authored-By: Velimir Mueller
EOF
)"
```

(Adjust the co-author line to taste — this commit is your code, not generated.)

---

## Task 3: Wire `invoke_with_retry` into `debate/nodes.py`

**Files:**
- Modify: `debate/nodes.py`

Four call-site swaps + one import. Each swap is mechanical: `llm.invoke(messages, config=...)` → `invoke_with_retry(llm, messages, config=...)`. No other logic changes.

- [ ] **Step 1: Add the import**

Open `debate/nodes.py`. Find the existing import block:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from debate.state import DebateState, Turn
```

Add one line (alphabetical ordering, so it goes after `debate.state`... or keep it grouped with `debate.` imports — your call). Final block:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from debate.rate_limit import invoke_with_retry
from debate.state import DebateState, Turn
```

- [ ] **Step 2: Swap call site 1 — inside `_run_with_tools` ReAct loop (around line 60)**

Find this block in `_run_with_tools`:

```python
    for i in range(MAX_TOOL_CALLS):
        response = llm.invoke(
            messages,
            config={**config, "run_name": f"{run_name_prefix}-iter{i}"},
        )
```

Change to:

```python
    for i in range(MAX_TOOL_CALLS):
        response = invoke_with_retry(
            llm,
            messages,
            config={**config, "run_name": f"{run_name_prefix}-iter{i}"},
        )
```

- [ ] **Step 3: Swap call site 2 — the "forced-close" invoke (around line 75)**

Find:

```python
    # Budget exhausted without prose: force a tool-free close.
    forced = _model_for(role).invoke(
        messages + [HumanMessage(content=(
            "You've used your research budget. Produce your final argument now, "
            "using only the evidence gathered above."
        ))],
        config={**config, "run_name": f"{run_name_prefix}-forced-close"},
    )
    return forced.content
```

Change to:

```python
    # Budget exhausted without prose: force a tool-free close.
    forced = invoke_with_retry(
        _model_for(role),
        messages + [HumanMessage(content=(
            "You've used your research budget. Produce your final argument now, "
            "using only the evidence gathered above."
        ))],
        config={**config, "run_name": f"{run_name_prefix}-forced-close"},
    )
    return forced.content
```

- [ ] **Step 4: Swap call site 3 — inside `moderator_node` (around line 118)**

Find:

```python
    llm_config = {**config, "run_name": f"moderator-{'open' if is_opening else 'distill'}-r{round_n}"}
    response = _model_for("moderator").invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        config=llm_config,
    )
```

Change to:

```python
    llm_config = {**config, "run_name": f"moderator-{'open' if is_opening else 'distill'}-r{round_n}"}
    response = invoke_with_retry(
        _model_for("moderator"),
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        config=llm_config,
    )
```

- [ ] **Step 5: Swap call site 4 — inside `judge_node` (around line 219)**

Find:

```python
    llm_config = {**config, "run_name": f"judge-decision-r{round_n}"}
    response = _model_for("judge").invoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user_prompt)],
        config=llm_config,
    )
```

Change to:

```python
    llm_config = {**config, "run_name": f"judge-decision-r{round_n}"}
    response = invoke_with_retry(
        _model_for("judge"),
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user_prompt)],
        config=llm_config,
    )
```

- [ ] **Step 6: Verify no other `llm.invoke`, `_model_for(...).invoke`, or similar patterns remain**

Run:
```bash
grep -n '\.invoke(' debate/nodes.py
```

Expected: zero lines. Any `.invoke(` left in `debate/nodes.py` means a call site was missed. (The four edits above should have covered everything, but this grep is a cheap confirmation.)

If the grep returns hits, each one needs the same swap treatment.

- [ ] **Step 7: Verify the module still imports**

Run:
```bash
.venv/bin/python -c "from debate.nodes import proposer_node, critic_node, moderator_node, judge_node; print('ok')"
```

Expected: `ok`. If you get `ImportError: cannot import name 'invoke_with_retry'`, Task 1 didn't commit or Step 1 above wasn't done.

- [ ] **Step 8: Run a baseline debate to confirm no behavioural regression**

Run:
```bash
.venv/bin/python main.py "Is Python a better first language than JavaScript?"
```

Expected:
- The debate runs to completion with a `VERDICT:` line printed.
- **No `[rate-limit]` messages on stderr** (because you weren't rate-limited).
- Trace URLs printed at the end.

If the debate crashes with `NameError: name 'invoke_with_retry' is not defined`, Step 1 import is missing or mistyped. If it crashes with `NotImplementedError`, you're hitting a real 429 and the stub from Task 2 is still in place — but Task 2 should have been completed before reaching Task 3; check `git log` to verify.

- [ ] **Step 9: Commit**

```bash
git add debate/nodes.py
git commit -m "$(cat <<'EOF'
feat: wire invoke_with_retry into all ChatAnthropic call sites

Four sites swapped in debate/nodes.py (ReAct iter, forced-close,
moderator, judge) from llm.invoke(...) to invoke_with_retry(llm, ...).
Each retry becomes its own LangSmith/Langfuse span — rate-limit events
stay visible in traces.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add env keys to `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Append the rate-limit block**

Open `.env.example`. At the end of the file (after the `TAVILY_API_KEY=` line), append:

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

(Note the leading blank line — keeps the block visually separated from the Tavily section.)

- [ ] **Step 2: Verify `.env.example` still parses**

Run:
```bash
.venv/bin/python -c "
from dotenv import dotenv_values
cfg = dotenv_values('.env.example')
print('RATE_LIMIT_MAX_RETRIES:', cfg.get('RATE_LIMIT_MAX_RETRIES'))
print('RATE_LIMIT_MAX_SLEEP_SECONDS:', cfg.get('RATE_LIMIT_MAX_SLEEP_SECONDS'))
print('RATE_LIMIT_BASE_SLEEP_SECONDS:', cfg.get('RATE_LIMIT_BASE_SLEEP_SECONDS'))
"
```

Expected:
```
RATE_LIMIT_MAX_RETRIES: 5
RATE_LIMIT_MAX_SLEEP_SECONDS: 90
RATE_LIMIT_BASE_SLEEP_SECONDS: 2
```

- [ ] **Step 3: (Optional — owner only) Set custom values in `.env` if desired**

Your local `.env` (untracked) is where you'd override. For the 10k-tokens/min tier, the defaults are sensible; no override strictly needed. If you want to experiment with a shorter retry budget, add e.g. `RATE_LIMIT_MAX_RETRIES=2` to `.env`.

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "$(cat <<'EOF'
docs: document rate-limit guardrail env keys in .env.example

Three optional keys with defaults as comments:
RATE_LIMIT_MAX_RETRIES, RATE_LIMIT_MAX_SLEEP_SECONDS,
RATE_LIMIT_BASE_SLEEP_SECONDS. Loaded at module import by
debate/rate_limit.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update README

**Files:**
- Modify: `README.md`

One-line entry in the "Project structure" file list. Keeps the doc honest about what lives in the repo.

- [ ] **Step 1: Add the `rate_limit.py` entry to the tree**

Open `README.md`. Find the "Project structure" section — the tree that currently looks like:

```
├── debate/
│   ├── __init__.py
│   ├── state.py            # DebateState + Turn TypedDicts
│   ├── prompts.py          # 5 system prompts — main iteration surface
│   ├── tools.py            # Tavily search + extract as LangChain Tool instances
│   ├── nodes.py            # node functions + model factory + ReAct loop helper
│   └── graph.py            # build_graph() — topology
```

Insert a line for `rate_limit.py`. Alphabetical ordering within the block would place it between `prompts.py` and `state.py`; or keep things grouped by role (config/types first, orchestration after) and place it after `nodes.py`. The project's current ordering is roughly "types → prompts → tools → orchestration → topology", so put it near the orchestration side:

```
├── debate/
│   ├── __init__.py
│   ├── state.py            # DebateState + Turn TypedDicts
│   ├── prompts.py          # 5 system prompts — main iteration surface
│   ├── tools.py            # Tavily search + extract as LangChain Tool instances
│   ├── rate_limit.py       # retry-on-429 guardrail around ChatAnthropic.invoke()
│   ├── nodes.py            # node functions + model factory + ReAct loop helper
│   └── graph.py            # build_graph() — topology
```

- [ ] **Step 2: Also update the per-module responsibilities bullet list below the tree**

The README has a second "Each module has one responsibility" block. Add an entry for `rate_limit.py` between the `tools.py` and `nodes.py` bullets:

Find:
```markdown
- `tools.py` — Tavily configuration, isolated from orchestration so retrieval knobs (`max_results`, `search_depth`) don't bleed into `nodes.py`.
- `nodes.py` — node functions that read state + config, call the LLM, return state deltas. Also houses `_run_with_tools()` (the ReAct loop) and `MAX_TOOL_CALLS`. The model factory is here too.
```

Change to:
```markdown
- `tools.py` — Tavily configuration, isolated from orchestration so retrieval knobs (`max_results`, `search_depth`) don't bleed into `nodes.py`.
- `rate_limit.py` — reactive retry-on-429 guardrail wrapping every `ChatAnthropic.invoke()` call. Sleep duration honours the `retry-after` header when present, exponential backoff otherwise. Knobs via `.env` (`RATE_LIMIT_*`).
- `nodes.py` — node functions that read state + config, call the LLM, return state deltas. Also houses `_run_with_tools()` (the ReAct loop) and `MAX_TOOL_CALLS`. The model factory is here too.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: mention debate/rate_limit.py in README project structure

Adds the new module to both the file tree and the per-module
responsibilities list.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: End-to-end verification

**Files:**
- None modified. Verification only.

Three scenarios to confirm the guardrail works. The first is the cheap baseline; the second and third use temporary monkey-patches to force 429s (because you can't reliably induce real rate limits on demand).

- [ ] **Step 1: Baseline — unmodified run, no 429s**

Run:
```bash
.venv/bin/python main.py "Should remote work be the default for knowledge workers?"
```

Expected:
- Debate runs to completion; `VERDICT:` line prints.
- **No `[rate-limit]` lines on stderr.**
- LangSmith and Langfuse trace URLs print.
- Trace spans look identical to pre-change runs (no extra retry spans).

This proves the guardrail is transparent when it's not needed.

- [ ] **Step 2: Simulated 429 — succeeds on first retry**

Create a throwaway script `scratch_sim_429_once.py` in the project root:

```python
"""Throwaway: simulate a 429 on first attempt, succeed on second.

Run with: .venv/bin/python scratch_sim_429_once.py
Delete after verification — do NOT commit.
"""

from dotenv import load_dotenv
load_dotenv()

import httpx
import anthropic
from debate.rate_limit import invoke_with_retry

# Build a tiny fake LLM that 429s once then succeeds.
class FakeLLM:
    def __init__(self):
        self.calls = 0
    def invoke(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            fake_response = httpx.Response(
                status_code=429,
                headers={"retry-after": "3"},
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            )
            raise anthropic.RateLimitError(
                message="fake rate limit", response=fake_response, body=None
            )
        return {"content": f"success on call {self.calls}"}

llm = FakeLLM()
result = invoke_with_retry(llm, [], config=None)
print("result:", result)
print("total calls:", llm.calls)
```

Run:
```bash
.venv/bin/python scratch_sim_429_once.py
```

Expected (on stderr):
```
[rate-limit] attempt 1/5, sleeping 3.0s (retry-after=3.0)
```
Then (on stdout), after roughly 3 seconds:
```
result: {'content': 'success on call 2'}
total calls: 2
```

If you see no `[rate-limit]` line, the guardrail isn't catching. If it sleeps longer than ~3s, `_compute_sleep_seconds` is ignoring the retry-after value.

- [ ] **Step 3: Simulated 429 — exhausts retries**

Edit the throwaway script so the fake LLM always raises:

```python
class FakeLLM:
    def invoke(self, messages, config=None):
        fake_response = httpx.Response(
            status_code=429,
            headers={"retry-after": "1"},  # short sleep so the test finishes quickly
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        raise anthropic.RateLimitError(
            message="fake persistent rate limit", response=fake_response, body=None
        )

llm = FakeLLM()
try:
    invoke_with_retry(llm, [], config=None)
except anthropic.RateLimitError as e:
    print("propagated as expected:", e)
```

Run:
```bash
.venv/bin/python scratch_sim_429_once.py
```

Expected (on stderr, five lines — the default `RATE_LIMIT_MAX_RETRIES=5`):
```
[rate-limit] attempt 1/5, sleeping 1.0s (retry-after=1.0)
[rate-limit] attempt 2/5, sleeping 1.0s (retry-after=1.0)
[rate-limit] attempt 3/5, sleeping 1.0s (retry-after=1.0)
[rate-limit] attempt 4/5, sleeping 1.0s (retry-after=1.0)
[rate-limit] attempt 5/5, sleeping 1.0s (retry-after=1.0)
```
Then (on stdout):
```
propagated as expected: fake persistent rate limit
```

Total runtime: roughly 5 seconds.

If it doesn't propagate the exception after exhaustion, the `if attempt == cfg.max_retries: raise` branch in `invoke_with_retry` isn't firing.

- [ ] **Step 4: Env-configurability round-trip**

Without committing it, temporarily add to your `.env`:
```
RATE_LIMIT_MAX_RETRIES=1
```

Re-run the throwaway from Step 3.

Expected: only **one** `[rate-limit]` line on stderr (because `max_retries=1` means one retry after the initial attempt), then the exception propagates. Total runtime ~1 second.

Remove the override from `.env` when done.

This confirms the env var actually takes effect.

- [ ] **Step 5: Clean up**

```bash
rm scratch_sim_429_once.py
```

Do NOT commit the scratch file. (A `.gitignore` entry isn't needed — just delete it.)

- [ ] **Step 6: Final smoke test + commit any remaining docs**

Re-run a real debate to confirm nothing was left in a broken state:

```bash
.venv/bin/python main.py "Is algorithmic curation preferable to editorial curation?"
```

Expected: clean run, VERDICT printed, no stderr rate-limit lines.

No commit needed here — all code was committed in earlier tasks.

---

## Summary

- Task 1: `debate/rate_limit.py` created with `RateLimitConfig`, `_parse_retry_after`, `invoke_with_retry`, and a `NotImplementedError` stub for `_compute_sleep_seconds`.
- Task 2: **[USER-WRITTEN]** Project owner implements `_compute_sleep_seconds` (~5-8 lines).
- Task 3: Four call-site swaps in `debate/nodes.py`. Plus the `invoke_with_retry` import.
- Task 4: `.env.example` gets the three new keys as commented defaults.
- Task 5: README project-structure update (two small edits).
- Task 6: Three verification scenarios — baseline, simulated single-retry, simulated exhaustion — plus env-config round-trip.

Five commits land on the branch. One throwaway script is created and deleted in Task 6 (never committed).

No changes to: graph topology, state shape, prompts, tools, `main.py`, `requirements.txt`.
