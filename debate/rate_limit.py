"""Reactive rate-limit guardrail for Anthropic API calls.

Wraps `ChatAnthropic.invoke()` with retry-on-429 behaviour. Sleeps honour the
`retry-after` header when present; otherwise fall back to exponential backoff.
All sleeps are capped at RATE_LIMIT_MAX_SLEEP_SECONDS. On exhaustion, the
original exception propagates — we surface rate-limit failures, not hide them.

Config is read from .env at module import time. Missing values use sensible
defaults. Non-numeric values raise ValueError at import time (fail fast, not
on first 429). Because config is loaded at import, `debate/rate_limit.py`
MUST be imported after `load_dotenv()` in main.py. After Task 3 wires it
into debate/nodes.py, this is satisfied transitively; until then, any direct
import of this module must follow load_dotenv().
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
        """Read config from env vars.

        Raises ValueError at import time for invalid values:
        - Non-numeric strings
        - Negative max_retries
        - Non-positive max_sleep_seconds or base_sleep_seconds
        """
        max_retries = int(os.getenv("RATE_LIMIT_MAX_RETRIES", "5"))
        max_sleep_seconds = float(os.getenv("RATE_LIMIT_MAX_SLEEP_SECONDS", "90"))
        base_sleep_seconds = float(os.getenv("RATE_LIMIT_BASE_SLEEP_SECONDS", "2"))
        if max_retries < 0:
            raise ValueError(
                f"RATE_LIMIT_MAX_RETRIES must be >= 0, got {max_retries}"
            )
        if max_sleep_seconds <= 0:
            raise ValueError(
                f"RATE_LIMIT_MAX_SLEEP_SECONDS must be > 0, got {max_sleep_seconds}"
            )
        if base_sleep_seconds <= 0:
            raise ValueError(
                f"RATE_LIMIT_BASE_SLEEP_SECONDS must be > 0, got {base_sleep_seconds}"
            )
        return cls(
            max_retries=max_retries,
            max_sleep_seconds=max_sleep_seconds,
            base_sleep_seconds=base_sleep_seconds,
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
