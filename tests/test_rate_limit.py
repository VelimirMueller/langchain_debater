"""Offline tests for the retry-on-429 guardrail in debate/rate_limit.py."""

import anthropic
import httpx
import pytest

import debate.rate_limit as rl

CFG = rl.RateLimitConfig(max_retries=3, max_sleep_seconds=10, base_sleep_seconds=2)


def _rate_limit_error(retry_after: str | None = None) -> anthropic.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, headers=headers, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


@pytest.mark.parametrize(
    ("retry_after", "attempt", "expected"),
    [
        (5.0, 0, 5.0),     # server hint wins
        (60.0, 0, 10.0),   # server hint clamped to max
        (0.0, 0, 1.0),     # zero hint floored
        (None, 0, 2.0),    # backoff: base * 2**0
        (None, 2, 8.0),    # backoff: base * 2**2
        (None, 5, 10.0),   # backoff clamped to max
    ],
)
def test_compute_sleep_seconds(retry_after, attempt, expected):
    assert rl._compute_sleep_seconds(retry_after, attempt, CFG) == expected


@pytest.mark.parametrize(("header", "expected"), [("30", 30.0), (None, None), ("Wed, 21 Oct 2026 07:28:00 GMT", None)])
def test_parse_retry_after(header, expected):
    assert rl._parse_retry_after(_rate_limit_error(header)) == expected


class FlakyModel:
    def __init__(self, errors: list[Exception]):
        self.errors = errors
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(rl.time, "sleep", slept.append)
    monkeypatch.setattr(rl, "_CONFIG", CFG)
    return slept


def test_retries_then_succeeds(no_sleep):
    model = FlakyModel([_rate_limit_error("3"), _rate_limit_error()])
    assert rl.invoke_with_retry(model, []) == "ok"
    assert model.calls == 3
    assert no_sleep == [3.0, 4.0]  # retry-after, then backoff for attempt 1


def test_raises_after_retries_exhausted(no_sleep):
    model = FlakyModel([_rate_limit_error() for _ in range(CFG.max_retries + 1)])
    with pytest.raises(anthropic.RateLimitError):
        rl.invoke_with_retry(model, [])
    assert model.calls == CFG.max_retries + 1


def test_other_errors_are_not_retried(no_sleep):
    model = FlakyModel([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        rl.invoke_with_retry(model, [])
    assert model.calls == 1
    assert no_sleep == []


@pytest.mark.parametrize(
    ("var", "value"),
    [("RATE_LIMIT_MAX_RETRIES", "-1"), ("RATE_LIMIT_MAX_SLEEP_SECONDS", "0"), ("RATE_LIMIT_BASE_SLEEP_SECONDS", "abc")],
)
def test_invalid_env_fails_fast(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError):
        rl.RateLimitConfig.load_from_env()
