"""Shared Meta Model API plumbing: client, retries, clear error messages, budget guard."""

from __future__ import annotations

import os
import re
import time
from typing import Callable, TypeVar

from .config import API_BACKOFF_SECONDS, API_MAX_ATTEMPTS, BASE_URL, SESSION_BUDGET_USD

T = TypeVar("T")


class ApiError(RuntimeError):
    """An API failure explained in plain words. Never cached."""


class BudgetExceeded(RuntimeError):
    pass


def client(api_key: str | None = None):
    from openai import OpenAI  # imported here so offline tests don't need the SDK configured

    key = api_key or os.environ.get("MODEL_API_KEY")
    if not key:
        raise ApiError("MODEL_API_KEY is not set. Put it in the environment or an untracked .env file.")
    # Our own retry loop handles 429/5xx, so turn off the SDK's.
    return OpenAI(base_url=BASE_URL, api_key=key, max_retries=0)


def _status(e: Exception) -> int | None:
    return getattr(e, "status_code", None)


def _retryable(e: Exception) -> bool:
    import openai

    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if getattr(e, "retryable", False):       # plain-HTTP callers (voice.py) mark their own
        return True
    s = _status(e)
    return s == 429 or (s is not None and s >= 500)


def explain(e: Exception, what: str) -> str:
    s = _status(e)
    msg = str(e)
    if s == 404:
        return (f"{what}: model or file not found (404). Run `python -m chronotrace check` "
                f"to see which models your key has. API said: {msg}")
    if s == 400:
        return (f"{what}: request rejected (400). For SAM, the prompt must be one short noun "
                f"phrase such as 'box' (no lists, commands or questions). API said: {msg}")
    if s in (401, 403):
        return f"{what}: key rejected ({s}). Check MODEL_API_KEY. API said: {msg}"
    if s == 429:
        return f"{what}: still rate limited after {API_MAX_ATTEMPTS} tries. Wait a minute and retry."
    return f"{what}: {type(e).__name__}: {msg}"


def with_retries(fn: Callable[[], T], what: str, attempts: int = API_MAX_ATTEMPTS,
                 backoff: float = API_BACKOFF_SECONDS, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print) -> T:
    """Run fn, retrying 429 / 5xx / network errors with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if not _retryable(e) or i == attempts - 1:
                raise ApiError(explain(e, what)) from e
            wait = backoff * (2 ** i)
            log(f"  {what}: {type(e).__name__}, retrying in {wait:.0f}s ({i + 2}/{attempts})")
            sleep(wait)
    raise AssertionError("unreachable")


class Budget:
    """Running estimate of this process's API spend, with a hard ceiling."""

    def __init__(self, limit: float = SESSION_BUDGET_USD):
        self.limit = limit
        self.spent = 0.0

    def reserve(self, amount: float, what: str) -> None:
        if self.spent + amount > self.limit:
            raise BudgetExceeded(
                f"{what} would bring estimated spend to ${self.spent + amount:.2f}, over the "
                f"${self.limit:.2f} session budget. Stopped before calling the API.")

    def add(self, amount: float) -> None:
        self.spent += amount


def spark_version(model_id: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", model_id))


def pick_spark_model(ids: list[str], preferred: str) -> str | None:
    """The preferred Spark id if the key has it, else the newest muse-spark-* listed."""
    if preferred in ids:
        return preferred
    sparks = [i for i in ids if i.startswith("muse-spark-")]
    return max(sparks, key=spark_version) if sparks else None


# One running total per process, shared by SAM, Spark and transcription.
SESSION = Budget()
