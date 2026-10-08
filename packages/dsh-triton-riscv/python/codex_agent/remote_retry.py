"""Bounded retries for observation, never for starting or mutating remote jobs."""
import subprocess
import time

from codex_agent.process_control import remaining

READ_ACTIONS = frozenset({"inspect", "collect"})
RETRY_DELAYS = (0.5, 1.0)


class TransportUnavailable(RuntimeError):
    pass


def observe_with_retry(action, call, *, on_retry=None):
    attempts = 1 + len(RETRY_DELAYS) if action in READ_ACTIONS else 1
    for attempt in range(attempts):
        remaining(900)
        try:
            return call()
        except (TransportUnavailable, subprocess.TimeoutExpired) as error:
            if attempt == attempts - 1:
                raise
            if on_retry:
                on_retry(attempt + 1, type(error).__name__)
            end = time.monotonic() + RETRY_DELAYS[attempt]
            while time.monotonic() < end:
                # Share the caller's wall-clock/cancellation budget, including backoff.
                time.sleep(min(remaining(0.1), max(0, end - time.monotonic())))
