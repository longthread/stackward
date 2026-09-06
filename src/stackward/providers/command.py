"""`command`: a generic provider that shells out to an externally-configured
CLI — what "adding a remote provider must not add a Python dependency" (the
brief's own wording) means in practice. This module imports nothing beyond
the standard library and `ProviderError` from this same package; the argv it
runs, and everything that binary needs to authenticate or reach a remote
store, comes entirely from `command` in configuration. No vendor or product
name, API shape, or authentication convention appears anywhere in this
module, by design — a real remote store's own CLI is exactly as far away
from this code as `config` says it is.

**Wire format.** `CommandSecretSource` runs `[*command, name]` and reads
stdout as the value (one trailing newline stripped, matching
`commands.set_secrets._pulumi_config_get`'s own convention for the same
reason); empty output means "not set". `CommandCredentialStore` runs
`[*command, profile]` and parses stdout as a JSON object of `name: value`
strings — the one already-generic wire format this codebase depends on
everywhere else (`json` is stdlib; no new format to invent or document
per-remote). Both raise `ProviderError` — never return an empty or partial
result to mean failure — on a non-zero exit, a timeout, or a missing
executable; `CommandCredentialStore` raises it too on stdout that is not a
JSON object of strings, since that is this provider failing to answer just
as much as a non-zero exit is.

**Never written to disk, and stdout is decoded, never re-encoded, for
`CommandSecretSource`.** A fetched value lives in this process's memory for
the one call that resolved it — matching `providers/__init__.py`'s
"fetched values are held in memory for the invocation only" — and `_run`
decodes the child's stdout with `errors="surrogateescape"` rather than
`"strict"`, so a value that is not valid UTF-8 becomes a `str` this module
can still return (Python's own encoding for `os.environ`, applied here for
the same reason) instead of this provider itself raising on a value it
never needed to interpret as text beyond passing it through.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence

from . import ProviderError

DEFAULT_TIMEOUT_SECONDS = 30.0


def _require_command(command: Sequence[str]) -> list[str]:
    if not command:
        raise ProviderError("command provider: 'command' must be a non-empty list")
    return list(command)


def _run(command: Sequence[str], arg: str, timeout: float) -> str:
    """Run `[*command, arg]` and return its stdout, decoded as UTF-8 with
    `surrogateescape` (see the module docstring).

    Raises `ProviderError` on anything short of a clean exit: a timeout, a
    missing executable, or a non-zero exit code. Never includes the child's
    stdout or stderr in the exception message — neither is this module's to
    vouch for as free of the very value it just ran to fetch.
    """
    try:
        completed = subprocess.run([*command, arg], capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"{command[0]!r} timed out") from exc
    except OSError as exc:
        raise ProviderError(f"cannot run {command[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        raise ProviderError(f"{command[0]!r} exited {completed.returncode}")
    return completed.stdout.decode("utf-8", errors="surrogateescape")


class CommandSecretSource:
    """Resolves one logical name by running `command <name>` and reading its
    stdout as the value. Empty stdout means "not set" — see the module
    docstring's wire format."""

    def __init__(self, command: Sequence[str], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._command = _require_command(command)
        self._timeout = timeout

    def resolve(self, name: str) -> str | None:
        text = _run(self._command, name, self._timeout)
        value = text[:-1] if text.endswith("\n") else text
        return value or None


class CommandCredentialStore:
    """Resolves a profile's bootstrap set by running `command <profile>` and
    parsing its stdout as a JSON object of `name: value` strings."""

    def __init__(self, command: Sequence[str], *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._command = _require_command(command)
        self._timeout = timeout

    def resolve(self, profile: str) -> dict[str, str]:
        text = _run(self._command, profile, self._timeout)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self._command[0]!r} did not print a JSON object") from exc
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
        ):
            raise ProviderError(f"{self._command[0]!r} must print a JSON object of strings")
        return payload
