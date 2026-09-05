"""`login`, `exec`, `shell`: point Pulumi at a profile's backend, and run
commands with that profile's credentials in the environment.

**Why this module exists at all.** `pulumi login` persists only the backend
URL. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `PULUMI_CONFIG_PASSPHRASE`
are read from the environment of *every subsequent* `pulumi` process -- `up`,
`preview`, `config get`, `stack ls`. An encrypted store only `stackward` can
read therefore breaks a plain `pulumi up` outright: `exec` is what makes the
store usable in practice, not an optional convenience layered on top of it.

**Secrets exist only in the child's environment.** `exec` and `shell` decrypt
a profile, build one `dict` holding the caller's own environment plus four
names, and hand it to `subprocess.run` as `env=`. Nothing here writes a
credential to disk, logs one, or lets one reach `argv` -- the values travel
from the sealed store into the child's environment block and nowhere else.
Every error path in this module names a profile, a key, a file or a backend
URL, and never a credential value.

**`shell` inherits, it does not replace.** Both `exec` and `shell` overlay
exactly four names onto a *copy* of the caller's own environment
(`PATH`, `HOME`, `TERM` and everything else the child would otherwise need
survive untouched); they never construct an environment from the four names
alone. `tests/test_session.py` asserts this by diffing parent and child
environments, not by counting keys in the child -- an implementation that
happened to also inherit the right ambient variables by accident would still
fail a diff-based test if it dropped or altered anything else.

**The backend guard applies to `exec` and `shell`, never to `login`.**
`login` is the command that *sets* Pulumi's persisted backend; gating it on
matching that same state would make switching profiles impossible. The guard
instead sits in front of the two commands that run someone else's code with a
profile's credentials live in its environment: before doing that, this module
compares the resolved profile's backend URL against Pulumi's own persisted
`current` backend (see `_current_backend`) and refuses on any mismatch, naming
both. Without it, a second profile is merely a convenience -- a `pulumi`
invocation that does not go through `stackward exec` (a stray script, a
forgotten shell alias, a plugin) would still silently use whichever backend
the last `login` pointed at.

**Reading Pulumi's current backend never touches the network.**
`pulumi whoami`, `pulumi about` and `pulumi whoami --json` were all tried
against a real (unreachable) S3 backend during development of this module,
and every one of them attempts to contact the backend itself -- exactly the
network dependency a pre-flight safety check must not have (a guard that can
hang or fail on network conditions unrelated to the mismatch it exists to
catch is not a guard). `pulumi login <url>` was also observed, empirically,
to write the URL verbatim to `<PULUMI_HOME>/credentials.json`'s `current`
field with no network access at all for a `file://` backend. This module
therefore reads that field directly. `PULUMI_HOME` (default `~/.pulumi`) is
Pulumi's own documented override, not knowledge about any particular
deployment of it.

**The store password is never read from `sys.stdin`.** `exec` decrypts a
profile before handing the *child's* stdin, stdout and stderr through
untouched -- a `stackward shell` session and a `stackward exec -- pulumi up`
that itself prompts for confirmation both depend on that descriptor being
free. `_read_store_password` checks `STACKWARD_PASSWORD` first (the
non-interactive path: CI has no controlling terminal to prompt against) and
otherwise prompts with `getpass.getpass`, which itself is verified, before
ever being called, to be able to open `/dev/tty` directly -- if it cannot,
this module raises rather than letting `getpass` fall back to its own
`sys.stdin` read, which would both consume bytes meant for the child and
echo them to the screen. `login` needs none of this: it resolves a backend
URL from `config` (plaintext, unencrypted) and never opens the credentials
store at all, which is also why a profile can be logged into before its
credentials have ever been set (see `store.py`'s own note on this).

**Exit status is translated, not passed through raw.** `subprocess`
represents a child killed by a signal as a *negative* return code; handing
that negative number to `sys.exit` gets reduced modulo 256 at the OS boundary
and no longer identifies the signal. `_exit_status` converts it to the
`128 + signal` form a POSIX shell's own `$?` would report, which is the one
representation `sys.exit` can carry through intact.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote, urlencode

from ..config import ConfigError, find_repo_config, load_config
from ..store import (
    Profile,
    StoreError,
    load_store_config,
    resolve_credentials,
    select_profile,
)
from .check_config import fail_closed

# Read once, never printed: the non-interactive escape hatch for `exec`/
# `shell` in an environment with no controlling terminal (CI, most notably).
# Never accepted as a CLI argument -- only ever an environment variable, per
# the same rule that keeps every other secret out of argv and out of `ps`.
ENV_STORE_PASSWORD = "STACKWARD_PASSWORD"

# The three names `resolve_credentials` must return for a profile before
# `exec`/`shell` can run anything -- the exact vocabulary `pulumi` itself
# reads from a child process's environment (see the module docstring). Named
# here, not in `store.py`: the store deliberately does not know which
# variables a backend needs, so this is the one place in the codebase that
# does.
AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
PULUMI_CONFIG_PASSPHRASE = "PULUMI_CONFIG_PASSPHRASE"
PULUMI_BACKEND_URL = "PULUMI_BACKEND_URL"

CREDENTIAL_NAMES = (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, PULUMI_CONFIG_PASSPHRASE)


class SessionError(Exception):
    """Something specific to this module stopped `login`/`exec`/`shell` from
    proceeding: no store password was available, a decrypted profile was
    missing a required credential name, a backend could not be determined, or
    the resolved backend does not match Pulumi's current one.

    Never carries a credential value -- every message here names a profile, a
    key name, a file path or a backend URL, never a secret."""


# Errors this module's own `cmd_*` entry points recognise and report with a
# one-line message rather than falling through to `fail_closed`'s generic
# "could not run (<type>)". `StoreError` alone already covers `ProfileError`
# and `PasswordError`, both of which subclass it.
_KNOWN_ERRORS = (StoreError, ConfigError, SessionError)


def _repo_profile() -> str | None:
    """The repository's own `profile =`, from the nearest `.stackward.toml`
    above the current working directory, or `None` when there is none.

    Wires the precedence tier `select_profile` has always accepted but that
    nothing supplied until this module existed: without this, a repository
    declaring its own backend -- the normal case, per `store.select_profile`'s
    own docstring -- would silently fall through to `default_profile` instead.
    """
    path = find_repo_config()
    if path is None:
        return None
    return load_config(path).profile


def _resolve_profile(explicit: str | None, *, directory: Path | None = None) -> Profile:
    """`select_profile`, with the repository tier wired in.

    `_repo_profile()` is skipped entirely when `explicit` is given: an
    explicit `--profile` is the highest-precedence source and answers the
    question completely on its own, so a `.stackward.toml` that happens to be
    unparsable elsewhere in the repository must not stop it from working.
    Every other source (`STACKWARD_PROFILE`, `default_profile`) still leaves
    `.stackward.toml` parsed and validated -- see `config.py`'s own "invalid
    is never treated as absent" rule, which this preserves for every case
    where the repository tier could actually be consulted.
    """
    config = load_store_config(directory)
    repo_profile = None if explicit is not None else _repo_profile()
    return select_profile(explicit, config=config, repo_profile=repo_profile)


def _compose_backend_url(profile: Profile) -> str:
    """The profile's backend URL, ready to hand to `pulumi login`/inject as
    `PULUMI_BACKEND_URL`.

    `backend_url` passes through **byte-for-byte** -- never normalised,
    re-encoded or reordered -- so that whatever a user (or a future backend
    scheme) wrote is exactly what `pulumi` receives. Composing the component
    form into a URL is this module's job precisely because Task 6's store
    deliberately left it undone (see `store.Profile`'s own docstring): the
    query-parameter spelling below is a fact about Pulumi's S3-compatible
    backend, not about a profile.

    The composed form is `s3://<bucket>[/<prefix>][?region=...&endpoint=...
    &s3ForcePathStyle=true]`. `s3ForcePathStyle` is only ever added alongside
    `endpoint`: virtual-hosted-style addressing (the default without a custom
    endpoint) does not work against most S3-compatible services, which is
    exactly the case an `endpoint` in the profile signals. An empty string for
    `prefix`, `region` or `endpoint` is treated the same as absent, since
    there is no way to distinguish "the operator wrote an empty prefix on
    purpose" from "the key was left blank" and the safe reading for an
    optional component is to omit it.

    Never called with a profile that has neither form: `store._build_profile`
    already refuses to construct one. The check below exists anyway for a
    `Profile` built by hand (as a test can), so this function fails closed
    rather than raising an unrelated `AttributeError`.
    """
    if profile.backend_url is not None:
        return profile.backend_url
    if not profile.bucket:
        raise SessionError(f"profile {profile.name!r} has no backend configured")

    url = f"s3://{quote(profile.bucket, safe='')}"
    if profile.prefix:
        url += "/" + quote(profile.prefix, safe="/")

    params: dict[str, str] = {}
    if profile.region:
        params["region"] = profile.region
    if profile.endpoint:
        params["endpoint"] = profile.endpoint
        params["s3ForcePathStyle"] = "true"
    if params:
        url += "?" + urlencode(params)
    return url


def _pulumi_home() -> Path:
    """`PULUMI_HOME`, or its documented default. Not environment-specific
    knowledge -- this is Pulumi's own override, the same way `XDG_CONFIG_HOME`
    is the platform's."""
    home = os.environ.get("PULUMI_HOME")
    return Path(home) if home else Path.home() / ".pulumi"


def _current_backend() -> str | None:
    """Pulumi's persisted backend URL, read directly from
    `<PULUMI_HOME>/credentials.json`'s `current` field -- never by shelling
    out to `pulumi`, which (see the module docstring) contacts the backend
    itself even for a read-only status query.

    Returns `None` only for the legitimate "never logged in" state: the file
    does not exist, or exists but does not yet have a `current` entry. Any
    other failure to read or parse it raises `SessionError` -- an unreadable
    or malformed credentials file must not be silently treated as "nothing to
    guard against", which is exactly the fail-open behaviour the backend guard
    exists to prevent.
    """
    path = _pulumi_home() / "credentials.json"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SessionError(f"cannot read {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionError(
            f"{path}: not a valid pulumi credentials file ({type(exc).__name__})"
        ) from exc
    if not isinstance(data, dict):
        raise SessionError(f"{path}: not a valid pulumi credentials file")

    current = data.get("current")
    if current is None:
        return None
    if not isinstance(current, str):
        raise SessionError(f"{path}: 'current' field is not a string")
    return current


def _check_backend_guard(resolved_url: str) -> None:
    """Refuse when Pulumi's persisted backend does not match `resolved_url`,
    naming both. Never logged into anything at all is not a mismatch -- it is
    an absence, and there is nothing yet for `resolved_url` to conflict with;
    `stackward login` (or a first `exec`/`shell`) is how it gets set."""
    current = _current_backend()
    if current is not None and current != resolved_url:
        raise SessionError(
            "backend mismatch: this profile resolves to backend "
            f"{resolved_url!r}, but pulumi is currently logged in to "
            f"{current!r}. Run `stackward login` for this profile first."
        )


def _tty_available() -> bool:
    """Whether `/dev/tty` can be opened directly, independent of `sys.stdin`.

    A dedicated, monkeypatchable seam rather than inlining the `os.open` call
    in `_read_store_password`: it is what lets that function's "no interactive
    terminal" branch be exercised without a real detached process, and it is
    the exact question `getpass.getpass` itself answers first -- checking it
    ourselves means we can refuse *before* `getpass` would otherwise fall back
    to reading (and echoing) `sys.stdin`, which -- for `exec`/`shell` -- is the
    child's stdin, not a spare input source of our own to consume.
    """
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return False
    os.close(fd)
    return True


def _read_store_password() -> str:
    """`STACKWARD_PASSWORD` if set, otherwise an interactive prompt.

    Never reads `sys.stdin`. The environment variable exists for the
    non-interactive case `exec`/`shell` must support (a CI job running
    `stackward exec -- pulumi up` has no controlling terminal to prompt
    against); the prompt exists for everyone else, and refuses outright,
    rather than degrading to a visible, stdin-consuming fallback, when
    `_tty_available` says `/dev/tty` cannot be opened.
    """
    from_env = os.environ.get(ENV_STORE_PASSWORD)
    if from_env is not None:
        return from_env
    if not _tty_available():
        raise SessionError(
            f"no store password available: set {ENV_STORE_PASSWORD}, or run "
            "this command from an interactive terminal"
        )
    return getpass.getpass("stackward credential store password: ")


def _require_credential_names(credentials: Mapping[str, str], profile: str) -> None:
    """Refuse, naming the missing key names (never a value), when a
    decrypted profile does not carry every name `exec`/`shell` must inject.
    An envelope holding extra names beyond the three required ones is
    accepted as-is; only those three are ever injected -- see `_child_env`."""
    missing = [name for name in CREDENTIAL_NAMES if name not in credentials]
    if missing:
        raise SessionError(
            f"profile {profile!r} is missing required credential(s): "
            f"{', '.join(missing)}"
        )


def _child_env(credentials: Mapping[str, str], backend_url: str) -> dict[str, str]:
    """The caller's own environment, plus exactly the four session names.

    A copy of `os.environ`, never `os.environ` itself -- overlaying onto the
    real mapping would leak the four names into every subsequent call in this
    same process, including this one's own `fail_closed` error path."""
    env = dict(os.environ)
    for name in CREDENTIAL_NAMES:
        env[name] = credentials[name]
    env[PULUMI_BACKEND_URL] = backend_url
    return env


def _prepare_env(
    explicit_profile: str | None, *, directory: Path | None = None
) -> dict[str, str]:
    """Resolve the profile, enforce the backend guard, decrypt its
    credentials and return the exact environment `exec`/`shell` must run
    their child in.

    Raises `StoreError`, `ConfigError` or `SessionError` and returns nothing
    partial: every step that can fail runs before `_child_env` builds
    anything, so a caller either gets a complete environment or an exception,
    never a dict missing one of the four names.
    """
    profile = _resolve_profile(explicit_profile, directory=directory)
    url = _compose_backend_url(profile)
    _check_backend_guard(url)
    password = _read_store_password()
    credentials = resolve_credentials(profile.name, password, directory=directory)
    _require_credential_names(credentials, profile.name)
    return _child_env(credentials, url)


def _exit_status(returncode: int) -> int:
    """A `subprocess` return code, translated to what a POSIX shell's `$?`
    would report for the same child.

    A normal POSIX exit is already 0-255 (the kernel only ever reports 8 bits
    of it) and passes through unchanged. A negative `returncode` is how
    `subprocess` represents a POSIX child killed by a signal; `sys.exit`
    cannot carry that negative number through intact (it is reduced modulo
    256 at the OS boundary, and no longer identifies the signal), so it is
    translated to `128 + signal` -- the one representation that survives.

    This function does not special-case Windows, where a terminated
    process's `returncode` can be a large *positive* `STATUS_*` value outside
    0-255: that value is returned unchanged, and is subject to whatever
    `sys.exit` itself does with a number that large on that platform, the
    same as it would be for any exit code this tool ever returns.
    """
    if returncode < 0:
        return 128 - returncode
    return returncode


def _run_child(command: list[str], env: dict[str, str] | None = None) -> int:
    """Run `command`, inheriting this process's stdin/stdout/stderr, and
    return its translated exit status.

    `env=None` means "inherit the caller's environment unchanged", the same
    meaning `subprocess.run` itself gives it -- used by `login`, which injects
    nothing. `exec`/`shell` always pass an explicit environment built by
    `_child_env`.

    A missing executable is reported by name (never by the full command line,
    which could hold an argument value, though never a credential -- those
    never reach argv at all) and mapped to exit code 2, the same "could not
    run" code every other refusal in this module uses.
    """
    try:
        completed = subprocess.run(command, env=env)
    except OSError as exc:
        print(f"error: cannot run {command[0]!r}: {exc}", file=sys.stderr)
        return 2
    return _exit_status(completed.returncode)


@fail_closed
def cmd_login(args: argparse.Namespace) -> int:
    """Entry point for `stackward login`.

    Resolves the profile's backend URL and runs `pulumi login <url>`. Never
    opens the credentials store -- a profile can be logged into before its
    credentials have been set at all (`store.py`'s own R3) -- and never
    subject to the backend guard, which exists to protect the commands that
    run with a profile's credentials live in the environment; `login` is the
    command that *sets* what the guard checks against, so gating it on
    itself would make switching profiles impossible.
    """
    try:
        profile = _resolve_profile(args.profile)
        url = _compose_backend_url(profile)
    except _KNOWN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pulumi = shutil.which("pulumi")
    if pulumi is None:
        print("error: pulumi executable not found on PATH", file=sys.stderr)
        return 2
    return _run_child([pulumi, "login", url])


@fail_closed
def cmd_exec(args: argparse.Namespace) -> int:
    """Entry point for `stackward exec -- <command...>`.

    An empty command (nothing after `--`, or no `--` at all) is a usage
    error, reported and returned as exit code 2 before anything is resolved
    or decrypted -- the same code every other pre-flight refusal in this
    module uses.
    """
    if not args.argv:
        print(
            "error: exec requires a command: stackward exec -- <command...>",
            file=sys.stderr,
        )
        return 2

    try:
        env = _prepare_env(args.profile)
    except _KNOWN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return _run_child(args.argv, env)


@fail_closed
def cmd_shell(args: argparse.Namespace) -> int:
    """Entry point for `stackward shell`: the same as `exec`, with the
    user's `$SHELL` as the command."""
    shell_path = os.environ.get("SHELL")
    if not shell_path:
        print("error: $SHELL is not set", file=sys.stderr)
        return 2

    try:
        env = _prepare_env(args.profile)
    except _KNOWN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return _run_child([shell_path], env)
