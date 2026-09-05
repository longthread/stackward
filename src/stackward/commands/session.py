"""`login`, `exec`, `shell`: point Pulumi at a profile's backend, and run
commands with that profile's credentials in the environment.

**Why this module exists at all.** `pulumi login` persists only the backend
URL. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `PULUMI_CONFIG_PASSPHRASE`
are read from the environment of *every subsequent* `pulumi` process -- `up`,
`preview`, `config get`, `stack ls`. An encrypted store only `stackward` can
read therefore breaks a plain `pulumi up` outright: `exec` is what makes the
store usable in practice, not an optional convenience layered on top of it.

**Secrets exist only in the child's environment -- and only the right ones.**
`exec` and `shell` decrypt a profile, build one `dict` holding the caller's
own environment plus four names, and hand it to `subprocess.run` as `env=`.
Nothing here writes a credential to disk, logs one, or lets one reach `argv`.
But "the caller's own environment" is not handed through unfiltered:
`STACKWARD_PASSWORD` -- the *store* password, which unlocks every profile,
not just the one being run with -- is scrubbed from every child's
environment, `login`'s included, by `_environ_without_store_password`. A
review of this module found the gap concretely: in the single-command form
`STACKWARD_PASSWORD=x stackward exec -- pulumi up`, the user's intent is
unambiguously "stackward only", and without scrubbing it, arbitrary
caller-supplied code would inherit a secret wider in scope than the one it
was actually invoked to receive. Every error path in this module names a
profile, a key, a file or a redacted backend URL, and never a credential
value.

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
deployment of it. `PULUMI_BACKEND_URL`, when already present in *this*
process's own environment, is checked first and wins over the persisted
file: Pulumi's own CLI honours that variable over a stored login (`exec`
relies on exactly this to override `login`'s persisted state for a single
child), so a stray `PULUMI_BACKEND_URL` a user has exported into their shell
is what a bare `pulumi` command would actually use -- and is therefore what
the guard must compare against, not the file underneath it.

**A backend URL is never put in argv, and is redacted wherever it is
printed.** Most schemes (`s3://`, `gs://`, `azblob://`, `file://`) carry no
secret, but `pulumi login --help` documents `postgres://user:password@host/db`
as a supported backend form, and Global Constraint 3 names `argv` visibility
explicitly. `login` therefore sets `PULUMI_BACKEND_URL` in `pulumi login`'s
environment and invokes it with **no** positional URL argument, rather than
`pulumi login <url>` -- verified, empirically, that a bare `pulumi login`
with only `PULUMI_BACKEND_URL` set logs in identically for a `file://`
backend, with no network access. The backend guard's mismatch message still
names both URLs (that is the point of it), but through `_redact_url`, which
replaces `user:pass@` with `<redacted>@` wherever it appears in a printed
backend URL.

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

**A `Ctrl-C` while the child is running prints no traceback.** CPython
re-raises `SIGINT` as `KeyboardInterrupt`, which `fail_closed` does not catch
(it inherits from `BaseException`, not `Exception`) -- an uncaught one
already exits 130 by Python's own default, so the exit status was never
wrong, but reaching the top of the process uncaught also prints a traceback,
which reads as a crash on `stackward exec -- pulumi up`'s single most common
interrupt path. `_run_child` catches it around the `subprocess.run` call and
returns `_exit_status(-signal.SIGINT)` -- the same 130, with nothing printed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from ..config import ConfigError, find_repo_config, load_config
from ..store import (
    Profile,
    StoreError,
    load_store_config,
    select_profile,
)
from .check_config import fail_closed

if TYPE_CHECKING:
    # Never imported at runtime -- see `_build_credential_store`'s own
    # docstring on why provider modules stay off this module's own top
    # level. `TYPE_CHECKING` is always `False` when this file actually runs,
    # so this line never touches `sys.modules`.
    from ..providers import CredentialStore

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


# Matches the userinfo segment of a URL -- `scheme://user[:pass]@` -- so it
# can be replaced wherever a backend URL is printed. A regex over the raw
# string rather than a full URL parse: `backend_url` is passed through
# verbatim and is never validated as a well-formed URL elsewhere in this
# codebase, and this needs to redact whatever shape actually shows up, not
# only the shapes a stricter parser would accept. `[^/@]*` is what confines
# the match to the first `@` -- the userinfo segment cannot itself contain an
# unencoded `/` or `@` -- so a later `@` in a path or query string is left
# alone.
_USERINFO = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@]*@")


def _redact_url(url: str) -> str:
    """`url`, with any `user[:pass]@` userinfo replaced by `<redacted>@`.

    Most backend schemes (`s3://`, `gs://`, `azblob://`, `file://`) never
    carry a secret in the URL itself, but `pulumi login --help` documents
    `postgres://user:password@host/db` as a supported backend form that
    does. This is the one function everywhere a backend URL is printed
    passes through, so that scheme's password is never one of the values
    that ends up on screen -- see the module docstring.
    """
    return _USERINFO.sub(r"\1<redacted>@", url, count=1)


def _current_backend() -> str | None:
    """What a bare `pulumi` invocation -- one not run through this module's
    own `exec`/`shell`, and so not handed `PULUMI_BACKEND_URL` by them --
    would use as its backend right now.

    Checks *this* process's own `PULUMI_BACKEND_URL` first: Pulumi's CLI
    honours that variable over a persisted login, so a value a user has
    already exported into their shell is what a stray `pulumi` command would
    actually use, and is exactly the drift the guard exists to catch (see
    the module docstring). Only when that is unset does this fall back to
    `<PULUMI_HOME>/credentials.json`'s `current` field -- read directly,
    never by shelling out to `pulumi`, which (see the module docstring)
    contacts the backend itself even for a read-only status query.

    Returns `None` only for the legitimate "nothing else says otherwise"
    state: no `PULUMI_BACKEND_URL`, and either no credentials file or one
    with no `current` entry yet. Any other failure to read or parse the file
    raises `SessionError` -- an unreadable or malformed credentials file must
    not be silently treated as "nothing to guard against", which is exactly
    the fail-open behaviour the backend guard exists to prevent.
    """
    from_env = os.environ.get(PULUMI_BACKEND_URL)
    if from_env:
        return from_env

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
    naming both (redacted -- see `_redact_url`). Never logged into anything
    at all is not a mismatch -- it is an absence, and there is nothing yet
    for `resolved_url` to conflict with; `stackward login` (or a first
    `exec`/`shell`) is how it gets set."""
    current = _current_backend()
    if current is not None and current != resolved_url:
        raise SessionError(
            "backend mismatch: this profile resolves to backend "
            f"{_redact_url(resolved_url)!r}, but pulumi is currently logged "
            f"in to {_redact_url(current)!r}. Run `stackward login` for "
            "this profile first."
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


def _environ_without_store_password() -> dict[str, str]:
    """A copy of the caller's own environment with `STACKWARD_PASSWORD`
    removed.

    Shared by `_child_env` (`exec`/`shell`) and `login`'s own child
    environment. The store password unlocks *every* profile in the store,
    while what either command hands to a child is scoped to one profile
    (`exec`/`shell`) or to nothing secret at all (`login`) -- inheriting it
    into arbitrary caller-supplied code, or even into `pulumi` itself, widens
    that scope for no reason any of the three commands need. A copy, never
    `os.environ` itself: mutating the real mapping here would affect every
    later call in this same process.
    """
    env = dict(os.environ)
    env.pop(ENV_STORE_PASSWORD, None)
    return env


def _child_env(credentials: Mapping[str, str], backend_url: str) -> dict[str, str]:
    """The caller's own environment, minus the store password, plus exactly
    the four session names."""
    env = _environ_without_store_password()
    for name in CREDENTIAL_NAMES:
        env[name] = credentials[name]
    env[PULUMI_BACKEND_URL] = backend_url
    return env


def _build_credential_store(
    profile: Profile, *, directory: Path | None
) -> "CredentialStore":
    """The `CredentialStore` `profile.credentials`'s `provider` selects
    (`"file"` when the table or the key is absent, preserving today's
    behaviour unchanged).

    Provider modules are imported here, function-locally -- never at this
    module's own top level -- for the reason `providers/__init__.py`'s
    module docstring states: `cli.py` imports this module unconditionally,
    so a module-scope import here would put a provider on every
    `stackward` invocation's import graph, `check-config` and `pre-commit`
    included, which is exactly what the gate-path invariant forbids. A
    caller that never calls this function -- which is every gate-path
    command -- never triggers any of these imports.

    `"file"` reads the store password now, at the same point `_prepare_env`
    always has -- after the backend guard, never before it (see
    `test_prepare_env_checks_the_backend_guard_before_reading_a_password`).
    Only `"file"` needs one; `"env"` and `"command"` never prompt.
    """
    provider = profile.credentials.get("provider", "file")
    if provider == "file":
        from ..providers.file import FileCredentialStore

        return FileCredentialStore(_read_store_password(), directory=directory)
    if provider == "env":
        from ..providers.env import EnvCredentialStore

        return EnvCredentialStore(CREDENTIAL_NAMES)
    if provider == "command":
        from ..providers.command import CommandCredentialStore

        command = profile.credentials.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) for item in command
        ):
            raise SessionError(
                f"profile {profile.name!r}: credentials provider 'command' needs "
                "a non-empty 'command' list of strings"
            )
        return CommandCredentialStore(command)
    raise SessionError(
        f"profile {profile.name!r}: unknown credentials provider {provider!r}"
    )


def _prepare_env(
    explicit_profile: str | None, *, directory: Path | None = None
) -> dict[str, str]:
    """Resolve the profile, enforce the backend guard, resolve its
    credentials through whichever `CredentialStore` its `provider` selects,
    and return the exact environment `exec`/`shell` must run their child in.

    Raises `StoreError`, `ConfigError` or `SessionError` and returns nothing
    partial: every step that can fail runs before `_child_env` builds
    anything, so a caller either gets a complete environment or an exception,
    never a dict missing one of the four names.
    """
    profile = _resolve_profile(explicit_profile, directory=directory)
    url = _compose_backend_url(profile)
    _check_backend_guard(url)
    credential_store = _build_credential_store(profile, directory=directory)

    from ..providers import ProviderError  # local: see `_build_credential_store`

    try:
        credentials = credential_store.resolve(profile.name)
    except ProviderError as exc:
        # `file` never raises this -- `store.StoreError` propagates
        # unchanged, per `providers.file`'s own docstring -- so this branch
        # is reachable only for a provider (`command`, or a future one) that
        # actually failed, never for `file`'s existing, already-tested
        # error paths.
        raise SessionError(str(exc)) from exc
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
    meaning `subprocess.run` itself gives it. No current caller in this
    module actually relies on that default -- `login` passes an explicit
    environment too now, with `STACKWARD_PASSWORD` scrubbed even though
    `pulumi login` never needs it, for the same reason `exec`/`shell` do --
    but it is kept as `_run_child`'s own general-purpose meaning rather than
    coupled to what today's callers happen to do.

    A missing executable is reported by name (never by the full command line,
    which could hold an argument value, though never a credential -- those
    never reach argv at all) and mapped to exit code 2, the same "could not
    run" code every other refusal in this module uses.

    A `Ctrl-C` reaching this process while the child runs surfaces here as
    `KeyboardInterrupt` (CPython's own re-raising of `SIGINT`) rather than as
    a return code `subprocess` reports -- `fail_closed` does not catch it
    (`BaseException`, not `Exception`), so left alone it would propagate to
    the top of the process and print a traceback before Python's own default
    handling exits 130. Caught here and translated the same way a
    signal-killed child is, for the same exit status with nothing printed.
    """
    try:
        completed = subprocess.run(command, env=env)
    except OSError as exc:
        print(f"error: cannot run {command[0]!r}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return _exit_status(-signal.SIGINT)
    return _exit_status(completed.returncode)


def _report(exc: Exception) -> int:
    """Print `exc`'s message as `error: ...` on stderr and return exit code
    2 -- the "could not run" code every pre-flight refusal in `cmd_login`,
    `cmd_exec` and `cmd_shell` uses. `exc` must be one of `_KNOWN_ERRORS`:
    none of those ever carry a credential value in their message, which is
    not a property of `Exception` in general -- an unanticipated exception
    is `fail_closed`'s job, not this function's."""
    print(f"error: {exc}", file=sys.stderr)
    return 2


@fail_closed
def cmd_login(args: argparse.Namespace) -> int:
    """Entry point for `stackward login`.

    Resolves the profile's backend URL and runs `pulumi login` with it set
    as `PULUMI_BACKEND_URL` in the child's environment -- never as a
    positional argument. `pulumi login --help` documents
    `postgres://user:password@host/db` as a supported backend form, and an
    argv value is visible in `ps` for the life of the call; verified,
    empirically, that a bare `pulumi login` (no positional URL) honours
    `PULUMI_BACKEND_URL` identically to the positional form, for a `file://`
    backend with no network access. Never opens the credentials store -- a
    profile can be logged into before its credentials have been set at all
    (`store.py`'s own R3) -- and never subject to the backend guard, which
    exists to protect the commands that run with a profile's credentials
    live in the environment; `login` is the command that *sets* what the
    guard checks against, so gating it on itself would make switching
    profiles impossible.
    """
    try:
        profile = _resolve_profile(args.profile)
        url = _compose_backend_url(profile)
    except _KNOWN_ERRORS as exc:
        return _report(exc)

    pulumi = shutil.which("pulumi")
    if pulumi is None:
        print("error: pulumi executable not found on PATH", file=sys.stderr)
        return 2
    env = _environ_without_store_password()
    env[PULUMI_BACKEND_URL] = url
    return _run_child([pulumi, "login"], env)


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
        return _report(exc)

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
        return _report(exc)

    return _run_child([shell_path], env)
