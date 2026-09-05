"""The credential store: which backend, whose credentials, and where they live.

A profile is a Pulumi state backend plus the credentials that unlock it. That
pairing is split across two files in `${XDG_CONFIG_HOME:-~/.config}/stackward/`:

    config       TOML, mode 0600 — backend identity
    credentials  JSON, mode 0600 — one sealed envelope per profile

The split is what makes `config` diffable, reviewable and safe to reason about
without decrypting anything: it says *which* bucket, endpoint and region a
profile names, and a user checking what a profile points at never has to open
the file that would show them a key if they mistyped a command.

**`config` is not, however, credential-free, and must not be treated as such.**
It once carried mode 0644 on the stated grounds that it "holds backend identity
and never a credential". That claim was wrong, and in the one place it mattered
most: `pulumi login --help` documents `postgres://user:password@host/db` as a
supported backend form, `Profile.backend_url` passes a URL through verbatim,
and `commands.session` exists partly to redact exactly that password wherever
it prints one. A supported, documented configuration therefore put a live
credential in a world-readable file. Both files are 0600 now, and `config` is
covered by the same permissiveness warning `credentials` always was. A
`backend_url` carrying userinfo is still accepted — refusing it would make a
Pulumi-supported backend unusable through this tool, and there is nowhere else
to put it, since `credentials` holds only the sealed name/value envelopes the
session commands inject — but it is a credential, in a file whose mode and
error messages now assume so.

**Everything else in this module follows from three refusals.**

*Never guess a backend.* Selection runs `--profile`, then `STACKWARD_PROFILE`,
then the repository's own `profile =`, then `default_profile`, and then it
errors. There is no built-in default and no fallback: a tool that guesses which
backend it is pointed at can publish state, or a credential, to the wrong place.
A name that comes from a source but does not exist in `config` is an error too —
it never falls through to the next source, because "the profile you asked for is
missing" and "you asked for no profile" are different problems with different
fixes.

*Never write a credential outside an envelope.* Values arrive as a mapping, go
into `crypto.seal_json`, and come back out only through `resolve_credentials`.
This module has no code path that prints one, logs one, or puts one in an
exception message; error text names profiles, files and key names only.

*Never leave a half-written store.* Every write goes through `atomic_write`:
a temporary file in the same directory, `fsync`, the mode, then `rename`. The
mode is set before the rename and not after, because "after" is a window in
which the finished file exists at its real path with the wrong permissions.
Password rotation builds the entire re-sealed document in memory first and then
performs exactly one such write, so it re-seals every envelope or none — a
rotation that failed halfway through would leave a store whose profiles need two
different passwords, and no way to tell which is which.

There is deliberately no CLI command here. This task's brief lists this module,
`crypto.py` and their tests; the commands that drive them (`login`, `exec`,
`credentials`) arrive with the tasks that own them, on top of this API.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .crypto import (
    CryptoError,
    DecryptionError,
    EmptyPasswordError,
    describe_value,
    seal_json,
    unseal_json,
)

CONFIG_FILENAME = "config"
CREDENTIALS_FILENAME = "credentials"

DIR_MODE = 0o700
# 0600, not 0644: a `backend_url` of the documented `postgres://user:password@
# host/db` form puts a live credential in `config`, so it gets the same
# owner-only mode `credentials` has. See the module docstring.
CONFIG_MODE = 0o600
CREDENTIALS_MODE = 0o600

ENV_PROFILE = "STACKWARD_PROFILE"

STORE_VERSION = 1

# The verifier is a store-level envelope holding a known plaintext, written when
# the store is initialised and opened before anything else. Without it a
# mistyped password at `credentials init` produces a store that looks fine and
# fails at first use, possibly weeks later and on a different machine; with it,
# a wrong password is reported as a wrong password, immediately, and is never
# confused with a corrupt profile envelope.
VERIFIER_KEY = "verifier"
VERIFIER_AAD = "stackward:verifier"
VERIFIER_PLAINTEXT = "stackward credential store"

# A profile name is used verbatim as the AAD of its envelope, so the set of
# valid names must not contain the verifier's AAD — otherwise a profile could be
# created whose envelope is interchangeable with the verifier's. Excluding ':'
# from the charset is what keeps the two namespaces disjoint; it is a mechanism,
# not a convention, and `test_store.py` proves it by trying the collision.
#
# '.' is excluded too, for a different reason: TOML reads a dot in a table
# header as nesting, so `[profile.a.b]` declares a table `b` inside a table `a`
# and not a profile named `a.b`. Allowing dotted names would mean a name that
# has two spellings — one of which silently parses as something else — for no
# benefit. What remains is a strict subset of the TOML bare-key charset — TOML
# would also allow a leading '_' or '-', which this does not — so every valid
# name is spellable as `[profile.<name>]` with no quoting and no ambiguity,
# but not every bare TOML key is a valid profile name.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

_TOP_LEVEL_KEYS = frozenset({"default_profile", "profile"})
_PROFILE_KEYS = frozenset(
    {"backend_url", "bucket", "prefix", "endpoint", "region", "credentials"}
)
_COMPONENT_KEYS = ("bucket", "prefix", "endpoint", "region")


class StoreError(Exception):
    """Anything that stops the store from answering. Callers print these, so no
    subclass ever interpolates a credential value into its message."""


class ProfileError(StoreError):
    """No profile could be selected, or the selected one does not exist.

    Separate from `StoreError` so a command can tell "you have not told me which
    backend to use" — which a user fixes with `--profile` or a config edit —
    apart from "the store itself is unusable"."""


class PasswordError(StoreError):
    """The store password is unusable: absent, or wrong.

    Both mean "the password given cannot open this store", which is a single
    problem to whoever reads it. Kept distinct from a profile envelope's
    `DecryptionError`, which is a different one: this says "retype your
    password", while a failure on a profile envelope *after* the verifier opened
    says the store itself is damaged or was tampered with."""


@dataclass(frozen=True)
class Profile:
    """One `[profile.<name>]` table: a backend, in one of the two forms.

    `backend_url` is passed through **verbatim** and is never parsed,
    normalised or reassembled by this tool — that is what lets `s3://`,
    `gs://`, `azblob://` and `file://` all work without this module knowing
    anything about any of them, and what keeps a scheme nobody has thought of
    yet working too.

    The component form (`bucket`/`prefix`/`endpoint`/`region`) describes an
    S3-compatible store in generic terms for users who would rather not
    hand-assemble a URL. Composing those components *into* a URL is
    deliberately not done here: the query-parameter spelling a Pulumi backend
    expects is knowledge about Pulumi, not about profiles, and it belongs with
    the command that runs `pulumi login`. This module carries the components as
    structured data and stops there.

    `credentials` is the raw, otherwise-unvalidated `[profile.<name>.
    credentials]` table -- `provider` (a string, defaulting to `"file"` when
    the table or the key is absent) and whatever else a chosen provider
    needs (a remote provider's `command`, say). This module validates only
    that the table exists and that `provider`, if present, is a string; it
    does not know which providers exist any more than `config.py`'s
    `_build_secrets` knows what `[secrets.*]` means -- that is
    `providers/__init__.py`'s vocabulary, not this one's, and this module
    does not import it (see that package's own module docstring for why).
    """

    name: str
    backend_url: str | None = None
    bucket: str | None = None
    prefix: str | None = None
    endpoint: str | None = None
    region: str | None = None
    credentials: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_components(self) -> bool:
        return any(getattr(self, key) is not None for key in _COMPONENT_KEYS)


@dataclass(frozen=True)
class StoreConfig:
    """A parsed `config` file: the default profile, and the profiles."""

    default_profile: str | None = None
    profiles: Mapping[str, Profile] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Locations and file handling
# ---------------------------------------------------------------------------


def store_dir(directory: Path | None = None) -> Path:
    """The store directory: `directory` when given, otherwise `cli.config_home()`.

    Named `directory`, not `home`: this is the store directory itself, the
    thing `config_home()` returns with `/stackward` already appended. A
    parameter called `home` invites a caller to pass a config home or
    `Path.home()` and silently get `<that>/credentials`.

    `cli` is imported inside the function rather than at module scope. The
    commands that will drive this store live under `commands/`, which `cli`
    imports at *its* module scope — so a module-level `from .cli import
    config_home` here would close a cycle (`cli` → `commands.x` → `store` →
    `cli`) the moment such a command exists, and would fail on a name `cli` has
    not defined yet at that point in its own execution. `config.py` avoids the
    same cycle the same way; see `enforce_min_version` there.
    """
    if directory is not None:
        return directory
    from .cli import config_home  # local: breaks an import cycle

    return config_home()


def config_path(directory: Path | None = None) -> Path:
    return store_dir(directory) / CONFIG_FILENAME


def credentials_path(directory: Path | None = None) -> Path:
    return store_dir(directory) / CREDENTIALS_FILENAME


def warn_if_permissive(path: Path, allowed: int) -> None:
    """Warn on stderr when `path` grants more than `allowed`.

    A warning and not an error: refusing to read a store because a `umask` or a
    restored backup widened a mode would lock a user out of their own
    credentials, and the value has already been exposed by then — refusing
    afterwards protects nothing. The warning goes to stderr so it cannot
    contaminate a command's stdout, which callers parse.
    """
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return  # absent or unreadable: the caller's own read reports it properly
    extra = mode & ~allowed
    if extra:
        print(
            f"warning: {path} has mode {mode:04o}, which is more permissive "
            f"than {allowed:04o}",
            file=sys.stderr,
        )


def ensure_store_dir(directory: Path | None = None) -> Path:
    """Create the store directory if absent, and make sure it is 0700.

    `chmod` runs unconditionally rather than relying on `mkdir(mode=...)`,
    whose result the process `umask` modifies — a `umask` of 0 would otherwise
    leave a world-readable directory holding a credentials file.
    """
    resolved = store_dir(directory)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        os.chmod(resolved, DIR_MODE)
    except OSError as exc:
        raise StoreError(f"{resolved}: cannot create store directory: {exc}") from exc
    return resolved


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    """Replace `path` with `data`, atomically, at exactly `mode`.

    A crash at any point leaves either the old file or the new one, never a
    truncated one: the content is written to a temporary file in the *same*
    directory (so the final step is a rename within one filesystem, which is
    atomic — across filesystems it would be a copy, and a copy can be
    interrupted), flushed and `fsync`ed so the bytes are on the medium before
    anything points at them, and only then renamed into place.

    The mode is set on the file descriptor before the rename. Doing it after
    would leave a window in which the completed file is visible at its real
    path under whatever `mkstemp` and the `umask` produced. `mkstemp` already
    creates at 0600, so that window would not currently be world-readable — but
    the ordering is what makes that true regardless of the mode being asked for,
    and it is the ordering, not the accident, that the next reader should rely
    on.

    On any failure the temporary file is removed, so a failed write leaves
    neither a damaged target nor debris beside it.
    """
    directory = path.parent
    handle, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt between mkstemp and
        # rename would otherwise leave a temporary file holding sealed
        # credentials in the store directory forever.
        tmp.unlink(missing_ok=True)
        raise

    try:
        # Make the rename itself durable. Best effort: not every platform
        # supports opening a directory for fsync, and by this point the data is
        # already in place — failing here would report an error for a write
        # that succeeded.
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# `config` — backend identity
# ---------------------------------------------------------------------------


def validate_profile_name(name: str, source: str) -> str:
    """Reject a name that is not spellable as a bare TOML key or that could
    collide with a reserved AAD. `source` names where the value came from, so
    the message points at the thing to edit."""
    if not name:
        raise ProfileError(f"{source}: profile name is empty")
    if not _PROFILE_NAME.match(name):
        raise ProfileError(
            f"{source}: invalid profile name {name!r}; names may contain "
            "letters, digits, '_' and '-', and must start with a letter or digit"
        )
    return name


def _require_str(table: Mapping[str, Any], key: str, where: str) -> str | None:
    if key not in table:
        return None
    value = table[key]
    if not isinstance(value, str):
        raise StoreError(f"{where}.{key} must be a string")
    return value


def _build_credentials_table(raw: Any, where: str) -> Mapping[str, Any]:
    """Shape only: a table, with `provider` a string if present at all.

    Everything else in the table (a remote provider's `command`, say) is
    passed through unexamined, for the same reason `config._build_secrets`
    defers `[secrets.*]`'s interior to the module that knows its semantics:
    this one does not know which providers exist, and inventing a whitelist
    here would either reject a provider this module has never heard of or
    have to be kept in step with `providers/__init__.py` by hand.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise StoreError(f"{where}.credentials must be a table")
    provider = raw.get("provider")
    if provider is not None and not isinstance(provider, str):
        raise StoreError(f"{where}.credentials.provider must be a string")
    return raw


def _build_profile(name: str, raw: Any) -> Profile:
    if not isinstance(raw, dict):
        raise StoreError(f"profile.{name} must be a table")
    where = f"profile.{name}"

    unknown = set(raw) - _PROFILE_KEYS
    if unknown:
        raise StoreError(f"unknown key {where}.{sorted(unknown)[0]}")

    profile = Profile(
        name=name,
        backend_url=_require_str(raw, "backend_url", where),
        bucket=_require_str(raw, "bucket", where),
        prefix=_require_str(raw, "prefix", where),
        endpoint=_require_str(raw, "endpoint", where),
        region=_require_str(raw, "region", where),
        credentials=_build_credentials_table(raw.get("credentials"), where),
    )

    # Exactly one form. Both would leave two answers to "which backend?" with
    # no rule for choosing; neither leaves none, and there is no default to fall
    # back on.
    if profile.backend_url is not None and profile.has_components:
        raise StoreError(
            f"{where}: set either backend_url or the component form "
            "(bucket/prefix/endpoint/region), not both"
        )
    if profile.backend_url is None and not profile.has_components:
        raise StoreError(
            f"{where}: needs a backend — either backend_url or the component "
            "form (bucket/prefix/endpoint/region)"
        )
    if profile.backend_url is None and profile.bucket is None:
        raise StoreError(f"{where}: the component form requires bucket")
    return profile


# The `(at line L, column C)` coordinate `tomllib` appends to every message
# it raises, and the only part of that message safe to quote back — see
# `_toml_position`.
_TOML_POSITION = re.compile(r"\(at (?:line \d+, column \d+|end of document)\)")


def _toml_position(exc: tomllib.TOMLDecodeError) -> str:
    """Just the coordinate out of a `TOMLDecodeError`, as ` (at line L,
    column C)`, or `""` when the message does not carry one.

    This exists because `config` can hold a credential and the parser's own
    message can quote the document. Both halves of that were previously
    believed false, and this call site quoted `exc` in full on the strength
    of it: "`config` holds backend identity and never a credential, so
    quoting tomllib's message -- which may include the offending line -- is
    safe here in a way it would not be for `credentials`."

    Neither half survived checking. A `backend_url` of the documented
    `postgres://user:password@host/db` form is a credential in `config` (see
    the module docstring). And `tomllib` does echo document text: most of its
    messages are a fixed description plus a coordinate, but not all of them
    -- `tomllib.loads("[a]\\nx=1\\n[a]\\n")` raises `Cannot declare
    ('a',) twice`, naming the key back. `credentials` is handled by exactly
    this rule already, one function below, and for exactly this reason; the
    two files differ in how *likely* a pasted secret is, not in whether the
    parser can read one back.

    The coordinate is kept because it is what makes the error actionable and
    it is structural, not quoted text. `TOMLDecodeError` exposes no
    `lineno`/`colno` before 3.13 and this project supports 3.11, so it is
    matched out of the message rather than read off the exception; a message
    shape this does not recognise simply yields no coordinate, which loses
    diagnostics and never discloses anything.
    """
    match = _TOML_POSITION.search(str(exc))
    return f" {match.group(0)}" if match else ""


def load_store_config(directory: Path | None = None) -> StoreConfig:
    """Parse `config`, or return an empty one when the file does not exist.

    Absent is a normal state for a user who has not set the tool up yet; the
    profile that cannot then be selected is reported by `select_profile`, which
    can say what to do about it. Present-but-invalid raises — a `config` with a
    typo must never be treated as though it were absent, since that would turn
    a broken profile into a silently missing one.
    """
    path = config_path(directory)
    if not path.exists():
        return StoreConfig()

    warn_if_permissive(path, CONFIG_MODE)
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise StoreError(f"{path}: cannot read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        # Position only, never the parser's message — see `_toml_position`.
        raise StoreError(f"{path}: invalid TOML{_toml_position(exc)}") from exc

    try:
        return _build_store_config(data)
    except StoreError as exc:
        raise StoreError(f"{path}: {exc}") from exc


def _build_store_config(data: Mapping[str, Any]) -> StoreConfig:
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise StoreError(f"unknown key {sorted(unknown)[0]!r}")

    default_profile = _require_str(data, "default_profile", "config")
    if default_profile is not None:
        validate_profile_name(default_profile, "default_profile")

    raw_profiles = data.get("profile", {})
    if not isinstance(raw_profiles, dict):
        raise StoreError("'profile' must be a table of tables")

    profiles: dict[str, Profile] = {}
    for name, raw in raw_profiles.items():
        validate_profile_name(name, "config")
        profiles[name] = _build_profile(name, raw)

    if default_profile is not None and default_profile not in profiles:
        raise StoreError(
            f"default_profile is {default_profile!r}, which has no "
            f"[profile.{default_profile}] table"
        )
    return StoreConfig(default_profile=default_profile, profiles=profiles)


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------


def select_profile(
    explicit: str | None = None,
    *,
    config: StoreConfig,
    repo_profile: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Profile:
    """Resolve which profile to use. First source that *speaks* wins.

    `--profile`, then `STACKWARD_PROFILE`, then the repository's own `profile =`
    (the normal case — a repository's stacks live in one backend, which is a
    fact about the repository and not about whoever's shell is running), then
    `default_profile`, then an error naming every source that could have
    answered.

    "Speaks" means *present*, not *valid*. A source that names a profile which
    does not exist is an error, not a miss: falling through to the next source
    would run the command against a different backend than the one asked for,
    which is the single worst thing this tool could do quietly. An empty value
    is likewise an error rather than a miss — `STACKWARD_PROFILE=""` in a CI
    environment is a broken variable, not an absent one, and treating it as
    absent would silently substitute the default.
    """
    environ = os.environ if environ is None else environ

    sources: list[tuple[str, str | None]] = [
        ("--profile", explicit),
        (ENV_PROFILE, environ.get(ENV_PROFILE)),
        ("the repository's .stackward.toml 'profile'", repo_profile),
        ("default_profile in the store config", config.default_profile),
    ]

    for source, value in sources:
        if value is None:
            continue
        name = validate_profile_name(value, source)
        profile = config.profiles.get(name)
        if profile is None:
            raise ProfileError(
                f"{source} selects profile {name!r}, which has no "
                f"[profile.{name}] table in {CONFIG_FILENAME}"
            )
        return profile

    raise ProfileError(
        "no profile selected and there is no default: pass --profile, set "
        f"{ENV_PROFILE}, set 'profile' in the repository's .stackward.toml, or "
        f"set default_profile in {CONFIG_FILENAME}"
    )


# ---------------------------------------------------------------------------
# `credentials` — the sealed envelopes
# ---------------------------------------------------------------------------


def _read_document(directory: Path | None = None) -> dict[str, Any]:
    path = credentials_path(directory)
    warn_if_permissive(path, CREDENTIALS_MODE)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise StoreError(
            f"{path}: no credential store here yet — initialise one first"
        ) from exc
    except OSError as exc:
        raise StoreError(f"{path}: cannot read: {exc}") from exc

    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Only the failure type, never the parser's message: the message quotes
        # the offending text, and this is the one file that holds ciphertext
        # the user may have pasted a plaintext into by mistake.
        raise StoreError(
            f"{path}: not a valid credential store ({type(exc).__name__})"
        ) from exc

    if not isinstance(document, dict):
        raise StoreError(f"{path}: not a valid credential store")
    if document.get("v") != STORE_VERSION:
        # `_describe`, not the raw value: this field is hand-editable and lives
        # in the one file expected to hold secrets, so a credential pasted into
        # it must not be read back out by the error that caught the mistake.
        raise StoreError(
            f"{path}: unsupported store version {describe_value(document.get('v'))} "
            f"(this build reads {STORE_VERSION})"
        )
    if not isinstance(document.get("profiles"), dict):
        raise StoreError(f"{path}: not a valid credential store")
    return document


def _write_document(document: Mapping[str, Any], directory: Path | None = None) -> None:
    ensure_store_dir(directory)
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    atomic_write(credentials_path(directory), payload, CREDENTIALS_MODE)


def _require_password(password: str, what: str = "a password") -> None:
    """Refuse an absent password, as a `PasswordError` naming which one.

    `crypto._normalise` is the real guarantee — it sits under every seal and
    every open, so no path can miss it. This exists on top of it only to phrase
    the refusal in the store's own vocabulary and to say *which* password was
    missing when a call takes two, which the crypto layer cannot know.
    """
    if not password.strip():
        raise PasswordError(f"{what} is required")


def _check_password(document: Mapping[str, Any], password: str) -> None:
    """Open the verifier, and fail with `PasswordError` if it does not.

    Everything that touches an envelope goes through here first, so a wrong
    password is always reported as a wrong password — before any profile
    envelope is attempted, and therefore before a decryption failure could be
    mistaken for a damaged store.
    """
    _require_password(password)
    envelope = document.get(VERIFIER_KEY)
    if envelope is None:
        raise StoreError(
            "credential store has no verifier: it was not written by this tool, "
            "or it is damaged"
        )
    try:
        plaintext = unseal_json(envelope, password, VERIFIER_AAD)
    except DecryptionError as exc:
        raise PasswordError("wrong password for the credential store") from exc
    except EmptyPasswordError as exc:
        # Unreachable while `_require_password` above stands, and mapped anyway:
        # `EmptyPasswordError` is a `CryptoError`, so the clause below would
        # otherwise recode "no password" as "the verifier is unusable".
        raise PasswordError("a password is required") from exc
    except CryptoError as exc:
        raise StoreError(f"credential store verifier is unusable: {exc}") from exc

    if plaintext != VERIFIER_PLAINTEXT:
        # The GCM tag already proves the plaintext is the one sealed under this
        # password and AAD, so this compares a value nothing else could have
        # produced. It is kept because it is what makes the verifier a
        # *known-plaintext* check rather than an implicit property of the tag,
        # and because a future format change that broke the invariant would
        # otherwise pass silently.
        raise StoreError("credential store verifier did not match")


def _require_credentials(payload: Any, profile: str) -> dict[str, str]:
    """A decrypted payload must be a flat mapping of names to strings.

    This is the one place in this module where decrypted credential values are
    in scope at the moment an exception is raised, so the message below
    interpolates only the profile name and a fixed string.

    Do not "improve" it by naming the offending entry. A key here would
    ordinarily be an environment variable name and harmless — but the payload
    is whatever was sealed, and a mapping built the wrong way round
    (`{value: name}`) puts a credential in the key position. The safety of this
    branch rests on interpolating neither half of an entry, not on an
    assumption about which half is sensitive.
    """
    if not isinstance(payload, dict):
        raise StoreError(f"credentials for profile {profile!r} are malformed")
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise StoreError(
                f"credentials for profile {profile!r}: every entry must be a "
                "string name with a string value"
            )
        result[key] = value
    return result


def init_store(password: str, *, directory: Path | None = None) -> None:
    """Create the store directory and an empty `credentials` holding a verifier.

    Refuses to overwrite an existing store: doing so would discard every sealed
    envelope in it, irrecoverably, in response to a command a user could plausibly
    run twice.
    """
    _require_password(password)
    ensure_store_dir(directory)
    path = credentials_path(directory)
    if path.exists():
        raise StoreError(f"{path}: a credential store already exists here")
    _write_document(
        {
            "v": STORE_VERSION,
            VERIFIER_KEY: seal_json(VERIFIER_PLAINTEXT, password, VERIFIER_AAD),
            "profiles": {},
        },
        directory,
    )


def store_profiles(directory: Path | None = None) -> list[str]:
    """Names that have a sealed envelope, sorted. Reads no password and opens
    nothing, so it is safe to call for a listing."""
    return sorted(_read_document(directory)["profiles"])


def set_credentials(
    profile: str,
    credentials: Mapping[str, str],
    password: str,
    *,
    directory: Path | None = None,
) -> None:
    """Seal `credentials` as `profile`'s envelope, replacing any existing one.

    The password is checked against the verifier first, so a mistyped password
    cannot add an envelope that nothing else in the store can open.
    """
    validate_profile_name(profile, "profile")
    for key, value in credentials.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise StoreError("credentials must be a mapping of strings to strings")

    document = _read_document(directory)
    _check_password(document, password)
    document["profiles"][profile] = seal_json(dict(credentials), password, profile)
    _write_document(document, directory)


def resolve_credentials(
    profile: str, password: str, *, directory: Path | None = None
) -> dict[str, str]:
    """The bootstrap values for `profile`: names to values, decrypted.

    This is the function the `file` credential-store provider will register in
    Task 11 — one name, a profile, a mapping back — so that adding an `env` or
    a remote provider adapts this rather than rewriting it. It deliberately does
    not know *which* names a caller wants: the environment variables a Pulumi
    backend needs are the calling command's business, and naming them here would
    put a vendor's vocabulary into the store.

    The password is not read from anywhere. Prompting, `getpass`, and any
    environment fallback belong to the command; a library that could prompt
    would be a library that can block a hook.
    """
    document = _read_document(directory)
    _check_password(document, password)

    envelope = document["profiles"].get(profile)
    if envelope is None:
        raise ProfileError(f"profile {profile!r} has no credentials in the store")

    try:
        payload = unseal_json(envelope, password, profile)
    except DecryptionError as exc:
        # The verifier already opened under this password, so the password is
        # right and this envelope is not the one that belongs here — the AAD
        # mismatch an envelope moved between profiles produces, or a modified
        # ciphertext. Either way it is damage, not a typo.
        raise StoreError(
            f"profile {profile!r}: sealed data does not belong to this profile, "
            "or has been modified"
        ) from exc
    except CryptoError as exc:
        raise StoreError(f"profile {profile!r}: {exc}") from exc

    return _require_credentials(payload, profile)


def rotate_password(
    old_password: str, new_password: str, *, directory: Path | None = None
) -> None:
    """Re-seal every envelope under `new_password`, or change nothing.

    The whole document is rebuilt in memory and written once, so a failure at
    any point — a damaged envelope, an interrupted process, a full disk —
    leaves the store exactly as it was, still opening under the old password.
    A partial rotation would be the worst outcome available: a store whose
    profiles need different passwords, with nothing recording which is which.

    Every envelope *in the credentials file* is re-sealed, not every profile in
    `config`. An envelope whose `[profile.<name>]` table has been removed is
    unreachable but not gone, and rotating past it would quietly destroy it.
    """
    _require_password(new_password, "the new password")
    document = _read_document(directory)
    _check_password(document, old_password)

    rotated: dict[str, Any] = {}
    for name, envelope in document["profiles"].items():
        try:
            payload = unseal_json(envelope, old_password, name)
        except CryptoError as exc:
            raise StoreError(
                f"profile {name!r}: cannot re-seal, its envelope did not open "
                f"({type(exc).__name__}); nothing has been changed"
            ) from exc
        rotated[name] = seal_json(payload, new_password, name)

    # Replace only the two fields this operation owns and keep the rest of the
    # document as it was read. Rebuilding from a literal would silently drop any
    # other top-level field, and would leave the two write paths disagreeing
    # about what a document is — `set_credentials` preserves the whole thing,
    # and rotation would be the one discarding.
    document[VERIFIER_KEY] = seal_json(VERIFIER_PLAINTEXT, new_password, VERIFIER_AAD)
    document["profiles"] = rotated
    _write_document(document, directory)
