"""`set-secrets`: publish a stack's declared secrets from local sources into
Pulumi's config.

**The manifest maps a config path to a *logical name*; where the value comes
from is a separate concern.** `.stackward.toml` declares, per project
directory, a `[secrets."<dir>".secret]` table of `<config path> = <logical
name>` pairs — this module resolves each name to a value (highest precedence
first: the process environment, then `[secrets.source].files` in file order,
later files overriding earlier ones) and publishes it. Nothing here decides
*how* a name becomes a value beyond that local, file-and-environment scheme:
`LocalSecretSource.resolve` is written to the one-method shape a future
`SecretSource` interface (Task 11's `dotenv`/`env` providers) will register,
so that interface adapts this module rather than rewriting it.

**Both `[secrets."<dir>"].secret` and `.plaintext` are published; `unmanaged`
is not.** The three sibling tables share the same `path = name` shape, but
they are not interchangeable: the split between `secret` and `plaintext` *is*
the declaration of whether a value is a credential, and a declared table that
this command silently never acts on would make that declaration meaningless
— a manifest author adding a `plaintext` entry with nothing telling them it
does nothing is exactly the failure mode this project treats as worse than
either "reject it" or "act on it." `secret` entries are published with
`pulumi config set --secret --path <path>`; `plaintext` entries with
`pulumi config set --plaintext --path <path>` — same resolution, same
skip-if-unset semantics, same delivery on stdin, in `_pulumi_config_set`'s one
code path (see below). `unmanaged` stays inert: it exists to record a path
deliberately *not* managed by this tool, together with a reason — a
"documentation-only" table by design, not an oversight.

**A value never reaches `argv`.** `pulumi config set --secret|--plaintext
--path <path>` is invoked with the resolved value on **stdin**, never as a
positional argument — a command line is visible in `ps` and in shell history
for the life of the call, the same concern Task 7's `login` already had to
solve for a backend URL. Stdin is used for `plaintext` too, even though its
values are not confidential: keeping one delivery mechanism for both is
simpler than forking a second one for the entries that do not strictly need
it. `pulumi config get --path <path>` (drift detection's read) has no such
argument at all, so it carries no value on either channel.

**A name whose value is unset is skipped, never cleared.** This module has no
code path that ever removes a Pulumi config value; it only ever calls
`config set` for a name it was actually given a non-empty value for. A value
that resolves to the empty string is treated identically to one that did not
resolve at all — an environment variable exported as `""` is not a value
worth publishing, and publishing it would silently clear whatever secret was
there before, which is exactly the destructive surprise the brief calls out.

**`[required]` promotes "skipped" to "failed".** A path named there — under
`[secrets."<dir>".required]`'s `paths` list, the **one** canonical shape this
module accepts; see `_parse_required` for why a second, alternate spelling is
refused rather than tolerated — must resolve to a non-empty value (as either
a `secret` or a `plaintext` entry) or the run reports a non-zero exit. It
does not, however, abort the loop: a timeout, a missing `pulumi` binary, and
an unresolved required path are all the same class of per-entry failure (see
`_publish_entries`), and every one of them lets every other declared entry
still get its own chance to be set. A rotation that stops partway and reports
success would be worse than one that finishes and reports which entries did
not land — the summary printed by `cmd_set_secrets` exists to say exactly
that, by name, every time.

**Drift detection never prints a value, and silence is a real, load-bearing
state — not merely "nothing to report".** `[[secrets."<dir>".drift_pairs]]`
declares a `bootstrap` and a `managed` logical name; the check compares the
*local* value of `bootstrap` (resolved the same way any other entry is)
against the *published* value at whichever config path `managed` names in
`secret` (read with `pulumi config get`, which decrypts — the list form,
`pulumi config`, renders every secret as `[secret]` without decrypting and
exits 0 under any passphrase, so it cannot answer this question at all). Three
states are silent on purpose, not merely "no branch happened to fire": the
local `bootstrap` value not resolving (nothing to compare against),
`managed`'s published value not being readable (a timeout, a non-zero exit, a
missing `pulumi`), and the two values actually agreeing. Only an *unreadable*
name — `managed` naming zero or more than one declared secret, which is a
manifest problem rather than a state-of-the-world one — warns, since that one
is worth a human's attention regardless of what the values turn out to be.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import paths
from ..config import Config, ConfigError, find_repo_config, load_config
from ..store import warn_if_permissive
from .check_config import fail_closed

# How long any single `pulumi` invocation (a `config set` or a `config get`)
# is allowed to run before this module gives up on it and reports that one
# entry as failed. A module-level variable, not a function default, so a test
# can lower it (`monkeypatch.setattr`) to exercise the timeout path without
# actually waiting anywhere near this long.
DEFAULT_TIMEOUT_SECONDS = 30.0

# The two quote characters `_parse_env_file` strips one matched pair of, and
# nothing else — this is a `KEY=VALUE` file, not a shell, and no other
# escaping is attempted.
_QUOTE_CHARS = ('"', "'")

_EXPORT_PREFIX = "export "


class SetSecretsError(Exception):
    """Something about the manifest, the source files, or an invocation of
    `pulumi` stops `set-secrets` from proceeding for reasons distinct from
    "the value was not there" (that is a skip, not an error — see the module
    docstring).

    Never carries a credential value: every message here names a config path,
    a logical name, a file, or a `pulumi` exit code, never anything read out
    of an envelope or an environment variable.
    """


@dataclass(frozen=True)
class DriftPair:
    """One declared `[[secrets."<dir>".drift_pairs]]` entry: two *logical
    names*, never config paths themselves. `managed`'s config path is found
    by inverting `ProjectManifest.secret` at check time (see `_check_drift`),
    not stored here — the manifest is the one place that mapping lives, and
    keeping a second copy of it in a `DriftPair` would let the two disagree
    silently if `secret` ever changed shape.
    """

    bootstrap: str
    managed: str


@dataclass(frozen=True)
class ProjectManifest:
    """One `[secrets."<dir>"]` table, parsed to only what this module acts
    on: `secret` and `plaintext` (each config path -> logical name; both are
    published, see the module docstring), `required` (a subset of `secret`'s
    and `plaintext`'s paths that must resolve), and `drift_pairs`.
    `unmanaged` is read by `config.py`'s shape check and never reaches this
    dataclass at all — it is documentation-only, by design.
    """

    secret: dict[str, str] = field(default_factory=dict)
    plaintext: dict[str, str] = field(default_factory=dict)
    required: frozenset[str] = frozenset()
    drift_pairs: tuple[DriftPair, ...] = ()


@dataclass(frozen=True)
class EntryOutcome:
    """What happened to one `secret` entry, for both the per-line report and
    the closing summary. `reason` is `None` exactly when `status` is `"set"`
    or `"would_set"` — every other status names why, in words that never
    include a value."""

    path: str
    name: str
    status: str  # "set" | "would_set" | "skipped" | "failed"
    reason: str | None = None


class LocalSecretSource:
    """Resolves a logical name to a value, or `None` — the environment first,
    then each parsed source file, later files overriding earlier ones.

    This is deliberately the one-method shape ("a name in, a value or `None`
    out") that Task 11's `SecretSource` interface will register `dotenv` and
    `env` implementations against. Building resolution this way now means
    that interface *adapts* this class — wraps it, or replaces its
    construction — rather than needing to rewrite the call sites in
    `_publish_entries`/`_check_drift` that already only ever call
    `.resolve(name)`.

    Presence, not truthiness, decides precedence: an environment variable
    exported as `""` still wins over a file that also defines the same name,
    because that is what "the process environment first" means. It is
    `_publish_entries` and `_check_drift` — not this class — that decide an
    empty resolved value is unusable; a resolver returning `None` and one
    returning `""` are different facts, and only the caller (which knows
    whether the empty case is a skip or a "nothing to compare" for drift)
    should be the one to treat them the same.
    """

    def __init__(
        self, environ: Mapping[str, str], file_values: list[dict[str, str]]
    ) -> None:
        self._environ = environ
        # In the order declared in `[secrets.source].files`; `resolve` walks
        # this in reverse so a later file overrides an earlier one.
        self._file_values = file_values

    def resolve(self, name: str) -> str | None:
        if name in self._environ:
            return self._environ[name]
        for values in reversed(self._file_values):
            if name in values:
                return values[name]
        return None


# ---------------------------------------------------------------------------
# Manifest parsing — `config.py` validates only that `secrets.<x>` is a
# table; everything below this line is this module's own semantics.
# ---------------------------------------------------------------------------


def _parse_path_name_table(raw: Any, where: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SetSecretsError(f"{where} must be a table")
    result: dict[str, str] = {}
    for path, name in raw.items():
        if not isinstance(name, str):
            raise SetSecretsError(f"{where}.{path!r} must map to a string logical name")
        result[path] = name
    return result


# The only key `[secrets."<dir>".required]` recognises. A second, alternate
# spelling of "these paths are required" (a bare `required = [...]` array, or
# a table whose own keys are the required paths) is deliberately not accepted
# alongside this one: a manifest format with two shapes for the same thing
# means a typo in one can silently read as the *other* shape's "nothing
# required" instead of as an error — an ambiguity a fail-closed check must
# not have. See `_parse_required`.
_REQUIRED_KEYS = frozenset({"paths"})


def _parse_required(raw: Any, where: str) -> frozenset[str]:
    """The **one** canonical `[secrets."<dir>".required]` shape: a table with
    exactly one key, `paths`, holding a list of config-path strings —

        [secrets."<dir>".required]
        paths = ["a.b", "c.d"]

    Absent entirely is a normal state (nothing required, an empty
    `frozenset`). Present but any other shape — a bare `required = [...]`
    array, a table whose own keys are the paths, an unrecognised key inside
    `[required]` — is a hard `SetSecretsError` naming `where`, never silently
    reinterpreted as one of the shapes this module used to also accept: this
    is `config.py`'s own "invalid is never treated as absent" rule, and its
    own "an unknown key is named and rejected" rule, both carried over here
    for the same reason they hold there.
    """
    if raw is None:
        return frozenset()
    if not isinstance(raw, dict):
        raise SetSecretsError(
            f"{where} must be a table with a 'paths' list, e.g. "
            f'[{where}]\npaths = ["a.b"]'
        )
    unknown = set(raw) - _REQUIRED_KEYS
    if unknown:
        raise SetSecretsError(f"unknown key {where}.{sorted(unknown)[0]}")
    raw_paths = raw.get("paths", [])
    if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
        raise SetSecretsError(f"{where}.paths must be a list of path strings")
    return frozenset(raw_paths)


def _parse_drift_pairs(raw: Any, where: str) -> tuple[DriftPair, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SetSecretsError(f"{where} must be a list of tables")
    pairs: list[DriftPair] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise SetSecretsError(f"{where}[{index}] must be a table")
        bootstrap = entry.get("bootstrap")
        managed = entry.get("managed")
        if not isinstance(bootstrap, str) or not isinstance(managed, str):
            raise SetSecretsError(
                f"{where}[{index}] needs string 'bootstrap' and 'managed' entries"
            )
        pairs.append(DriftPair(bootstrap=bootstrap, managed=managed))
    return tuple(pairs)


def parse_project_manifest(raw: dict[str, Any], project: str) -> ProjectManifest:
    """Build a `ProjectManifest` from `config.secrets[project]`.

    Every config path this module will actually publish or forward to
    `pulumi` — every key in `secret` and in `plaintext`, since both are
    published — is validated with `paths.parse` here, at load time; this is
    the one path grammar the whole tool shares (see `paths.py`), and using it
    as the validator is what "do not invent string handling" means in
    practice. `unmanaged` keys are never parsed this way: this module never
    forwards them to `pulumi`, and they may legitimately hold a wildcard
    shape (`"<path>.*"`) that is not a `--path` at all — see the module
    docstring.
    """
    secret = _parse_path_name_table(raw.get("secret"), f"secrets.{project!r}.secret")
    plaintext = _parse_path_name_table(raw.get("plaintext"), f"secrets.{project!r}.plaintext")
    for table_name, table in (("secret", secret), ("plaintext", plaintext)):
        for path in table:
            try:
                paths.parse(path)
            except ValueError as exc:
                raise SetSecretsError(
                    f"secrets.{project!r}.{table_name}: invalid config path {path!r}: {exc}"
                ) from exc

    required = _parse_required(raw.get("required"), f"secrets.{project!r}.required")
    unknown = required - set(secret) - set(plaintext)
    if unknown:
        raise SetSecretsError(
            f"secrets.{project!r}.required.paths names {sorted(unknown)[0]!r}, which "
            "is not declared under 'secret' or 'plaintext'"
        )

    drift_pairs = _parse_drift_pairs(
        raw.get("drift_pairs"), f"secrets.{project!r}.drift_pairs"
    )

    return ProjectManifest(
        secret=secret, plaintext=plaintext, required=required, drift_pairs=drift_pairs
    )


def select_project(config: Config, repo_root: Path, cwd: Path) -> str:
    """Which `[secrets."<dir>"]` table this invocation publishes.

    A single declared project (besides the reserved `source` table) is used
    unconditionally — "a single-project repository uses one table" per the
    brief, with no need to make that table's name match anything about
    `cwd`. With more than one, the project is the repository-root-relative
    directory of `cwd` itself; `find_repo_config` always searches upward
    *from* `cwd`, so `cwd` is guaranteed to be at or below `repo_root` and
    `os.path.relpath` never has to represent an escape above it.
    """
    candidates = sorted(name for name in config.secrets if name != "source")
    if not candidates:
        raise SetSecretsError("no [secrets.\"<dir>\"] table declared in .stackward.toml")
    if len(candidates) == 1:
        return candidates[0]

    relative = os.path.relpath(cwd.resolve(), repo_root.resolve())
    project = "." if relative == "." else relative.replace(os.sep, "/")
    if project in candidates:
        return project
    raise SetSecretsError(
        f"cannot tell which project's secrets to publish from {cwd}; declared "
        f"projects: {', '.join(candidates)}"
    )


# ---------------------------------------------------------------------------
# Value sources: the process environment, and `[secrets.source].files`
# ---------------------------------------------------------------------------


def substitute_stack(filename: str, stack: str | None) -> str:
    """`{stack}` in `filename`, replaced with `stack` — or a `SetSecretsError`
    naming the file when the manifest needs a stack this invocation was never
    given one for. A plain `.replace`, not `str.format`: a source filename is
    not a template a manifest author is expected to write other `{...}`
    placeholders into, and `.format` would raise its own confusing error on
    one that happened to contain an unrelated brace.
    """
    if "{stack}" not in filename:
        return filename
    if stack is None:
        raise SetSecretsError(
            f"secrets.source.files entry {filename!r} needs {{stack}} substituted, "
            "but no --stack was given"
        )
    return filename.replace("{stack}", stack)


def resolve_source_files(config: Config, repo_root: Path, stack: str | None) -> list[Path]:
    """`[secrets.source].files`, each `{stack}`-substituted and made absolute
    against `repo_root` when not already absolute. A missing `[secrets.source]`
    table, or one with no `files`, resolves to an empty list — value
    resolution then falls back to the process environment alone, which is a
    normal, working configuration, not an error."""
    source = config.secrets.get("source", {})
    if not isinstance(source, dict):
        raise SetSecretsError("secrets.source must be a table")

    raw_files = source.get("files", [])
    if not isinstance(raw_files, list) or not all(isinstance(f, str) for f in raw_files):
        raise SetSecretsError("secrets.source.files must be a list of strings")

    resolved: list[Path] = []
    for raw in raw_files:
        name = substitute_stack(raw, stack)
        candidate = Path(name)
        resolved.append(candidate if candidate.is_absolute() else repo_root / candidate)
    return resolved


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse one `KEY=VALUE` file: blank lines and `#` comments are ignored, a
    leading `export ` is stripped from the key side, and exactly one matched
    pair of surrounding quotes is stripped from the value.

    Warns on stderr (via `store.warn_if_permissive`, the same message shape
    used for the credential store's own files) when `path` is more permissive
    than `0o600` — this file can hold a bootstrap secret in plaintext on disk,
    so a mode that lets another local user read it is worth flagging, even
    though refusing to read it outright would only lock a user out of their
    own secrets after the exposure already happened.

    A non-blank, non-comment line with no `=` is a `SetSecretsError` naming
    the file and line number, never the line's own text — this is fail-closed
    on a genuinely malformed bootstrap file rather than silently skipping a
    line that was probably meant to define something.
    """
    warn_if_permissive(path, 0o600)
    try:
        text = path.read_text()
    except OSError as exc:
        raise SetSecretsError(f"{path}: cannot read: {exc}") from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(_EXPORT_PREFIX):
            line = line[len(_EXPORT_PREFIX) :]
        if "=" not in line:
            raise SetSecretsError(f"{path}:{line_number}: not a KEY=VALUE line")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTE_CHARS:
            value = value[1:-1]
        if not key:
            raise SetSecretsError(f"{path}:{line_number}: empty key")
        values[key] = value
    return values


# ---------------------------------------------------------------------------
# Talking to `pulumi`
# ---------------------------------------------------------------------------


# The two publishable manifest tables, and the `pulumi config set` flag each
# one is published with. `_publish_entries` never calls `_pulumi_config_set`
# with a `mode` outside this mapping's keys.
_MODE_FLAGS = {"secret": "--secret", "plaintext": "--plaintext"}


def _pulumi_config_set(
    pulumi: str,
    path: str,
    value: str,
    *,
    mode: str,
    stack: str | None,
    timeout: float,
) -> None:
    """`pulumi config set --secret|--plaintext --path <path>`, `value` on
    **stdin** — never as a positional argument, which would put it in `argv`
    and therefore in `ps` output and shell history for the life of the call.
    `mode` (`"secret"` or `"plaintext"`, via `_MODE_FLAGS`) picks the flag;
    the two are mutually exclusive on one invocation, and a `secret` entry is
    never sent with `--plaintext` or vice versa — each entry carries its own
    table's mode all the way from `manifest.secret`/`manifest.plaintext`
    through to this call.

    Raises `subprocess.TimeoutExpired` and `OSError` (a missing `pulumi`
    binary, among other things) unchanged, for `_publish_entries` to
    translate into a per-entry failure without this function needing to know
    that vocabulary. A non-zero exit is instead raised here as
    `SetSecretsError` naming only the exit code — `pulumi`'s own stdout and
    stderr are captured and never included in any message or re-printed:
    they are not this module's to vouch for as free of the very value it
    just piped to this process on stdin.
    """
    args = [pulumi, "config", "set", _MODE_FLAGS[mode], "--path", path]
    if stack:
        args += ["--stack", stack]
    completed = subprocess.run(args, input=value.encode(), timeout=timeout, capture_output=True)
    if completed.returncode != 0:
        raise SetSecretsError(f"pulumi exited {completed.returncode}")


def _pulumi_config_get(
    pulumi: str, path: str, *, stack: str | None, timeout: float
) -> str | None:
    """The published, decrypted value at `path`, or `None` when it could not
    be read — a non-zero exit, a timeout, or a missing `pulumi` binary all
    collapse to the same `None` here, on purpose: drift detection treats
    every one of those as "nothing to compare", silently, per the module
    docstring. Never raises, and never prints its own stdout — the decrypted
    value is returned to the caller and nowhere else.

    `pulumi config` (the list form) is not usable for this: it renders every
    secret as `[secret]` **without** decrypting, and exits 0 under any
    passphrase at all — a caller could not tell "the values agree" from "this
    command told me nothing." Only `pulumi config get --path <path>`
    decrypts, which is why this function -- and not the list form -- is what
    drift detection is built on.
    """
    args = [pulumi, "config", "get", "--path", path]
    if stack:
        args += ["--stack", stack]
    try:
        completed = subprocess.run(args, timeout=timeout, capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    # `pulumi config get` terminates its output with a trailing newline;
    # stripping it is what makes the comparison in `_check_drift` compare the
    # actual value rather than failing on every equal pair.
    return completed.stdout.rstrip("\n")


# ---------------------------------------------------------------------------
# Publishing and drift detection
# ---------------------------------------------------------------------------


def _publish_entries(
    manifest: ProjectManifest,
    source: LocalSecretSource,
    *,
    pulumi: str | None,
    stack: str | None,
    dry_run: bool,
    timeout: float,
) -> list[EntryOutcome]:
    """Walk every `secret` entry (in declared order), then every `plaintext`
    entry (in declared order), resolving and then (unless `dry_run`)
    publishing each one with its own table's mode — one `EntryOutcome` per
    entry, regardless of whether it succeeded.

    Every failure mode here — an unresolved required path, a `pulumi` timeout,
    a missing `pulumi` binary, a non-zero `pulumi` exit — is recorded on its
    own `EntryOutcome` and the loop continues to the next entry rather than
    raising: a half-finished rotation that stops on the first failure and
    exits non-zero has already left every entry after it unset, and reports
    that failure no differently than one that finishes and simply fails to
    report which entries did not land. The order entries are declared within
    each table (a `dict`, so insertion order — the order the TOML was written
    in) is preserved throughout, since it is what makes "the failing entry
    was in the middle" a meaningful, reproducible scenario to test.
    """
    entries = [(path, name, "secret") for path, name in manifest.secret.items()]
    entries += [(path, name, "plaintext") for path, name in manifest.plaintext.items()]

    outcomes: list[EntryOutcome] = []
    for path, name, mode in entries:
        value = source.resolve(name)
        if not value:
            if path in manifest.required:
                outcomes.append(EntryOutcome(path, name, "failed", "required but unresolved"))
            else:
                outcomes.append(EntryOutcome(path, name, "skipped", "unresolved"))
            continue

        if dry_run:
            outcomes.append(EntryOutcome(path, name, "would_set", None))
            continue

        if pulumi is None:
            outcomes.append(
                EntryOutcome(path, name, "failed", "pulumi executable not found on PATH")
            )
            continue

        try:
            _pulumi_config_set(pulumi, path, value, mode=mode, stack=stack, timeout=timeout)
        except subprocess.TimeoutExpired:
            outcomes.append(EntryOutcome(path, name, "failed", "pulumi timed out"))
        except OSError as exc:
            outcomes.append(EntryOutcome(path, name, "failed", f"cannot run pulumi: {exc}"))
        except SetSecretsError as exc:
            outcomes.append(EntryOutcome(path, name, "failed", str(exc)))
        else:
            outcomes.append(EntryOutcome(path, name, "set", None))
    return outcomes


def _config_path_for_name(secret: dict[str, str], name: str) -> str | None:
    """The one config path in `secret` mapping to `name`, or `None` when zero
    or more than one do — `managed` in a drift pair must name exactly one
    declared secret for `pulumi config get` to have anything to read."""
    matches = [path for path, mapped in secret.items() if mapped == name]
    return matches[0] if len(matches) == 1 else None


def _check_drift(
    manifest: ProjectManifest,
    source: LocalSecretSource,
    *,
    pulumi: str | None,
    stack: str | None,
    timeout: float,
) -> None:
    """Warn, on stderr, for each declared drift pair whose local `bootstrap`
    value and published `managed` value disagree — see the module docstring
    for exactly which states are silent on purpose. Never called under
    `--dry-run`: drift detection reads `pulumi config get`, which is a real
    invocation, and dry-run's contract is to run nothing at all."""
    if pulumi is None:
        return
    for pair in manifest.drift_pairs:
        bootstrap_value = source.resolve(pair.bootstrap)
        if not bootstrap_value:
            continue

        managed_path = _config_path_for_name(manifest.secret, pair.managed)
        if managed_path is None:
            print(
                f"warning: drift pair {pair.bootstrap!r}/{pair.managed!r}: "
                f"{pair.managed!r} does not name exactly one declared secret",
                file=sys.stderr,
            )
            continue

        published = _pulumi_config_get(pulumi, managed_path, stack=stack, timeout=timeout)
        if published is None:
            continue

        if published != bootstrap_value:
            print(
                f"warning: drift: {pair.bootstrap!r} (bootstrap) and "
                f"{pair.managed!r} (managed, at {managed_path!r}) differ",
                file=sys.stderr,
            )


def _print_outcome(outcome: EntryOutcome) -> None:
    if outcome.status == "set":
        print(f"set {outcome.path!r} ({outcome.name})")
    elif outcome.status == "would_set":
        print(f"would set {outcome.path!r} ({outcome.name})")
    elif outcome.status == "skipped":
        print(f"skip {outcome.path!r} ({outcome.name}): {outcome.reason}")
    else:
        print(f"error: {outcome.path!r} ({outcome.name}): {outcome.reason}", file=sys.stderr)


@fail_closed
def cmd_set_secrets(args: argparse.Namespace) -> int:
    """Entry point for `stackward set-secrets`.

    Exit 0 when every declared entry was set or validly skipped; exit 2 when
    anything was not set that should have been (a required-but-unresolved
    path, a `pulumi` timeout, a missing `pulumi` binary, a non-zero `pulumi`
    exit) or the manifest/config could not be loaded at all. Never exit 1 —
    this tool reserves that code for `check-config`'s "a credential was
    found," a different claim this command never makes.
    """
    try:
        config_path = find_repo_config()
        if config_path is None:
            raise SetSecretsError("no .stackward.toml found above the current directory")
        config = load_config(config_path)
        repo_root = config_path.parent
        project = select_project(config, repo_root, Path.cwd())
        manifest = parse_project_manifest(config.secrets[project], project)
        files = resolve_source_files(config, repo_root, args.stack)
        file_values = [parse_env_file(f) for f in files if f.exists()]
        source = LocalSecretSource(os.environ, file_values)
    except (ConfigError, SetSecretsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pulumi = None if args.dry_run else shutil.which("pulumi")

    outcomes = _publish_entries(
        manifest,
        source,
        pulumi=pulumi,
        stack=args.stack,
        dry_run=args.dry_run,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    for outcome in outcomes:
        _print_outcome(outcome)

    if not args.dry_run:
        _check_drift(
            manifest, source, pulumi=pulumi, stack=args.stack, timeout=DEFAULT_TIMEOUT_SECONDS
        )

    failed = [outcome for outcome in outcomes if outcome.status == "failed"]
    set_count = sum(1 for outcome in outcomes if outcome.status in ("set", "would_set"))
    skipped_count = sum(1 for outcome in outcomes if outcome.status == "skipped")
    print(f"summary: {set_count} set, {skipped_count} skipped, {len(failed)} failed")
    if failed:
        named = ", ".join(f"{outcome.path!r} ({outcome.name})" for outcome in failed)
        print(f"not set: {named}", file=sys.stderr)
        return 2
    return 0
