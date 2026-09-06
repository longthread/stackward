"""Repository policy: `.stackward.toml`.

Every later command needs to know, about *this* repository specifically:
which keys and parent categories count as sensitive, which pydantic classes
declare secrets for which stack, which secret manifest and value sources
apply, and the oldest `stackward` release this repository's policy is known
to work with. None of that is knowledge this tool ships with — it all
arrives through this one file, found by walking up from the working
directory to the repository root. That is the seam that keeps `stackward`
generic: swap the file, not the tool.

Two states a caller must tell apart: **absent** (`find_repo_config` returns
`None`; there is no policy to apply, and it is on the caller to decide what
that means for the command at hand) and **present but invalid**
(`load_config` raises `ConfigError`). There is no third state where an
invalid file quietly falls back to defaults — a typo that silently disabled
part of a credential gate would be worse than one that refused to run at
all.

What the gate commands decide for the absent case is: refuse. `check-config`
and `pre-commit` both exit 2 naming the minimal file to create, because
`CheckConfig()`'s own defaults include `model_net = "none"`, and defaulting
to that would make the weaker of the two nets a *silent fallback* rather
than the declared mode it is meant to be. This module still reports absence
rather than raising, because `cli.cmd_doctor` genuinely needs to run in a
repository that has no policy yet — reporting that fact is most of what it
is for.

**Reading and parsing are separate.** `load_config_text` validates text;
`load_config` is the thin file reader in front of it. `commands.pre_commit`
needs the second half without the first, because its policy comes out of the
git index rather than the working tree.

**`read_min_version` is deliberately narrower than either.** It answers only
"what floor does this file declare", for a caller that must get an answer
even when the rest of the file is something this binary cannot validate --
which is exactly the case a floor is declared for. See it for why that is not
a second, laxer config loader.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__

CONFIG_FILENAME = ".stackward.toml"

# Case-insensitive substrings that make a leaf's key suspect on their own,
# with no help from an ancestor. `[check].sensitive_keys` in the repo config
# EXTENDS this set — see `_extend`/`_replace` below for why keys extend but
# parents replace.
BUILTIN_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "passphrase",
        "token",
        "secret",
        "apikey",
        "api_key",
        "jwt",
        "credential",
        "private_key",
        "access_key_id",
        "secret_access_key",
    }
)

# No parent key is sensitive by default. Unlike `sensitive_keys` above, this
# tool ships no built-in guess at what a "sensitive parent" category is
# called in any particular repository's config shape: any non-empty default
# would be a guess about someone's config shape, which is exactly the kind of
# environment-specific knowledge this module exists to keep out of a public,
# generic tool. A repo that wants whole categories of keys flagged declares
# `sensitive_parents` itself, which then REPLACES this (empty) default
# outright.
BUILTIN_SENSITIVE_PARENTS: frozenset[str] = frozenset()

_MODEL_NET_VALUES = frozenset({"artifact", "none"})
_TOP_LEVEL_KEYS = frozenset({"profile", "python", "min_version", "check", "secrets"})
_CHECK_KEYS = frozenset(
    {
        "model_net",
        "stack_models",
        "declared_paths_fn",
        "sensitive_keys",
        "sensitive_parents",
        "allowed_references",
    }
)


class ConfigError(Exception):
    """`.stackward.toml` is present but the current invocation cannot proceed
    because of it — invalid content, or (via `MinVersionError`) a
    `min_version` the installed `stackward` does not meet.

    Never raised for a missing file — that is `find_repo_config` returning
    `None`, which is a normal state callers decide how to handle. The caller
    must not treat an invalid or unsatisfied config the same as "no config"
    and fall back to defaults.
    """


class MinVersionError(ConfigError):
    """The installed `stackward` is older than this repository's `min_version`."""


@dataclass(frozen=True)
class CheckConfig:
    """The `[check]` table: what the credential-scanning nets look for."""

    model_net: str = "none"
    stack_models: dict[str, str] = field(default_factory=dict)
    # `"module:function"`, or None for this tool's own marking convention.
    # See `_require_module_class` for why the shape is validated here.
    declared_paths_fn: str | None = None
    sensitive_keys: frozenset[str] = BUILTIN_SENSITIVE_KEYS
    sensitive_parents: frozenset[str] = BUILTIN_SENSITIVE_PARENTS
    allowed_references: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    """A parsed, validated `.stackward.toml`."""

    profile: str | None = None
    python: str | None = None
    min_version: str | None = None
    check: CheckConfig = field(default_factory=CheckConfig)
    # Raw `[secrets]` content: one entry per direct member, which is either
    # the reserved "source" table or an arbitrary project-directory table.
    # See `_build_secrets` for why this module validates no deeper than
    # "a table of tables" here.
    secrets: dict[str, dict[str, Any]] = field(default_factory=dict)


def find_repo_config(start: Path | None = None) -> Path | None:
    """Nearest .stackward.toml at or above `start`, stopping at the repo root.

    Returns None rather than falling back to a default: a tool that guesses
    which backend it is talking to is worse than one that refuses.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        config = candidate / CONFIG_FILENAME
        if config.is_file():
            return config
        if (candidate / ".git").exists():
            break
    return None


def _parse_version(text: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints for comparison.

    String comparison gets this wrong: `"0.10.0" > "0.9.0"` is False under
    lexicographic ordering (`"1" < "9"` character by character) but True
    numerically, which is the comparison a version floor actually needs.
    """
    parts = text.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"not a numeric dotted version: {text!r}")
    return tuple(int(part) for part in parts)


def _extend(builtin: frozenset[str], declared: list[str] | None) -> frozenset[str]:
    """`sensitive_keys` EXTENDS the built-ins.

    A repo can only ever add to the set of key names that make a leaf
    suspect. Silently letting a repo drop a built-in key would weaken the
    gate without anyone having deliberately chosen that.
    """
    if declared is None:
        return builtin
    return builtin | frozenset(declared)


def _replace(builtin: frozenset[str], declared: list[str] | None) -> frozenset[str]:
    """`sensitive_parents` REPLACES the built-ins.

    Unlike a key name, a whole parent *category* may have to stay in
    plaintext on purpose in some repository, and there is no "extend but
    exclude one" operation available to opt a single built-in parent back
    out. Replacing the list wholesale is the only way to do that, and it is
    what a repo reaching for this knob wants. This asymmetry with `_extend`
    above is deliberate, not an inconsistency — do not "fix" it into
    matching behaviour.
    """
    if declared is None:
        return builtin
    return frozenset(declared)


def _require_str(data: dict[str, Any], key: str, name: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string")
    return value


def _require_str_list(data: dict[str, Any], key: str, name: str) -> list[str] | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{name} must be a list of strings")
    return value


def _require_module_class(value: str, name: str) -> None:
    """Validate the `"module:Class"` shape the brief states explicitly for
    `stack_models` values.

    Unlike `[secrets.*]`'s interior (left unspecified by the brief, and
    deferred to Task 8 — see `_build_secrets`), this shape *is* specified
    here, so it gets the same fail-closed treatment as `model_net`'s enum:
    a malformed value (a dot instead of a colon, an empty module or class
    name) is rejected at load time, naming the key, rather than loading
    silently and surfacing later as a confusing `ImportError` deep inside
    Task 10, far from the file that caused it. This only checks shape —
    it never imports the module or resolves the class.
    """
    parts = value.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ConfigError(f"{name} must be of the form 'module:Class', got {value!r}")


def _build_check(raw: Any) -> CheckConfig:
    if not isinstance(raw, dict):
        raise ConfigError("'check' must be a table")

    unknown = set(raw) - _CHECK_KEYS
    if unknown:
        raise ConfigError(f"unknown key 'check.{sorted(unknown)[0]}'")

    model_net = raw.get("model_net", "none")
    if not isinstance(model_net, str) or model_net not in _MODEL_NET_VALUES:
        raise ConfigError(
            f"check.model_net must be one of {sorted(_MODEL_NET_VALUES)!r}, "
            f"got {model_net!r}"
        )

    stack_models_raw = raw.get("stack_models", {})
    if not isinstance(stack_models_raw, dict):
        raise ConfigError("check.stack_models must be a table of string to string")
    stack_models: dict[str, str] = {}
    for namespace, target in stack_models_raw.items():
        # TOML table keys are always strings; only the value needs checking.
        if not isinstance(target, str):
            raise ConfigError(
                f"check.stack_models.{namespace!r} must be a string of the "
                "form 'module:Class'"
            )
        _require_module_class(target, f"check.stack_models.{namespace!r}")
        stack_models[namespace] = target

    declared_paths_fn = _require_str(
        raw, "declared_paths_fn", "check.declared_paths_fn"
    )
    if declared_paths_fn is not None:
        # The escape hatch for a repository whose marking convention is not
        # this tool's `json_schema_extra={"secret": True}`. Same
        # `"module:Attribute"` shape as `stack_models`, and validated here for
        # the same reason: a malformed value would otherwise surface as a
        # confusing ImportError inside the generator subprocess, far from the
        # file that caused it.
        _require_module_class(declared_paths_fn, "check.declared_paths_fn")

    sensitive_keys = _require_str_list(raw, "sensitive_keys", "check.sensitive_keys")
    sensitive_parents = _require_str_list(
        raw, "sensitive_parents", "check.sensitive_parents"
    )
    allowed_references = _require_str_list(
        raw, "allowed_references", "check.allowed_references"
    )

    return CheckConfig(
        model_net=model_net,
        stack_models=stack_models,
        declared_paths_fn=declared_paths_fn,
        sensitive_keys=_extend(BUILTIN_SENSITIVE_KEYS, sensitive_keys),
        sensitive_parents=_replace(BUILTIN_SENSITIVE_PARENTS, sensitive_parents),
        allowed_references=list(allowed_references or []),
    )


def _build_secrets(raw: Any) -> dict[str, dict[str, Any]]:
    """Shape only: a table whose every direct member is itself a table.

    This is the boundary drawn in Task 2's brief: Task 8 owns the manifest's
    *semantics* — what `secret`/`plaintext`/`unmanaged` mean inside a
    project's table, what `[secrets.source]`'s own keys are — and this
    module does not yet know what a valid entry looks like, so it does not
    check for one. Unlike `[check]` above, whose keys this module enumerates
    and rejects unknown ones for, `[secrets.*]` content is passed through
    unexamined on purpose. That is not the same inconsistency either: it
    holds because `[check]`'s keys are fully specified in this task's
    brief and `[secrets.*]`'s are not — don't "fix" this one to match, and
    don't loosen `[check]` to match this one.
    """
    if not isinstance(raw, dict):
        raise ConfigError("'secrets' must be a table")

    result: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ConfigError(f"secrets.{name!r} must be a table")
        result[name] = value
    return result


def _min_version_of(data: dict[str, Any]) -> str | None:
    """The `min_version` a parsed document declares, validated, or `None`.

    One definition rather than two, because two callers need the identical
    answer for different reasons: `_build_config` below, which is building a
    whole `Config` and by then knows every other key is one it recognises;
    and `read_min_version`, which needs the floor out of a document this
    binary may be *unable* to validate the rest of. See `read_min_version`
    for why that second case exists at all.
    """
    min_version = _require_str(data, "min_version", "min_version")
    if min_version is not None:
        try:
            _parse_version(min_version)
        except ValueError as exc:
            raise ConfigError(f"min_version: {exc}") from exc
    return min_version


def _build_config(data: dict[str, Any]) -> Config:
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown key {sorted(unknown)[0]!r}")

    profile = _require_str(data, "profile", "profile")
    python = _require_str(data, "python", "python")
    min_version = _min_version_of(data)

    return Config(
        profile=profile,
        python=python,
        min_version=min_version,
        check=_build_check(data.get("check", {})),
        secrets=_build_secrets(data.get("secrets", {})),
    )


# The `(at line L, column C)` coordinate `tomllib` appends to most of its
# messages, and the only part of such a message that is safe to quote back —
# see `toml_position`.
_TOML_POSITION = re.compile(r"\(at (?:line \d+, column \d+|end of document)\)")


def toml_position(exc: tomllib.TOMLDecodeError) -> str:
    """Just the coordinate out of a `TOMLDecodeError`, as ` (at line L,
    column C)`, or `""` when the message does not carry one.

    **`tomllib` echoes document text.** Most of its messages are a fixed
    description plus a coordinate, but not all: `tomllib.loads("[a]\\nx=1\\n
    [a]\\n")` raises `Cannot declare ('a',) twice`, naming the key back. So
    a call site that interpolates `exc` is quoting whatever the parser
    decided to quote, which is a property of the input file rather than a
    decision this code made — and Global Constraint 4 forbids printing the
    matching text.

    That was found first in `store.py`, whose `config` file can hold a
    credential (a `backend_url` of the documented
    `postgres://user:password@host/db` form), and the same defect was still
    in `load_config_text` below, guarded only by the claim that
    `.stackward.toml` holds path and key *names* and never a value. The
    claim is a reasonable reading of what that file is *for*, and it is not
    something this error path can check: the text it is reporting on is, by
    definition, a file that failed to parse — half-pasted, mid-edit, or
    written by someone who misunderstood it. "The file should not contain a
    credential" is exactly the assumption a mistake violates, and the error
    that catches the mistake is the last place that should read one back.

    One definition, in `config.py` rather than in `store.py`, because both
    modules need it and only this direction of import is available:
    `store.py` imports `crypto`, and `config.py` is on the gate path, where
    Global Constraint 2 forbids reaching either. So `store` imports this,
    and never the reverse.

    The coordinate is kept because it is what makes the error actionable and
    it is structural, not quoted text. `TOMLDecodeError` exposes no
    `lineno`/`colno` before 3.13 and this project supports 3.11, so it is
    matched out of the message rather than read off the exception; a message
    shape this does not recognise simply yields no coordinate, which loses
    diagnostics and never discloses anything.
    """
    match = _TOML_POSITION.search(str(exc))
    return f" {match.group(0)}" if match else ""


def load_config_text(text: str, origin: str) -> Config:
    """Parse and validate `text` as `.stackward.toml` content.

    Parsing is separated from reading a file on purpose, and this is the
    seam that makes it so. `commands.pre_commit` must load the policy the
    *index* holds -- `git show ":.stackward.toml"` -- because a policy read
    from the working tree could relax the gate for a commit that does not
    itself carry the relaxation. Having only a path-taking loader forced
    that command to read the working tree, which was a fail-open: the
    content it scanned came from the index while the rules it scanned under
    came from wherever the developer had most recently typed. There is no
    round-trip through a temporary file here either, for the same reason
    `pre_commit` never writes a staged blob to disk.

    `origin` is what the raised `ConfigError` names, so a caller can label
    where the text came from in the terms a person would use to reproduce
    it: a filesystem path for `load_config`, and `":.stackward.toml"` -- the
    literal argument to `git show` -- for the index.

    Raises `ConfigError` naming the offending key on anything invalid: an
    unknown key at any recognised level, a value of the wrong type, or an
    unparseable `min_version`. Never falls back to defaults — a config that
    fails to validate must not be treated as though it did not exist.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # Position only, never the parser's message — see `toml_position`,
        # and `store.load_store_config`, which is the same refusal for the
        # same reason.
        raise ConfigError(f"{origin}: invalid TOML{toml_position(exc)}") from exc

    try:
        return _build_config(data)
    except ConfigError as exc:
        raise ConfigError(f"{origin}: {exc}") from exc


def _read_config_text(path: Path) -> str:
    """`path`'s content as text, or `ConfigError`.

    An unreadable file and one that is not UTF-8 are both `ConfigError`,
    never a silent fall back to defaults: TOML is defined as UTF-8, so bytes
    that are not are a broken config file, not an absent one.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path}: cannot read: not valid UTF-8: {exc}") from exc


def load_config(path: Path) -> Config:
    """Parse and validate the file at `path` as `.stackward.toml`.

    A thin reader in front of `load_config_text`, which does all the
    validation; see there for what is raised and why the two are separate.
    """
    return load_config_text(_read_config_text(path), str(path))


def read_min_version(path: Path) -> str | None:
    """The `min_version` the file at `path` declares — and *nothing else
    about that file*.

    This exists because a full `load_config` cannot answer the question in
    the one case the floor is most needed for. `_build_config` rejects an
    unrecognised top-level key before it gets as far as building anything,
    so a repository that declares both a floor this binary does not meet
    **and** a key this binary does not know — which is precisely
    forward-compatibility, the situation `min_version` exists to make
    survivable — used to report `unknown key 'future_key'` and let the
    command run, instead of saying "upgrade stackward". The unknown key is
    a *consequence* of the unmet floor there, not an independent problem,
    and reporting the consequence sends the reader to fix the wrong thing.

    So the floor is read on its own terms: parse the document, take
    `min_version`, validate that one value, and form no opinion on any
    other key. Everything else about the file is still checked, with better
    words, by whichever command actually needs it — both gate commands
    refuse outright on a policy they cannot parse or validate.

    Raises `ConfigError` for an unreadable file, invalid TOML, or a
    `min_version` that is not a parseable version string. Its sole caller,
    `cli._min_version_refusal`, prints none of those: it treats every one of
    them as "there is no floor I can establish here" and lets the command
    run. The messages are shaped anyway, rather than left as bare
    exceptions, so that a second caller cannot inherit an error it would be
    unable to report.
    """
    text = _read_config_text(path)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # Position only, never the parser's message — see `toml_position`.
        raise ConfigError(f"{path}: invalid TOML{toml_position(exc)}") from exc
    try:
        return _min_version_of(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def enforce_min_version(
    config: Config | None,
    command: str,
    *,
    installed: str = __version__,
) -> None:
    """Raise `MinVersionError` when `installed` is older than `config.min_version`.

    Called from `cli.main` between argument parsing and dispatch, so the
    floor applies to every command rather than to whichever ones remembered
    to ask. Commands in `cli.VERSION_CHECK_EXEMPT` always run: blocking
    `doctor` would make the upgrade instruction this raises unreachable from
    the one command that diagnoses the problem.

    The instruction names `install.sh`, the installer this project actually
    ships, rather than a `stackward self-update` subcommand -- there is no
    such subcommand, and an error message whose only advice is a command
    that does not exist is worse than no advice at all.

    `cli` is imported locally rather than at module level because `cli`
    imports `find_repo_config` from this module — importing `cli` here at
    module scope would make the two modules need each other to finish
    executing.
    """
    from .cli import VERSION_CHECK_EXEMPT  # local: breaks an import cycle

    if config is None or config.min_version is None:
        return
    if command in VERSION_CHECK_EXEMPT:
        return
    if _parse_version(installed) < _parse_version(config.min_version):
        raise MinVersionError(
            f"this repository requires stackward >= {config.min_version} "
            f"(installed: {installed}); upgrade by re-running the installer: "
            "curl -fsSL "
            "https://raw.githubusercontent.com/longthread/stackward/main/install.sh"
            " | sh"
        )
