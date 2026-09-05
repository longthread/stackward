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
"""

from __future__ import annotations

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
# called in any particular repository's config shape — inventing one (e.g.
# copying a name like "environment_variables" out of somebody's existing
# gate script) would be exactly the kind of environment-specific knowledge
# this module exists to keep out of a public, generic tool. A repo that
# wants whole categories of keys flagged declares `sensitive_parents`
# itself, which then REPLACES this (empty) default outright.
BUILTIN_SENSITIVE_PARENTS: frozenset[str] = frozenset()

_MODEL_NET_VALUES = frozenset({"artifact", "none"})
_TOP_LEVEL_KEYS = frozenset({"profile", "python", "min_version", "check", "secrets"})
_CHECK_KEYS = frozenset(
    {
        "model_net",
        "stack_models",
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


def _build_config(data: dict[str, Any]) -> Config:
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown key {sorted(unknown)[0]!r}")

    profile = _require_str(data, "profile", "profile")
    python = _require_str(data, "python", "python")
    min_version = _require_str(data, "min_version", "min_version")
    if min_version is not None:
        try:
            _parse_version(min_version)
        except ValueError as exc:
            raise ConfigError(f"min_version: {exc}") from exc

    return Config(
        profile=profile,
        python=python,
        min_version=min_version,
        check=_build_check(data.get("check", {})),
        secrets=_build_secrets(data.get("secrets", {})),
    )


def load_config(path: Path) -> Config:
    """Parse and validate `path` as `.stackward.toml`.

    Raises `ConfigError` naming the offending key on anything invalid: an
    unknown key at any recognised level, a value of the wrong type, or an
    unparseable `min_version`. Never falls back to defaults — a config that
    fails to validate must not be treated as though it did not exist.
    """
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        # `.stackward.toml` holds path and key *names* only, never a
        # credential value, so it is safe for this message to include
        # whatever tomllib quotes from the offending line.
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc

    try:
        return _build_config(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def enforce_min_version(
    config: Config | None,
    command: str,
    *,
    installed: str = __version__,
) -> None:
    """Raise `MinVersionError` when `installed` is older than `config.min_version`.

    Commands in `cli.VERSION_CHECK_EXEMPT` (`doctor`, `self-update`) always
    run: blocking either one would make the upgrade instruction this raises
    unreachable. `cli` is imported locally rather than at module level
    because `cli` imports `find_repo_config` from this module — importing
    `cli` here at module scope would make the two modules need each other to
    finish executing.
    """
    from .cli import VERSION_CHECK_EXEMPT  # local: breaks an import cycle

    if config is None or config.min_version is None:
        return
    if command in VERSION_CHECK_EXEMPT:
        return
    if _parse_version(installed) < _parse_version(config.min_version):
        raise MinVersionError(
            f"this repository requires stackward >= {config.min_version} "
            f"(installed: {installed}); upgrade with: stackward self-update"
        )
