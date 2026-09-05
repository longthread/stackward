"""The heuristic net: a key-name/parent-name scan for plaintext credentials.

This is the check that runs unconditionally, with no repository cooperation
required — it needs nothing beyond the `[check]` policy and the parsed YAML
document. It walks the *parsed structure*, never a rendered path string: a
dict key may legally contain `.` or `[`, and re-parsing a rendered path to
decide sensitivity would silently lose track of such a key. `paths.render`
is the only thing that turns a walked position back into text, for exactly
the paths this module reports.

Content-level, not file-level, by design. `find_plaintext_credentials` takes
already-parsed document data plus policy and returns findings; it does no
I/O, imports nothing beyond `..config` and `..paths`, and in particular
never imports `yaml` or `cryptography`. That keeps it callable both from a
file on disk (`commands.check_config`) and from a staged git blob
(`commands.pre_commit`, checking `git show ":<path>"` output) without either
caller having to round-trip through a temp file first.

Two rules a plausible implementation gets wrong — see `_is_encrypted` and
`_is_empty` for why, in detail:

- "Encrypted" is a property of the leaf's *parent mapping*, not of the
  leaf's own path text. `path.endswith(".secure")` cannot tell a genuine
  `{"secure": ciphertext}` apart from a malformed `{"secure": ..., "other":
  ...}` sitting next to it — both end in `.secure`.
- "Empty" is `None` or `""`, tested so that `True`/`False` also pass but the
  integers `0` and `1` do not — a plain `value in (None, "", False, True)`
  gets this backwards, because `0 == False` and `1 == True` in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import CheckConfig
from ..paths import render

# The one shape that means "this value is encrypted, not stored in
# plaintext" in a Pulumi stack config: the leaf's parent mapping has exactly
# one key, named `secure`. Anything else — extra sibling keys included — is
# not this shape, however much a stringly-typed check might want it to be.
_SECURE_WRAPPER_KEYS = frozenset({"secure"})


class DocumentError(ValueError):
    """The document cannot be scanned at all: it is not a mapping.

    Raised, never swallowed — a YAML document that failed to parse into a
    key/value structure is a reason to refuse the check, not a reason to
    report zero findings. Callers map this to the "could not run" exit
    code, distinct from both "clean" and "findings present".
    """


def find_plaintext_credentials(document: Any, check: CheckConfig) -> list[str]:
    """Walk `document` for plaintext credentials under `check`'s policy.

    `document` is already-parsed YAML data (e.g. the result of
    `yaml.safe_load`) — this function does no parsing and no file I/O, so it
    works identically whether the document came from a file on disk or a
    `git show ":<path>"` blob.

    A leaf is a finding when its own key matches `check.sensitive_keys`
    (case-insensitive substring) or any strict ancestor key is a member of
    `check.sensitive_parents` — unless the leaf's parent mapping is the
    `{"secure": ...}` encryption wrapper, the value is empty (`None`, `""`,
    `True` or `False`), or the leaf's rendered path is one of
    `check.allowed_references` (a path that names another secret rather
    than holding one).

    Returns the rendered paths of every finding, sorted. Never returns or
    inspects a value for anything other than its structure and identity —
    no finding is reported by value, only by path.

    Raises `DocumentError` if `document` is not a mapping at the top level.
    """
    if not isinstance(document, dict):
        raise DocumentError(
            "stack config must be a mapping at the top level, got "
            f"{type(document).__name__}"
        )

    ctx = _ScanContext(
        check=check,
        allowed_references=frozenset(check.allowed_references),
        findings=[],
    )
    _walk(document, [], frozenset(), ctx)
    return sorted(ctx.findings)


@dataclass(frozen=True)
class _ScanContext:
    """The parts of a scan that stay the same as `_walk`/`_check_leaf`
    recurse, bundled so their signatures carry only what actually varies
    per call: the node, its path, and its ancestor set.

    `findings` is a plain mutable list, appended to in place — `frozen`
    only stops `_walk`/`_check_leaf` from *reassigning* `ctx.findings` to a
    different list, which they never need to do.
    """

    check: CheckConfig
    allowed_references: frozenset[str]
    findings: list[str]


def _walk(
    node: Any,
    path: list[str | int],
    ancestors: frozenset[str],
    ctx: _ScanContext,
) -> None:
    """Recurse through `node`, appending a rendered path to `ctx.findings`
    for every leaf that qualifies. `ancestors` holds every dict key seen
    strictly above the current position — never the current key itself,
    since `sensitive_parents` matching is defined over ancestors, and a
    key's own membership in `sensitive_parents` only matters for what is
    *beneath* it.
    """
    if isinstance(node, dict):
        for raw_key, value in node.items():
            # YAML mapping keys are ordinarily strings already; this only
            # guards the corner cases (a YAML 1.1 bareword like `on:`
            # parsing as a bool, a stray int key) from crashing `render`,
            # which requires str|int segments and rejects bool outright.
            key = raw_key if isinstance(raw_key, str) else str(raw_key)
            child_path = [*path, key]
            if isinstance(value, (dict, list)):
                _walk(value, child_path, ancestors | {key}, ctx)
            else:
                _check_leaf(node, key, value, child_path, ancestors, ctx)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child_path = [*path, index]
            if isinstance(value, (dict, list)):
                _walk(value, child_path, ancestors, ctx)
            else:
                _check_leaf(None, None, value, child_path, ancestors, ctx)


def _check_leaf(
    parent: dict[str, Any] | None,
    own_key: str | None,
    value: Any,
    path: list[str | int],
    ancestors: frozenset[str],
    ctx: _ScanContext,
) -> None:
    """Decide whether the scalar `value` at `path` is a finding.

    `parent` is the mapping directly containing this leaf (`None` when the
    leaf sits directly in a list, which cannot be the `{"secure": ...}`
    shape). `own_key` is that mapping's key for this leaf (`None` for a
    list element, which has no name to match `sensitive_keys` against —
    only an ancestor dict key further up can make such a leaf sensitive).
    """
    sensitive_by_key = own_key is not None and _matches_sensitive_key(
        own_key, ctx.check.sensitive_keys
    )
    sensitive_by_ancestor = bool(ancestors & ctx.check.sensitive_parents)
    if not (sensitive_by_key or sensitive_by_ancestor):
        return
    if parent is not None and _is_encrypted(parent):
        return
    if _is_empty(value):
        return
    rendered = render(path)
    if rendered in ctx.allowed_references:
        return
    ctx.findings.append(rendered)


def _matches_sensitive_key(key: str, sensitive_keys: frozenset[str]) -> bool:
    lowered = key.lower()
    return any(pattern.lower() in lowered for pattern in sensitive_keys)


def _is_encrypted(parent: dict[str, Any]) -> bool:
    """The leaf's parent mapping is exactly `{"secure": ...}` — one key,
    named `secure`.

    Tested against the mapping's shape, never against the leaf's own
    rendered path: `path.endswith(".secure")` cannot distinguish this from
    a malformed `{"secure": ..., "other": ...}`, whose `.secure` child is
    not encrypted (nothing tells a reader the sibling `other` was not the
    plaintext value that leaked) and must still be flagged.
    """
    return set(parent.keys()) == _SECURE_WRAPPER_KEYS


def _is_empty(value: Any) -> bool:
    """`None`, `""`, `True` and `False` are excluded; every integer,
    including `0` and `1`, is not.

    `None`/`True`/`False` are tested by identity and `""` by an
    exact-type-safe equality (`isinstance` first) — never by a containment
    test like `value in (None, "", False, True)`. That test uses `==` under
    the hood, and in Python `0 == False` and `1 == True`, so it would
    silently also exclude the integers `0` and `1` — which occur as real
    config leaves (e.g. a numeric value written in plaintext) and must
    still be findings.
    """
    if value is None or value is True or value is False:
        return True
    return isinstance(value, str) and value == ""
