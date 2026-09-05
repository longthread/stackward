"""The model net: declared secrets, carried across a process boundary as a graph.

The heuristic net (`nets.heuristic`) names a credential by the *shape of its
key*. This net names one because a repository *declared* it — a pydantic
field marked `json_schema_extra={"secret": True}`. The two are unioned, so a
credential either of them can name is caught.

**Why an artifact at all.** A shipped `stackward` binary bundles its own
interpreter and cannot import the consuming repository's classes. The
declarations therefore have to cross that boundary as data, written by
`stackward sync-declared-secrets` (which runs under the *repository's*
interpreter — see `commands.sync_declared`) and read back here. Nothing in
this module imports pydantic, and the shipped binary must never bundle it.

**Why a graph and not a list of path patterns.** This is the decision the
whole design turns on, and the counterexample is concrete::

    class Node(BaseModel):
        token: str = Field(json_schema_extra={"secret": True})
        children: list["Node"] = []

Any generator that emits a flat list of patterns needs a cycle guard, so it
emits `['token']` and stops. The *data* that model describes is unbounded:
`token`, `children[0].token`, `children[0].children[0].token`, … The flat
form therefore fails **open** at exactly the depth it must fail closed. So
the artifact is a graph — models, their fields, which fields are marked, and
which child model a field's values belong to — and `find_declared_credentials`
walks that graph and the document *together*. Recursion and mutual recursion
then work by construction rather than by enumeration, because the walk is
driven by the data, which is finite.

Two semantics the equivalence is false without:

- **Prefix, not exact.** A marked field covers everything beneath it,
  whatever its annotation. A marked `peer: P` covers `peer.url` and `peer.k`
  alike — see `_descend`, which stops consulting the graph entirely once
  coverage is on.
- **Structural, not string-glob.** A mapping key may legally contain `.` and
  `[`. This module walks the parsed document alongside the graph and calls
  `paths.render` only to report a position it has already walked to; it never
  builds or re-parses a path string to decide anything, exactly as
  `nets.heuristic` does not.

**Suppressions are shared with the heuristic net**, through
`heuristic.is_encrypted` and `heuristic.is_empty`: a declared field whose
value is the `{"secure": ...}` envelope is encrypted, and one that is absent
or blank holds nothing. Reimplementing either here would let the two nets
drift into disagreeing about whether the same leaf is a credential.

**Freshness, and what it does and does not close.** A committed artifact goes
stale. `load_model_net` reads it from the **git index**
(`git show ":.stackward/declared-secrets.json"`), never the working tree, so
an unstaged regeneration cannot make a stale committed artifact pass; and it
compares the **git blob id** of every recorded source against the index, in
one `git ls-files -s`. The recorded sources come from an MRO walk, so a mark
inherited from a base class in another file is covered, and from the module
that defines any `Annotated` alias carrying a mark; for a contributor outside
the repository the dependency lockfile's blob id stands in, since `git show`
cannot reach site-packages.

That is a *layer*, not a proof, and the boundary is worth stating plainly
rather than overclaiming: blob ids detect **modification of a recorded
source**. They cannot detect a **newly added** file that ought to contribute
but was never recorded — nothing in the index distinguishes that from a file
this repository does not use. Regenerating the artifact in CI and failing on
any diff is what closes that, and it is the mechanism to rely on; this one
catches the common case cheaply, at commit time.

**Every failure to load is exit 2, never exit 1.** A missing, unreadable,
malformed or stale artifact under `model_net = "artifact"` means the check
could not run — reporting it as "a credential was found" would hide a
fail-closed refusal inside a normal finding, and reporting it as clean would
be worse. `model_net = "none"` is a *declared* mode that skips this net
entirely; it is never reached as a fallback from a failure above.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from ..config import CheckConfig
from ..paths import render
from .heuristic import DocumentError, is_empty, is_encrypted

# Repository-relative, POSIX-separated, and the same string on both sides:
# `commands.sync_declared` writes here and `load_model_net` reads `":<this>"`
# out of the index. A `:<path>` revision that does not start with `./` is
# resolved from the repository's top level, which is what makes one constant
# correct for both.
ARTIFACT_PATH = ".stackward/declared-secrets.json"

# Named in every refusal this module raises. A fail-closed error that does not
# say how to clear itself is a fail-closed error someone disables.
REGEN_COMMAND = "stackward sync-declared-secrets"

# Bumped when the artifact's shape changes **or when the generator's coverage
# does**. An unrecognised version is a hard refusal, not a best-effort parse,
# and the refusal has to run in both directions: a newer generator's graph read
# by an older matcher would be under-walked, and an older generator's graph read
# by a newer matcher is under-*covered* — internally consistent, fresh against
# every source it records, and quietly describing less than the repository
# declares.
#
# That second direction is why this is 2 rather than 1. Version 1's generator
# answered "this cannot hold a model" for any class it did not recognise, so a
# pydantic dataclass, a `TypedDict` or a `NamedTuple` holding a marked field
# left the graph with no refusal. Fixing the generator does nothing for an
# artifact already committed by the old one — nothing in it is stale — so the
# version is what forces every repository to regenerate once, and the refusal
# names the command that does it.
ARTIFACT_VERSION = 2

# The document key holding a Pulumi stack's configuration. Findings are
# rendered from the document root (`config.<namespace>.<...>`) so that a path
# this net reports and a path the heuristic net reports are the same string
# for the same leaf, and the union of the two can be deduplicated.
_CONFIG_KEY = "config"


class ModelNetError(Exception):
    """The model net could not be loaded: the artifact is missing from the
    index, unreadable, malformed, of an unrecognised version, or stale.

    Always maps to exit code 2 — "the check could not run" — never to 1,
    which means specifically "a credential was found".
    """


@dataclass(frozen=True)
class ModelRef:
    """A field's values are instances of the model with this id."""

    model: str


@dataclass(frozen=True)
class ListOf:
    """A field's values are a list; every element walks `item`."""

    item: TypeNode


@dataclass(frozen=True)
class DictOf:
    """A field's values are a mapping; every *value* walks `value`.

    The mapping's own keys are data, not field names, and are never matched
    against anything — that is what keeps a key containing `.` or `[` from
    being lost.
    """

    value: TypeNode


# `None` means "nothing beneath this position is a declared model", which is
# the overwhelmingly common case (every scalar field, and every container of
# scalars). Representing it as absence keeps the committed artifact small and
# its diffs meaningful.
TypeNode = Union[ModelRef, ListOf, DictOf]


@dataclass(frozen=True)
class DeclaredField:
    """One field of one model, as the matcher needs it.

    `secret` is the mark. `child` is where the field's values go next in the
    graph, and is `None` both for a scalar field and for a marked field —
    once a field is marked, its annotation stops mattering, because coverage
    is prefix-based and everything beneath it is a finding regardless.
    """

    secret: bool
    child: TypeNode | None


@dataclass(frozen=True)
class ModelNet:
    """The parsed graph: which model each config namespace starts at, and
    every model reachable from those roots."""

    roots: dict[str, str]
    models: dict[str, dict[str, DeclaredField]]


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelNetError(f"artifact: {what} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise ModelNetError(f"artifact: {what} has a non-string key")
    return value


def _parse_type_node(raw: Any, where: str) -> TypeNode | None:
    """One `child` entry, validated rather than trusted.

    An unknown `kind`, a missing member or a dangling model reference is a
    refusal: a graph this module cannot fully walk would silently under-report
    exactly the nesting it exists to reach.
    """
    if raw is None:
        return None
    node = _require_mapping(raw, f"{where}: child")
    kind = node.get("kind")
    if kind == "model":
        model = node.get("model")
        if not isinstance(model, str):
            raise ModelNetError(f"artifact: {where}: 'model' must be a string")
        return ModelRef(model)
    if kind == "list":
        return ListOf(_require_node(node.get("item"), f"{where}: list item"))
    if kind == "dict":
        return DictOf(_require_node(node.get("value"), f"{where}: dict value"))
    raise ModelNetError(f"artifact: {where}: unknown child kind {kind!r}")


def _require_node(raw: Any, where: str) -> TypeNode:
    """A container's interior, which must itself be a node.

    A container whose interior holds no model is written as a scalar field by
    the generator — `list[str]` is not `{"kind": "list", "item": null}` — so
    a null interior here is a malformed artifact, not an empty one.
    """
    node = _parse_type_node(raw, where)
    if node is None:
        raise ModelNetError(f"artifact: {where} must not be null")
    return node


def parse_net(payload: Any) -> tuple[ModelNet, dict[str, str]]:
    """Validate a decoded artifact and return its graph and its `sources` map.

    Raises `ModelNetError` on anything it cannot fully account for: an
    unrecognised version, a missing or misshapen section, a root or a child
    naming a model the artifact does not define, or an empty `sources` map
    (which would leave freshness unverifiable, and so unverified).
    """
    document = _require_mapping(payload, "top level")

    version = document.get("version")
    if version != ARTIFACT_VERSION:
        raise ModelNetError(
            f"artifact: unsupported version {version!r} "
            f"(this stackward reads version {ARTIFACT_VERSION}); "
            f"regenerate with: {REGEN_COMMAND}"
        )

    raw_models = _require_mapping(document.get("models"), "'models'")
    models: dict[str, dict[str, DeclaredField]] = {}
    for model_id, raw_fields in raw_models.items():
        fields: dict[str, DeclaredField] = {}
        for name, raw_field in _require_mapping(
            raw_fields, f"model {model_id!r}"
        ).items():
            entry = _require_mapping(raw_field, f"model {model_id!r} field {name!r}")
            secret = entry.get("secret", False)
            if not isinstance(secret, bool):
                raise ModelNetError(
                    f"artifact: model {model_id!r} field {name!r}: "
                    "'secret' must be a boolean"
                )
            child = _parse_type_node(
                entry.get("child"), f"model {model_id!r} field {name!r}"
            )
            fields[name] = DeclaredField(secret=secret, child=child)
        models[model_id] = fields

    raw_roots = _require_mapping(document.get("roots"), "'roots'")
    roots: dict[str, str] = {}
    for namespace, model_id in raw_roots.items():
        if not isinstance(model_id, str):
            raise ModelNetError(f"artifact: root {namespace!r} must name a model")
        roots[namespace] = model_id

    for model_id in _referenced_models(roots, models):
        if model_id not in models:
            raise ModelNetError(f"artifact: undefined model {model_id!r}")

    sources: dict[str, str] = {}
    for path, blob in _require_mapping(document.get("sources"), "'sources'").items():
        if not isinstance(blob, str):
            raise ModelNetError(f"artifact: source {path!r} must name a blob id")
        sources[path] = blob
    if not sources:
        raise ModelNetError(
            "artifact: 'sources' is empty, so freshness cannot be verified; "
            f"regenerate with: {REGEN_COMMAND}"
        )

    return ModelNet(roots=roots, models=models), sources


def _referenced_models(
    roots: dict[str, str], models: dict[str, dict[str, DeclaredField]]
) -> set[str]:
    """Every model id named anywhere in the graph, roots included."""
    referenced = set(roots.values())
    for fields in models.values():
        for field in fields.values():
            node = field.child
            while node is not None:
                if isinstance(node, ModelRef):
                    referenced.add(node.model)
                    break
                node = node.item if isinstance(node, ListOf) else node.value
    return referenced


@dataclass(frozen=True)
class _MatchContext:
    """What stays the same as `_walk` recurses, so its signature carries only
    what varies: the node, its path, its graph position and its coverage."""

    net: ModelNet
    allowed_references: frozenset[str]
    findings: list[str]


def find_declared_credentials(
    document: Any, net: ModelNet, check: CheckConfig
) -> list[str]:
    """Walk `document` against `net`, returning the rendered path of every
    declared field found holding a plaintext value.

    Takes an already-parsed document and does no I/O, exactly like
    `heuristic.find_plaintext_credentials` — so the same function serves a
    file read from disk and a blob read from the index, with no caller
    round-tripping content through a temporary file to reuse it.

    Matching starts at `document["config"]`, whose keys are Pulumi's
    `<project>:<key>` namespaces; a namespace the artifact declares a root
    for is walked against that root model, and one it does not is left to
    the heuristic net. Findings are rendered from the *document* root, so a
    path reported here and a path reported by the heuristic net are the same
    string for the same leaf.

    Raises `DocumentError` if `document` is not a mapping — the same refusal
    the heuristic net makes, for the same reason.
    """
    if not isinstance(document, dict):
        raise DocumentError(
            "stack config must be a mapping at the top level, got "
            f"{type(document).__name__}"
        )

    config = document.get(_CONFIG_KEY)
    if not isinstance(config, dict):
        return []

    ctx = _MatchContext(
        net=net,
        allowed_references=frozenset(check.allowed_references),
        findings=[],
    )
    for namespace, model_id in sorted(net.roots.items()):
        if namespace not in config:
            continue
        value = config[namespace]
        path: list[str | int] = [_CONFIG_KEY, namespace]
        if isinstance(value, (dict, list)):
            _walk(
                value,
                path,
                ModelRef(model_id),
                False,
                frozenset({id(config), id(value)}),
                ctx,
            )
        # A namespace whose value is a scalar cannot be a model instance;
        # there is no field beneath it to have been marked.
    return sorted(set(ctx.findings))


def _walk(
    node: Any,
    path: list[str | int],
    tnode: TypeNode | None,
    covered: bool,
    visiting: frozenset[int],
    ctx: _MatchContext,
) -> None:
    """Recurse through `node`, driven jointly by the data and by `tnode`.

    `covered` is true once a marked field has been entered. From that point
    the graph is not consulted again: every scalar beneath is a finding,
    whatever the marked field's annotation was, which is what makes coverage
    prefix-based rather than exact.

    `visiting` holds `id()` of every container open on *this* descent path,
    for the same reason `heuristic._walk` carries one: a YAML anchor/alias
    pair can make a container contain itself, and revisiting it would recurse
    forever producing a longer distinct path each time. It is scoped to the
    current path rather than "every container ever seen", so two sibling
    branches legitimately sharing one object are each still walked.
    """
    if isinstance(node, dict):
        for raw_key, value in node.items():
            key = raw_key if isinstance(raw_key, str) else str(raw_key)
            child_covered, child_node = _descend(ctx, tnode, key, covered)
            if not child_covered and child_node is None:
                continue
            child_path = [*path, key]
            if isinstance(value, (dict, list)):
                if id(value) in visiting:
                    continue
                _walk(
                    value,
                    child_path,
                    child_node,
                    child_covered,
                    visiting | {id(value)},
                    ctx,
                )
            elif child_covered:
                _report(node, value, child_path, ctx)
    elif isinstance(node, list):
        child_covered, child_node = _element(tnode, covered)
        if not child_covered and child_node is None:
            return
        for index, value in enumerate(node):
            child_path = [*path, index]
            if isinstance(value, (dict, list)):
                if id(value) in visiting:
                    continue
                _walk(
                    value,
                    child_path,
                    child_node,
                    child_covered,
                    visiting | {id(value)},
                    ctx,
                )
            elif child_covered:
                _report(None, value, child_path, ctx)


def _descend(
    ctx: _MatchContext, tnode: TypeNode | None, key: str, covered: bool
) -> tuple[bool, TypeNode | None]:
    """Where mapping key `key` lands: still covered, and what graph position.

    Once `covered` is on it stays on and the graph position is irrelevant.
    Otherwise the key is a *field name* only when the current position is a
    model; under a `DictOf` the key is data and never matched against
    anything, which is exactly what keeps a key containing `.` or `[` from
    being lost the way a dotted glob would lose it.
    """
    if covered:
        return True, None
    if isinstance(tnode, ModelRef):
        field = ctx.net.models[tnode.model].get(key)
        if field is None:
            return False, None
        if field.secret:
            return True, None
        return False, field.child
    if isinstance(tnode, DictOf):
        return False, tnode.value
    return False, None


def _element(tnode: TypeNode | None, covered: bool) -> tuple[bool, TypeNode | None]:
    """Where a list element lands. A list carries no names, so nothing here
    can newly become covered — only an already-covered ancestor, or a
    `ListOf` position, gives an element anywhere to go."""
    if covered:
        return True, None
    if isinstance(tnode, ListOf):
        return False, tnode.item
    return False, None


def _report(
    parent: dict[str, Any] | None,
    value: Any,
    path: list[str | int],
    ctx: _MatchContext,
) -> None:
    """Record a covered leaf, unless it is encrypted, empty, or an allowed
    reference.

    `parent` is the mapping directly containing the leaf, or `None` for a
    list element (which cannot be the `{"secure": ...}` envelope). Reports the
    rendered path only — never the value, and never anything derived from it.
    """
    if parent is not None and is_encrypted(parent):
        return
    if is_empty(value):
        return
    rendered = render(path)
    if rendered in ctx.allowed_references:
        return
    ctx.findings.append(rendered)


# --------------------------------------------------------------------------
# Loading the artifact. Everything above this line is pure: parsed data in,
# findings out, no I/O. Below it is the part that has to talk to git, because
# freshness is a property of the index and cannot be read from anywhere else.
# --------------------------------------------------------------------------


def run_git(args: list[str]) -> bytes:
    """Run `git <args>` and return stdout, raising `ModelNetError` on failure.

    Runs in the current working directory; every invocation that must be
    anchored to the repository passes `-C <root>` explicitly rather than
    through a parameter here, so there is one way it is done rather than two.

    Shared with `commands.sync_declared` so that both sides of the artifact —
    the writer resolving blob ids and the reader verifying them — use one
    invocation style and one failure mode. git's own stderr is included: the
    only invocations made through this helper are `rev-parse`, `ls-files` and
    `show` of the artifact path, and a *failing* run of any of them has not
    produced blob content to leak.

    Runs under `LC_ALL=C`, like `commands.pre_commit`'s own git calls. Nothing
    here branches on git's message text — every caller branches on
    `returncode` — so this is hardening rather than a live fix, but a stderr
    line quoted verbatim into a `ModelNetError` should read the same in a bug
    report as it did on the machine that hit it.

    The environment is **merged**, never replaced. git exports `GIT_DIR` and
    `GIT_INDEX_FILE` to a hook process, and this helper is what reads the
    index the commit is actually being gated on; a bare `env={"LC_ALL": "C"}`
    would silently point every `ls-files` and `show` here at a different index
    than `pre-commit` is checking — a fail-open no output assertion would
    catch.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError as exc:
        raise ModelNetError(f"cannot run git {' '.join(args)}: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ModelNetError(f"git {' '.join(args)} failed: {stderr}")
    return proc.stdout


def repo_toplevel() -> Path:
    """The repository root, so every git invocation below can be anchored to
    it with `-C`. `git ls-files`' pathspecs are resolved relative to the
    current directory, so a check run from a subdirectory would otherwise
    match nothing and read as "every source is missing"."""
    raw = run_git(["rev-parse", "--show-toplevel"])
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ModelNetError(f"repository path is not valid UTF-8: {exc}") from exc
    if not text:
        raise ModelNetError("git did not report a repository root")
    # Resolved, because callers compare it against `Path(...).resolve()` of a
    # source file to decide whether that file is inside the repository. On a
    # platform where the repository sits under a symlinked directory, git's
    # own answer and a resolved module path are two different strings for one
    # directory, and the comparison would fail for every file.
    return Path(text).resolve()


def index_blob_ids(root: Path, paths: list[str]) -> dict[str, str]:
    """Blob id of each of `paths` as the **index** currently holds it.

    One `git ls-files -s` for the whole set, which is both cheaper and more
    exact than hashing file content: it reads what would actually be
    committed, and it needs no assumption about line endings, filters, or the
    working tree matching the index at all.

    Paths absent from the index are simply absent from the result — it is the
    caller that decides whether that is a refusal, because the two callers
    mean different things by it. An unmerged entry (stage other than 0) is a
    refusal here, since there is no single blob to compare against.
    """
    if not paths:
        return {}
    raw = run_git(["-C", str(root), "ls-files", "-s", "-z", "--", *paths])
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelNetError(f"index listing is not valid UTF-8: {exc}") from exc

    found: dict[str, str] = {}
    for entry in text.split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) != 3 or not path:
            raise ModelNetError("could not parse git ls-files output")
        _mode, blob, stage = parts
        if stage != "0":
            raise ModelNetError(
                f"{path}: unresolved merge conflict in the index; "
                "resolve conflicts and re-stage before checking"
            )
        found[path] = blob
    return found


def read_artifact_from_index(root: Path) -> Any:
    """The artifact as the **index** holds it, decoded from JSON.

    Read with `git show ":<path>"`, never `Path.read_text()`. An unstaged
    regeneration sitting only in the working tree must not let the stale
    artifact that is actually about to be committed pass — which is the whole
    reason this check exists rather than trusting the file on disk.
    """
    try:
        raw = run_git(["-C", str(root), "show", f":{ARTIFACT_PATH}"])
    except ModelNetError as exc:
        # git's own message here is about a revision it could not resolve and
        # says nothing about what to do — and "not in the index" is by far the
        # most likely reason to reach it: the artifact was never generated, or
        # was generated and never staged. Say that, and say how to fix it.
        raise ModelNetError(
            f"{ARTIFACT_PATH} is not in the git index, and "
            'check.model_net = "artifact" requires it. Generate it with: '
            f"{REGEN_COMMAND}, then: git add {ARTIFACT_PATH} "
            f"(git reported: {str(exc).splitlines()[0]})"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelNetError(
            f"{ARTIFACT_PATH}: staged content is not valid UTF-8: {exc}"
        ) from exc
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ModelNetError(f"{ARTIFACT_PATH}: invalid JSON: {exc}") from exc


def verify_sources(root: Path, sources: dict[str, str]) -> None:
    """Refuse unless every recorded source is in the index at its recorded
    blob id.

    A mismatch means a file that contributed a mark has changed since the
    artifact was written, and the artifact may no longer describe the models
    — so it is a hard refusal naming the regeneration command, never a
    downgrade to running without this net. Reports the first offending path
    in sorted order, so the message is stable across runs.
    """
    found = index_blob_ids(root, sorted(sources))
    for path in sorted(sources):
        actual = found.get(path)
        if actual is None:
            raise ModelNetError(
                f"{ARTIFACT_PATH} is stale: {path} is no longer in the index; "
                f"regenerate with: {REGEN_COMMAND}"
            )
        if actual != sources[path]:
            raise ModelNetError(
                f"{ARTIFACT_PATH} is stale: {path} has changed since it was "
                f"generated; regenerate with: {REGEN_COMMAND}"
            )


def _verify_roots(net: ModelNet, check: CheckConfig) -> None:
    """The artifact must start from exactly the namespaces the policy declares.

    A second, independent guard on the same drift the recorded blob id of
    `.stackward.toml` catches — and it needs no blob id at all, so it still
    holds if that file is somehow not among the sources. A namespace declared
    in `check.stack_models` with no root in the artifact is not a smaller
    answer; it is *no* answer for that namespace, silently.

    **Key sets only.** A model id is `module:QualName` while the config names
    `module:Class`, and those diverge for a nested class — comparing values
    would refuse a correct artifact.
    """
    declared = set(check.stack_models)
    present = set(net.roots)
    if declared == present:
        return
    missing = sorted(declared - present)
    extra = sorted(present - declared)
    detail = []
    if missing:
        detail.append(f"declares {missing} that it does not cover")
    if extra:
        detail.append(f"covers {extra} that it no longer declares")
    raise ModelNetError(
        f"{ARTIFACT_PATH} is stale: this repository {' and '.join(detail)}; "
        f"regenerate with: {REGEN_COMMAND}"
    )


def load_model_net(check: CheckConfig) -> ModelNet | None:
    """The model net for this repository, or `None` when it is switched off.

    `model_net = "none"` returns `None` without touching git or the
    filesystem: it is a *declared* mode, and this function must be
    indistinguishable from a repository that never had an artifact. Every
    other outcome under `model_net = "artifact"` — no artifact in the index,
    unreadable content, a malformed or unsupported graph, a stale source —
    raises `ModelNetError`, which callers map to exit code 2.
    """
    if check.model_net == "none":
        return None
    if check.model_net != "artifact":
        raise ModelNetError(f"unknown check.model_net mode {check.model_net!r}")

    root = repo_toplevel()
    net, sources = parse_net(read_artifact_from_index(root))
    verify_sources(root, sources)
    _verify_roots(net, check)
    return net
