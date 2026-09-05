"""`check-config`: the credential nets, run over one or more stack config files.

Registered in `cli.build_parser` as `stackward check-config FILE...`. Exit
codes are deliberately distinct: 0 clean, 1 a credential was found, 2 the
check could not run at all (an unreadable path, unparsable YAML, a
`.stackward.toml` this module could not load, a document that is not a
mapping, or a model-net artifact that is missing, malformed or stale) —
never conflated, so a fail-closed refusal reads the same way whichever check
made it: "the gate did not get to answer".

**Two nets, unioned.** `nets.heuristic` names a credential by the shape of
its key and needs no cooperation from the repository. `nets.model` names one
the repository *declared*, by walking a committed graph of its own pydantic
models against the document. A leaf either net names is a finding, and a leaf
both name is reported once — `scan_file` unions the two through a set, so a
path found twice does not print twice.

**A stale model net is exit 2, not exit 1.** `load_model_net` raises rather
than returning an empty net when the artifact is missing, unreadable or no
longer matches the sources it was generated from, and this module maps that
to "could not run". Reporting it as a finding would file a fail-closed
refusal under "a credential was found"; reporting it as clean would be worse
still. `model_net = "none"` is the *declared* way to run without that net,
and is never reached as a fallback from a failure.
"""

from __future__ import annotations

import argparse
import functools
import re
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

from ..config import CheckConfig, ConfigError, find_repo_config, load_config
from ..nets.heuristic import DocumentError, find_plaintext_credentials
from ..nets.model import ModelNet, ModelNetError, find_declared_credentials, load_model_net

# Matches a repr-quoted fragment the way PyYAML's `%r` interpolation
# produces one — e.g. `'2'`, `` '`' ``, `'id001'`. Python's `repr()` flips
# to double quotes whenever the value contains `'` and no `"` (so `%r` on a
# lone apostrophe is `"'"`, not `'''`), which is why both delimiter forms
# are matched here: matching only `'...'` leaves exactly that one trigger
# character unredacted. See `describe_yaml_error` for why every such
# fragment gets redacted rather than trusted.
_QUOTED_FRAGMENT = re.compile(r"'[^']*'|\"[^\"]*\"")


class CheckError(Exception):
    """A file could not be checked at all: an unreadable path, unparsable
    YAML, or a document that is not a mapping. Maps to exit code 2, never
    1 — that code means "a credential was found", which this is not.

    Reused as-is by `commands.pre_commit` for the same "could not run"
    conditions on a staged blob (undecodable content, unparsable YAML, a
    non-mapping document, an unmerged index entry, a git command that
    failed) — one exception type for one meaning, not a second class that
    would have to be kept consistent with this one by hand."""


def fail_closed(
    command: Callable[[argparse.Namespace], int],
) -> Callable[[argparse.Namespace], int]:
    """Wrap a command entry point so any exception it does not already
    handle itself becomes exit code 2, never left to propagate.

    Both `cmd_check_config` and `cmd_pre_commit` already wrap each
    per-file scan individually — see either function's own per-file
    `except CheckError` / catch-all `except Exception` pair — but that
    only protects the scan step itself. Policy loading
    (`_load_check_policy`, built on `find_repo_config`/`load_config`) and,
    in `pre_commit`, the staged-file listing (`_staged_paths`) both run
    *outside* that loop, each guarded only by `except ConfigError` or
    `except CheckError` respectively. A review of this module found the
    gap concretely: `find_repo_config` walks upward through parent
    directories with bare `.is_file()`/`.exists()` calls and no
    try/except of its own, so a `PermissionError` on a non-traversable
    parent surfaces as a plain `OSError` — neither `ConfigError` nor
    `CheckError` — which would otherwise escape both commands uncaught
    and exit 1 by Python's own default: the code this tool reserves
    exclusively for "a credential was found", misreporting exactly that
    for an invocation that never got far enough to check anything.

    This decorator is the outer boundary that catches precisely what the
    inner, per-file guards do not, from one definition shared by both
    commands rather than two independently-maintained copies of the same
    try/except (or, worse, three or more scattered call-site patches that
    would leave the next command built this way exposed by default).
    Never prints `str(exc)` — the exception type only — for the same
    reason `describe_yaml_error` never echoes `str(exc)` on a YAML error:
    an unanticipated exception's text is not something this boundary can
    vouch for as free of file content.
    """

    @functools.wraps(command)
    def wrapped(args: argparse.Namespace) -> int:
        try:
            return command(args)
        except Exception as exc:  # noqa: BLE001 - the fail-closed boundary itself
            print(f"error: could not run ({type(exc).__name__})", file=sys.stderr)
            return 2

    return wrapped


def scan_file(
    path: Path, check: CheckConfig, net: ModelNet | None = None
) -> list[str]:
    """Thin file-path wrapper over both nets, unioned.

    Reads `path` from the working tree and parses it as YAML before handing
    the result to the content-level nets. This is deliberately the *only*
    place that happens: both nets take already-parsed document data, so a
    caller that must check staged content instead of the working tree —
    `pre-commit`, reading `git show ":<path>"` — calls them directly rather
    than going through this wrapper, and never round-trips a blob through a
    temp file to get there.

    `net` is `None` when the repository declares `model_net = "none"`, or
    has no `[check]` policy at all; the heuristic net always runs. The two
    result lists are unioned through a set, because both nets can name the
    same leaf — a field marked `secret` whose key is also called `password`
    is the ordinary case, not a corner one — and a finding printed twice
    reads as two problems.

    Raises `CheckError` for anything that means the check could not run:
    the path cannot be read, the content is not valid YAML, or the parsed
    document is not a mapping.
    """
    try:
        text = path.read_text()
    except OSError as exc:
        raise CheckError(f"{path}: cannot read: {exc}") from exc

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CheckError(f"{path}: invalid YAML: {describe_yaml_error(exc)}") from exc

    try:
        found = set(find_plaintext_credentials(document, check))
        if net is not None:
            found |= set(find_declared_credentials(document, net, check))
    except DocumentError as exc:
        raise CheckError(f"{path}: {exc}") from exc
    return sorted(found)


def describe_yaml_error(exc: yaml.YAMLError) -> str:
    """A `YAMLError` message built only from PyYAML's own description and
    the error position, with every repr-quoted fragment in it redacted —
    never from `str(exc)`.

    Public (not `_`-prefixed) specifically so `commands.pre_commit` can
    reuse it for the same reason: a staged blob it parses can hold a
    credential just as easily as a file on disk can, so a YAML parse
    failure there needs the exact same redaction, not a second,
    independently-written copy of it.

    `str(exc)` on a `MarkedYAMLError` calls `Mark.__str__`, which calls
    `Mark.get_snippet()` and emits the *whole source line* around the
    error — reason enough on its own never to use it here, in the one file
    in the whole system expected to hold a credential.

    But `exc.problem` is not unconditionally safe either, which an earlier
    version of this docstring claimed and which a review disproved: several
    PyYAML scanner/parser messages interpolate one raw character or a
    user-chosen name via `%r` — "found character %r that cannot start any
    token", "found unknown escape character %r", an anchor/alias/tag-handle
    name in a duplicate/undefined-alias error. A value like `"hunter\2ok"`
    yields `found unknown escape character '2'` — one character of a real
    credential; `repr()` flips to double quotes whenever that character is
    itself an apostrophe (`"'"`, not `'''`), which is why `_QUOTED_FRAGMENT`
    matches both delimiter forms rather than only `'...'`.

    This removes every document character PyYAML is *currently known* to
    interpolate via `%r`, by shape (any repr-quoted run) rather than by
    enumerating today's message text — that is a mechanism with a stated
    boundary, not a guarantee that no PyYAML wording could ever leak a
    character some other way. `tests/test_heuristic.py`'s parameterized
    redaction test is what would catch a future wording change that
    escapes this pattern; this docstring does not promise it can't happen.
    Contrast `config.py`'s `load_config`, which safely echoes tomllib's
    message in full because `.stackward.toml` holds only path and key
    *names* — this file holds the opposite, so it gets the more careful,
    still-bounded treatment.
    """
    if isinstance(exc, yaml.MarkedYAMLError) and exc.problem is not None:
        problem = _QUOTED_FRAGMENT.sub("<redacted>", exc.problem)
        if exc.problem_mark is not None:
            return (
                f"{problem} (line {exc.problem_mark.line + 1}, "
                f"column {exc.problem_mark.column + 1})"
            )
        return problem
    return "invalid YAML syntax"


def _load_check_policy() -> CheckConfig:
    """The `[check]` policy for the repository containing the current
    working directory, or the built-in defaults when there is none —
    absence is a normal state (see `config.py`), not an error."""
    config_path = find_repo_config()
    if config_path is None:
        return CheckConfig()
    return load_config(config_path).check


@fail_closed
def cmd_check_config(args: argparse.Namespace) -> int:
    """Entry point for `stackward check-config FILE...`.

    Findings print as `<file>: plaintext credential at '<path>'`, sorted,
    one per line — file and config path only, never the value. They print
    even when *another* file in the same invocation errors: a real finding
    on `good.yaml` must never be hidden behind a parse error on
    `broken.yaml` that happens to sort after it, so the exit code alone
    carries the could-not-run signal. A could-not-run error on any file
    still wins the exit code over findings elsewhere: it exits 2, since
    exit 1 means specifically "a credential was found", which an
    invocation that did not fully run cannot claim to know.

    Anything a per-file scan raises other than `CheckError` — a file that
    is not valid UTF-8, or an unanticipated bug elsewhere in this tool —
    is also mapped to the could-not-run exit code rather than left to
    propagate: Python's default exit code for an uncaught exception is 1,
    which here would misreport "a credential was found" for a file this
    command never actually finished evaluating. The message never includes
    `str(exc)`, for the same reason `describe_yaml_error` avoids it: an
    unanticipated exception's text is not something this module can vouch
    for as free of file content.

    That per-file guard only covers the scan loop. Anything policy-loading
    raises other than `ConfigError` — this function's own `except
    ConfigError` below only catches that one type — is caught instead by
    the `@fail_closed` decorator wrapping this function, for the same
    "never exit 1 for a reason other than a real finding" guarantee.

    The model net is loaded once, before the loop, and a failure to load it
    exits 2 without scanning anything. That ordering is the point: a stale
    artifact means this command does not know what the repository declared,
    so reporting the *other* net's findings and exiting 1 would announce a
    complete answer it does not have.
    """
    try:
        check = _load_check_policy()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        net = load_model_net(check)
    except ModelNetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    findings: list[tuple[str, str]] = []
    errored = False
    for file_arg in args.files:
        try:
            paths = scan_file(Path(file_arg), check, net)
        except CheckError as exc:
            print(f"error: {exc}", file=sys.stderr)
            errored = True
            continue
        except Exception as exc:  # noqa: BLE001 - fail closed on anything unanticipated
            print(
                f"error: {file_arg}: could not check ({type(exc).__name__})",
                file=sys.stderr,
            )
            errored = True
            continue
        findings.extend((file_arg, path) for path in paths)

    if findings:
        for file_arg, path in sorted(findings):
            print(f"{file_arg}: plaintext credential at '{path}'")

    if errored:
        return 2
    return 1 if findings else 0
