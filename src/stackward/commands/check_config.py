"""`check-config`: the heuristic net, run over one or more stack config files.

Registered in `cli.build_parser` as `stackward check-config FILE...`. Exit
codes are deliberately distinct: 0 clean, 1 a credential was found, 2 the
check could not run at all (an unreadable path, unparsable YAML, a
`.stackward.toml` this module could not load, or a document that is not a
mapping) — never conflated, so a later fail-closed refusal (also exit 2)
reads the same way this one does: "the gate did not get to answer".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from ..config import CheckConfig, ConfigError, find_repo_config, load_config
from ..nets.heuristic import DocumentError, find_plaintext_credentials

# Matches a repr-quoted fragment the way PyYAML's `%r` interpolation
# produces one — e.g. `'2'`, `` '`' ``, `'id001'`. Python's `repr()` flips
# to double quotes whenever the value contains `'` and no `"` (so `%r` on a
# lone apostrophe is `"'"`, not `'''`), which is why both delimiter forms
# are matched here: matching only `'...'` leaves exactly that one trigger
# character unredacted. See `_describe_yaml_error` for why every such
# fragment gets redacted rather than trusted.
_QUOTED_FRAGMENT = re.compile(r"'[^']*'|\"[^\"]*\"")


class CheckError(Exception):
    """A file could not be checked at all: an unreadable path, unparsable
    YAML, or a document that is not a mapping. Maps to exit code 2, never
    1 — that code means "a credential was found", which this is not."""


def scan_file(path: Path, check: CheckConfig) -> list[str]:
    """Thin file-path wrapper over `find_plaintext_credentials`.

    Reads `path` from the working tree and parses it as YAML before handing
    the result to the content-level net. This is deliberately the *only*
    place that happens: `find_plaintext_credentials` itself takes
    already-parsed document data, so a caller that must check staged
    content instead of the working tree — `pre-commit`, reading
    `git show ":<path>"` — calls that function directly rather than going
    through this wrapper, and never round-trips a blob through a temp file
    to get there.

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
        raise CheckError(f"{path}: invalid YAML: {_describe_yaml_error(exc)}") from exc

    try:
        return find_plaintext_credentials(document, check)
    except DocumentError as exc:
        raise CheckError(f"{path}: {exc}") from exc


def _describe_yaml_error(exc: yaml.YAMLError) -> str:
    """A `YAMLError` message built only from PyYAML's own description and
    the error position, with every repr-quoted fragment in it redacted —
    never from `str(exc)`.

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
    `str(exc)`, for the same reason `_describe_yaml_error` avoids it: an
    unanticipated exception's text is not something this module can vouch
    for as free of file content.
    """
    try:
        check = _load_check_policy()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    findings: list[tuple[str, str]] = []
    errored = False
    for file_arg in args.files:
        try:
            paths = scan_file(Path(file_arg), check)
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
