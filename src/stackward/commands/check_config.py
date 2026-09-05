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
import sys
from pathlib import Path

import yaml

from ..config import CheckConfig, ConfigError, find_repo_config, load_config
from ..nets.heuristic import DocumentError, find_plaintext_credentials


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
        raise CheckError(f"{path}: invalid YAML: {exc}") from exc

    try:
        return find_plaintext_credentials(document, check)
    except DocumentError as exc:
        raise CheckError(f"{path}: {exc}") from exc


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
    one per line — file and config path only, never the value. A
    could-not-run error on any file takes priority over findings on
    another: it is reported and the whole invocation exits 2, since exit 1
    means specifically "a credential was found", which an invocation that
    did not fully run cannot claim to know.
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
        findings.extend((file_arg, path) for path in paths)

    if errored:
        return 2
    if findings:
        for file_arg, path in sorted(findings):
            print(f"{file_arg}: plaintext credential at '{path}'")
        return 1
    return 0
