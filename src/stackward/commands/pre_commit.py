"""`pre-commit`: the gate, run as a git hook body over *staged* content.

Registered in `cli.build_parser` as `stackward pre-commit`, taking no
arguments -- it always operates on the current repository's index, exactly
as a git hook would invoke it. Installing this as the actual hook body is
Task 5's job (`commands.install_hooks`); this module only implements what
runs once invoked.

This is the layer that actually *prevents* a credential from being
published: it runs before the commit object exists, so a block here means
the credential never enters history at all. A CI-side check (elsewhere in
this project's plan) can only ever report one that is already public.
Consequently every rule below leans the same way when in doubt: refuse
rather than guess, because the cost of a false positive here is a rerun,
and the cost of a false negative is permanent.

**Staged, never working-tree.** Every path this module reads goes through
`git show ":<path>"` (see `_read_staged_blob`), which is stage 0 of the
index as it stands right now -- not `Path.read_text()`. An unstaged fix
sitting only in the working tree must not let an already-staged credential
through, and a working-tree file that merely *looks* clean must not be
mistaken for what will actually be committed.

**Content-level, not file-level.** `_scan_staged_config` calls
`nets.heuristic.find_plaintext_credentials` directly with an
already-parsed document, the same content-level entry point
`commands.check_config.scan_file` calls for a file on disk. Neither this
module nor that one ever round-trips a blob through a temporary file to
reuse the other's file-path wrapper -- doing that here specifically would
write staged (possibly credential-bearing) content to disk outside the
index, and would reintroduce the working-tree-vs-index confusion this
module exists to avoid.

**No credential resolution.** This module imports `yaml`, `..config` and
`..nets.heuristic` -- never `cryptography` and never any credential
provider. The gate decides whether a value looks like a plaintext
credential; it never fetches, decrypts, or connects anywhere to do it.

**Path discovery from any working directory.** `git diff --cached
--name-only` and `git show ":<path>"` both resolve paths relative to the
repository's top level regardless of the caller's current working
directory (verified empirically: a path printed by the first command,
handed unchanged to the second, resolves correctly whether invoked from
the repo root or a subdirectory several levels down). `_load_check_policy`
relies on the same property transitively through `find_repo_config`, which
walks upward from `Path.cwd()` to the repository root and checks
`<candidate>/.stackward.toml` before checking whether `<candidate>/.git`
exists -- so the repo root's own `.stackward.toml` is always found on the
same pass that discovers `.git`, never one step too late. Between the two,
this command needs no special handling for "the hook ran from a
subdirectory": there is no such thing, only "the hook ran somewhere inside
the repository."
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

import yaml

from ..config import CheckConfig, ConfigError, find_repo_config, load_config
from ..nets.heuristic import DocumentError, find_plaintext_credentials
from .check_config import CheckError, describe_yaml_error, fail_closed

_STATE_EXPORT_NAME = "state.json"
_STATE_EXPORT_SUFFIX = ".stack-export.json"

_STACK_CONFIG_PREFIX = "Pulumi."
_STACK_CONFIG_SUFFIX = ".yaml"


def _run_git(args: list[str]) -> bytes:
    """Run `git <args>` from the current working directory and return its
    stdout as bytes.

    Raises `CheckError` for anything that means this check cannot proceed:
    the `git` executable missing, the current directory not inside a work
    tree, or the invoked subcommand failing for any git-reported reason
    (including `git show` on a path with no stage-0 entry -- an unmerged
    path). The message includes git's own stderr text verbatim: the only
    two invocations this module ever makes are `diff --cached --name-only`
    and `show ":<path>"`, and a *failing* run of either never gets far
    enough to have printed blob content -- a non-zero exit here means git
    could not produce output at all, so its diagnostic is always about a
    path or a ref, never about what a blob contains.
    """
    try:
        proc = subprocess.run(["git", *args], capture_output=True, check=False)
    except OSError as exc:
        raise CheckError(f"cannot run git {' '.join(args)}: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise CheckError(f"git {' '.join(args)} failed: {stderr}")
    return proc.stdout


def _staged_paths(diff_filter: str) -> list[str]:
    """Staged paths matching `diff_filter`, exactly as git names them --
    relative to the repository's top level regardless of the caller's own
    working directory (see the module docstring).

    Always `-z`, split on NUL: without it, git quotes and C-escapes any
    path containing a space, a newline, or a non-ASCII byte, which would
    corrupt exactly the paths this command exists to protect.

    Called with `"ACMRT"` for the files actually being committed, and
    separately with `"U"` to find anything unmerged.

    `R` matters: a rename plus an edit stages as `R`, and a filter that
    omitted it would let that change bypass every check below. `T`
    (type-change) matters for the same reason, discovered in review of
    the brief's own literal `"ACMR"`: converting an *already-tracked*
    `Pulumi.<stack>.yaml` into a symlink stages as `T`, not `A`/`M`/`R` --
    without `T`, that file would vanish from this listing exactly the way
    an unmerged path does, and the credential scan below would never see
    it at all. A *new* symlink stages as `A`, which `"ACMRT"` already
    covered even before `T` was added; `T` closes the gap for a symlink
    replacing a file git was already tracking as ordinary content.

    Deliberately never called combined with `"U"` (`"ACMRTU"`): an
    unmerged path must be caught by the dedicated check in
    `cmd_pre_commit` and refused before anything else runs, not folded
    into the "files to scan" list, where finding it there could be
    mistaken for having already decided what to do with it.
    """
    raw = _run_git(
        ["diff", "--cached", "--name-only", "-z", f"--diff-filter={diff_filter}"]
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckError(f"staged path list is not valid UTF-8: {exc}") from exc
    return [path for path in text.split("\0") if path]


def _is_state_export(path: str) -> bool:
    """`state.json`, or a name ending `.stack-export.json` -- matched by
    basename, so one nested under any directory still counts. Either shape
    is a full Pulumi state export: every resource input the stack has,
    credentials included, in plaintext, by design rather than by mistake.
    """
    name = PurePosixPath(path).name
    return name == _STATE_EXPORT_NAME or name.endswith(_STATE_EXPORT_SUFFIX)


def _is_pulumi_stack_config(path: str) -> bool:
    """`Pulumi.<stack>.yaml`, matched by basename on the `Pulumi.` prefix
    and `.yaml` suffix rather than a restrictive character class --
    `<stack>` may itself contain dots (`Pulumi.prod.eu.yaml` is a legal
    stack name).

    The empty-stack case is excluded on purpose. `Pulumi.yaml` -- the
    *project* file, which holds no stack config at all -- also satisfies a
    naive `name.startswith("Pulumi.") and name.endswith(".yaml")`, because
    the two literals overlap on the single "." they share: all 11
    characters of `"Pulumi.yaml"` are covered by `"Pulumi."` (the first 7)
    and by `".yaml"` (the last 5) at the same time, with nothing left over
    for a stack name to occupy. Requiring a non-empty slice between the two
    literals is what tells `Pulumi.yaml` apart from `Pulumi.dev.yaml`.
    """
    name = PurePosixPath(path).name
    if not (name.startswith(_STACK_CONFIG_PREFIX) and name.endswith(_STACK_CONFIG_SUFFIX)):
        return False
    stack = name[len(_STACK_CONFIG_PREFIX) : -len(_STACK_CONFIG_SUFFIX)]
    return bool(stack)


def _read_staged_blob(path: str) -> str:
    """The staged (index) content of `path`, decoded as UTF-8 -- never the
    working tree. `git show ":<path>"` reads stage 0 of the index exactly
    as it stands right now, so a working-tree edit made after `git add`
    (or never staged at all) cannot hide a credential that is already
    staged, and cannot be mistaken for one that is not.

    Raises `CheckError` if git cannot produce the blob at all -- most
    notably a path with no stage-0 entry, i.e. unmerged, though
    `cmd_pre_commit`'s dedicated check should already have refused before
    this is ever reached for such a path -- or if the bytes it does
    produce are not valid UTF-8.
    """
    raw = _run_git(["show", f":{path}"])
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckError(f"{path}: staged content is not valid UTF-8: {exc}") from exc


def _scan_staged_config(path: str, check: CheckConfig) -> list[str]:
    """Content-level scan of `path`'s *staged* blob.

    Mirrors `check_config.scan_file` exactly, except the source of text is
    `_read_staged_blob` (the index) rather than `Path.read_text()` (the
    working tree): both parse with `yaml.safe_load` and hand the resulting
    document straight to `find_plaintext_credentials`, and both wrap
    every way that can fail in `CheckError` -- unreadable/undecodable
    content, invalid YAML (via `describe_yaml_error`, which redacts what
    PyYAML would otherwise interpolate into its own message), or a
    document that is not a mapping.
    """
    text = _read_staged_blob(path)
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CheckError(f"{path}: invalid YAML: {describe_yaml_error(exc)}") from exc
    try:
        return find_plaintext_credentials(document, check)
    except DocumentError as exc:
        raise CheckError(f"{path}: {exc}") from exc


def _load_check_policy() -> CheckConfig:
    """The `[check]` policy for the repository containing the current
    working directory, or the built-in defaults when there is none --
    absence is a normal state (see `config.py`), not an error.

    The same three lines as `check_config`'s private helper of the same
    name, built entirely from this module's own public imports
    (`find_repo_config`, `load_config`); not worth an inter-command import
    for something this small.
    """
    config_path = find_repo_config()
    if config_path is None:
        return CheckConfig()
    return load_config(config_path).check


@fail_closed
def cmd_pre_commit(_args: argparse.Namespace) -> int:
    """Entry point for `stackward pre-commit` -- the hook body's whole job.

    Exit codes, never conflated (see the module docstring in
    `check_config.py` for why this distinction matters to a later,
    fail-closed refusal): **0** clean; **1** a credential was found by the
    heuristic net, or a Pulumi state export was staged (refused outright,
    since a state export is credentials by construction, not by finding);
    **2** the check could not run at all -- an unmerged index entry, a git
    command failing, staged content that is not decodable or not valid
    YAML, a document that is not a mapping, an invalid `.stackward.toml`,
    or an unanticipated exception. Python's own default for an uncaught
    exception is exit 1, which here would misreport "a credential was
    found" for a file this command never actually finished evaluating --
    every per-file scan below is wrapped accordingly, and the `@fail_closed`
    decorator above is the outer boundary that catches anything else this
    function does not: policy loading (`_load_check_policy`) and the
    staged-file listing (`_staged_paths`) both run *outside* the per-file
    loop, guarded only by `except ConfigError` / `except CheckError`
    respectively, so a different, unanticipated exception from either --
    a bare `OSError` from `find_repo_config` walking through an
    unreadable parent directory, say -- needs `@fail_closed` to avoid
    escaping uncaught and exiting 1 by Python's own default.

    Checks run in this fixed order, each a hard gate before the next:

    1. Unmerged index entries. Not a content check at all -- a
       precondition on whether the index can be trusted enough to read.
       `--diff-filter=ACMRT`, used for everything below, silently
       *excludes* unmerged paths (git reports them as bare status `U`,
       which matches none of `A`, `C`, `M`, `R`, `T`), so without this
       dedicated check an unmerged file would never reach either check
       below and this command would report "clean" for a commit git
       itself already refuses to allow. Checked for *any* unmerged path,
       not only ones that look like a stack config or state export: git's
       own refusal to commit ("Committing is not possible because you
       have unmerged files") is likewise unconditional on which path is
       unmerged, and this command can be invoked directly, bypassing that
       refusal.
    2. Staged Pulumi state exports (`state.json`, `*.stack-export.json`) --
       refused outright, before the credential scan below even starts.
    3. The heuristic net, over every staged `Pulumi.<stack>.yaml`.

    Never prints a credential value: a finding is reported only by its
    rendered config path, exactly like `check-config`.
    """
    try:
        unmerged = _staged_paths("U")
    except CheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if unmerged:
        for path in sorted(unmerged):
            print(f"error: {path}: unresolved merge conflict in the index", file=sys.stderr)
        print("error: resolve conflicts and re-stage before committing", file=sys.stderr)
        return 2

    try:
        staged = _staged_paths("ACMRT")
    except CheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    state_exports = sorted(path for path in staged if _is_state_export(path))
    if state_exports:
        print(
            "State exports (state.json, *.stack-export.json) hold every "
            "resource input a stack has, credentials included, in plaintext."
        )
        for path in state_exports:
            print(f"{path}: Pulumi state export staged -- refused")
            print(f"  fix: git restore --staged '{path}'")
        print("git commit --no-verify bypasses this hook.")
        return 1

    try:
        check = _load_check_policy()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stack_configs = sorted(path for path in staged if _is_pulumi_stack_config(path))

    findings: list[tuple[str, str]] = []
    errored = False
    for path in stack_configs:
        try:
            found = _scan_staged_config(path, check)
        except CheckError as exc:
            print(f"error: {exc}", file=sys.stderr)
            errored = True
            continue
        except Exception as exc:  # noqa: BLE001 - fail closed on anything unanticipated
            print(
                f"error: {path}: could not check ({type(exc).__name__})",
                file=sys.stderr,
            )
            errored = True
            continue
        findings.extend((path, finding) for finding in found)

    if findings:
        for file_path, finding_path in sorted(findings):
            print(f"{file_path}: plaintext credential at '{finding_path}'")
            print(f"  fix: pulumi config set --secret --path '{finding_path}'")
        print("git commit --no-verify bypasses this hook.")

    if errored:
        return 2
    return 1 if findings else 0
