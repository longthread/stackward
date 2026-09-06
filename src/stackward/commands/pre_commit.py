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

**Staged, never working-tree -- policy included.** Every path this module
reads goes through `git show ":<path>"` (see `_read_staged_blob`), which is
stage 0 of the index as it stands right now -- not `Path.read_text()`. An
unstaged fix sitting only in the working tree must not let an already-staged
credential through, and a working-tree file that merely *looks* clean must
not be mistaken for what will actually be committed.

`.stackward.toml` is read the same way (`_index_config_path`,
`_load_check_policy`), and it took a review to notice it had not been.
Reading content from the index while reading the *rules* from the working
tree is a fail-open with a working exploit: stage a credential, then edit
the working-tree `.stackward.toml` to add an `allowed_references` entry for
it -- staging nothing -- and this command exits 0. The credential is
committed under a relaxation that is not itself being committed, and the
next clone of the repository contains the leak but not the excuse. Three
knobs weaken from that direction (`allowed_references` adds a suppression,
`model_net = "none"` drops a whole net, and `sensitive_parents` *replaces*
rather than extends, so the worktree can shrink what the index declared);
`nets.model.verify_sources` cannot close any of them, because it compares an
index blob against a recorded blob and a worktree-only edit touches neither,
and under `model_net = "none"` it does not run at all. Reading policy from
the index closes all three at once: a relaxation only takes effect in the
same commit that carries it.

**Content-level, not file-level.** `_scan_staged_config` calls
`nets.heuristic.find_plaintext_credentials` and
`nets.model.find_declared_credentials` directly with an already-parsed
document, the same content-level entry points
`commands.check_config.scan_file` calls for a file on disk. Neither this
module nor that one ever round-trips a blob through a temporary file to
reuse the other's file-path wrapper -- doing that here specifically would
write staged (possibly credential-bearing) content to disk outside the
index, and would reintroduce the working-tree-vs-index confusion this
module exists to avoid.

**Both nets run here, not only the heuristic one.** This is the layer that
prevents publication; `check-config`, invoked by hand, reports a leak that
may already be committed. A model net running only on manual invocation
would be absent at the one moment it counts, so the stronger of the two nets
would not be protecting anything. The two are unioned through a set, so a
leaf both name is reported once.

**The declared-secrets artifact is read from the index too**, by
`nets.model.load_model_net` -- the same rule as every other path this module
reads, and for the same reason: an unstaged regeneration must not vouch for
the stale artifact that is actually about to be committed. A missing,
malformed or stale artifact under `model_net = "artifact"` exits 2, never 1;
`model_net = "none"` skips that net without reading anything at all.

**No credential resolution.** This module imports `yaml`, `..config` and
`..nets.heuristic` -- never `cryptography` and never any credential
provider. The gate decides whether a value looks like a plaintext
credential; it never fetches, decrypts, or connects anywhere to do it.

**Path discovery from any working directory.** `git diff --cached
--name-only` and `git show ":<path>"` both resolve paths relative to the
repository's top level regardless of the caller's current working
directory (verified empirically: a path printed by the first command,
handed unchanged to the second, resolves correctly whether invoked from
the repo root or a subdirectory several levels down). `_index_config_path`
reproduces `config.find_repo_config`'s upward walk in those same
repository-relative terms, using `git rev-parse --show-prefix` for where
the caller is standing -- so a `.stackward.toml` in a subdirectory is found
from below it and ignored from beside it, exactly as the working-tree walk
would. Between the two, this command needs no special handling for "the hook
ran from a subdirectory": there is no such thing, only "the hook ran
somewhere inside the repository."
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import PurePosixPath

import yaml

from ..config import CONFIG_FILENAME, CheckConfig, ConfigError, load_config_text
from ..nets.heuristic import DocumentError, find_plaintext_credentials
from ..nets.model import (
    ModelNet,
    ModelNetError,
    find_declared_credentials,
    load_model_net,
)
from .check_config import (
    CheckError,
    MissingPolicyError,
    describe_yaml_error,
    fail_closed,
    missing_policy_message,
)

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
    invocations this module makes are `diff --cached --name-only`,
    `rev-parse --show-prefix`, `ls-files` and `show ":<path>"`, and a
    *failing* run of any of them never gets far enough to have printed blob
    content -- a non-zero exit here means git could not produce output at
    all, so its diagnostic is always about a path or a ref, never about what
    a blob contains.

    Runs under `LC_ALL=C`. Everything below branches on `returncode` alone
    and never on git's message text, so this is hardening rather than a live
    fix here -- but git's diagnostics are gettext-marked, and a message
    quoted into a `CheckError` should read the same in a bug report as it did
    on the machine that hit it. The environment is *merged*, never replaced:
    git sets `GIT_INDEX_FILE`, `GIT_DIR` and friends for a hook process, and
    dropping them would make this command read a different index than the
    commit it is gating.
    """
    env = {**os.environ, "LC_ALL": "C"}
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, check=False, env=env
        )
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


def _scan_staged_config(
    path: str, check: CheckConfig, net: ModelNet | None
) -> list[str]:
    """Content-level scan of `path`'s *staged* blob, by both nets.

    Mirrors `check_config.scan_file` exactly, except the source of text is
    `_read_staged_blob` (the index) rather than `Path.read_text()` (the
    working tree): both parse with `yaml.safe_load`, hand the resulting
    document straight to the two content-level nets, union the results
    through a set (a leaf both name is one problem, not two), and wrap
    every way that can fail in `CheckError` -- unreadable/undecodable
    content, invalid YAML (via `describe_yaml_error`, which redacts what
    PyYAML would otherwise interpolate into its own message), or a
    document that is not a mapping.

    `net` is `None` when the repository declares `model_net = "none"`; the
    heuristic net always runs. There is no third case where it is `None`
    because no policy was found -- that is a refusal now, not a scan.
    """
    text = _read_staged_blob(path)
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


def _index_config_path() -> str | None:
    """Repository-relative path of the nearest `.stackward.toml` **in the
    index** at or above the current directory, or `None` if the index holds
    none.

    The index-side twin of `config.find_repo_config`, and it walks in the
    same direction for the same reason: a policy file beside the caller
    beats one at the repository root. `git rev-parse --show-prefix` gives
    the current directory as a repository-relative prefix -- git's own
    answer, in git's own byte encoding, so no assumption is needed about
    where the process's `cwd` sits relative to a symlinked work tree, and
    (verified) it is not subject to `core.quotePath` mangling the way
    `ls-files` output would be without `-z`.

    Candidates are matched with `:(literal,top)` pathspec magic. `top` is
    load-bearing: `ls-files`' pathspecs resolve against the *caller's*
    directory, not the repository root, so a hook invoked from a
    subdirectory would otherwise look for `<subdir>/<subdir>/.stackward.toml`
    and find nothing. `literal` is belt-and-braces -- verified empirically
    that git compares a pathspec's literal text against each path before
    falling back to wildmatch, so an exact-path pathspec containing `[`, `*`
    or `?` matches its own directory either way -- and it is kept because
    the intent here is an exact path, and nothing downstream should have to
    depend on that fallback ordering staying as it is. What actually makes
    a metacharacter harmless is the line below: a candidate is accepted only
    by exact string equality against this list, never by taking whatever git
    happened to print. `--full-name` makes the answer repository-relative,
    which is what `git show ":<path>"` then wants, and `-z` keeps
    `core.quotePath` from escaping a non-ASCII path into something no
    candidate would equal.
    """
    raw_prefix = _run_git(["rev-parse", "--show-prefix"])
    try:
        prefix = raw_prefix.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckError(f"repository path is not valid UTF-8: {exc}") from exc
    # Split rather than strip: a directory name may legally begin or end
    # with a space, and `.strip()` would silently rename it.
    parts = [part for part in prefix.rstrip("\n").split("/") if part]

    # Nearest first, ending at the repository root -- `find_repo_config`'s
    # order, so the two helpers cannot disagree about which file wins.
    candidates = [
        "/".join([*parts[:depth], CONFIG_FILENAME])
        for depth in range(len(parts), -1, -1)
    ]
    listed = _run_git(
        [
            "ls-files",
            "-z",
            "--full-name",
            "--",
            *(f":(literal,top){candidate}" for candidate in candidates),
        ]
    )
    try:
        text = listed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckError(f"index path list is not valid UTF-8: {exc}") from exc
    present = {path for path in text.split("\0") if path}
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def _load_check_policy() -> CheckConfig:
    """The `[check]` policy this commit will actually carry, read from the
    **index** -- never from the working tree.

    Not the same three lines as `check_config`'s helper of the same name,
    and the difference is the point rather than an oversight: `check-config`
    answers "is this file on disk clean?" and so reads its policy from disk,
    while this command answers "is what I am about to commit clean?" and so
    reads both the content and the rules from the index. Reading the rules
    from the working tree was a fail-open with a working exploit -- see the
    module docstring for it and for the three knobs that weaken from that
    direction.

    A policy file present in the working tree but never staged is therefore
    *not* a policy as far as this command is concerned, which is the same
    rule it already applies to every stack config it scans. Absence is a
    refusal (`MissingPolicyError`), never the built-in defaults: those carry
    `model_net = "none"`, so returning them here would drop the stronger net
    in exactly the repositories that had never said anything about it.
    """
    config_path = _index_config_path()
    if config_path is None:
        raise MissingPolicyError(
            missing_policy_message(
                where="in the index",
                remedy=(
                    f"then stage it: git add {CONFIG_FILENAME}  "
                    "(this command reads its policy from staged content, "
                    "exactly like the files it scans, so a relaxation cannot "
                    "take effect without being committed alongside what it "
                    "relaxes)"
                ),
            )
        )
    # Labelled the way a person would reproduce it: `git show ":<path>"`.
    return load_config_text(_read_staged_blob(config_path), f":{config_path}").check


@fail_closed
def cmd_pre_commit(_args: argparse.Namespace) -> int:
    """Entry point for `stackward pre-commit` -- the hook body's whole job.

    Exit codes, never conflated (see the module docstring in
    `check_config.py` for why this distinction matters to a later,
    fail-closed refusal): **0** clean; **1** a credential was found by either
    net, or a Pulumi state export was staged (refused outright, since a
    state export is credentials by construction, not by finding);
    **2** the check could not run at all -- an unmerged index entry, a git
    command failing, staged content that is not decodable or not valid
    YAML, a document that is not a mapping, a `.stackward.toml` that is
    invalid *or absent from the index*, a missing or stale
    declared-secrets artifact under `model_net = "artifact"`, or an
    unanticipated exception. Python's own default for an uncaught
    exception is exit 1, which here would misreport "a credential was
    found" for a file this command never actually finished evaluating --
    every per-file scan below is wrapped accordingly, and the `@fail_closed`
    decorator above is the outer boundary that catches anything else this
    function does not: policy loading (`_load_check_policy`) and the
    staged-file listing (`_staged_paths`) both run *outside* the per-file
    loop, guarded only by `except (ConfigError, CheckError)` / `except
    CheckError` respectively, so a different, unanticipated exception from
    either -- an `OSError` from the `git` invocation layer that neither
    converts, say -- needs `@fail_closed` to avoid escaping uncaught and
    exiting 1 by Python's own default.

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
       refused outright, before the credential scan below even starts, and
       deliberately *before* the policy is loaded: this refusal needs no
       policy to make (a state export is credentials by construction, not
       by finding), and exit 1 is the more specific answer than the exit 2
       a missing or unparseable policy would produce. A repository that has
       not written a `.stackward.toml` yet still gets told it staged a
       state export, rather than being told about its config first.
    2a. The `[check]` policy, loaded from the index (`_load_check_policy`).
       A repository with none is refused here, exit 2, naming the minimal
       file to create -- never scanned under `CheckConfig()`'s defaults.
    3. The declared-secrets artifact, loaded from the index. Loaded once,
       before the scan loop, and a failure to load it exits 2 having
       reported nothing: a stale artifact means this command does not know
       what the repository declared, so printing the *other* net's findings
       and exiting 1 would claim a complete answer it does not have.
    4. Both nets, unioned, over every staged `Pulumi.<stack>.yaml`.

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
    except (ConfigError, CheckError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        net = load_model_net(check)
    except ModelNetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stack_configs = sorted(path for path in staged if _is_pulumi_stack_config(path))

    findings: list[tuple[str, str]] = []
    errored = False
    for path in stack_configs:
        try:
            found = _scan_staged_config(path, check, net)
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
