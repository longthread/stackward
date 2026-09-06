"""`hooks install`: write the `pre-commit` gate into this repository's git
hooks directory.

Registered in `cli.build_parser` as `stackward hooks install`, taking no
arguments -- like `pre-commit` itself, it always operates on the repository
containing the current working directory.

**The hook body has no logic.** It locates the `stackward` executable and
execs `stackward pre-commit` -- nothing else. Git does not version hooks, so
whatever text this command writes into a hook file sits there, unchanged,
until someone runs this command again; a hook body that duplicated any of
`commands.pre_commit`'s actual rules would silently drift from that module
the first time either one changed, in every clone that had installed it
before the drift. Keeping the hook body to "find the binary, exec it" means
every fix and every new rule reaches an already-installed hook automatically,
through the same `stackward` upgrade that would have shipped it anyway --
nothing about the hook file itself has to change, or even be reinstalled.

**Never hardcode `.git/hooks`.** `core.hooksPath` is routinely redirected --
by husky, by the `pre-commit` framework, by a org-wide global config -- and a
directory git does not consult is a *silent* no-op: nothing errors, the
command reports success, and the next commit with a plaintext credential in
it sails straight through a gate everyone believes is installed. `_hooks_dir`
resolves the real location with `git rev-parse --git-path hooks`, exactly
once, and every other function in this module treats that resolved `Path` as
the hooks directory rather than reconstructing or assuming it.

**Refuse a tracked hooks directory outright.** Some repositories commit their
hook *scripts* to a tracked directory (`.githooks/`, say) and point
`core.hooksPath` at it precisely so every clone gets the same hooks without a
separate install step. Writing this tool's hook into a directory like that
would commit a file that `exec`s a `stackward` binary the next clone may not
have installed at all -- turning a shared convenience into a hook that blocks
every commit in every clone until someone notices and reverts it. `_is_tracked`
checks this with `git ls-files --error-unmatch <dir>` (a tracked-directory
pathspec matches if *any* file under it is tracked; git's ordinary,
unredirected `.git/hooks` never matches, because paths under `.git/` are not
part of the repository's tracked namespace at all) and `cmd_install_hooks`
refuses before writing anything -- printing the reason and pointing at
chaining from the tracked hook instead of overwriting it.

**Absolute path, with a `PATH` fallback baked into the hook itself.** The
obvious one-liner, `exec stackward pre-commit`, depends on `stackward` being
on `PATH` *at the moment git invokes the hook* -- and several everyday commit
paths (GUI clients, IDEs, cron) run git with a minimal environment that does
not include the shell profile where `PATH` normally gets extended. `exec`
against a name that is not found there exits 127, and a hook that exits
non-zero blocks the commit -- exactly the experience that teaches people to
reach for `git commit --no-verify` and never look back. `_running_executable`
embeds the absolute path this command was itself invoked with (see its own
docstring for why that differs between a frozen build and a console-script
install), and the hook body it writes still falls back to a `PATH` lookup if
that embedded path is not executable when the hook actually runs -- covering
the case where `stackward` moved, or this hook was installed from one clone's
install layout and is now running against another's.

**A fixed marker line, backup rather than refuse or overwrite.** Some other
tool, or a human, may already have written a hook at the same path (`husky`,
`pre-commit`, or a hand-written script). This command must not silently
destroy that -- but it also must not refuse forever, since re-running this
command is exactly how the hook body picks up a `stackward` upgrade. The
answer is the fixed line `_MARKER`: a hook that already carries it was
written by this command and is safe to overwrite in place (this is what
makes running the command twice leave exactly one hook, with no growing pile
of backups); a hook that lacks it is somebody else's and is moved aside to
`<hook>.backup.<UTC timestamp>` -- never overwritten, never dropped, and
never a reason to refuse the install outright.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import CONFIG_FILENAME, find_repo_config
from .check_config import MINIMAL_POLICY, fail_closed

_HOOK_NAME = "pre-commit"

# Exact line a hook this command wrote always contains, and the sole thing
# that tells "ours, safe to overwrite in place" apart from "somebody else's,
# back it up" -- see the module docstring. Never change this text without
# also handling hooks written by a prior version that still carry the old
# one; nothing in this codebase (yet) needs that migration.
_MARKER = "# managed by: stackward hooks install"


class InstallError(Exception):
    """Installation cannot proceed for a reason that is safe to print
    verbatim: git is missing, the current directory is not inside a git
    working tree, or a git command genuinely failed. Never raised for a
    plaintext credential -- this module never reads staged or working-tree
    *content* at all, only git's own metadata about paths."""


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `git <args>` in the current working directory, under `LC_ALL=C`,
    and hand the completed process back for the caller to interpret.

    **Every** git invocation in this module goes through here, including the
    one whose exit code is not simply pass/fail (`_is_tracked`). That is the
    point of it existing separately from `_run_git`: `_is_tracked` has to
    read git's own stderr *text* to tell "this path is not tracked" apart
    from "git could not answer", and it previously called `subprocess.run`
    directly to do so -- a bypass that already produced one bug, and that
    hid this one. Git's diagnostics are gettext-marked, so under any
    non-English locale those strings are translated, `_is_tracked` matches
    neither, and `hooks install` raises and exits 2 in *every* repository --
    not merely a repository with a redirected `core.hooksPath`. The command
    that installs the gate stops working, for a reason that has nothing to
    do with the repository it is run in.

    The environment is *merged*, never replaced, exactly as
    `commands.pre_commit._run_git` merges it: `git` sets `GIT_DIR`,
    `GIT_INDEX_FILE` and friends for a hook process, and a bare
    `env={"LC_ALL": "C"}` would drop them and point this command at a
    different repository than the one it was invoked for.

    Raises `InstallError` only for the `git` executable being missing --
    every other interpretation of the result belongs to the caller.
    """
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError as exc:
        raise InstallError(f"cannot run git {' '.join(args)}: {exc}") from exc


def _run_git(args: list[str]) -> str:
    """Run `git <args>` in the current working directory and return its
    stdout, stripped of surrounding whitespace.

    Raises `InstallError` for anything that means installation cannot
    proceed: the `git` executable missing, or the invoked subcommand
    failing for any git-reported reason (most notably, the current
    directory not being inside a git working tree at all)."""
    proc = _git(args)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise InstallError(f"git {' '.join(args)} failed: {stderr}")
    return proc.stdout.strip()


def _hooks_dir() -> Path:
    """The repository's actual hooks directory -- `git rev-parse --git-path
    hooks`, resolved to an absolute path, never `.git/hooks` hardcoded. See
    the module docstring for why a redirected `core.hooksPath` makes that
    hardcoding a silent no-op rather than a loud failure."""
    return Path(_run_git(["rev-parse", "--git-path", "hooks"])).resolve()


def _is_tracked(path: Path) -> bool:
    """Whether any file under `path` (already an absolute `Path`) is
    tracked by git, tested exactly as the brief specifies: `git ls-files
    --error-unmatch <dir>` -- a directory pathspec matches if git tracks
    anything underneath it. Passing the pre-resolved absolute path, rather
    than a relative one, keeps this correct regardless of what the current
    working directory happens to be. The same command answers for a *file*
    path, where it degenerates to "is this in the index" -- which is what
    `_warn_when_no_policy` asks it about `.stackward.toml`, so that both
    callers get their answer from one place rather than from two spellings
    of `git ls-files` that could come to disagree.

    Git's own `.git/hooks`, unredirected, never matches: paths under
    `.git/` are not part of the tracked namespace at all, so this reliably
    distinguishes that default location from a `core.hooksPath`
    deliberately pointed at a tracked directory.

    A non-zero exit is deliberately *not* read as "not tracked" on its
    own -- only two specific, recognised git failures are: the ordinary
    in-repository case (`error: pathspec '...' did not match any
    file(s) known to git`, exit 1) and a `core.hooksPath` that resolves
    outside the repository entirely (`fatal: '...' is outside repository
    at '...'`, exit 128 -- reachable because `_hooks_dir` resolves
    symlinks, which can make the resolved path diverge from git's own
    notion of the repository root). Both are matched on git's own
    reported text, not the exit code alone, because exit 128 in
    particular is shared with a broad class of unrelated fatal errors
    (a corrupt index, a permissions problem) where whether the directory
    is tracked is genuinely unanswered. Treating *any* of those as "safe
    to write here" would be exactly the silent bypass of a mandated
    refusal this function exists to prevent, and would make this the one
    place in the module that inverts `_run_git`'s own rule of raising on
    any non-zero exit rather than guessing -- so an unrecognised failure
    raises `InstallError` here too, mapped by the caller to the same
    could-not-determine exit code as every other git failure in this
    command.

    Both recognised strings are gettext-marked in git's own source, so the
    match below is only meaningful under a fixed locale. That is why this
    goes through `_git`, which sets `LC_ALL=C` -- see its docstring; run
    under a translated locale without it, *every* invocation in *every*
    repository falls through to the raise below and `hooks install` exits
    2."""
    proc = _git(["ls-files", "--error-unmatch", str(path)])
    if proc.returncode == 0:
        return True
    stderr = proc.stderr
    if "did not match any file" in stderr or "is outside repository" in stderr:
        return False
    raise InstallError(
        f"could not determine whether {path} is tracked by git: {stderr.strip()}"
    )


def _running_executable() -> str:
    """Absolute path to the running `stackward`, embedded verbatim as the
    hook body's primary way of finding the binary (the `PATH` lookup in
    `_hook_body` is only the runtime fallback).

    A frozen PyInstaller build's `sys.executable` *is* the compiled
    `stackward` binary -- the same fact `cli.cmd_doctor` already reports
    via `getattr(sys, "frozen", False)`. For an ordinary source or
    console-script install, `sys.executable` is the Python interpreter
    instead, which a bare `exec` cannot usefully invoke on its own; there,
    `sys.argv[0]` is normally the console-script file itself -- an
    executable file with its own shebang -- and resolving it to an
    absolute path is what actually names the command that is running.

    `sys.argv[0]` is not always that, though: invoked as `python -m
    stackward`, it resolves to `.../stackward/__main__.py`, a plain
    source file with no shebang and no execute bit -- embedding it would
    make the hook silently fall through to its own `PATH` fallback on
    every single run, which is exactly the dependency this whole scheme
    exists to avoid relying on (see the module docstring). Checked with
    `os.access(..., os.X_OK)` rather than assumed, and resolved with
    `shutil.which("stackward")` when it fails: on a normal install that
    finds the very console script `sys.argv[0]` would have pointed to
    when invoked the usual way.
    """
    if getattr(sys, "frozen", False):
        return sys.executable
    argv0 = Path(sys.argv[0]).resolve()
    if argv0.is_file() and os.access(argv0, os.X_OK):
        return str(argv0)
    found = shutil.which("stackward")
    return found if found is not None else str(argv0)


def _hook_body(executable_path: str) -> str:
    """The full text of the hook file: the marker, an edit-me-not notice,
    and nothing that decides anything about a commit. `executable_path` is
    embedded as a single-quoted, shell-escaped literal (safe even if it
    contains a space) rather than interpolated into a double-quoted
    string, which would let a `$`, backtick, or backslash in the path be
    reinterpreted by the shell instead of taken literally.
    """
    quoted = "'" + executable_path.replace("'", "'\\''") + "'"
    return (
        "#!/bin/sh\n"
        f"{_MARKER}\n"
        "# Installed by `stackward hooks install` -- do not edit this file by\n"
        "# hand; re-run that command instead (it is safe to run again, and\n"
        "# picks up a newer stackward automatically). Everything this hook\n"
        "# actually decides lives in `stackward pre-commit`, not here -- a\n"
        "# rule added or fixed there reaches every clone through a normal\n"
        "# stackward upgrade, with no change to this file at all.\n"
        "set -eu\n"
        "\n"
        f"STACKWARD={quoted}\n"
        'if [ ! -x "$STACKWARD" ]; then\n'
        '    STACKWARD="$(command -v stackward 2>/dev/null || true)"\n'
        "fi\n"
        'if [ -z "$STACKWARD" ]; then\n'
        '    echo "stackward: could not find the stackward executable" >&2\n'
        '    echo "  (checked the path recorded at install time, then PATH)" >&2\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        'exec "$STACKWARD" pre-commit\n'
    )


def _is_stackward_hook(path: Path) -> bool:
    """Whether the file at `path` is a hook this command wrote -- i.e.
    carries `_MARKER` on a line of its own. Never raises: an unreadable or
    binary existing file simply is not recognised as ours, which is the
    correct, safe answer either way -- it gets backed up rather than
    mistaken for a hook this command can overwrite in place."""
    try:
        content = path.read_text()
    except OSError:
        return False
    return any(line.strip() == _MARKER for line in content.splitlines())


def _backup_path(hook_path: Path) -> Path:
    """`<hook>.backup.<UTC timestamp>`, microsecond-resolution so two
    installs in the same repository within the same second cannot collide
    and silently overwrite one backup with another."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return hook_path.with_name(f"{hook_path.name}.backup.{stamp}")


def _install_hook(hooks_dir: Path, executable_path: str) -> Path | None:
    """Write the `pre-commit` hook into `hooks_dir`, creating the directory
    if it does not yet exist (a `core.hooksPath` redirected to a directory
    nothing has populated yet, most notably).

    Returns the path an existing *foreign* hook was backed up to, or
    `None` when there was nothing to back up -- either no hook was present,
    or the one present already carried `_MARKER` and was simply overwritten
    in place. That distinction is what makes installing twice leave exactly
    one hook and create no second backup: the second run finds its own
    marker on the first run's hook and overwrites it directly, rather than
    backing it up as if it were foreign.

    Always `chmod 0o755` after writing, explicitly -- never assumed from a
    default `open()` mode, which a restrictive umask could leave non-
    executable, and never inherited from a prior file at the same path.
    """
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / _HOOK_NAME

    backup_path: Path | None = None
    if hook_path.exists() or hook_path.is_symlink():
        if not _is_stackward_hook(hook_path):
            backup_path = _backup_path(hook_path)
            hook_path.rename(backup_path)

    hook_path.write_text(_hook_body(executable_path))
    hook_path.chmod(0o755)
    return backup_path


@fail_closed
def cmd_install_hooks(_args: argparse.Namespace) -> int:
    """Entry point for `stackward hooks install`.

    Exit codes: **0** installed (fresh, updated in place, or an existing
    foreign hook backed up first); **1** refused because the resolved hooks
    directory is tracked by git (see the module docstring for why writing
    there is unsafe); **2** could not even determine where to install, or
    whether the resolved hooks directory is tracked -- `git` missing, the
    current directory not inside a git working tree, or an unrecognised
    `git ls-files` failure (see `_is_tracked`).
    Reusing `check_config.fail_closed` here for the same reason it wraps
    `cmd_check_config` and `cmd_pre_commit`: an unanticipated exception
    must not fall through to Python's own default exit code of 1, which
    this tool reserves for a specific, different meaning everywhere else it
    appears.

    A repository with no policy file gets a **warning** after the hook is
    written, never a refusal, and the exit code stays 0. Installing the hook
    before writing the policy is a legitimate order to do things in, and
    exit 1 and 2 here mean specific things about the *install* that this is
    not one of -- but without the warning, a repository that installed the
    gate and never declared a policy discovers that at somebody's first
    blocked commit (`pre-commit` refuses a repository that declares none)
    rather than at the moment it could still be fixed in one line.
    """
    try:
        hooks_dir = _hooks_dir()
        tracked = _is_tracked(hooks_dir)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if tracked:
        print(f"error: hooks directory is tracked by git: {hooks_dir}", file=sys.stderr)
        print(
            "Installing here would commit a hook that execs a stackward binary "
            "other clones may not have, blocking their commits. Chain from the "
            "tracked hook instead: have it invoke `stackward pre-commit` itself "
            "(with the same PATH fallback this command would have used) rather "
            "than letting this command overwrite it.",
            file=sys.stderr,
        )
        return 1

    executable = _running_executable()
    hook_path = hooks_dir / _HOOK_NAME
    backup_path = _install_hook(hooks_dir, executable)
    if backup_path is not None:
        print(f"Existing hook backed up to {backup_path}")
    print(f"Installed pre-commit hook -> {hook_path}")
    _warn_when_no_policy()
    return 0


def _warn_when_no_policy() -> None:
    """Warn on stderr when the hook just installed will not find a policy.

    **The question is what `pre-commit` will see, not what is on disk.**
    That gate reads its `[check]` policy from the *index* -- deliberately,
    since a policy read from the working tree could relax the rules for a
    commit that does not itself carry the relaxation -- so a
    `.stackward.toml` that exists but has never been staged is, to the hook
    being installed here, no policy at all. Asking `find_repo_config` alone
    reads the working tree and answers a different question than the one
    this warning is for: the repository got no warning and then blocked
    every commit. That state is a *more* likely one right after `hooks
    install` than a wholly absent policy, since writing the file and
    installing the hook is one sitting and staging it is a separate act.

    Three outcomes, and the third is the reason the other two are worth
    distinguishing:

    * Nothing in the working tree: say so, and name the minimal file.
    * Present but not in the index: say *that*, and name `git add` -- the
      remedy is one word, and "you have no policy" would be visibly false
      to someone looking at the file.
    * In the index: silence, whatever the file's content. A
      `.stackward.toml` that does not parse is present as far as this is
      concerned -- reporting it is the gate's job, at the point it actually
      reads it, and duplicating that judgement here would give one
      repository two different verdicts on the same file.

    One bounded gap, stated so it is a decision and not an oversight: a
    policy that is in the index but deleted from the working tree gets the
    "declares none" message, though the hook would in fact find it. Closing
    it means a second copy of `pre_commit._index_config_path`'s
    nearest-first candidate search, and two copies of that would be a worse
    defect than the wrong wording on a state nobody reaches by accident.

    Deliberately *after* the hook is written and deliberately not an error:
    see `cmd_install_hooks`'s own docstring. That is also why a git failure
    here is silence rather than a raise -- this function runs after the
    install has already succeeded, inside a `@fail_closed` command, so an
    escaping `InstallError` would report exit 2 for a hook that is on disk
    and working. Nothing is lost: the gate itself reports a policy it
    cannot find, in better words, at the moment it matters.

    On stderr, not stdout, for the same reason `store.warn_if_permissive`
    puts its warning there -- this command's stdout says where the hook
    landed, and a caller reading that must not have to filter advisory text
    out of it.
    """
    path = find_repo_config()
    if path is None:
        print(
            f"warning: this repository declares no {CONFIG_FILENAME}, and "
            "`stackward pre-commit` refuses a repository that declares none "
            "-- the hook is installed, but every commit will be blocked "
            "until one exists.",
            file=sys.stderr,
        )
        print(
            f"Create {CONFIG_FILENAME} at the repository root with at least:\n"
            + "\n".join(f"    {line}" for line in MINIMAL_POLICY.splitlines()),
            file=sys.stderr,
        )
        return

    try:
        if _is_tracked(path):
            return
    except InstallError:
        return

    print(
        f"warning: {path} exists but is not staged, and `stackward "
        "pre-commit` reads its policy from the index rather than the "
        "working tree -- the hook is installed, but every commit will be "
        "blocked until the policy is staged too.",
        file=sys.stderr,
    )
    print(f"Stage it: git add {CONFIG_FILENAME}", file=sys.stderr)
