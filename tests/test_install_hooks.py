"""Tests for `hooks install` -- writing the `pre-commit` gate into a
repository's real hooks directory.

Every test here uses a real temporary git repository, created with `git
init`. This command's entire reason to exist is a handful of git behaviours
that are easy to get wrong silently -- `core.hooksPath` redirection, a
tracked hooks directory, an existing foreign hook -- so, exactly as in
`test_pre_commit.py`, git is never mocked.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stackward.cli import main
from stackward.commands import install_hooks as install_hooks_module

# ---------------------------------------------------------------------------
# Real-git-repository fixtures and helpers.
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path: Path, isolated_git: Path) -> Path:
    """A real, minimal git repository with one initial commit that declares
    a `.stackward.toml`.

    The policy file is committed rather than merely written because
    `pre-commit` -- which the end-to-end tests here drive through a real
    `git commit` -- now reads its policy from the index and refuses a
    repository that declares none (Global Constraint 3: "A missing policy
    file is a refusal, not a skip"). Without it, the commit-blocking tests
    below would still see a non-zero exit, but for the wrong reason, and the
    clean-commit test would fail outright. `hooks install` itself reads no
    policy, so the tests that only assert on installation are unaffected
    either way.
    """
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / ".stackward.toml").write_text('[check]\nmodel_net = "none"\n')
    _git(path, "add", ".stackward.toml")
    _git(path, "commit", "-q", "-m", "init")
    return path


def install(repo: Path, monkeypatch) -> int:
    monkeypatch.chdir(repo)
    return main(["hooks", "install"])


def backups(hooks_dir: Path) -> list[Path]:
    return sorted(hooks_dir.glob("pre-commit.backup.*"))


# ---------------------------------------------------------------------------
# Default location: `.git/hooks`, resolved rather than hardcoded.
# ---------------------------------------------------------------------------


def test_default_location_installs_into_git_hooks(repo, monkeypatch, capsys):
    assert install(repo, monkeypatch) == 0
    hook_path = repo / ".git" / "hooks" / "pre-commit"
    assert hook_path.is_file()
    assert install_hooks_module._MARKER in hook_path.read_text()
    assert str(hook_path) in capsys.readouterr().out


def test_installed_hook_actually_blocks_a_real_commit_with_a_credential(
    repo, monkeypatch, real_executable
):
    """A file existing at the expected path is not proof git runs it --
    that gap is exactly the "silent no-op" the brief warns about. This
    drives a real `git commit`, through the real `stackward` this test
    interpreter has installed, against the real default hooks directory."""
    assert install(repo, monkeypatch) == 0

    (repo / "Pulumi.dev.yaml").write_text("config:\n  myproject:dbPassword: hunter2\n")
    _git(repo, "add", "Pulumi.dev.yaml")
    result = subprocess.run(
        ["git", "commit", "-m", "add credential"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "plaintext credential" in result.stdout + result.stderr


def test_installed_hook_allows_a_real_clean_commit(repo, monkeypatch, real_executable):
    assert install(repo, monkeypatch) == 0

    (repo / "Pulumi.dev.yaml").write_text("name: myproject\n")
    _git(repo, "add", "Pulumi.dev.yaml")
    result = subprocess.run(
        ["git", "commit", "-m", "add clean config"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# core.hooksPath redirection: never hardcode `.git/hooks`.
# ---------------------------------------------------------------------------


def test_redirected_hooks_path_installs_there_not_in_dot_git(repo, monkeypatch):
    (repo / "myhooks").mkdir()
    _git(repo, "config", "core.hooksPath", "myhooks")

    assert install(repo, monkeypatch) == 0

    redirected_hook = repo / "myhooks" / "pre-commit"
    default_hook = repo / ".git" / "hooks" / "pre-commit"
    assert redirected_hook.is_file()
    assert not default_hook.exists()


def test_redirected_hooks_path_directory_is_created_if_missing(repo, monkeypatch):
    """The redirected directory need not exist yet -- e.g. a fresh clone
    that has set core.hooksPath but never run any install step."""
    _git(repo, "config", "core.hooksPath", "myhooks")
    assert not (repo / "myhooks").exists()

    assert install(repo, monkeypatch) == 0
    assert (repo / "myhooks" / "pre-commit").is_file()


def test_hooks_path_absolute_and_outside_the_worktree_is_treated_as_untracked(
    tmp_path, repo, monkeypatch
):
    """`git ls-files --error-unmatch` on a path outside the repository
    fails with "is outside repository", a different git error than the
    in-repo untracked case (`did not match any file(s)`) exercised
    elsewhere in this file. `_is_tracked` recognises both messages as
    "not tracked" -- this proves that answer is still correct for this
    differently-shaped git failure, not merely right by accident of a
    single error message this suite happens to have checked."""
    external = tmp_path / "external-hooks"
    external.mkdir()
    _git(repo, "config", "core.hooksPath", str(external))

    assert install(repo, monkeypatch) == 0
    assert (external / "pre-commit").is_file()


def test_unrecognised_ls_files_failure_exits_2_rather_than_installing(
    repo, monkeypatch, capsys
):
    """`_is_tracked` must not treat *every* non-zero `git ls-files` exit
    as "not tracked" -- only the two recognised, specifically-worded
    failures. A `git ls-files` failure for an unrelated reason (a corrupt
    index, a permissions problem -- simulated here) leaves the question
    genuinely unanswered, and installing anyway would be exactly the
    silent bypass of a mandated refusal `_is_tracked` exists to prevent.
    This must exit 2 (could not determine), never 0 (installed) or 1
    (the different, definitive "is tracked" refusal)."""
    real_run = subprocess.run

    def failing_run(args, **kwargs):
        if args[:2] == ["git", "ls-files"]:
            return subprocess.CompletedProcess(
                args,
                returncode=128,
                stdout="",
                stderr="fatal: index file corrupt (simulated)\n",
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_run)
    assert install(repo, monkeypatch) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "index file corrupt" in captured.err
    # Refused to guess either way: no hook was written.
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()


def test_is_tracked_raises_install_error_for_an_unrecognised_failure(
    tmp_path, monkeypatch
):
    """Direct unit test of the boundary `_is_tracked` itself draws,
    independent of the command-level wiring covered above."""
    real_run = subprocess.run

    def failing_run(args, **kwargs):
        if args[:2] == ["git", "ls-files"]:
            return subprocess.CompletedProcess(
                args, returncode=128, stdout="", stderr="fatal: something else\n"
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_run)
    with pytest.raises(install_hooks_module.InstallError, match="something else"):
        install_hooks_module._is_tracked(tmp_path)


def test_redirected_hooks_path_hook_actually_runs_on_a_real_commit(
    repo, monkeypatch, real_executable
):
    """The specific case "don't hardcode `.git/hooks`" exists for: proves
    the installed hook does anything at all when git is configured to
    look somewhere else entirely."""
    (repo / "myhooks").mkdir()
    _git(repo, "config", "core.hooksPath", "myhooks")
    assert install(repo, monkeypatch) == 0

    (repo / "Pulumi.dev.yaml").write_text("config:\n  myproject:dbPassword: hunter2\n")
    _git(repo, "add", "Pulumi.dev.yaml")
    result = subprocess.run(
        ["git", "commit", "-m", "add credential"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "plaintext credential" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Refusal on a tracked hooks directory.
# ---------------------------------------------------------------------------


def test_refuses_when_hooks_directory_is_tracked_by_git(repo, monkeypatch, capsys):
    (repo / "githooks").mkdir()
    (repo / "githooks" / "pre-commit").write_text("#!/bin/sh\necho tracked\n")
    _git(repo, "add", "githooks")
    _git(repo, "commit", "-q", "-m", "commit shared hooks")
    _git(repo, "config", "core.hooksPath", "githooks")

    original = (repo / "githooks" / "pre-commit").read_text()
    assert install(repo, monkeypatch) == 1

    captured = capsys.readouterr()
    assert "tracked" in captured.err.lower()
    assert "githooks" in captured.err
    # Refused, not silently overwritten or backed up.
    assert (repo / "githooks" / "pre-commit").read_text() == original
    assert backups(repo / "githooks") == []


def test_tracked_refusal_suggests_chaining_from_the_tracked_hook(repo, monkeypatch, capsys):
    (repo / "githooks").mkdir()
    (repo / "githooks" / "pre-commit").write_text("#!/bin/sh\n")
    _git(repo, "add", "githooks")
    _git(repo, "commit", "-q", "-m", "commit shared hooks")
    _git(repo, "config", "core.hooksPath", "githooks")

    install(repo, monkeypatch)
    err = capsys.readouterr().err
    assert "chain" in err.lower()
    assert "pre-commit" in err  # names what to chain to: `stackward pre-commit`


# ---------------------------------------------------------------------------
# An existing foreign hook is backed up, never refused or overwritten blind.
# ---------------------------------------------------------------------------


def test_existing_foreign_hook_is_backed_up(repo, monkeypatch, capsys):
    hooks_dir = repo / ".git" / "hooks"
    hook_path = hooks_dir / "pre-commit"
    foreign_content = "#!/bin/sh\necho this is somebody else's hook\n"
    hook_path.write_text(foreign_content)
    hook_path.chmod(0o755)

    assert install(repo, monkeypatch) == 0

    found = backups(hooks_dir)
    assert len(found) == 1
    assert found[0].read_text() == foreign_content
    # The installed hook now carries this tool's marker.
    assert install_hooks_module._MARKER in hook_path.read_text()
    assert str(found[0]) in capsys.readouterr().out


def backup_stamp(hooks_dir: Path) -> str:
    [backup] = backups(hooks_dir)
    assert backup.name.startswith("pre-commit.backup.")
    return backup.name.removeprefix("pre-commit.backup.")


@pytest.fixture
def local_time_far_from_utc():
    """Move this process's *local* time twelve hours away from UTC.

    `XXX-12` is a POSIX `TZ` string -- a made-up zone abbreviation and a
    numeric offset, no geography and no place name -- so nothing here depends
    on the tz database or on where the suite happens to run. `time.tzset()`
    is what makes `datetime.now()` (which reads local time) actually observe
    it.

    Restored by hand rather than through `monkeypatch`, because `tzset()`
    caches the zone in the C library: undoing the variable without calling
    `tzset()` again would leave every later test in the session running with
    a shifted local clock.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "XXX-12"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def test_backup_filename_matches_hook_dot_backup_dot_timestamp(repo, monkeypatch):
    """The suffix parses as the full timestamp, not merely "starts with
    eight digits".

    `stamp[:8].isdigit()` was satisfied by any date-shaped prefix, so
    neither the microseconds (what stops two installs in the same second
    from colliding) nor the format was pinned by it.
    """
    hooks_dir = repo / ".git" / "hooks"
    (hooks_dir / "pre-commit").write_text("#!/bin/sh\necho foreign\n")

    before = datetime.now(timezone.utc)
    install(repo, monkeypatch)
    after = datetime.now(timezone.utc)

    stamp = backup_stamp(hooks_dir)
    parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
    assert before <= parsed <= after
    assert parsed.microsecond or "." in stamp  # sub-second resolution is present


def test_backup_timestamp_is_utc_and_not_local_time(
    repo, monkeypatch, local_time_far_from_utc
):
    """The `Z` in the filename has to mean UTC.

    Nothing pinned it: every existing assertion on this name held equally
    for `datetime.now()`, and on a machine whose local zone *is* UTC the two
    are indistinguishable -- which is most CI images, so the gap would never
    have surfaced there either. With local time twelve hours away, a
    timestamp taken in local time cannot fall inside a UTC bracket around
    the call.

    It matters because these names sort, and because two clones on machines
    in different zones would otherwise produce backups whose apparent order
    is wrong -- for files whose whole job is to be the recoverable copy of
    somebody else's hook.
    """
    hooks_dir = repo / ".git" / "hooks"
    (hooks_dir / "pre-commit").write_text("#!/bin/sh\necho foreign\n")

    before = datetime.now(timezone.utc)
    install(repo, monkeypatch)
    after = datetime.now(timezone.utc)

    parsed = datetime.strptime(backup_stamp(hooks_dir), "%Y%m%dT%H%M%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    assert before <= parsed <= after
    # Guard against the fixture silently not taking effect, which would make
    # the assertion above prove nothing.
    assert abs((datetime.now() - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()) > 3600


# ---------------------------------------------------------------------------
# Idempotence: installing twice leaves one hook and creates no second backup.
# ---------------------------------------------------------------------------


def test_installing_twice_over_a_foreign_hook_creates_only_one_backup(repo, monkeypatch):
    hooks_dir = repo / ".git" / "hooks"
    (hooks_dir / "pre-commit").write_text("#!/bin/sh\necho foreign\n")

    assert install(repo, monkeypatch) == 0
    assert install(repo, monkeypatch) == 0

    assert len(backups(hooks_dir)) == 1
    assert (hooks_dir / "pre-commit").is_file()


def test_installing_twice_from_a_clean_repo_leaves_exactly_one_hook(repo, monkeypatch):
    assert install(repo, monkeypatch) == 0
    assert install(repo, monkeypatch) == 0

    hooks_dir = repo / ".git" / "hooks"
    assert backups(hooks_dir) == []
    assert sum(1 for p in hooks_dir.iterdir() if p.name == "pre-commit") == 1


def test_second_install_recognises_its_own_marker_and_overwrites_in_place(
    repo, monkeypatch
):
    """The mechanism idempotence depends on: a hook already carrying
    `_MARKER` is treated as ours and overwritten directly, not backed up
    as though it were foreign."""
    install(repo, monkeypatch)
    hook_path = repo / ".git" / "hooks" / "pre-commit"
    assert install_hooks_module._is_stackward_hook(hook_path) is True

    install(repo, monkeypatch)
    assert backups(repo / ".git" / "hooks") == []


# ---------------------------------------------------------------------------
# Mode 0o755, set explicitly.
# ---------------------------------------------------------------------------


def test_installed_hook_mode_is_0o755(repo, monkeypatch):
    install(repo, monkeypatch)
    hook_path = repo / ".git" / "hooks" / "pre-commit"
    mode = stat.S_IMODE(hook_path.stat().st_mode)
    assert mode == 0o755


def test_hook_mode_is_0o755_even_when_replacing_a_non_executable_foreign_hook(
    repo, monkeypatch
):
    hooks_dir = repo / ".git" / "hooks"
    hook_path = hooks_dir / "pre-commit"
    hook_path.write_text("#!/bin/sh\necho foreign\n")
    hook_path.chmod(0o644)

    install(repo, monkeypatch)

    mode = stat.S_IMODE(hook_path.stat().st_mode)
    assert mode == 0o755


# ---------------------------------------------------------------------------
# The hook body embeds an absolute path to the running executable.
# ---------------------------------------------------------------------------


def test_hook_body_contains_an_absolute_path(repo, monkeypatch):
    install(repo, monkeypatch)
    hook_path = repo / ".git" / "hooks" / "pre-commit"
    body = hook_path.read_text()

    stackward_line = next(
        line for line in body.splitlines() if line.startswith("STACKWARD=")
    )
    quoted_path = stackward_line.removeprefix("STACKWARD=")
    raw_path = quoted_path[1:-1].replace("'\\''", "'")  # undo shell single-quoting
    assert Path(raw_path).is_absolute()


def test_running_executable_returns_an_absolute_path():
    assert Path(install_hooks_module._running_executable()).is_absolute()


def test_running_executable_falls_back_to_path_when_argv0_is_not_executable(
    tmp_path, monkeypatch
):
    """`python -m stackward` sets `sys.argv[0]` to `.../stackward/__main__.py`
    -- a plain source file, with no execute bit and no shebang. Embedding
    that path verbatim would make the hook silently fall through to its
    own `PATH` fallback on *every* run, which is exactly the dependency
    this whole scheme exists to avoid relying on -- so a non-executable
    `argv[0]` must be recognised and a real `PATH` lookup used instead."""
    non_executable = tmp_path / "__main__.py"
    non_executable.write_text("# not executable\n")
    monkeypatch.setattr(sys, "argv", [str(non_executable)])

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    real_stub = fake_bin / "stackward"
    real_stub.write_text("#!/bin/sh\n")
    real_stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    assert install_hooks_module._running_executable() == str(real_stub)


def test_running_executable_falls_back_to_argv0_when_nothing_else_is_found(
    tmp_path, monkeypatch
):
    """No usable fallback exists either: better to embed the best guess
    available (which the hook's own `PATH` lookup can still recover from
    at runtime) than to raise here and block installation entirely."""
    non_executable = tmp_path / "__main__.py"
    non_executable.write_text("# not executable\n")
    monkeypatch.setattr(sys, "argv", [str(non_executable)])

    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    assert install_hooks_module._running_executable() == str(non_executable.resolve())


# ---------------------------------------------------------------------------
# The hook actually locates and execs `stackward pre-commit` -- proven by
# running the generated shell script directly, not merely inspecting text.
# ---------------------------------------------------------------------------


def _make_stub(path: Path, out_file: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"echo \"invoked with: $*\" > {out_file}\n"
        "exit 0\n"
    )
    path.chmod(0o755)


def test_hook_body_execs_the_embedded_absolute_path_with_pre_commit(tmp_path):
    stub = tmp_path / "stackward-stub"
    out_file = tmp_path / "out.txt"
    _make_stub(stub, out_file)

    hook_script = tmp_path / "generated-hook"
    hook_script.write_text(install_hooks_module._hook_body(str(stub)))
    hook_script.chmod(0o755)

    result = subprocess.run([str(hook_script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert out_file.read_text().strip() == "invoked with: pre-commit"


def test_hook_body_falls_back_to_path_when_embedded_path_is_missing(tmp_path):
    """A bare `exec stackward` from an environment with a minimal `PATH`
    is what this fallback exists to avoid -- but when the embedded path
    genuinely no longer exists (the binary moved, or this hook was
    installed by a different clone's layout), falling back to `PATH`
    keeps the hook working rather than failing with exit 127."""
    missing_path = tmp_path / "does" / "not" / "exist" / "stackward"

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    out_file = tmp_path / "out.txt"
    _make_stub(fake_bin / "stackward", out_file)

    hook_script = tmp_path / "generated-hook"
    hook_script.write_text(install_hooks_module._hook_body(str(missing_path)))
    hook_script.chmod(0o755)

    env = {"PATH": str(fake_bin)}
    result = subprocess.run(
        [str(hook_script)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0
    assert out_file.read_text().strip() == "invoked with: pre-commit"


def test_hook_body_fails_clearly_when_nothing_is_found(tmp_path):
    missing_path = tmp_path / "does" / "not" / "exist" / "stackward"
    hook_script = tmp_path / "generated-hook"
    hook_script.write_text(install_hooks_module._hook_body(str(missing_path)))
    hook_script.chmod(0o755)

    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    result = subprocess.run(
        [str(hook_script)], capture_output=True, text=True, env={"PATH": str(empty_bin)}
    )
    assert result.returncode == 1
    assert "could not find the stackward executable" in result.stderr


# ---------------------------------------------------------------------------
# The marker line: fixed, and how "ours" is told apart from "foreign".
# ---------------------------------------------------------------------------


def test_marker_is_a_whole_line_in_the_generated_body():
    body = install_hooks_module._hook_body("/usr/bin/stackward")
    assert install_hooks_module._MARKER in body.splitlines()


def test_a_hook_without_the_marker_is_not_recognised_as_ours(tmp_path):
    foreign = tmp_path / "pre-commit"
    foreign.write_text("#!/bin/sh\necho not stackward\n")
    assert install_hooks_module._is_stackward_hook(foreign) is False


# ---------------------------------------------------------------------------
# Fail-closed: not a git repository, or git itself unavailable.
# ---------------------------------------------------------------------------


def test_outside_a_git_repository_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["hooks", "install"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


def test_git_executable_missing_exits_2(repo, monkeypatch, capsys):
    real_run = subprocess.run

    def failing_run(args, **kwargs):
        if args[:1] == ["git"]:
            raise OSError("git not found (simulated)")
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_run)
    assert install(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "git not found" in captured.err


def test_git_rev_parse_failure_exits_2(repo, monkeypatch, capsys):
    real_run = subprocess.run

    def failing_run(args, **kwargs):
        if args[:3] == ["git", "rev-parse", "--git-path"]:
            return subprocess.CompletedProcess(
                args, returncode=128, stdout="", stderr="fatal: simulated failure\n"
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_run)
    assert install(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "simulated failure" in captured.err


def test_unanticipated_exception_exits_2_not_1(repo, monkeypatch, capsys):
    """Python's default exit code for an uncaught exception is 1 -- the
    code `pre-commit` reserves for "a credential was found" elsewhere in
    this tool. An unrelated bug here must not exit 1, which `@fail_closed`
    (shared with `check_config`/`pre_commit`) is what prevents."""

    def boom(_hooks_dir, _executable_path):
        raise RuntimeError("unanticipated failure")

    monkeypatch.setattr(install_hooks_module, "_install_hook", boom)
    assert install(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RuntimeError" in captured.err


# ---------------------------------------------------------------------------
# Git is invoked under a fixed locale, and the fixtures are hermetic.
# ---------------------------------------------------------------------------


def test_every_git_invocation_runs_under_a_fixed_locale(repo, monkeypatch):
    """`LC_ALL=C` on every git call, with the environment merged, not
    replaced.

    Hardening for `_run_git`, which branches on the exit code -- but a live
    bug for `_is_tracked`, which branches on git's *message text*. Both
    strings it recognises (`did not match any file`, `is outside
    repository`) are gettext-marked in git's own source, so under any
    non-English locale neither matches, `_is_tracked` raises, and `hooks
    install` exits 2 in **every** repository, not only one with a redirected
    `core.hooksPath`. The command that installs the gate stops working.

    The merge half matters as much as the override: `git` sets `GIT_DIR` and
    friends for a hook process, and a bare `env={"LC_ALL": "C"}` would drop
    them and point this command at a different repository than the one it
    was invoked for.
    """
    monkeypatch.setenv("STACKWARD_TEST_MARKER", "inherited")
    seen: list[dict[str, str] | None] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        if args and args[0] == "git":
            seen.append(kwargs.get("env"))
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    assert install(repo, monkeypatch) == 0

    assert seen, "no git invocation was recorded"
    for env in seen:
        assert env is not None, "a git invocation inherited the ambient locale"
        assert env.get("LC_ALL") == "C"
        assert env.get("STACKWARD_TEST_MARKER") == "inherited"


def test_hooks_install_survives_a_git_that_translates_its_diagnostics(
    repo, monkeypatch, capsys
):
    """The live half of the bug above, at the level where it actually bites.

    `_is_tracked` recognises "not tracked" by matching two English phrases,
    and both are gettext-marked in git's own source. Simply exporting a
    non-English `LC_ALL` here would prove nothing on a machine with no git
    translations installed -- which is most CI images, and this one -- so
    the translation is modelled at the subprocess boundary instead, exactly
    as gettext behaves: git's stderr comes back translated **unless** the
    invocation asked for the C locale.

    Deterministic everywhere, and it goes red the moment the `LC_ALL=C`
    override is dropped: `_is_tracked` then matches neither phrase, raises,
    and `hooks install` exits 2 in an ordinary repository with an ordinary
    `.git/hooks` -- the command that installs the gate, refusing to install
    it, for a reason that has nothing to do with the repository.
    """
    real_run = subprocess.run
    translations = {
        "did not match any file": "ne correspond a aucun fichier",
        "is outside repository": "est en dehors du depot",
    }

    def translating_run(args, **kwargs):
        proc = real_run(args, **kwargs)
        env = kwargs.get("env") or os.environ
        if list(args[:1]) == ["git"] and env.get("LC_ALL") != "C":
            stderr = proc.stderr
            if isinstance(stderr, str):
                for english, translated in translations.items():
                    stderr = stderr.replace(english, translated)
                return subprocess.CompletedProcess(
                    args, proc.returncode, proc.stdout, stderr
                )
        return proc

    monkeypatch.setattr(subprocess, "run", translating_run)

    assert install(repo, monkeypatch) == 0
    assert (repo / ".git" / "hooks" / "pre-commit").is_file()
    assert capsys.readouterr().err == ""


def test_the_repo_fixture_ignores_a_global_core_hookspath(repo, isolated_git):
    """The fixture's isolation, asserted rather than assumed.

    A global `core.hooksPath` -- husky, lefthook, `pre-commit`, or an
    org-wide `/etc/gitconfig` -- silently moves where git looks for hooks,
    so without isolation the tests in this file examine a directory the
    command was never pointed at, and sixty-odd tests across this file and
    `test_pre_commit.py` error out.

    The positive control is the point: it proves git *would* honour the file
    written below, so the first assertion is about the fixture suppressing
    it and not about the file being ineffective.
    """
    hostile = isolated_git / ".gitconfig"
    hostile.write_text("[core]\n\thooksPath = /nonexistent/global-hooks\n")

    def hooks_path(env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=repo,
            capture_output=True,
            text=True,
            env=env,
        )

    control = hooks_path({**os.environ, "GIT_CONFIG_GLOBAL": str(hostile)})
    assert control.stdout.strip() == "/nonexistent/global-hooks"

    isolated = hooks_path(dict(os.environ))
    assert isolated.returncode == 1
    assert isolated.stdout.strip() == ""


# ---------------------------------------------------------------------------
# No policy the *hook* will find: a warning, never a refusal.
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_without_policy(repo, monkeypatch) -> Path:
    """`repo` with its `.stackward.toml` removed from the working tree.

    `find_repo_config` reads the filesystem, not the index -- deliberately,
    since `hooks install` is not the gate and does not scan staged content
    -- so removing the file is enough to make the repository policy-less
    from this command's point of view.
    """
    (repo / ".stackward.toml").unlink()
    return repo


def test_no_policy_warns_on_stderr_and_still_installs(repo_without_policy, monkeypatch, capsys):
    """A repository that has not declared a policy yet gets told so, at the
    moment it can still be fixed in one line -- rather than at somebody's
    first blocked commit, since `pre-commit` refuses a repository that
    declares none.

    A warning, not a refusal: installing the hook before writing the policy
    is a legitimate order to do things in, and this command's exit 1 and 2
    mean specific things about the *install*, which this is not one of.
    """
    assert install(repo_without_policy, monkeypatch) == 0
    assert (repo_without_policy / ".git" / "hooks" / "pre-commit").is_file()

    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert ".stackward.toml" in captured.err
    # Names the minimal file to write, not merely that one is missing.
    for line in install_hooks_module.MINIMAL_POLICY.splitlines():
        assert line in captured.err
    # stdout stays the machine-readable "where the hook went" line.
    assert "warning" not in captured.out.lower()
    assert str(repo_without_policy / ".git" / "hooks" / "pre-commit") in captured.out


def test_a_repository_that_declares_a_policy_gets_no_warning(repo, monkeypatch, capsys):
    """The other half: the warning must not fire for the normal case, or it
    is noise everyone learns to ignore."""
    assert (repo / ".stackward.toml").is_file()
    assert install(repo, monkeypatch) == 0
    assert "warning" not in capsys.readouterr().err.lower()


@pytest.fixture
def repo_with_unstaged_policy(repo) -> Path:
    """`repo` with its `.stackward.toml` still on disk but no longer in the
    index.

    `--cached` removes the index entry and leaves the working-tree file
    alone, which is the state a repository is in between writing a policy
    and staging it -- and that is the moment somebody runs `hooks install`.
    """
    _git(repo, "rm", "-q", "--cached", ".stackward.toml")
    return repo


def test_an_unstaged_policy_warns_that_it_is_unstaged(
    repo_with_unstaged_policy, monkeypatch, capsys
):
    """The warning has to answer the question the hook will ask.

    `pre-commit` reads its policy from the index, so a file sitting
    unstaged in the working tree is no policy to it -- and this repository
    used to be told nothing at all, then block every commit. Saying
    "declares none" here would be worse than silence: it is visibly false
    to someone looking straight at the file, which is how a warning teaches
    people to stop reading warnings.
    """
    assert install(repo_with_unstaged_policy, monkeypatch) == 0
    assert (repo_with_unstaged_policy / ".git" / "hooks" / "pre-commit").is_file()

    err = capsys.readouterr().err
    assert "warning" in err.lower()
    assert "not staged" in err
    assert f"git add {install_hooks_module.CONFIG_FILENAME}" in err
    # Not the absent-policy wording: the file is right there.
    assert "declares no" not in err


def test_the_policy_warning_does_not_change_the_exit_code_of_a_refusal(
    repo_without_policy, monkeypatch, capsys
):
    """A tracked hooks directory is still exit 1, and the hook was not
    written -- so there is nothing to warn about and the refusal is not
    diluted by a second message about a different problem."""
    (repo_without_policy / "githooks").mkdir()
    (repo_without_policy / "githooks" / "pre-commit").write_text("#!/bin/sh\n")
    _git(repo_without_policy, "add", "githooks")
    _git(repo_without_policy, "commit", "-q", "-m", "commit shared hooks")
    _git(repo_without_policy, "config", "core.hooksPath", "githooks")

    assert install(repo_without_policy, monkeypatch) == 1
    assert "no .stackward.toml" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# A repository that demands a newer stackward than this one.
# ---------------------------------------------------------------------------


def test_an_unmet_min_version_floor_refuses_before_any_hook_is_written(
    repo, monkeypatch, capsys
):
    """`hooks install` is the command with the most to lose from a floor
    check that quietly gives up.

    Every other command refuses a `.stackward.toml` it cannot validate, so
    an unreported floor still costs them only the wrong *message*. This one
    reads no policy at all: with the refusal silenced it installs the hook
    and exits 0, and the repository is then gated by a binary its own
    policy says is too old to understand the policy. The floor is what
    catches that, and it has to survive the config carrying a key this
    binary does not know -- which is the shape a repository demanding a
    newer stackward actually has.

    Staged, not merely written, so that the missing-policy warning has
    nothing to say here and the only thing under test is the floor.
    """
    # Top-level keys before the `[check]` table -- after it they would be
    # members of it, and this would test something else entirely.
    (repo / ".stackward.toml").write_text(
        'min_version = "9.9.9"\n'
        'future_key = "a key only a newer stackward knows"\n'
        '[check]\nmodel_net = "none"\n'
    )
    _git(repo, "add", ".stackward.toml")

    assert install(repo, monkeypatch) == 2
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()
    assert "requires stackward >= 9.9.9" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Wiring.
# ---------------------------------------------------------------------------


def test_hooks_install_is_registered_and_dispatches(repo, monkeypatch):
    assert install(repo, monkeypatch) == 0


def test_hooks_with_no_subcommand_prints_help_and_exits_2(capsys):
    assert main(["hooks"]) == 2
    assert "usage:" in capsys.readouterr().out


def test_unexpected_argument_is_a_usage_error_exiting_2():
    with pytest.raises(SystemExit) as exit_info:
        main(["hooks", "install", "unexpected-argument"])
    assert exit_info.value.code == 2
