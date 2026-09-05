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
import shutil
import stat
import subprocess
import sys
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
def repo(tmp_path: Path) -> Path:
    """A real, minimal git repository with one initial commit."""
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "commit", "-q", "--allow-empty", "-m", "init")
    return path


def install(repo: Path, monkeypatch) -> int:
    monkeypatch.chdir(repo)
    return main(["hooks", "install"])


def backups(hooks_dir: Path) -> list[Path]:
    return sorted(hooks_dir.glob("pre-commit.backup.*"))


def _real_stackward_executable() -> str:
    """The actual installed `stackward` console script for this test
    interpreter -- not a stub. Used only by the end-to-end tests below,
    which prove the generated hook body really gets invoked by a real
    `git commit`, not merely that a file landed at the expected path: a
    file at the right path that git never actually runs is precisely the
    "silent no-op" this command's own hooks-directory resolution exists to
    prevent, and asserting on the file alone cannot catch that failure
    mode."""
    candidate = Path(sys.executable).parent / "stackward"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which("stackward")
    if found:
        return found
    pytest.skip("no installed `stackward` console script found for an end-to-end test")


@pytest.fixture
def real_executable(monkeypatch) -> str:
    """Make `_install_hook` embed the real `stackward` console script
    instead of whatever `sys.argv[0]` happens to be under pytest (its own
    interpreter) -- see `_real_stackward_executable`."""
    executable = _real_stackward_executable()
    monkeypatch.setattr(install_hooks_module, "_running_executable", lambda: executable)
    return executable


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
    elsewhere in this file. `_is_tracked` treats any non-zero exit as "not
    tracked" -- this proves that answer is still correct for this
    differently-shaped git failure, not merely right by accident of a
    single error message this suite happens to have checked."""
    external = tmp_path / "external-hooks"
    external.mkdir()
    _git(repo, "config", "core.hooksPath", str(external))

    assert install(repo, monkeypatch) == 0
    assert (external / "pre-commit").is_file()


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


def test_backup_filename_matches_hook_dot_backup_dot_timestamp(repo, monkeypatch):
    hooks_dir = repo / ".git" / "hooks"
    (hooks_dir / "pre-commit").write_text("#!/bin/sh\necho foreign\n")

    install(repo, monkeypatch)

    [backup] = backups(hooks_dir)
    assert backup.name.startswith("pre-commit.backup.")
    # The suffix is a real timestamp, not an arbitrary string.
    stamp = backup.name.removeprefix("pre-commit.backup.")
    assert stamp  # non-empty
    assert stamp[:8].isdigit()  # YYYYMMDD at minimum


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
