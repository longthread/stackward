"""Fixtures shared by the tests that drive a real `git`.

Two of them, for two reasons that are not the same. `isolated_git` (below)
belongs here because four files need it and a judgement about which git
configuration a test may see must not exist in four copies. This one
belongs here because the rule about when a missing console script is a
skip and when it is a failure is a judgement that must not drift either.

Two test files need the *installed* `stackward` console script — the one
that proves a git hook really runs, rather than that a file landed at the
expected path. The rule about when its absence is a skip and when it is a
failure is a judgement that must not exist in two places and drift: a second
copy that forgot the `CI` branch would let a CI run go green having lost
exactly the coverage these tests are for.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from stackward.commands import install_hooks as install_hooks_module


def real_stackward_executable() -> str:
    """The actual installed `stackward` console script for this test
    interpreter -- not a stub.

    Used only by the end-to-end tests, which prove the generated hook body
    really gets invoked by a real `git commit`, not merely that a file landed
    at the expected path: a file at the right path that git never actually
    runs is precisely the "silent no-op" the hooks-directory resolution exists
    to prevent, and asserting on the file alone cannot catch that failure
    mode.

    Skipping when nothing is found keeps a local run convenient (an
    interpreter not launched through this project's own `uv`-managed venv
    genuinely may not have one), but `uv sync` always produces
    `.venv/bin/stackward`, so CI must never quietly go green having lost
    exactly the coverage the real-git-repository approach leans on hardest.
    Skip locally; fail loudly under `CI`.
    """
    candidate = Path(sys.executable).parent / "stackward"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which("stackward")
    if found:
        return found
    message = "no installed `stackward` console script found for an end-to-end test"
    if os.environ.get("CI"):
        pytest.fail(f"{message} -- `uv sync` should have installed one")
    pytest.skip(message)


@pytest.fixture
def real_executable(monkeypatch) -> str:
    """Make `_install_hook` embed the real `stackward` console script instead
    of whatever `sys.argv[0]` happens to be under pytest (its own
    interpreter) -- see `real_stackward_executable`."""
    executable = real_stackward_executable()
    monkeypatch.setattr(install_hooks_module, "_running_executable", lambda: executable)
    return executable


@pytest.fixture
def isolated_git(tmp_path: Path, monkeypatch) -> Path:
    """Cut every `git` a test runs off from the machine's own git config.

    Without this, a developer (or a CI image) with a *global*
    `core.hooksPath` -- what husky, lefthook and the `pre-commit` framework
    all set, and precisely the configuration `hooks install` exists to
    handle -- makes `git init`/`git commit` inherit that hook, and dozens of
    tests across four files error out for a reason that has nothing to do
    with the code under test. A suite whose result depends on the
    developer's `~/.gitconfig` is not testing what it claims to.

    Two consequences are worse than a confusing error, and are why this is
    a fixture every git-touching test takes rather than a convention:

    * A global `core.hooksPath` silently moves where `hooks install`
      *writes*, so tests asserting on `.git/hooks/pre-commit` would be
      examining a directory the command was never pointed at.
    * It also moves where `hooks install` writes to a directory **outside
      `tmp_path`**. Reproduced on this branch: with a global `core.hooksPath`
      set, `tests/test_model_net.py` installed a stackward pre-commit hook
      into the developer's own global hooks directory, where it stayed after
      the run and then ran against every unrelated repository on the
      machine. That is an escape, not merely a failure.

    All four sources git consults are redirected, not only the global one:
    `GIT_CONFIG_SYSTEM` covers `/etc/gitconfig`, which is exactly where an
    org-wide `hooksPath` lives; `HOME` and `XDG_CONFIG_HOME` cover
    `~/.gitconfig` and `$XDG_CONFIG_HOME/git/config`, which git still reads
    on a path that leaves `GIT_CONFIG_GLOBAL` unset. `/dev/null` is git's
    own documented spelling of "this configuration file does not exist" for
    the two `GIT_CONFIG_*` variables.

    The redirect is set with `monkeypatch`, so it is still in place while
    the *test body* runs `git commit` -- not only while the fixture built
    the repository. It lives here, in `conftest.py`, because it was
    previously two identical copies in two files and two other files had
    none; a third copy is how the two that had it drift apart, and having
    none is what let the escape above happen.
    """
    home = tmp_path / "git-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    return home
