"""Fixtures shared by the end-to-end tests that drive a real `git commit`.

Two test files now need the *installed* `stackward` console script — the one
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
