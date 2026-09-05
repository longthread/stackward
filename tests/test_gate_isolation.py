"""Global Constraint 2, tested as one property rather than three halves.

    "The gate resolves no credentials. `check-config` and `pre-commit` read
    files only. No code reachable from them may prompt, open a network
    connection, or import a credential provider."

`README.md` states the same thing under the heading "These are constraints,
not aspirations — each has a test", and adds the consequence users actually
care about: a commit "cannot be blocked by an expired session or a flaky
connection".

Two reviews found that claim under-tested in three separate ways, and this
module exists because each gap was invisible from inside the test that was
supposed to cover it:

* `tests/test_model_net.py` checked `cryptography` and `stackward.store`, but
  imported the two command modules **directly**, bypassing `cli.py` — which
  imported `commands.check_passphrase` (-> `cryptography`) and
  `commands.session` (-> `store` -> `crypto`) at its own module scope. Every
  real invocation goes through `cli.py`.
* `tests/test_providers.py` did go through `cli.main`, but grepped only for
  `stackward.providers*`, so it saw none of that.
* Nothing at all tested "never prompt" or "never touch the network".
  Inserting a real `store.resolve_credentials(...)` call into
  `cmd_check_config` survived the entire suite, and blocked on a `getpass`
  prompt at a terminal.

So the import check is one assertion over the union of every forbidden
module, taken through the dispatch path a `git commit` really takes, and the
prompting and network checks are their own tests over both gate commands.
"""

from __future__ import annotations

import getpass
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from stackward.cli import main

MINIMAL_POLICY = '[check]\nmodel_net = "none"\n'

# Every module the gate must not reach, as prefixes. `stackward.providers`
# and `cryptography` are matched with their submodules too, so a provider or
# a backend added later is covered without editing this list.
FORBIDDEN_PREFIXES = (
    "stackward.providers",
    "stackward.store",
    "stackward.crypto",
    "cryptography",
)

_PROBE = """
import sys
from stackward.cli import main
code = main(["check-config", {target!r}])
forbidden = sorted(
    module
    for module in sys.modules
    for prefix in {prefixes!r}
    if module == prefix or module.startswith(prefix + ".")
)
print("exit", code)
print("forbidden", forbidden)
"""


def test_the_gate_path_reaches_no_credential_or_provider_code(tmp_path):
    """One assertion over the union, through `cli.main`.

    A subprocess, because this test process has already imported half the
    tool through other test modules in the same session — asserting against
    its own `sys.modules` would prove nothing.

    The exit code is asserted alongside the module list on purpose: a
    `check-config` that refused before doing any work would import nothing
    either, and would pass this test while proving the opposite of what it
    claims.
    """
    (tmp_path / ".stackward.toml").write_text(MINIMAL_POLICY)
    target = tmp_path / "Pulumi.dev.yaml"
    target.write_text("name: myproject\n")

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(target=str(target), prefixes=FORBIDDEN_PREFIXES)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "exit 0" in proc.stdout
    assert "forbidden []" in proc.stdout


class GateViolation(BaseException):
    """Raised by the `sealed` fixture when the gate does something Global
    Constraint 2 forbids. See that fixture for why it is not an
    `Exception`."""


@pytest.fixture
def sealed(monkeypatch):
    """Make prompting and networking impossible, loudly.

    `GateViolation` derives from `BaseException`, not `Exception`, and that
    is the whole trick: `check_config.fail_closed` converts any `Exception`
    into `return 2`, so a control raised as an `AssertionError` inside a
    gate command is silently swallowed and the test then asserts on an exit
    code that looks like an ordinary refusal — it would pass whether the
    forbidden call happened or not. (That trap has already caught one
    reviewer on this project.)

    `KeyboardInterrupt` would also survive the boundary, but pytest treats
    it as "the human wants out" and aborts the whole session instead of
    failing the one test, which turns a precise signal into a confusing
    one. A private `BaseException` subclass survives `fail_closed` *and* is
    reported as an ordinary failure naming this test.
    """

    def refuse(reason):
        def refusing(*_args, **_kwargs):
            raise GateViolation(reason)

        return refusing

    monkeypatch.setattr(getpass, "getpass", refuse("the gate prompted for a password"))
    monkeypatch.setattr("builtins.input", refuse("the gate prompted on stdin"))
    monkeypatch.setattr(socket, "socket", refuse("the gate opened a socket"))
    monkeypatch.setattr(
        socket, "create_connection", refuse("the gate opened a connection")
    )


def test_check_config_neither_prompts_nor_connects(sealed, tmp_path, monkeypatch):
    (tmp_path / ".stackward.toml").write_text(MINIMAL_POLICY)
    target = tmp_path / "Pulumi.dev.yaml"
    target.write_text("name: myproject\n")
    monkeypatch.chdir(tmp_path)

    assert main(["check-config", str(target)]) == 0


def test_check_config_neither_prompts_nor_connects_on_a_finding(
    sealed, tmp_path, monkeypatch
):
    """The finding path, not only the clean one: reporting a credential is
    where a credential-resolving call would most plausibly be added."""
    (tmp_path / ".stackward.toml").write_text(MINIMAL_POLICY)
    target = tmp_path / "Pulumi.dev.yaml"
    target.write_text("config:\n  myproject:dbPassword: hunter2\n")
    monkeypatch.chdir(tmp_path)

    assert main(["check-config", str(target)]) == 1


def _repo_with_policy(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / ".stackward.toml").write_text(MINIMAL_POLICY)
    subprocess.run(
        ["git", "add", ".stackward.toml"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, capture_output=True
    )
    return repo


def test_pre_commit_neither_prompts_nor_connects(sealed, tmp_path, monkeypatch):
    """The half that matters most: `pre-commit` is the command a `git
    commit` runs, so a prompt here blocks the commit on a terminal and hangs
    it everywhere else."""
    repo = _repo_with_policy(tmp_path)
    (repo / "Pulumi.dev.yaml").write_text("name: myproject\n")
    subprocess.run(
        ["git", "add", "Pulumi.dev.yaml"], cwd=repo, check=True, capture_output=True
    )
    monkeypatch.chdir(repo)

    assert main(["pre-commit"]) == 0


def test_pre_commit_neither_prompts_nor_connects_on_a_finding(
    sealed, tmp_path, monkeypatch
):
    repo = _repo_with_policy(tmp_path)
    (repo / "Pulumi.dev.yaml").write_text("config:\n  myproject:dbPassword: hunter2\n")
    subprocess.run(
        ["git", "add", "Pulumi.dev.yaml"], cwd=repo, check=True, capture_output=True
    )
    monkeypatch.chdir(repo)

    assert main(["pre-commit"]) == 1
