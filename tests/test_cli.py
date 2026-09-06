"""Tests for command dispatch and configuration discovery."""

from __future__ import annotations

import argparse
import ast
import inspect

import pytest

from stackward import __version__, cli
from stackward.cli import (
    VERSION_CHECK_EXEMPT,
    _import_command,
    build_parser,
    config_home,
    crypto_selftest,
    main,
)


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_doctor_runs_and_reveals_no_secrets(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out
    # doctor reports *where* things are, never what is in them.
    for forbidden in ("PASSPHRASE", "SECRET_ACCESS_KEY", "secure:"):
        assert forbidden not in out


def test_upgrade_instruction_is_not_self_blocking():
    """A repo demanding a newer stackward must not block the upgrade path.

    `doctor` is how you diagnose the problem and `self-update` is how you fix
    it; if a min_version check gated either, the tool would print an
    instruction it had just made impossible to follow.
    """
    assert {"doctor", "self-update"} <= VERSION_CHECK_EXEMPT


def test_crypto_selftest_exercises_the_backend():
    """A real round-trip, not an import check.

    The characteristic PyInstaller failure is a bundle that builds cleanly and
    then cannot load a native extension on a machine that is not the build
    machine — which an import check can still pass.
    """
    result = crypto_selftest()
    assert result.startswith("cryptography ")
    assert "Argon2id" in result and "AES-256-GCM" in result


def test_doctor_fails_when_crypto_is_unavailable(monkeypatch, capsys):
    """doctor is what a release smoke test runs, so a broken credential layer
    must make it exit non-zero rather than print a line and return success."""
    monkeypatch.setattr(
        "stackward.cli.crypto_selftest", lambda: "UNAVAILABLE: ImportError: boom"
    )
    assert main(["doctor"]) == 1
    assert "UNAVAILABLE" in capsys.readouterr().out


def test_config_home_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_home() == tmp_path / "stackward"


def test_config_home_falls_back_to_dot_config(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert config_home() == tmp_path / ".config" / "stackward"


def test_check_config_is_registered_and_dispatches(tmp_path, monkeypatch):
    """Wiring only — the rules `check-config` enforces are covered in
    tests/test_heuristic.py.

    The `.stackward.toml` and the `chdir` are what make this a dispatch
    test rather than a policy test: a repository that declares no policy is
    now refused with exit 2 before any file is opened, so without them this
    would assert the wrong exit code for a reason that has nothing to do
    with whether the subcommand is registered.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / ".stackward.toml").write_text('[check]\nmodel_net = "none"\n')
    monkeypatch.chdir(tmp_path)
    clean = tmp_path / "Pulumi.dev.yaml"
    clean.write_text("name: myproject\n")
    assert main(["check-config", str(clean)]) == 0


def test_doctor_still_works_when_repo_config_is_malformed(tmp_path, monkeypatch, capsys):
    """`doctor` is the command you run to diagnose a broken
    `.stackward.toml`, so dispatch must not load it before routing to
    `doctor` — that would make the one command that reports the problem
    unable to run at all."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".stackward.toml").write_text("this is not valid toml [[[")
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 0
    assert str(tmp_path / ".stackward.toml") in capsys.readouterr().out


def test_an_unmet_floor_is_reported_beside_a_key_this_binary_does_not_know(
    tmp_path, monkeypatch, capsys
):
    """The floor must win over the unknown key, because the unknown key is
    a *consequence* of the unmet floor.

    A repository that demands a newer stackward is very likely to be using
    something that newer stackward added, so an unrecognised top-level key
    next to an unmet `min_version` is the expected shape of this config
    rather than an exotic one -- it is forward-compatibility, which is the
    situation the floor exists to make survivable. Reporting `unknown key
    'future_key'` here names the symptom and sends the reader to delete the
    one line that would have explained what to do.

    The exit code is 2 either way (`check-config` refuses a config it
    cannot validate), so this asserts on the *message*: that is the whole
    difference, and an exit-code-only assertion would hold whether the
    refusal reads the floor first or not.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / ".stackward.toml").write_text(
        'min_version = "9.9.9"\nfuture_key = "a key only a newer stackward knows"\n'
    )
    monkeypatch.chdir(tmp_path)

    assert main(["check-config", "Pulumi.dev.yaml"]) == 2
    err = capsys.readouterr().err
    assert "requires stackward >= 9.9.9" in err
    assert "future_key" not in err


# ---------------------------------------------------------------------------
# Lazy dispatch, and the frozen bundle
# ---------------------------------------------------------------------------


def _dispatched_module_names(parser: argparse.ArgumentParser) -> set[str]:
    """Every command module reached through `cli._dispatch`, read off the
    parser rather than from a list a test keeps in step by hand.

    `_dispatch` stamps its thunk's `__qualname__` as `<module>.<function>`;
    a directly registered entry point (`cmd_doctor`, `cmd_check_config`) has
    no dot in its qualname and is therefore not one of these.
    """
    found: set[str] = set()
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for sub in action.choices.values():
            found |= _dispatched_module_names(sub)
            func = sub.get_default("func")
            qualname = getattr(func, "__qualname__", "")
            if func is not None and "." in qualname:
                found.add(qualname.partition(".")[0])
    return found


def test_every_lazily_dispatched_command_module_can_actually_be_imported():
    """A command registered through `_dispatch` but missing from
    `_import_command`'s chain would raise `RuntimeError` the first time
    anyone ran it, and nothing else in the suite would notice: every other
    test either calls an entry point directly or goes through a subcommand
    that already works.

    The membership check is a vacuity guard -- a `_dispatched_module_names`
    that silently returned nothing would satisfy the loop below against any
    implementation at all.
    """
    names = _dispatched_module_names(build_parser())
    assert {"session", "credentials"} <= names

    for name in names:
        assert _import_command(name).__name__ == f"stackward.commands.{name}"


def test_a_command_module_is_never_named_dynamically():
    """The one property no in-process test can observe, asserted against the
    module's own syntax tree instead.

    PyInstaller resolves imports by reading `import` statements out of
    bytecode; it cannot resolve a module name assembled at runtime. While
    dispatch used `importlib.import_module(f".commands.{name}", ...)`, every
    lazily dispatched command -- `login`, `exec`, `shell`, `set-secrets`,
    `sync-declared-secrets`, `check-passphrase` -- was simply absent from
    the shipped binary and died on `ModuleNotFoundError` with a traceback
    and exit code 1, the code reserved for "a credential was found". Source
    checkouts, this suite included, cannot see that: they import from a real
    filesystem where the module is there to be found.

    The tree, not the text: this module's own docstrings quote the old call
    on purpose, and a substring search over the source would fail on the
    explanation of the fix rather than on the defect.

    So this asserts the mechanism, and the release workflow's smoke step
    asserts the consequence by running a lazily dispatched command against
    the real bundle. Neither is sufficient alone: this one cannot prove the
    bundle works, and that one only runs on a release.
    """
    tree = ast.parse(inspect.getsource(cli))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "import_module" not in called
    assert "__import__" not in {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
