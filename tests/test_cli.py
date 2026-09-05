"""Tests for command dispatch and configuration discovery."""

from __future__ import annotations

import pytest

from stackward import __version__
from stackward.cli import (
    VERSION_CHECK_EXEMPT,
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
