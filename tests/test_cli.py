"""Tests for command dispatch and configuration discovery."""

from __future__ import annotations

import pytest

from stackward import __version__
from stackward.cli import (
    VERSION_CHECK_EXEMPT,
    config_home,
    find_repo_config,
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


def test_config_home_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_home() == tmp_path / "stackward"


def test_config_home_falls_back_to_dot_config(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert config_home() == tmp_path / ".config" / "stackward"


def test_repo_config_found_from_a_subdirectory(tmp_path):
    (tmp_path / ".git").mkdir()
    config = tmp_path / ".stackward.toml"
    config.write_text("")
    nested = tmp_path / "deploy" / "nested"
    nested.mkdir(parents=True)
    assert find_repo_config(nested) == config


def test_repo_config_absent_returns_none_rather_than_guessing(tmp_path):
    """Refusing beats defaulting: a tool that guesses which backend it is
    pointed at can publish a credential to the wrong place."""
    (tmp_path / ".git").mkdir()
    assert find_repo_config(tmp_path) is None


def test_search_stops_at_the_repo_root(tmp_path):
    """A config outside the repository must not be picked up — it would make
    behaviour depend on where the repo happens to be checked out."""
    (tmp_path / ".stackward.toml").write_text("")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    assert find_repo_config(repo) is None
