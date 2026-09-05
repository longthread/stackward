"""Tests for `.stackward.toml` discovery, parsing and validation.

This is the seam that keeps `stackward` generic: every fact a repository
wants applied — what counts as sensitive, which stack models declare
secrets, the oldest release its policy works with — arrives only through
this file. Absent means "no policy" (`find_repo_config` returns `None`);
present-but-invalid must raise, never fall back to a default.
"""

from __future__ import annotations

import pytest

from stackward.config import (
    BUILTIN_SENSITIVE_KEYS,
    Config,
    ConfigError,
    MinVersionError,
    _extend,
    _replace,
    enforce_min_version,
    find_repo_config,
    load_config,
)


def write_config(tmp_path, text: str):
    path = tmp_path / ".stackward.toml"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# find_repo_config — moved here from tests/test_cli.py along with the
# function itself (Task 2 brief).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Each recognised key, parsed.
# ---------------------------------------------------------------------------


def test_profile_key_is_parsed(tmp_path):
    config = load_config(write_config(tmp_path, 'profile = "staging"\n'))
    assert config.profile == "staging"


def test_python_key_is_parsed(tmp_path):
    config = load_config(write_config(tmp_path, 'python = ".venv/bin/python"\n'))
    assert config.python == ".venv/bin/python"


def test_min_version_key_is_parsed(tmp_path):
    config = load_config(write_config(tmp_path, 'min_version = "0.4.0"\n'))
    assert config.min_version == "0.4.0"


def test_check_model_net_is_parsed(tmp_path):
    config = load_config(write_config(tmp_path, '[check]\nmodel_net = "artifact"\n'))
    assert config.check.model_net == "artifact"


def test_check_model_net_defaults_to_none(tmp_path):
    config = load_config(write_config(tmp_path, "[check]\n"))
    assert config.check.model_net == "none"


def test_check_stack_models_is_parsed(tmp_path):
    config = load_config(
        write_config(
            tmp_path,
            '[check.stack_models]\n"infra" = "infra.config:StackConfig"\n',
        )
    )
    assert config.check.stack_models == {"infra": "infra.config:StackConfig"}


def test_check_sensitive_keys_is_parsed_and_extends_the_builtins(tmp_path):
    config = load_config(
        write_config(tmp_path, '[check]\nsensitive_keys = ["jwtsecret"]\n')
    )
    # Both the declared addition and a built-in survive — extension, not
    # replacement.
    assert "jwtsecret" in config.check.sensitive_keys
    assert "password" in config.check.sensitive_keys


def test_check_sensitive_parents_is_parsed(tmp_path):
    config = load_config(
        write_config(tmp_path, '[check]\nsensitive_parents = ["environment_variables"]\n')
    )
    assert config.check.sensitive_parents == frozenset({"environment_variables"})


def test_check_allowed_references_is_parsed(tmp_path):
    config = load_config(
        write_config(tmp_path, '[check]\nallowed_references = ["a.other_secret"]\n')
    )
    assert config.check.allowed_references == ["a.other_secret"]


def test_secrets_project_table_is_parsed(tmp_path):
    config = load_config(
        write_config(
            tmp_path,
            '[secrets."infra"]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
        )
    )
    assert config.secrets["infra"]["secret"] == {"db.password": "DB_PASSWORD"}


def test_secrets_source_table_is_parsed(tmp_path):
    config = load_config(
        write_config(
            tmp_path,
            '[secrets.source]\nfiles = [".env", ".env.{stack}.local"]\n',
        )
    )
    assert config.secrets["source"]["files"] == [".env", ".env.{stack}.local"]


# ---------------------------------------------------------------------------
# Unknown keys are loud, at every level this module validates.
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_errors_naming_the_key(tmp_path):
    with pytest.raises(ConfigError) as exc_info:
        load_config(write_config(tmp_path, 'bogus = "oops"\n'))
    assert "bogus" in str(exc_info.value)


def test_unknown_check_key_errors_naming_the_key(tmp_path):
    """`[check]`'s keys are fully enumerated in the brief, so a typo there
    must be as loud as an unknown top-level key — this module does not stop
    validating strictly the moment it descends one table."""
    with pytest.raises(ConfigError) as exc_info:
        load_config(write_config(tmp_path, '[check]\nbogus = "oops"\n'))
    assert "bogus" in str(exc_info.value)


# ---------------------------------------------------------------------------
# The extend-vs-replace asymmetry, proven for both lists.
#
# `sensitive_parents`' shipped built-in default is the empty set (see
# `BUILTIN_SENSITIVE_PARENTS` in config.py), so a load_config-level test
# alone cannot distinguish "replaced an empty set" from "extended an empty
# set" — both produce the same result. `_extend`/`_replace` are tested
# directly, against a synthetic non-empty `builtin`, to prove the mechanism
# itself rather than relying on the shipped default happening to be
# non-empty.
# ---------------------------------------------------------------------------


def test_extend_unions_declared_with_builtin():
    builtin = frozenset({"password"})
    assert _extend(builtin, ["apikey"]) == frozenset({"password", "apikey"})


def test_extend_returns_builtin_unchanged_when_nothing_declared():
    builtin = frozenset({"password"})
    assert _extend(builtin, None) == builtin


def test_replace_discards_the_builtin_entirely():
    builtin = frozenset({"environment_variables"})
    result = _replace(builtin, ["only_this_one"])
    assert result == frozenset({"only_this_one"})
    assert "environment_variables" not in result


def test_replace_returns_builtin_unchanged_when_nothing_declared():
    builtin = frozenset({"environment_variables"})
    assert _replace(builtin, None) == builtin


def test_sensitive_keys_extension_survives_through_load_config(tmp_path):
    """End-to-end proof, against the real (non-empty) built-in default: a
    declared sensitive_keys list adds to BUILTIN_SENSITIVE_KEYS rather than
    supplanting it."""
    config = load_config(
        write_config(tmp_path, '[check]\nsensitive_keys = ["custom_marker"]\n')
    )
    assert BUILTIN_SENSITIVE_KEYS <= config.check.sensitive_keys
    assert "custom_marker" in config.check.sensitive_keys


# ---------------------------------------------------------------------------
# min_version enforcement.
# ---------------------------------------------------------------------------


def test_min_version_accepts_an_equal_installed_version():
    config = Config(min_version="0.4.0")
    enforce_min_version(config, "check-config", installed="0.4.0")  # must not raise


def test_min_version_accepts_a_higher_installed_version():
    config = Config(min_version="0.4.0")
    enforce_min_version(config, "check-config", installed="0.5.0")  # must not raise


def test_min_version_rejects_a_lower_installed_version():
    config = Config(min_version="0.4.0")
    with pytest.raises(MinVersionError) as exc_info:
        enforce_min_version(config, "check-config", installed="0.3.0")
    message = str(exc_info.value)
    assert "0.4.0" in message
    assert "self-update" in message


def test_min_version_compares_numerically_not_lexicographically():
    """"0.10.0" > "0.9.0" is False under string ordering but True
    numerically — string comparison would wrongly block here."""
    config = Config(min_version="0.9.0")
    enforce_min_version(config, "check-config", installed="0.10.0")  # must not raise


def test_min_version_numeric_comparison_rejects_when_genuinely_lower():
    config = Config(min_version="0.10.0")
    with pytest.raises(MinVersionError):
        enforce_min_version(config, "check-config", installed="0.9.0")


def test_min_version_with_fewer_components_than_installed_still_compares():
    config = Config(min_version="0.2")
    with pytest.raises(MinVersionError):
        enforce_min_version(config, "check-config", installed="0.1.1")


def test_min_version_absent_never_raises():
    config = Config(min_version=None)
    enforce_min_version(config, "check-config", installed="0.0.1")  # must not raise


def test_min_version_config_none_never_raises():
    enforce_min_version(None, "check-config", installed="0.0.1")  # must not raise


def test_doctor_is_exempt_from_min_version():
    config = Config(min_version="99.0.0")
    enforce_min_version(config, "doctor", installed="0.1.1")  # must not raise


def test_self_update_is_exempt_from_min_version():
    config = Config(min_version="99.0.0")
    enforce_min_version(config, "self-update", installed="0.1.1")  # must not raise


# ---------------------------------------------------------------------------
# General fail-closed behaviour: present-but-invalid always raises, never
# falls back to a default.
# ---------------------------------------------------------------------------


def test_malformed_toml_raises_naming_the_path(tmp_path):
    path = write_config(tmp_path, "this is not valid toml [[[\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert str(path) in str(exc_info.value)


def test_invalid_model_net_value_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, '[check]\nmodel_net = "bogus"\n'))


def test_malformed_min_version_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, 'min_version = "not-a-version"\n'))


def test_secrets_value_that_is_not_a_table_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, 'secrets = "oops"\n'))


def test_secrets_member_that_is_not_a_table_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, '[secrets]\ninfra = "oops"\n'))
