"""Tests for `.stackward.toml` discovery, parsing and validation.

This is the seam that keeps `stackward` generic: every fact a repository
wants applied — what counts as sensitive, which stack models declare
secrets, the oldest release its policy works with — arrives only through
this file. Absent means "no policy" (`find_repo_config` returns `None`);
present-but-invalid must raise, never fall back to a default.
"""

from __future__ import annotations

import tomllib

import pytest

from leakcheck import assert_no_leak

from stackward import __version__
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
    load_config_text,
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


def test_check_stack_models_value_with_no_colon_raises(tmp_path):
    """The brief states the `"module:Class"` shape explicitly (unlike
    `[secrets.*]`'s interior), so a dot-instead-of-colon typo must be
    rejected here rather than surfacing later as an ImportError in Task 10."""
    with pytest.raises(ConfigError) as exc_info:
        load_config(
            write_config(
                tmp_path,
                '[check.stack_models]\n"infra" = "infra.config.StackConfig"\n',
            )
        )
    assert "infra" in str(exc_info.value)


def test_check_stack_models_value_with_two_colons_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(
            write_config(
                tmp_path,
                '[check.stack_models]\n"infra" = "infra:config:StackConfig"\n',
            )
        )


def test_check_stack_models_value_with_empty_module_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(
            write_config(tmp_path, '[check.stack_models]\n"infra" = ":StackConfig"\n')
        )


def test_check_stack_models_value_with_empty_class_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(
            write_config(tmp_path, '[check.stack_models]\n"infra" = "infra.config:"\n')
        )


def test_check_declared_paths_fn_defaults_to_none(tmp_path):
    """Absent means this tool's own marking convention, not a disabled net."""
    config = load_config(write_config(tmp_path, "[check]\n"))
    assert config.check.declared_paths_fn is None


def test_check_declared_paths_fn_is_parsed(tmp_path):
    config = load_config(
        write_config(
            tmp_path,
            '[check]\ndeclared_paths_fn = "conventions.marks:secret_fields"\n',
        )
    )
    assert config.check.declared_paths_fn == "conventions.marks:secret_fields"


def test_check_declared_paths_fn_with_no_colon_raises(tmp_path):
    """Same shape as `stack_models`, rejected here for the same reason: a
    malformed value would otherwise surface as a confusing ImportError deep
    inside the generator subprocess, far from the file that caused it."""
    with pytest.raises(ConfigError) as exc_info:
        load_config(
            write_config(
                tmp_path,
                '[check]\ndeclared_paths_fn = "conventions.marks.secret_fields"\n',
            )
        )
    assert "declared_paths_fn" in str(exc_info.value)


def test_check_declared_paths_fn_must_be_a_string(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, "[check]\ndeclared_paths_fn = 3\n"))


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
        write_config(tmp_path, '[check]\nsensitive_parents = ["example_parent"]\n')
    )
    assert config.check.sensitive_parents == frozenset({"example_parent"})


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
    builtin = frozenset({"example_parent"})
    result = _replace(builtin, ["only_this_one"])
    assert result == frozenset({"only_this_one"})
    assert "example_parent" not in result


def test_replace_returns_builtin_unchanged_when_nothing_declared():
    builtin = frozenset({"example_parent"})
    assert _replace(builtin, None) == builtin


def test_sensitive_parents_replacement_survives_through_load_config(tmp_path, monkeypatch):
    """End-to-end proof that `_build_check`'s call site actually wires
    `_replace` (not `_extend`) to `sensitive_parents`.

    The shipped `BUILTIN_SENSITIVE_PARENTS` is empty, so a test against the
    real default can't tell the two apart: extending or replacing an empty
    set both yield exactly the declared list. Monkeypatching the builtin to
    a non-empty synthetic value makes the two operations diverge, so this
    proves the wiring itself, not just the `_extend`/`_replace` helpers in
    isolation — mirrors
    `test_sensitive_keys_extension_survives_through_load_config` below,
    which gets this proof for free against the real (non-empty)
    `sensitive_keys` builtin.
    """
    monkeypatch.setattr(
        "stackward.config.BUILTIN_SENSITIVE_PARENTS", frozenset({"builtin_only"})
    )
    config = load_config(
        write_config(tmp_path, '[check]\nsensitive_parents = ["declared_only"]\n')
    )
    assert config.check.sensitive_parents == frozenset({"declared_only"})
    assert "builtin_only" not in config.check.sensitive_parents


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
    # The instruction has to name something that exists. `stackward
    # self-update` does not: `cli.build_parser` registers no such
    # subcommand, so the one message this feature prints used to end by
    # telling the reader to run a command that would exit 2 as a usage
    # error. `install.sh` is the upgrade path this project actually ships.
    assert "install.sh" in message
    assert "self-update" not in message


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


# ---------------------------------------------------------------------------
# Branches that had no test at all: each `raise`/`read` below survived the
# whole suite when mutated into a silent default, which is the exact failure
# shape this module's docstring says must not exist ("no third state where an
# invalid file quietly falls back to defaults").
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["sensitive_keys", "sensitive_parents", "allowed_references"]
)
def test_a_list_valued_check_key_of_the_wrong_type_raises(tmp_path, key):
    """`_require_str_list`'s rejection. Mutating its `raise` to `return
    None` made every one of these load as "key absent" — which for
    `sensitive_keys` means the built-ins alone, and for `sensitive_parents`
    means the empty default: a policy that declared a whole sensitive
    category, mistyped, would silently protect nothing."""
    with pytest.raises(ConfigError) as exc_info:
        load_config(write_config(tmp_path, f"[check]\n{key} = 5\n"))
    assert f"check.{key}" in str(exc_info.value)


def test_a_list_valued_check_key_containing_a_non_string_raises(tmp_path):
    """The `all(isinstance(...))` half of the same guard: a list is not
    enough, its members have to be names."""
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, '[check]\nsensitive_keys = ["ok", 5]\n'))


def test_a_check_table_that_is_not_a_table_raises(tmp_path):
    """`_build_check`'s type guard. Mutating it to `raw = {}` made
    `check = "oops"` load as an empty `[check]` table — every declared key
    silently discarded, including `model_net`."""
    with pytest.raises(ConfigError) as exc_info:
        load_config(write_config(tmp_path, 'check = "oops"\n'))
    assert "check" in str(exc_info.value)


def test_an_unreadable_config_raises_rather_than_defaulting(tmp_path):
    """`load_config`'s `OSError` branch. Mutating it to `return Config()`
    turned a config the tool could not read into a config that said
    nothing — the single worst outcome available, since `Config()` carries
    `model_net = "none"`."""
    unreadable = tmp_path / ".stackward.toml"
    unreadable.mkdir()  # a directory where a file is expected
    with pytest.raises(ConfigError) as exc_info:
        load_config(unreadable)
    assert "cannot read" in str(exc_info.value)


def test_a_config_that_is_not_utf8_raises_rather_than_defaulting(tmp_path):
    """TOML is defined as UTF-8, so bytes that are not are a broken config
    file, not an absent one. `Path.read_text` raises `UnicodeDecodeError`,
    which is a `ValueError` and not an `OSError` — caught separately, or it
    would escape `load_config` as something no caller expects."""
    path = tmp_path / ".stackward.toml"
    path.write_bytes(b'profile = "\xff\xfe"\n')
    with pytest.raises(ConfigError):
        load_config(path)


# ---------------------------------------------------------------------------
# Parsing separated from reading: what `commands.pre_commit` needs to load
# the policy the git index holds rather than the one on disk.
# ---------------------------------------------------------------------------


def test_load_config_text_parses_without_touching_the_filesystem():
    config = load_config_text('[check]\nmodel_net = "artifact"\n', ":.stackward.toml")
    assert config.check.model_net == "artifact"


def test_load_config_text_names_its_origin_in_an_error():
    """The origin label is how a person reproduces the failure: for the
    index reader it is literally the argument to `git show`."""
    with pytest.raises(ConfigError) as exc_info:
        load_config_text("this is not valid toml [[[", ":.stackward.toml")
    assert ":.stackward.toml" in str(exc_info.value)


# ---------------------------------------------------------------------------
# A parse failure reports a position, never the parser's own message.
#
# `tomllib` quotes document text in some of its messages, and this call site
# interpolated `exc` in full on the strength of a claim about what
# `.stackward.toml` is allowed to contain -- which is exactly the assumption
# a file that failed to parse has already broken. `store.py` fixed the same
# defect for the store's own `config`; both now go through
# `config.toml_position`.
# ---------------------------------------------------------------------------

# Opaque -- see `tests/leakcheck.py`. The messages under test print the
# origin label and the words "invalid TOML", and a marker built out of real
# words would share an eight-character run with them.
PASTED_INTO_THE_POLICY = "Xr4Nb8Kw2Vd6Ty9Qm3Zs7Fj"


def test_a_duplicate_declaration_is_reported_by_position_and_never_quoted_back():
    """`tomllib` really does read a document's own text back.

    Not hypothetical, and not reachable through the malformed-syntax tests
    above: nearly every `TOMLDecodeError` is a fixed description plus a
    coordinate, and only a handful name something from the file. A
    re-declared table is the one this project has actually confirmed --
    `Cannot declare ('a',) twice` -- so it is the shape worth pinning.
    """
    text = (
        f'[secrets."{PASTED_INTO_THE_POLICY}"]\n'
        f'[secrets."{PASTED_INTO_THE_POLICY}"]\n'
    )
    # Guard: if a future tomllib stops quoting the key, this test would pass
    # against the unfixed call site too, and would need to be rewritten
    # rather than quietly kept.
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        assert PASTED_INTO_THE_POLICY in str(exc), "tomllib no longer quotes the key"

    with pytest.raises(ConfigError) as exc_info:
        load_config_text(text, ":.stackward.toml")

    message = str(exc_info.value)
    assert_no_leak(message, PASTED_INTO_THE_POLICY, what="text from the policy file")
    assert message.startswith(":.stackward.toml: invalid TOML")
    assert "line 2" in message


def test_a_parser_message_is_reduced_to_its_coordinate(monkeypatch):
    """The general rule, independent of which messages this `tomllib` happens
    to produce: whatever the parser says, only the coordinate survives."""

    def raise_quoting_the_document(*_args, **_kwargs):
        raise tomllib.TOMLDecodeError(
            f"Cannot declare ('{PASTED_INTO_THE_POLICY}',) twice "
            "(at line 3, column 3)"
        )

    monkeypatch.setattr(tomllib, "loads", raise_quoting_the_document)
    with pytest.raises(ConfigError) as exc_info:
        load_config_text("irrelevant", "<memory>")

    message = str(exc_info.value)
    assert_no_leak(message, PASTED_INTO_THE_POLICY, what="the parser's own message")
    assert message == "<memory>: invalid TOML (at line 3, column 3)"


def test_a_message_with_no_coordinate_loses_the_coordinate_not_the_secrecy(monkeypatch):
    """The fallback must degrade to *no position*, never to the whole
    message -- which is what this call site used to do unconditionally.
    `TOMLDecodeError`'s text is not an API and this project supports three
    Python versions, so an unrecognised shape has to be survivable."""

    def raise_an_unparseable_shape(*_args, **_kwargs):
        raise tomllib.TOMLDecodeError(
            f"a shape from some future release: {PASTED_INTO_THE_POLICY}"
        )

    monkeypatch.setattr(tomllib, "loads", raise_an_unparseable_shape)
    with pytest.raises(ConfigError) as exc_info:
        load_config_text("irrelevant", "<memory>")

    message = str(exc_info.value)
    assert_no_leak(message, PASTED_INTO_THE_POLICY, what="the parser's own message")
    assert message == "<memory>: invalid TOML"


def test_load_config_applies_the_same_validation_as_load_config_text(tmp_path):
    """The file reader must not be a second, drifting implementation: it is
    a reader in front of the same validator, so an invalid key is rejected
    identically whichever entry point saw it."""
    text = '[check]\nmodel_net = "bogus"\n'
    with pytest.raises(ConfigError):
        load_config_text(text, "<memory>")
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, text))


# ---------------------------------------------------------------------------
# `min_version` reaches a real invocation.
#
# The floor was built here and never wired: `enforce_min_version` had zero
# production call sites, and `cli.main` dispatched straight to `args.func`.
# The plan states Task 2's goal as "Load and validate `.stackward.toml`, and
# enforce `min_version`" — the second half of which no command performed.
# These tests go through `cli.main`, because that is the only place the
# wiring exists and a unit test of `enforce_min_version` cannot see it.
# ---------------------------------------------------------------------------


def _repo_declaring(tmp_path, monkeypatch, text: str):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".stackward.toml").write_text(text)
    (tmp_path / "Pulumi.dev.yaml").write_text("name: myproject\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "Pulumi.dev.yaml"


def test_main_refuses_a_command_below_the_repositorys_min_version(
    tmp_path, monkeypatch, capsys
):
    """A repository whose policy needs a newer `stackward` than the one
    installed must not have that policy interpreted by this one. Exit 2 —
    could-not-run, never 1 — and the message names the required version and
    how to upgrade."""
    from stackward.cli import main as cli_main

    target = _repo_declaring(
        tmp_path, monkeypatch, 'min_version = "99.0.0"\n[check]\nmodel_net = "none"\n'
    )
    assert cli_main(["check-config", str(target)]) == 2
    err = capsys.readouterr().err
    assert "99.0.0" in err
    assert "install.sh" in err


def test_main_dispatches_normally_when_the_floor_is_met(tmp_path, monkeypatch, capsys):
    """The other half: a satisfied floor must be invisible. Without this, the
    test above would pass just as well if every command exited 2."""
    from stackward.cli import main as cli_main

    target = _repo_declaring(
        tmp_path, monkeypatch, 'min_version = "0.0.1"\n[check]\nmodel_net = "none"\n'
    )
    assert cli_main(["check-config", str(target)]) == 0
    assert capsys.readouterr().out == ""


def test_doctor_still_runs_under_an_unmet_min_version(tmp_path, monkeypatch, capsys):
    """`VERSION_CHECK_EXEMPT` reaching a real invocation. `doctor` is how you
    diagnose the problem; a floor that blocked it would print an instruction
    from a command it had just made unreachable."""
    from stackward.cli import main as cli_main

    _repo_declaring(tmp_path, monkeypatch, 'min_version = "99.0.0"\n')
    assert cli_main(["doctor"]) == 0
    assert __version__ in capsys.readouterr().out


def test_a_malformed_config_does_not_break_dispatch(tmp_path, monkeypatch, capsys):
    """Task 2's carry-forward ruling, re-pinned against the new call site:
    the floor check runs before every command, so a `.stackward.toml` that
    will not parse must make it stand aside rather than take `doctor` down
    with it. The gate commands still refuse — with their own, better
    message — because they load the same file themselves."""
    from stackward.cli import main as cli_main

    (tmp_path / ".git").mkdir()
    (tmp_path / ".stackward.toml").write_text("this is not valid toml [[[")
    monkeypatch.chdir(tmp_path)
    assert cli_main(["doctor"]) == 0
    assert __version__ in capsys.readouterr().out
