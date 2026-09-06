"""Tests for `set-secrets`.

Two things every test in this file is careful about, for the reasons
`test_session.py` states them:

*No assertion depends on a credential value appearing in captured output.*
Every place a real value could plausibly leak — a successful publish, a
failed one, a drift mismatch, a `pulumi` failure that echoes something odd on
its own stdout/stderr — has a test that asserts the value's absence from
`capfd`-level output (real child process writes included, not only this
process's own `print` calls), never a test that merely checks the call
succeeded.

*Several tests exist to fail against a plausible wrong implementation, not
merely to exercise a right one.* The stdin/argv test uses a real child
process that dumps its own `sys.argv` and `sys.stdin` to separate files,
rather than inspecting a mocked `subprocess.run` call, because asserting on
the `input=` keyword only proves the call was *constructed* with the value on
stdin, not that a real child process could actually read it there. The
timeout test puts the failing entry in the *middle* of three, because
checking only the exit code and the summary text would pass an
implementation that aborts on the first failure, provided that failure
happened to be the last entry.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from stackward.cli import main
from stackward.commands import set_secrets
from stackward.commands.set_secrets import (
    DriftPair,
    LocalSecretSource,
    ProjectManifest,
    SetSecretsError,
    _config_path_for_name,
    _parse_drift_pairs,
    _parse_required,
    _publish_entries,
    parse_env_file,
    parse_project_manifest,
    resolve_source_files,
    select_project,
    substitute_stack,
)
from stackward.config import Config, find_repo_config, load_config

from leakcheck import assert_no_leak

# Placeholder values only -- see GC4 in the task brief.
#
# They read as noise on purpose. The leak assertions below go through
# `leakcheck.assert_no_leak`, which fails on *any* eight-character run of the
# value as well as on the whole string, so a marker sharing an eight-character
# run with something the command legitimately prints -- a config path, a
# logical name, a `tmp_path` component -- would fail on output that disclosed
# nothing. The previous spelling, `marker-db-password-...`, shared the run
# `password` with the config path `db.password` that every one of these tests
# prints, which is exactly that false positive; and a leak check that cries
# wolf is a leak check that gets deleted.
#
# No two of these share an eight-character run with each other either, so a
# test asserting one marker's absence cannot fire on another marker's
# legitimate presence.
MARKER_DB_PASSWORD = "qz7m4x-1f8b3d9k-6t2v5r0w"
MARKER_PUBLISHED = "hj9c5n-2p6y8s4g-7l1e3a0u"
MARKER_BOOTSTRAP = "wd3r7v-5k9z1q6h-8n4j2c0m"
# The value a `plaintext`-table entry carries. Spelled opaquely for the same
# reason as the markers above: the old literal shared the run `plaintext` with
# the `--plaintext` flag the same argv legitimately carries.
MARKER_PLAINTEXT = "np4b8k-2z6r1x9d-5c7h3j0t"

# One per parameter case of the stdin/argv test below, so a leak under one
# invocation shape cannot be excused by a match against another shape's value.
# Opaque rather than descriptive for the reason above: a marker spelling out
# its own case (`case-plaintext-no-stack-...`) shares the run `plaintext` with
# the `--plaintext` flag that same argv legitimately carries.
ARGV_CASE_MARKERS = {
    ("secret", "no-stack"): "b4k9w2-7t5m1x8r",
    ("secret", "with-stack"): "c6p3z8-2h9v4n1s",
    ("plaintext", "no-stack"): "d1r7y5-8j2c6q0f",
    ("plaintext", "with-stack"): "e9s2u4-3g7l5b1k",
}


# ---------------------------------------------------------------------------
# A real `pulumi` stand-in, driven by environment variables -- never a mock
# of `subprocess.run`, so the stdin/argv test proves a real child process can
# read what this module sent it.
# ---------------------------------------------------------------------------

_PULUMI_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys, time

    argv = sys.argv[1:]


    def arg_after(flag):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return None


    dump_argv_to = os.environ.get("STUB_DUMP_ARGV_TO")
    if dump_argv_to:
        with open(dump_argv_to, "a") as f:
            f.write(json.dumps(argv) + "\\n")

    path = arg_after("--path")

    if "set" in argv:
        data = sys.stdin.buffer.read()
        dump_stdin_to = os.environ.get("STUB_DUMP_STDIN_TO")
        if dump_stdin_to:
            with open(dump_stdin_to, "ab") as f:
                f.write(data + b"\\n")
        slow_path = os.environ.get("STUB_SLOW_PATH")
        if slow_path and path == slow_path:
            time.sleep(float(os.environ.get("STUB_SLEEP_SECONDS", "5")))
        echo = os.environ.get("STUB_SET_ECHO")
        if echo:
            sys.stderr.write(echo)
        sys.exit(int(os.environ.get("STUB_SET_EXIT_CODE", "0")))

    if "get" in argv:
        table = json.loads(os.environ.get("STUB_GET_VALUES", "{}"))
        exit_code = int(os.environ.get("STUB_GET_EXIT_CODE", "0"))
        if path in table:
            # A real `pulumi config get` terminates its output with a
            # trailing newline -- reproduced here on purpose, so a
            # comparison that forgot to strip it would fire on every equal
            # pair instead of staying silent.
            sys.stdout.write(table[path] + "\\n")
        sys.exit(exit_code)

    sys.exit(1)
    """
)


@pytest.fixture
def stub_pulumi(tmp_path) -> Path:
    path = tmp_path / "pulumi-stub.py"
    path.write_text(_PULUMI_STUB)
    path.chmod(0o755)
    return path


def fake_pulumi(monkeypatch, executable: Path | None) -> None:
    """Make `shutil.which("pulumi")` resolve to `executable`, or to nothing
    at all when `executable` is `None` -- the seam every end-to-end test in
    this file uses in place of the real binary."""
    monkeypatch.setattr(
        set_secrets.shutil,
        "which",
        lambda name: str(executable) if (executable and name == "pulumi") else None,
    )


@pytest.fixture(autouse=True)
def fast_timeout(monkeypatch):
    """A short default timeout for every test in this file -- the real
    default (`DEFAULT_TIMEOUT_SECONDS = 30.0`) would make the timeout test
    below take thirty real seconds for no benefit."""
    monkeypatch.setattr(set_secrets, "DEFAULT_TIMEOUT_SECONDS", 0.3)


# ---------------------------------------------------------------------------
# A repo tree `find_repo_config` will search from -- mirrors
# `test_session.py`'s own `repo` fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    return root


def write_repo_config(repo: Path, text: str) -> Path:
    path = repo / ".stackward.toml"
    path.write_text(text)
    return path


def load_repo_config(repo: Path) -> Config:
    path = find_repo_config(repo)
    assert path is not None
    return load_config(path)


def recorded_pulumi_calls(argv_dump: Path, *values: str) -> list[list[str]]:
    """Every `pulumi` invocation the stub recorded -- having first asserted
    that none of `values` appears in any argument of any of them.

    Global Constraint 6 ("values reach external commands on stdin, never
    argv") used to be proved for exactly one invocation shape: one test, one
    `secret` entry, no `--stack`. `--stack` is the production shape for any
    real rotation, and `plaintext` goes through the same
    `_pulumi_config_set` code path, so appending the value to argv under
    either of those conditions passed the whole suite. Every test in this
    file that dumps argv now reads it back through this function, so the
    absence assertion cannot be forgotten by the next test that dumps argv
    for some unrelated reason.

    `assert_no_leak`, not `value not in arg`: a truncated or re-encoded value
    in `ps` output discloses the credential just as completely as the whole
    string does. Checked against the joined argv as well as each argument
    separately -- the join is what would catch a value split across two
    arguments, which `ps` renders back as one line anyway.
    """
    calls = [json.loads(line) for line in argv_dump.read_text().splitlines()]
    assert calls, "no pulumi invocation was recorded"
    for call in calls:
        for value in values:
            assert_no_leak(" ".join(call), value, what="the resolved value")
            for arg in call:
                assert_no_leak(arg, value, what="the resolved value")
    return calls


# ---------------------------------------------------------------------------
# `parse_env_file`
# ---------------------------------------------------------------------------


def test_parse_env_file_ignores_blanks_and_comments(tmp_path):
    path = tmp_path / ".env"
    path.write_text("\n# a comment\n\nDB_PASSWORD=" + MARKER_DB_PASSWORD + "\n")
    path.chmod(0o600)
    assert parse_env_file(path) == {"DB_PASSWORD": MARKER_DB_PASSWORD}


def test_parse_env_file_strips_leading_export(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f"export DB_PASSWORD={MARKER_DB_PASSWORD}\n")
    path.chmod(0o600)
    assert parse_env_file(path) == {"DB_PASSWORD": MARKER_DB_PASSWORD}


@pytest.mark.parametrize("quote", ['"', "'"])
def test_parse_env_file_strips_one_matched_pair_of_quotes(tmp_path, quote):
    path = tmp_path / ".env"
    path.write_text(f"DB_PASSWORD={quote}{MARKER_DB_PASSWORD}{quote}\n")
    path.chmod(0o600)
    assert parse_env_file(path) == {"DB_PASSWORD": MARKER_DB_PASSWORD}


def test_parse_env_file_only_strips_one_matched_pair(tmp_path):
    """A doubly-quoted value keeps its inner pair -- only *one* layer is
    stripped, per the brief's own wording."""
    path = tmp_path / ".env"
    path.write_text('DB_PASSWORD=""nested""\n')
    path.chmod(0o600)
    assert parse_env_file(path) == {"DB_PASSWORD": '"nested"'}


def test_parse_env_file_keeps_a_hash_inside_a_value(tmp_path):
    """`#` opens a comment only at the *start* of a line.

    No test anywhere wrote a value containing one, so truncating at an inline
    `#` -- the behaviour several `.env` parsers actually have -- passed the
    whole suite. A `#` is ordinary in a generated credential, and truncating
    at it publishes a different, shorter value that looks entirely correct in
    every log and every `pulumi config` listing: the failure only ever
    surfaces as an authentication error somewhere else, later.
    """
    path = tmp_path / ".env"
    value = "p4x#z9q#mk2"
    path.write_text(f"DB_PASSWORD={value}\n")
    path.chmod(0o600)
    assert parse_env_file(path) == {"DB_PASSWORD": value}


def test_parse_env_file_splits_at_the_first_equals_only(tmp_path):
    """A value may contain `=`, and everything after the first one is the
    value.

    Also untested until now, and equally ordinary: base64 padding ends in
    `=`, and a connection string or a query-shaped token carries them
    throughout. Splitting at the last `=` instead would move part of the
    value into the key, so the name would silently resolve to nothing and the
    entry would be reported as a clean skip.
    """
    path = tmp_path / ".env"
    value = "p4x=z9q=="
    path.write_text(f"DB_PASSWORD={value}\n")
    path.chmod(0o600)
    assert parse_env_file(path) == {"DB_PASSWORD": value}


@pytest.mark.parametrize(
    "value",
    [
        '"p4xz9q',      # an opening quote with nothing closing it
        'p4xz9q"',      # a closing quote with nothing opening it
        "\'p4xz9q\"",    # two quote characters, but not a matched pair
        "xp4xz9qx",     # a matched pair -- of something that is not a quote
    ],
)
def test_parse_env_file_strips_quotes_only_as_an_anchored_matched_pair(tmp_path, value):
    """Both halves of the anchoring, each on its own: the two characters must
    be *equal*, and they must be *quotes*.

    Dropping either half silently eats the first and last character of a
    value that was never quoted -- which, for a credential, means publishing
    something that is wrong by two characters and looks right everywhere.
    Each case here fails under exactly one of the two weakenings and passes
    under the correct rule.
    """
    path = tmp_path / ".env"
    path.write_text(f"DB_PASSWORD={value}\n")
    path.chmod(0o600)
    assert parse_env_file(path) == {"DB_PASSWORD": value}


def test_parse_env_file_warns_on_a_permissive_mode(tmp_path, capfd):
    path = tmp_path / ".env"
    path.write_text(f"DB_PASSWORD={MARKER_DB_PASSWORD}\n")
    path.chmod(0o644)
    parse_env_file(path)
    err = capfd.readouterr().err
    assert str(path) in err
    assert "0644" in err or "644" in err
    assert_no_leak(err, MARKER_DB_PASSWORD, what="the file's own value")


def test_parse_env_file_no_warning_at_exactly_0600(tmp_path, capfd):
    path = tmp_path / ".env"
    path.write_text(f"DB_PASSWORD={MARKER_DB_PASSWORD}\n")
    path.chmod(0o600)
    parse_env_file(path)
    assert capfd.readouterr().err == ""


def test_parse_env_file_malformed_line_raises(tmp_path):
    path = tmp_path / ".env"
    path.write_text("this is not key=value... wait\nDB_PASSWORD\n")
    path.chmod(0o600)
    with pytest.raises(SetSecretsError, match=r"\.env:2"):
        parse_env_file(path)


def test_parse_env_file_unreadable_path_raises(tmp_path):
    with pytest.raises(SetSecretsError):
        parse_env_file(tmp_path / "does-not-exist.env")


# ---------------------------------------------------------------------------
# `substitute_stack`
# ---------------------------------------------------------------------------


def test_substitute_stack_passes_through_a_filename_with_no_placeholder():
    assert substitute_stack(".env", None) == ".env"
    assert substitute_stack(".env", "prod") == ".env"


def test_substitute_stack_replaces_the_placeholder():
    assert substitute_stack(".env.{stack}.local", "prod") == ".env.prod.local"


def test_substitute_stack_raises_when_no_stack_was_given():
    with pytest.raises(SetSecretsError, match=r"\.env\.\{stack\}"):
        substitute_stack(".env.{stack}", None)


# ---------------------------------------------------------------------------
# `resolve_source_files`
# ---------------------------------------------------------------------------


def test_resolve_source_files_absent_source_table_is_an_empty_list(tmp_path):
    assert resolve_source_files(Config(secrets={}), tmp_path, None) == []


def test_resolve_source_files_absent_files_key_is_an_empty_list(tmp_path):
    config = Config(secrets={"source": {}})
    assert resolve_source_files(config, tmp_path, None) == []


def test_resolve_source_files_rejects_a_non_list_files_value(tmp_path):
    config = Config(secrets={"source": {"files": "not-a-list"}})
    with pytest.raises(SetSecretsError):
        resolve_source_files(config, tmp_path, None)


def test_resolve_source_files_rejects_a_files_list_of_non_strings(tmp_path):
    config = Config(secrets={"source": {"files": [1, 2]}})
    with pytest.raises(SetSecretsError):
        resolve_source_files(config, tmp_path, None)


def test_resolve_source_files_resolves_a_relative_entry_against_repo_root(tmp_path):
    config = Config(secrets={"source": {"files": [".env"]}})
    assert resolve_source_files(config, tmp_path, None) == [tmp_path / ".env"]


def test_resolve_source_files_leaves_an_absolute_entry_unchanged(tmp_path):
    absolute = tmp_path / "elsewhere" / ".env"
    config = Config(secrets={"source": {"files": [str(absolute)]}})
    # A repo root that is NOT an ancestor of `absolute` -- proves the
    # absolute entry is not joined onto it at all, not merely that the join
    # happens to produce the same path by coincidence.
    unrelated_root = tmp_path / "a-different-repo-root"
    assert resolve_source_files(config, unrelated_root, None) == [absolute]


def test_resolve_source_files_substitutes_stack_per_entry(tmp_path):
    config = Config(secrets={"source": {"files": [".env.{stack}.local"]}})
    resolved = resolve_source_files(config, tmp_path, "placeholder-stack")
    assert resolved == [tmp_path / ".env.placeholder-stack.local"]


# ---------------------------------------------------------------------------
# `_parse_required` -- the one canonical `[required]` shape
# ---------------------------------------------------------------------------


def test_parse_required_accepts_the_canonical_paths_table():
    assert _parse_required({"paths": ["a.b", "c.d"]}, "x") == frozenset({"a.b", "c.d"})


def test_parse_required_absent_is_empty():
    assert _parse_required(None, "x") == frozenset()


def test_parse_required_rejects_a_bare_list():
    """The shape this module used to also accept (`required = [...]`) is now
    a hard error, not a silently-tolerated alternate spelling."""
    with pytest.raises(SetSecretsError, match="x"):
        _parse_required(["a.b", "c.d"], "x")


def test_parse_required_rejects_a_table_whose_own_keys_are_the_paths():
    """The other shape this module used to also accept
    (`[secrets."<dir>".required]` with the paths as the table's own keys) is
    now a hard error too -- 'paths' is the only recognised key."""
    with pytest.raises(SetSecretsError, match="a.b"):
        _parse_required({"a.b": True, "c.d": "anything"}, "x")


def test_parse_required_rejects_an_unknown_key_alongside_paths():
    with pytest.raises(SetSecretsError, match="extra"):
        _parse_required({"paths": ["a.b"], "extra": "oops"}, "x")


def test_parse_required_rejects_a_paths_list_of_non_strings():
    with pytest.raises(SetSecretsError):
        _parse_required({"paths": [1, 2]}, "x")


def test_parse_required_rejects_a_non_table():
    with pytest.raises(SetSecretsError):
        _parse_required("a.b", "x")


# ---------------------------------------------------------------------------
# `_parse_drift_pairs`
# ---------------------------------------------------------------------------


def test_parse_drift_pairs_builds_pairs():
    raw = [{"bootstrap": "BOOT", "managed": "MANAGED"}]
    assert _parse_drift_pairs(raw, "x") == (DriftPair(bootstrap="BOOT", managed="MANAGED"),)


def test_parse_drift_pairs_absent_is_empty():
    assert _parse_drift_pairs(None, "x") == ()


def test_parse_drift_pairs_rejects_a_missing_field():
    with pytest.raises(SetSecretsError):
        _parse_drift_pairs([{"bootstrap": "BOOT"}], "x")


def test_parse_drift_pairs_rejects_a_non_list():
    with pytest.raises(SetSecretsError):
        _parse_drift_pairs({"bootstrap": "BOOT", "managed": "MANAGED"}, "x")


# ---------------------------------------------------------------------------
# `parse_project_manifest`
# ---------------------------------------------------------------------------


def test_parse_project_manifest_builds_the_secret_table():
    raw = {"secret": {"db.password": "DB_PASSWORD"}}
    manifest = parse_project_manifest(raw, ".")
    assert manifest.secret == {"db.password": "DB_PASSWORD"}
    assert manifest.plaintext == {}
    assert manifest.required == frozenset()
    assert manifest.drift_pairs == ()


def test_parse_project_manifest_builds_the_plaintext_table():
    raw = {"plaintext": {"registry.user": "REGISTRY_USER"}}
    manifest = parse_project_manifest(raw, ".")
    assert manifest.plaintext == {"registry.user": "REGISTRY_USER"}


def test_parse_project_manifest_rejects_a_malformed_config_path_in_secret():
    """`paths.parse` is the one grammar every config path in the manifest is
    validated with -- a key using invalid path syntax must fail here, at load
    time, rather than surfacing later as a `pulumi` invocation error."""
    raw = {"secret": {"a[": "SOME_NAME"}}
    with pytest.raises(SetSecretsError):
        parse_project_manifest(raw, ".")


def test_parse_project_manifest_rejects_a_malformed_config_path_in_plaintext():
    """`plaintext` is published too now, so its keys are validated exactly
    like `secret`'s."""
    raw = {"plaintext": {"a[": "SOME_NAME"}}
    with pytest.raises(SetSecretsError):
        parse_project_manifest(raw, ".")


def test_parse_project_manifest_does_not_validate_unmanaged_keys():
    """`unmanaged` may legitimately hold a wildcard shape that is not a
    `--path` at all -- see the module docstring -- so it must never be run
    through `paths.parse`, unlike `secret` and `plaintext`."""
    raw = {
        "secret": {"db.password": "DB_PASSWORD"},
        "unmanaged": {"legacy.*": "reason, not a logical name"},
    }
    manifest = parse_project_manifest(raw, ".")
    assert manifest.secret == {"db.password": "DB_PASSWORD"}


def test_parse_project_manifest_required_path_not_in_secret_or_plaintext_raises():
    raw = {
        "secret": {"db.password": "DB_PASSWORD"},
        "required": {"paths": ["nonexistent.path"]},
    }
    with pytest.raises(SetSecretsError, match="nonexistent.path"):
        parse_project_manifest(raw, ".")


def test_parse_project_manifest_required_path_in_secret_is_accepted():
    raw = {
        "secret": {"db.password": "DB_PASSWORD"},
        "required": {"paths": ["db.password"]},
    }
    manifest = parse_project_manifest(raw, ".")
    assert manifest.required == frozenset({"db.password"})


def test_parse_project_manifest_required_path_in_plaintext_is_accepted():
    """`required` now ranges over both publishable tables, since both are
    actually published."""
    raw = {
        "plaintext": {"registry.user": "REGISTRY_USER"},
        "required": {"paths": ["registry.user"]},
    }
    manifest = parse_project_manifest(raw, ".")
    assert manifest.required == frozenset({"registry.user"})


def test_parse_project_manifest_rejects_a_path_declared_in_both_secret_and_plaintext():
    """The split between `secret` and `plaintext` *is* the declaration of
    whether a value is a credential -- a path in both tables is a
    contradiction in that declaration, and must be refused rather than
    silently resolved toward the more dangerous (plaintext) reading."""
    raw = {
        "secret": {"db.password": "DB_PASSWORD"},
        "plaintext": {"db.password": "DB_PASSWORD_AGAIN"},
    }
    with pytest.raises(SetSecretsError, match="db.password"):
        parse_project_manifest(raw, ".")


def test_parse_project_manifest_secret_value_must_be_a_string():
    with pytest.raises(SetSecretsError):
        parse_project_manifest({"secret": {"db.password": 123}}, ".")


# ---------------------------------------------------------------------------
# `select_project`
# ---------------------------------------------------------------------------


def test_select_project_uses_the_single_declared_table_regardless_of_cwd(repo):
    write_repo_config(
        repo,
        '[secrets.source]\nfiles = []\n\n[secrets."."]\nsecret = { "a.b" = "NAME" }\n',
    )
    config = load_repo_config(repo)
    unrelated = repo / "not-a-declared-project"
    unrelated.mkdir()
    assert select_project(config, repo, unrelated) == "."


def test_select_project_multi_table_selects_by_cwd(repo):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "a.b" = "NAME" }\n\n'
        '[secrets."infra/foo"]\nsecret = { "c.d" = "OTHER" }\n',
    )
    config = load_repo_config(repo)
    subdir = repo / "infra" / "foo"
    subdir.mkdir(parents=True)
    assert select_project(config, repo, subdir) == "infra/foo"
    assert select_project(config, repo, repo) == "."


def test_select_project_multi_table_no_match_raises_naming_candidates(repo):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "a.b" = "NAME" }\n\n'
        '[secrets."infra/foo"]\nsecret = { "c.d" = "OTHER" }\n',
    )
    config = load_repo_config(repo)
    elsewhere = repo / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(SetSecretsError, match="infra/foo"):
        select_project(config, repo, elsewhere)


def test_select_project_with_no_declared_projects_raises():
    config = Config(secrets={"source": {"files": []}})
    with pytest.raises(SetSecretsError):
        select_project(config, Path("/tmp"), Path("/tmp"))


# ---------------------------------------------------------------------------
# `LocalSecretSource` -- precedence
# ---------------------------------------------------------------------------


def test_resolves_a_name_present_in_only_one_file():
    source = LocalSecretSource({}, [{"NAME": "from-file-1"}])
    assert source.resolve("NAME") == "from-file-1"


def test_later_file_overrides_an_earlier_one():
    source = LocalSecretSource(
        {}, [{"NAME": "from-file-1"}, {"NAME": "from-file-2"}]
    )
    assert source.resolve("NAME") == "from-file-2"


def test_environment_wins_over_every_file():
    source = LocalSecretSource(
        {"NAME": "from-env"}, [{"NAME": "from-file-1"}, {"NAME": "from-file-2"}]
    )
    assert source.resolve("NAME") == "from-env"


def test_full_precedence_chain_across_env_and_two_files():
    """One test, three sources, three distinct names -- proving each tier
    actually wins where it should, not merely that the highest tier beats a
    single lower one."""
    source = LocalSecretSource(
        {"ONLY_ENV": "env-value", "IN_ALL_THREE": "env-wins"},
        [
            {"ONLY_FILE_1": "file-1-value", "IN_ALL_THREE": "file-1-loses", "IN_BOTH_FILES": "file-1-loses"},
            {"IN_ALL_THREE": "file-2-loses", "IN_BOTH_FILES": "file-2-wins"},
        ],
    )
    assert source.resolve("ONLY_ENV") == "env-value"
    assert source.resolve("ONLY_FILE_1") == "file-1-value"
    assert source.resolve("IN_BOTH_FILES") == "file-2-wins"
    assert source.resolve("IN_ALL_THREE") == "env-wins"


def test_unresolvable_name_returns_none():
    source = LocalSecretSource({}, [{"OTHER": "value"}])
    assert source.resolve("NAME") is None


# ---------------------------------------------------------------------------
# `_config_path_for_name`
# ---------------------------------------------------------------------------


def test_config_path_for_name_finds_the_one_match():
    assert _config_path_for_name({"a.b": "NAME"}, "NAME") == "a.b"


def test_config_path_for_name_is_none_when_absent():
    assert _config_path_for_name({"a.b": "OTHER"}, "NAME") is None


def test_config_path_for_name_is_none_when_ambiguous():
    assert _config_path_for_name({"a.b": "NAME", "c.d": "NAME"}, "NAME") is None


# ---------------------------------------------------------------------------
# `_publish_entries` -- skip-if-unset, without touching `pulumi` at all
# ---------------------------------------------------------------------------


def test_unset_entry_is_skipped_and_pulumi_is_never_invoked(monkeypatch):
    manifest = ProjectManifest(secret={"a.b": "UNSET_NAME"})
    source = LocalSecretSource({}, [])

    def poison(*args, **kwargs):
        raise AssertionError("pulumi must not be invoked for an unresolved entry")

    monkeypatch.setattr(set_secrets, "_pulumi_config_set", poison)
    outcomes = _publish_entries(
        manifest, source, pulumi="/usr/bin/pulumi", stack=None, dry_run=False, timeout=1.0
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "skipped"


def test_empty_string_value_is_treated_as_unset(monkeypatch):
    """An environment variable exported as `""` must not be published -- that
    would clear whatever secret is already there, the destructive surprise
    the brief calls out by name."""
    manifest = ProjectManifest(secret={"a.b": "EMPTY_NAME"})
    source = LocalSecretSource({"EMPTY_NAME": ""}, [])

    def poison(*args, **kwargs):
        raise AssertionError("pulumi must not be invoked for an empty value")

    monkeypatch.setattr(set_secrets, "_pulumi_config_set", poison)
    outcomes = _publish_entries(
        manifest, source, pulumi="/usr/bin/pulumi", stack=None, dry_run=False, timeout=1.0
    )
    assert outcomes[0].status == "skipped"


def test_required_and_unset_is_a_failure_not_a_skip():
    manifest = ProjectManifest(secret={"a.b": "NAME"}, required=frozenset({"a.b"}))
    source = LocalSecretSource({}, [])
    outcomes = _publish_entries(
        manifest, source, pulumi=None, stack=None, dry_run=False, timeout=1.0
    )
    assert outcomes[0].status == "failed"
    assert "required" in outcomes[0].reason


# ---------------------------------------------------------------------------
# End-to-end via `main()`, with a real `pulumi` stand-in.
# ---------------------------------------------------------------------------


def test_no_repo_config_is_a_clean_failure(repo, monkeypatch, capfd):
    monkeypatch.chdir(repo)
    assert main(["set-secrets"]) == 2
    err = capfd.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_source_files_are_read_from_disk_through_the_full_manifest_pipeline(
    repo, monkeypatch, stub_pulumi, tmp_path
):
    """`resolve_source_files` is real logic -- default-to-empty, type
    validation, `{stack}` substitution per entry, repo-root-relative
    resolution -- and this is the one test that exercises it through the
    actual manifest-to-disk path, rather than through hand-built
    `LocalSecretSource` dicts (as `test_full_precedence_chain_across_env_and_
    two_files` does for the resolver itself). One test closes four gaps at
    once: two *real* files on disk where the second overrides a name from
    the first; `--stack` supplied so `{stack}` substitution is exercised as
    it is actually wired, not just as a standalone function call; a third
    declared file that is genuinely absent from disk, proving the
    silent-skip ruling end-to-end; and the result is read off a real child
    process's stdin, not this module's own belief about what it resolved.
    """
    write_repo_config(
        repo,
        '[secrets.source]\n'
        'files = [".env.base", ".env.{stack}.local", ".env.absent"]\n\n'
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n',
    )
    (repo / ".env.base").write_text("DB_PASSWORD=from-base-file\n")
    (repo / ".env.base").chmod(0o600)
    (repo / ".env.placeholder-stack.local").write_text("DB_PASSWORD=from-stack-local-file\n")
    (repo / ".env.placeholder-stack.local").chmod(0o600)
    # ".env.absent" is deliberately never created.

    monkeypatch.chdir(repo)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    fake_pulumi(monkeypatch, stub_pulumi)

    stdin_dump = tmp_path / "stdin.log"
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_dump))

    assert main(["set-secrets", "--stack", "placeholder-stack"]) == 0

    stdin_content = stdin_dump.read_bytes()
    assert b"from-stack-local-file" in stdin_content
    assert b"from-base-file" not in stdin_content


def test_an_absolute_source_file_path_is_read_regardless_of_repo_root(
    repo, monkeypatch, stub_pulumi, tmp_path
):
    outside = tmp_path / "outside-the-repo" / ".env"
    outside.parent.mkdir(parents=True)
    outside.write_text("DB_PASSWORD=from-an-absolute-path\n")
    outside.chmod(0o600)

    write_repo_config(
        repo,
        f'[secrets.source]\nfiles = ["{outside}"]\n\n'
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    fake_pulumi(monkeypatch, stub_pulumi)

    stdin_dump = tmp_path / "stdin.log"
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_dump))

    assert main(["set-secrets"]) == 0
    assert b"from-an-absolute-path" in stdin_dump.read_bytes()


def test_dry_run_never_invokes_a_subprocess(repo, monkeypatch, capfd):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)

    def poison(*args, **kwargs):
        raise AssertionError("--dry-run must never invoke a subprocess")

    monkeypatch.setattr(set_secrets.subprocess, "run", poison)
    assert main(["set-secrets", "--dry-run"]) == 0
    out = capfd.readouterr().out
    assert "would set" in out
    assert "db.password" in out
    assert_no_leak(out, MARKER_DB_PASSWORD, what="the resolved value")


def test_dry_run_reports_a_required_unresolved_path_as_a_failure(repo, monkeypatch, capfd):
    write_repo_config(
        repo,
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n\n'
        '[secrets.".".required]\n'
        'paths = ["db.password"]\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    assert main(["set-secrets", "--dry-run"]) == 2
    err = capfd.readouterr().err
    assert "db.password" in err


@pytest.mark.parametrize("table", ["secret", "plaintext"])
@pytest.mark.parametrize(
    ("stack_args", "stack_label"),
    [([], "no-stack"), (["--stack", "placeholder-stack"], "with-stack")],
    ids=["no-stack", "with-stack"],
)
def test_a_value_reaches_pulumi_on_stdin_and_never_in_argv(
    repo, monkeypatch, capfd, stub_pulumi, tmp_path, table, stack_args, stack_label
):
    """Global Constraint 6, over every shape this command actually produces.

    This was one test: one `secret` entry, no `--stack`. Both of the
    conditions it did not cover are the ordinary ones -- `--stack` is how any
    real rotation is invoked, and `plaintext` entries go through the same
    `_pulumi_config_set` call with a different flag -- and appending the
    value to argv under either of them passed the entire suite. `--stack` in
    particular is the more dangerous shape: it is the branch that *already*
    appends to `args`, so a value appended one line later reads as part of
    the same edit.

    Each case carries its own marker, so a leak under one parameter cannot be
    excused by a match against another parameter's value; and the marker is
    resolved from the environment, so what lands on stdin is the same string
    the assertion is made about.
    """
    value = ARGV_CASE_MARKERS[table, stack_label]
    write_repo_config(
        repo,
        f'[secrets."."]\n{table} = {{ "managed.value" = "VALUE_NAME" }}\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("VALUE_NAME", value)
    fake_pulumi(monkeypatch, stub_pulumi)

    argv_dump = tmp_path / "argv.log"
    stdin_dump = tmp_path / "stdin.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_dump))

    assert main(["set-secrets", *stack_args]) == 0

    calls = recorded_pulumi_calls(argv_dump, value)
    assert len(calls) == 1
    call_argv = calls[0]
    assert f"--{table}" in call_argv
    assert "--path" in call_argv and "managed.value" in call_argv
    assert ("--stack" in call_argv) is bool(stack_args)

    # The other half of the same claim: it really did reach the child, on
    # stdin, where `ps` cannot see it. Absent from argv alone would also be
    # satisfied by never sending it at all.
    assert value.encode() in stdin_dump.read_bytes()

    captured = capfd.readouterr()
    assert_no_leak(captured.out, value, what="the resolved value")
    assert_no_leak(captured.err, value, what="the resolved value")


def test_a_plaintext_entry_is_published_with_plaintext_and_not_secret(
    repo, monkeypatch, capfd, stub_pulumi, tmp_path
):
    """`plaintext` entries are published too (Ruling 1) -- with
    `--plaintext`, never `--secret`. Still delivered on stdin: one code path
    for both tables, per the ruling."""
    write_repo_config(
        repo,
        '[secrets."."]\nplaintext = { "registry.user" = "REGISTRY_USER" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("REGISTRY_USER", MARKER_PLAINTEXT)
    fake_pulumi(monkeypatch, stub_pulumi)

    argv_dump = tmp_path / "argv.log"
    stdin_dump = tmp_path / "stdin.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_dump))

    assert main(["set-secrets"]) == 0

    (call_argv,) = recorded_pulumi_calls(argv_dump, MARKER_PLAINTEXT)
    assert "--plaintext" in call_argv
    assert "--secret" not in call_argv
    assert "--path" in call_argv and "registry.user" in call_argv

    assert MARKER_PLAINTEXT.encode() in stdin_dump.read_bytes()


def test_a_secret_entry_is_never_published_with_plaintext(
    repo, monkeypatch, stub_pulumi, tmp_path
):
    """The converse of the test above: a `secret`-table entry always carries
    `--secret` and never `--plaintext`, even now that both tables are
    published through the same `_pulumi_config_set` code path."""
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    fake_pulumi(monkeypatch, stub_pulumi)

    argv_dump = tmp_path / "argv.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))

    assert main(["set-secrets"]) == 0
    (call_argv,) = recorded_pulumi_calls(argv_dump, MARKER_DB_PASSWORD)
    assert "--secret" in call_argv
    assert "--plaintext" not in call_argv


def test_secret_and_plaintext_entries_are_both_published_in_one_run(
    repo, monkeypatch, stub_pulumi, tmp_path
):
    """Both tables are live at once, each with its own flag -- not merely
    that either works in isolation."""
    write_repo_config(
        repo,
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n'
        'plaintext = { "registry.user" = "REGISTRY_USER" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    monkeypatch.setenv("REGISTRY_USER", MARKER_PLAINTEXT)
    fake_pulumi(monkeypatch, stub_pulumi)

    argv_dump = tmp_path / "argv.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))

    assert main(["set-secrets"]) == 0
    calls = recorded_pulumi_calls(argv_dump, MARKER_DB_PASSWORD, MARKER_PLAINTEXT)
    assert len(calls) == 2

    secret_call = next(c for c in calls if "db.password" in c)
    plaintext_call = next(c for c in calls if "registry.user" in c)
    assert "--secret" in secret_call and "--plaintext" not in secret_call
    assert "--plaintext" in plaintext_call and "--secret" not in plaintext_call


def test_stack_flag_is_forwarded_to_pulumi(repo, monkeypatch, capfd, stub_pulumi, tmp_path):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    fake_pulumi(monkeypatch, stub_pulumi)

    argv_dump = tmp_path / "argv.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))

    assert main(["set-secrets", "--stack", "placeholder-stack"]) == 0
    (call_argv,) = recorded_pulumi_calls(argv_dump, MARKER_DB_PASSWORD)
    assert "--stack" in call_argv
    assert "placeholder-stack" in call_argv

    captured = capfd.readouterr()
    assert_no_leak(captured.out, MARKER_DB_PASSWORD, what="the resolved value")
    assert_no_leak(captured.err, MARKER_DB_PASSWORD, what="the resolved value")


def test_stack_flag_omitted_when_not_given(repo, monkeypatch, capfd, stub_pulumi, tmp_path):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    fake_pulumi(monkeypatch, stub_pulumi)

    argv_dump = tmp_path / "argv.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))

    assert main(["set-secrets"]) == 0
    (call_argv,) = recorded_pulumi_calls(argv_dump, MARKER_DB_PASSWORD)
    assert "--stack" not in call_argv

    captured = capfd.readouterr()
    assert_no_leak(captured.out, MARKER_DB_PASSWORD, what="the resolved value")
    assert_no_leak(captured.err, MARKER_DB_PASSWORD, what="the resolved value")


def test_required_unresolved_path_fails_the_run(repo, monkeypatch, stub_pulumi, capfd):
    write_repo_config(
        repo,
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n\n'
        '[secrets.".".required]\n'
        'paths = ["db.password"]\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    fake_pulumi(monkeypatch, stub_pulumi)

    assert main(["set-secrets"]) == 2
    err = capfd.readouterr().err
    assert "db.password" in err


def test_a_non_canonical_required_shape_is_rejected_not_silently_ignored(
    repo, monkeypatch, capfd
):
    """Ruling 2: `required = [...]` (the bare-array shape this module used to
    also accept) is now a hard load-time error -- specifically *not* silently
    read as "nothing required".

    The exit code alone cannot distinguish "rejected at load time" from "the
    old shape was quietly tolerated": in this exact manifest the tolerated
    reading would still end up treating 'db.password' as required (the list
    happens to contain a valid path), still fail to resolve it, and still
    exit 2 through the ordinary per-entry failure path -- printing a
    "summary:" line on the way. A genuine load-time rejection never reaches
    that loop at all, so *that* absence, not the exit code, is the property
    this test actually has to prove -- the same trap the brief warns several
    of this task's tests are prone to.
    """
    write_repo_config(
        repo,
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n'
        'required = ["db.password"]\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    assert main(["set-secrets"]) == 2
    captured = capfd.readouterr()
    assert "summary:" not in captured.out
    assert "error:" in captured.err
    assert "paths" in captured.err


def test_a_scalar_where_the_secret_table_belongs_is_rejected_not_read_as_empty(
    repo, monkeypatch, capfd
):
    """`secret = "DB_PASSWORD"` -- the shape a manifest author writes when
    they forget the `path = name` mapping -- is a hard load-time refusal.

    Read instead as "this project declares no entries", it produces `0 set,
    0 skipped, 0 failed` and exit 0: a rotation that published nothing,
    reported in the language of success. Exit 0 with no work done is the
    worst of the three possible outcomes here, because nothing downstream has
    any reason to look again.

    The `summary:` line's absence, not the exit code, is what proves the
    refusal happened at load time rather than in the publishing loop.
    """
    write_repo_config(repo, '[secrets."."]\nsecret = "DB_PASSWORD"\n')
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)

    assert main(["set-secrets"]) == 2
    captured = capfd.readouterr()
    assert "summary:" not in captured.out
    assert "must be a table" in captured.err
    assert "secret" in captured.err
    assert_no_leak(captured.out, MARKER_DB_PASSWORD, what="the resolved value")
    assert_no_leak(captured.err, MARKER_DB_PASSWORD, what="the resolved value")


def test_the_summary_line_counts_every_outcome(repo, monkeypatch, capfd, stub_pulumi):
    """One run with one of each outcome, and the exact counts asserted.

    The `summary:` line is the whole report for a run nobody reads closely,
    and no test anywhere asserted its numbers -- hard-coding
    `0 set, 0 skipped, 0 failed` passed all 781. Three entries in a fixed,
    known declaration order: one that resolves, one that does not and is not
    required, and one that does not and is.
    """
    write_repo_config(
        repo,
        '[secrets."."]\n'
        "secret = { "
        '"a.set" = "NAME_SET", '
        '"b.skip" = "NAME_SKIP", '
        '"c.fail" = "NAME_FAIL" '
        "}\n\n"
        '[secrets.".".required]\n'
        'paths = ["c.fail"]\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("NAME_SET", "a-value-that-resolves")
    monkeypatch.delenv("NAME_SKIP", raising=False)
    monkeypatch.delenv("NAME_FAIL", raising=False)
    fake_pulumi(monkeypatch, stub_pulumi)

    assert main(["set-secrets"]) == 2
    captured = capfd.readouterr()
    assert "summary: 1 set, 1 skipped, 1 failed" in captured.out


def test_a_path_declared_in_both_secret_and_plaintext_is_rejected_not_downgraded(
    repo, monkeypatch, capfd, stub_pulumi
):
    """End-to-end version of the overlap check: without it, `_publish_entries`
    would publish 'db.password' twice -- `--secret` first, `--plaintext`
    last, with the last write winning -- leaving a value its author declared
    a credential sitting unencrypted in Pulumi's config. This must fail
    before either write happens, not merely end up correct by luck of write
    order."""
    write_repo_config(
        repo,
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n'
        'plaintext = { "db.password" = "DB_PASSWORD_AGAIN" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    monkeypatch.setenv("DB_PASSWORD_AGAIN", MARKER_DB_PASSWORD)
    fake_pulumi(monkeypatch, stub_pulumi)

    assert main(["set-secrets"]) == 2
    captured = capfd.readouterr()
    assert "summary:" not in captured.out
    assert "db.password" in captured.err
    assert_no_leak(captured.out, MARKER_DB_PASSWORD, what="the resolved value")
    assert_no_leak(captured.err, MARKER_DB_PASSWORD, what="the resolved value")


def test_a_missing_pulumi_binary_fails_that_entry_and_is_reported(repo, monkeypatch, capfd):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    fake_pulumi(monkeypatch, None)

    assert main(["set-secrets"]) == 2
    captured = capfd.readouterr()
    assert "not found on PATH" in captured.err
    assert "db.password" in captured.err
    assert_no_leak(captured.out, MARKER_DB_PASSWORD, what="the resolved value")
    assert_no_leak(captured.err, MARKER_DB_PASSWORD, what="the resolved value")


def test_a_timeout_in_the_middle_does_not_abort_the_remaining_entries(
    repo, monkeypatch, capfd, stub_pulumi, tmp_path
):
    """Three entries, the timing-out one in the middle, in a fixed, known
    declaration order -- proving the first and third entries both still get
    set, not merely that the run finishes and reports *a* failure."""
    write_repo_config(
        repo,
        '[secrets."."]\n'
        "secret = { "
        '"a.first" = "NAME_FIRST", '
        '"b.slow" = "NAME_SLOW", '
        '"c.last" = "NAME_LAST" '
        "}\n",
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("NAME_FIRST", "value-first")
    monkeypatch.setenv("NAME_SLOW", "value-slow")
    monkeypatch.setenv("NAME_LAST", "value-last")
    fake_pulumi(monkeypatch, stub_pulumi)

    stdin_dump = tmp_path / "stdin.log"
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_dump))
    monkeypatch.setenv("STUB_SLOW_PATH", "b.slow")
    # fast_timeout sets DEFAULT_TIMEOUT_SECONDS to 0.3s; sleep well past it.
    monkeypatch.setenv("STUB_SLEEP_SECONDS", "2")

    assert main(["set-secrets"]) == 2
    out, err = (lambda c: (c.out, c.err))(capfd.readouterr())
    assert "set 'a.first'" in out
    assert "set 'c.last'" in out
    assert "b.slow" in err
    assert "timed out" in err
    assert "not set:" in err and "b.slow" in err.split("not set:")[1]

    # The two entries either side of the slow one really did reach a real
    # child process on stdin -- not merely that this module *believes* they
    # were set. (The slow entry's own value also reaches the stub's stdin
    # before it sleeps and is killed on timeout -- a real subprocess cannot
    # un-receive what it was already sent, so that is not asserted here; what
    # matters is that this module never treats that entry as having landed.)
    stdin_lines = stdin_dump.read_bytes().splitlines()
    assert b"value-first" in stdin_lines
    assert b"value-last" in stdin_lines


def test_a_pulumi_failure_never_echoes_the_value_it_was_given(
    repo, monkeypatch, capfd, stub_pulumi
):
    """Even when `pulumi` itself misbehaves and writes something odd to its
    own stderr, this module must not relay it -- it is not this module's to
    vouch for as free of the value it just sent on stdin."""
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv("STUB_SET_EXIT_CODE", "1")
    monkeypatch.setenv("STUB_SET_ECHO", f"pulumi choked on {MARKER_DB_PASSWORD}")

    assert main(["set-secrets"]) == 2
    captured = capfd.readouterr()
    # `assert_no_leak`, not `not in`: the reason this module builds for a
    # non-zero exit (`pulumi exited 1`) is printed by `_print_outcome`, and a
    # reason quoting even the first eight characters of the value would pass a
    # whole-string check while disclosing the credential.
    assert_no_leak(captured.out, MARKER_DB_PASSWORD, what="the resolved value")
    assert_no_leak(captured.err, MARKER_DB_PASSWORD, what="the resolved value")


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


_DRIFT_MANIFEST = (
    '[secrets."."]\n'
    'secret = { "managed.path" = "MANAGED_NAME" }\n\n'
    '[[secrets.".".drift_pairs]]\n'
    'bootstrap = "BOOTSTRAP_NAME"\n'
    'managed = "MANAGED_NAME"\n'
)

_AMBIGUOUS_DRIFT_MANIFEST = (
    '[secrets."."]\n'
    "secret = { "
    '"a.path" = "MANAGED_NAME", '
    '"b.path" = "MANAGED_NAME" '
    "}\n\n"
    '[[secrets.".".drift_pairs]]\n'
    'bootstrap = "BOOTSTRAP_NAME"\n'
    'managed = "MANAGED_NAME"\n'
)

# `managed` lives in `plaintext`, not `secret` -- proves the inversion in
# `_check_drift` covers both publishable tables (Also-fix 3), not just
# `secret`.
_PLAINTEXT_MANAGED_DRIFT_MANIFEST = (
    '[secrets."."]\n'
    'plaintext = { "managed.path" = "MANAGED_NAME" }\n\n'
    '[[secrets.".".drift_pairs]]\n'
    'bootstrap = "BOOTSTRAP_NAME"\n'
    'managed = "MANAGED_NAME"\n'
)

# Two drift pairs, each unambiguously resolvable -- regression coverage for
# the `published` variable-shadowing bug: a pre-loop `published` (the merged
# path->name dict) and the loop's own `published` (`_pulumi_config_get`'s
# `str | None` result) used to share one name, so the second and every later
# pair called `_config_path_for_name` with a leftover `str` instead of the
# dict and crashed with `AttributeError`.
_TWO_DRIFT_PAIRS_MANIFEST = (
    '[secrets."."]\n'
    "secret = { "
    '"managed.path.one" = "MANAGED_NAME_ONE", '
    '"managed.path.two" = "MANAGED_NAME_TWO" '
    "}\n\n"
    '[[secrets.".".drift_pairs]]\n'
    'bootstrap = "BOOTSTRAP_NAME_ONE"\n'
    'managed = "MANAGED_NAME_ONE"\n\n'
    '[[secrets.".".drift_pairs]]\n'
    'bootstrap = "BOOTSTRAP_NAME_TWO"\n'
    'managed = "MANAGED_NAME_TWO"\n'
)


def test_drift_warns_when_the_values_differ(repo, monkeypatch, capfd, stub_pulumi):
    write_repo_config(repo, _DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("MANAGED_NAME", MARKER_PUBLISHED)
    monkeypatch.setenv("BOOTSTRAP_NAME", MARKER_BOOTSTRAP)
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv(
        "STUB_GET_VALUES", json.dumps({"managed.path": "a-different-published-value"})
    )

    assert main(["set-secrets"]) == 0
    captured = capfd.readouterr()
    assert "drift" in captured.err
    assert "BOOTSTRAP_NAME" in captured.err
    assert "MANAGED_NAME" in captured.err
    assert "managed.path" in captured.err
    # The drift warning names the two logical names and the config path, and
    # must never quote either value -- not whole, and not the leading run of
    # one, which a `not in` check would have let through.
    for stream in (captured.err, captured.out):
        assert_no_leak(stream, MARKER_PUBLISHED, what="the published value")
        assert_no_leak(stream, MARKER_BOOTSTRAP, what="the bootstrap value")


def test_drift_is_silent_when_the_values_agree(repo, monkeypatch, capfd, stub_pulumi):
    write_repo_config(repo, _DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("MANAGED_NAME", MARKER_PUBLISHED)
    monkeypatch.setenv("BOOTSTRAP_NAME", MARKER_PUBLISHED)
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv("STUB_GET_VALUES", json.dumps({"managed.path": MARKER_PUBLISHED}))

    assert main(["set-secrets"]) == 0
    captured = capfd.readouterr()
    assert "drift" not in captured.err
    assert "warning" not in captured.err


def test_drift_comparison_removes_only_the_clis_own_trailing_newline(
    repo, monkeypatch, capfd, stub_pulumi
):
    """Also-fix 4: a published value that itself legitimately ends in a
    newline must still compare equal to a local value that also does --
    `.removesuffix("\\n")` removes exactly the one trailing newline `pulumi
    config get` itself appends, where `.rstrip("\\n")` would also eat the
    value's own, firing a spurious warning over a difference that was never
    real. The stub appends its own trailing newline on top of the value
    (simulating the CLI), so this only passes if exactly one newline is
    stripped, not every one."""
    write_repo_config(repo, _DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    value_ending_in_newline = "value-with-its-own-trailing-newline\n"
    monkeypatch.setenv("MANAGED_NAME", value_ending_in_newline)
    monkeypatch.setenv("BOOTSTRAP_NAME", value_ending_in_newline)
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv(
        "STUB_GET_VALUES", json.dumps({"managed.path": value_ending_in_newline})
    )

    assert main(["set-secrets"]) == 0
    captured = capfd.readouterr()
    assert "drift" not in captured.err
    assert "warning" not in captured.err


def test_drift_is_silent_when_the_published_value_is_unreadable(
    repo, monkeypatch, capfd, stub_pulumi
):
    write_repo_config(repo, _DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("MANAGED_NAME", MARKER_PUBLISHED)
    monkeypatch.setenv("BOOTSTRAP_NAME", MARKER_BOOTSTRAP)
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv("STUB_GET_EXIT_CODE", "1")

    assert main(["set-secrets"]) == 0
    captured = capfd.readouterr()
    assert captured.err == ""


def test_drift_warns_when_the_managed_name_is_ambiguous(repo, monkeypatch, capfd, stub_pulumi):
    write_repo_config(repo, _AMBIGUOUS_DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("BOOTSTRAP_NAME", MARKER_BOOTSTRAP)
    fake_pulumi(monkeypatch, stub_pulumi)

    assert main(["set-secrets"]) == 0
    captured = capfd.readouterr()
    assert "MANAGED_NAME" in captured.err
    assert_no_leak(captured.err, MARKER_BOOTSTRAP, what="the bootstrap value")


def test_drift_finds_a_managed_name_declared_in_plaintext(repo, monkeypatch, capfd, stub_pulumi):
    """Also-fix 3: `_check_drift` inverts over `secret ∪ plaintext`, since
    both are published now (Ruling 1). Before the fix, this exact manifest
    -- `managed` correctly and unambiguously declared, just under
    `plaintext` -- would wrongly hit the "does not name exactly one declared
    secret" branch instead of comparing the values at all."""
    write_repo_config(repo, _PLAINTEXT_MANAGED_DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("MANAGED_NAME", MARKER_PLAINTEXT)
    monkeypatch.setenv("BOOTSTRAP_NAME", MARKER_BOOTSTRAP)
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv(
        "STUB_GET_VALUES", json.dumps({"managed.path": "a-different-published-value"})
    )

    assert main(["set-secrets"]) == 0
    captured = capfd.readouterr()
    assert "does not name exactly one" not in captured.err
    assert "drift" in captured.err
    assert "managed.path" in captured.err
    assert_no_leak(captured.err, MARKER_BOOTSTRAP, what="the bootstrap value")


def test_a_second_drift_pair_is_evaluated_and_the_run_reaches_its_summary(
    repo, monkeypatch, capfd, stub_pulumi
):
    """Regression test for the `published` variable-shadowing bug: a
    pre-loop `published` (the merged `secret ∪ plaintext` dict) and the
    loop's own `published` (`_pulumi_config_get`'s result) shared one name,
    so every drift pair after the first called `_config_path_for_name` with
    a leftover `str` and crashed with `AttributeError` -- which `fail_closed`
    then turned into a bare exit 2 *after* every `pulumi config set` call had
    already succeeded and been printed, with the closing `summary:` line
    never reached at all. Two pairs, both unambiguously resolvable and both
    genuinely differing (so both warnings must fire, not just "no crash"),
    proves the second pair is actually evaluated rather than merely not
    crashing on it by accident.
    """
    write_repo_config(repo, _TWO_DRIFT_PAIRS_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("MANAGED_NAME_ONE", "local-value-one")
    monkeypatch.setenv("MANAGED_NAME_TWO", "local-value-two")
    monkeypatch.setenv("BOOTSTRAP_NAME_ONE", "local-value-one")
    monkeypatch.setenv("BOOTSTRAP_NAME_TWO", "local-value-two")
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv(
        "STUB_GET_VALUES",
        json.dumps(
            {
                "managed.path.one": "published-value-one-differs",
                "managed.path.two": "published-value-two-differs",
            }
        ),
    )

    assert main(["set-secrets"]) == 0
    captured = capfd.readouterr()
    assert "AttributeError" not in captured.err
    assert "MANAGED_NAME_ONE" in captured.err
    assert "MANAGED_NAME_TWO" in captured.err
    assert "summary:" in captured.out


def test_drift_is_never_checked_under_dry_run(repo, monkeypatch, capfd, stub_pulumi, tmp_path):
    write_repo_config(repo, _DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("MANAGED_NAME", MARKER_PUBLISHED)
    monkeypatch.setenv("BOOTSTRAP_NAME", "a-different-value")
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv(
        "STUB_GET_VALUES", json.dumps({"managed.path": "a-different-published-value"})
    )
    argv_dump = tmp_path / "argv.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))

    assert main(["set-secrets", "--dry-run"]) == 0
    assert not argv_dump.exists()
    captured = capfd.readouterr()
    assert "drift" not in captured.err


def test_drift_is_silent_when_the_bootstrap_value_does_not_resolve(
    repo, monkeypatch, capfd, stub_pulumi
):
    write_repo_config(repo, _DRIFT_MANIFEST)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("MANAGED_NAME", MARKER_PUBLISHED)
    monkeypatch.delenv("BOOTSTRAP_NAME", raising=False)
    fake_pulumi(monkeypatch, stub_pulumi)
    monkeypatch.setenv(
        "STUB_GET_VALUES", json.dumps({"managed.path": "a-different-published-value"})
    )

    assert main(["set-secrets"]) == 0
    assert capfd.readouterr().err == ""
