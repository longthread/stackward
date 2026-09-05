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
    _normalize_required,
    _parse_drift_pairs,
    _publish_entries,
    parse_env_file,
    parse_project_manifest,
    select_project,
    substitute_stack,
)
from stackward.config import Config, find_repo_config, load_config

# Placeholder values only -- see GC4 in the task brief. Distinctive enough
# that an accidental substring match elsewhere in captured output would be
# implausible.
MARKER_DB_PASSWORD = "marker-db-password-1a2b3c4d"
MARKER_PUBLISHED = "marker-published-value-9c0d1e2f"
MARKER_BOOTSTRAP = "marker-bootstrap-value-3f4e5d6c"


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


def test_parse_env_file_warns_on_a_permissive_mode(tmp_path, capfd):
    path = tmp_path / ".env"
    path.write_text(f"DB_PASSWORD={MARKER_DB_PASSWORD}\n")
    path.chmod(0o644)
    parse_env_file(path)
    err = capfd.readouterr().err
    assert str(path) in err
    assert "0644" in err or "644" in err
    assert MARKER_DB_PASSWORD not in err


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
# `_normalize_required`
# ---------------------------------------------------------------------------


def test_normalize_required_accepts_a_list_of_paths():
    assert _normalize_required(["a.b", "c.d"], "x") == frozenset({"a.b", "c.d"})


def test_normalize_required_accepts_a_table_of_paths():
    assert _normalize_required({"a.b": True, "c.d": "anything"}, "x") == frozenset(
        {"a.b", "c.d"}
    )


def test_normalize_required_absent_is_empty():
    assert _normalize_required(None, "x") == frozenset()


def test_normalize_required_rejects_a_list_of_non_strings():
    with pytest.raises(SetSecretsError):
        _normalize_required([1, 2], "x")


def test_normalize_required_rejects_other_shapes():
    with pytest.raises(SetSecretsError):
        _normalize_required("a.b", "x")


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
    assert manifest.required == frozenset()
    assert manifest.drift_pairs == ()


def test_parse_project_manifest_rejects_a_malformed_config_path():
    """`paths.parse` is the one grammar every config path in the manifest is
    validated with -- a key using invalid path syntax must fail here, at load
    time, rather than surfacing later as a `pulumi` invocation error."""
    raw = {"secret": {"a[": "SOME_NAME"}}
    with pytest.raises(SetSecretsError):
        parse_project_manifest(raw, ".")


def test_parse_project_manifest_does_not_validate_unmanaged_or_plaintext_keys():
    """`unmanaged` may legitimately hold a wildcard shape that is not a
    `--path` at all -- see the module docstring -- so it must never be run
    through `paths.parse`."""
    raw = {
        "secret": {"db.password": "DB_PASSWORD"},
        "plaintext": {"registry.user": "REGISTRY_USER"},
        "unmanaged": {"legacy.*": "reason, not a logical name"},
    }
    manifest = parse_project_manifest(raw, ".")
    assert manifest.secret == {"db.password": "DB_PASSWORD"}


def test_parse_project_manifest_required_path_not_in_secret_raises():
    raw = {"secret": {"db.password": "DB_PASSWORD"}, "required": ["nonexistent.path"]}
    with pytest.raises(SetSecretsError, match="nonexistent.path"):
        parse_project_manifest(raw, ".")


def test_parse_project_manifest_required_path_in_secret_is_accepted():
    raw = {"secret": {"db.password": "DB_PASSWORD"}, "required": ["db.password"]}
    manifest = parse_project_manifest(raw, ".")
    assert manifest.required == frozenset({"db.password"})


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
    assert MARKER_DB_PASSWORD not in out


def test_dry_run_reports_a_required_unresolved_path_as_a_failure(repo, monkeypatch, capfd):
    write_repo_config(
        repo,
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n'
        'required = ["db.password"]\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    assert main(["set-secrets", "--dry-run"]) == 2
    err = capfd.readouterr().err
    assert "db.password" in err


def test_value_reaches_the_subprocess_via_stdin_and_never_via_argv(
    repo, monkeypatch, capfd, stub_pulumi, tmp_path
):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    fake_pulumi(monkeypatch, stub_pulumi)

    argv_dump = tmp_path / "argv.log"
    stdin_dump = tmp_path / "stdin.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_dump))

    assert main(["set-secrets"]) == 0

    argv_calls = [json.loads(line) for line in argv_dump.read_text().splitlines()]
    assert len(argv_calls) == 1
    call_argv = argv_calls[0]
    assert "--secret" in call_argv
    assert "--path" in call_argv and "db.password" in call_argv
    assert not any(MARKER_DB_PASSWORD in arg for arg in call_argv)

    stdin_content = stdin_dump.read_bytes()
    assert MARKER_DB_PASSWORD.encode() in stdin_content

    captured = capfd.readouterr()
    assert MARKER_DB_PASSWORD not in captured.out
    assert MARKER_DB_PASSWORD not in captured.err


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
    call_argv = json.loads(argv_dump.read_text().splitlines()[0])
    assert "--stack" in call_argv
    assert "placeholder-stack" in call_argv

    captured = capfd.readouterr()
    assert MARKER_DB_PASSWORD not in captured.out
    assert MARKER_DB_PASSWORD not in captured.err


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
    call_argv = json.loads(argv_dump.read_text().splitlines()[0])
    assert "--stack" not in call_argv

    captured = capfd.readouterr()
    assert MARKER_DB_PASSWORD not in captured.out
    assert MARKER_DB_PASSWORD not in captured.err


def test_required_unresolved_path_fails_the_run(repo, monkeypatch, stub_pulumi, capfd):
    write_repo_config(
        repo,
        '[secrets."."]\n'
        'secret = { "db.password" = "DB_PASSWORD" }\n'
        'required = ["db.password"]\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    fake_pulumi(monkeypatch, stub_pulumi)

    assert main(["set-secrets"]) == 2
    err = capfd.readouterr().err
    assert "db.password" in err


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
    assert MARKER_DB_PASSWORD not in captured.out
    assert MARKER_DB_PASSWORD not in captured.err


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
    assert MARKER_DB_PASSWORD not in captured.out
    assert MARKER_DB_PASSWORD not in captured.err


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
    assert MARKER_PUBLISHED not in captured.err
    assert MARKER_BOOTSTRAP not in captured.err
    assert MARKER_PUBLISHED not in captured.out
    assert MARKER_BOOTSTRAP not in captured.out


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
    assert MARKER_BOOTSTRAP not in captured.err


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
