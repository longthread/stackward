"""Tests for `login`, `exec`, `shell`.

Two things this file is careful about, for the same reasons `test_store.py`
states them:

*No assertion depends on a credential value appearing in captured output.*
Where a value must not appear, the assertion is on absence in `capfd` output
(file-descriptor level, so a real child process's own writes are covered, not
only this process's `print` calls) -- never on a value being displayed.

*Several tests exist to fail against a plausible wrong implementation, not
merely to exercise a right one.* The environment-diff tests fail against an
implementation that replaces the environment instead of overlaying four names
onto it; the backend-guard ordering test fails against an implementation that
prompts for a password before checking the guard; the repo-profile test fails
if `select_profile`'s `repo_profile` parameter is never wired to anything (the
gap `store.py`'s own report flagged as dead code).

Every real child process spawned here is a single, environment-variable-driven
Python script (`_STUB`, written once per test by the `stub` fixture) rather
than `pulumi` itself: this module must never depend on the `pulumi` binary
being installed to prove exit-status, signal, or environment propagation --
those are properties of *this* code, not of Pulumi's.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from stackward import store
from stackward.cli import main
from stackward.commands import session
from stackward.commands.session import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    ENV_STORE_PASSWORD,
    PULUMI_BACKEND_URL,
    PULUMI_CONFIG_PASSPHRASE,
    SessionError,
)
from stackward.store import Profile, ProfileError, StoreError

PASSWORD = "the store password"
WRONG_PASSWORD = "not the store password"

# Placeholder values only -- see GC4 in the task brief. Distinctive enough
# that an accidental substring match elsewhere in a test's own output would
# be implausible.
MARKER_KEY = "marker-access-key-1a2b3c4d"
MARKER_SECRET = "marker-secret-key-5e6f7a8b"
MARKER_PASSPHRASE = "marker-passphrase-9c0d1e2f"

FULL_CREDENTIALS = {
    AWS_ACCESS_KEY_ID: MARKER_KEY,
    AWS_SECRET_ACCESS_KEY: MARKER_SECRET,
    PULUMI_CONFIG_PASSPHRASE: MARKER_PASSPHRASE,
}
ALL_MARKERS = (MARKER_KEY, MARKER_SECRET, MARKER_PASSPHRASE)


@pytest.fixture(autouse=True)
def cheap_kdf(monkeypatch):
    """Lower the Argon2id cost for this module only -- see `test_store.py`'s
    identical fixture. This file performs several real seal/open round trips
    per test (`init_store`, `set_credentials`, `resolve_credentials`)."""
    from stackward import crypto

    monkeypatch.setattr(crypto, "ARGON2ID_MEMORY_KIB", 8192)
    monkeypatch.setattr(crypto, "ARGON2ID_ITERATIONS", 1)
    monkeypatch.setattr(crypto, "ARGON2ID_LANES", 1)


@pytest.fixture(autouse=True)
def no_real_pulumi_state(monkeypatch, tmp_path):
    """Point `PULUMI_HOME` at an empty, per-test directory so no test in this
    file can read or depend on this machine's real `~/.pulumi`."""
    monkeypatch.setenv("PULUMI_HOME", str(tmp_path / "pulumi-home"))


@pytest.fixture
def store_directory(tmp_path) -> Path:
    return tmp_path / "stackward"


def write_store_config(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / store.CONFIG_FILENAME
    path.write_text(text)
    path.chmod(store.CONFIG_MODE)
    return path


def seed_profile(
    directory: Path,
    name: str,
    *,
    backend_url: str,
    credentials: dict[str, str] | None = None,
    password: str = PASSWORD,
) -> None:
    """A store holding one profile in `config`, with `credentials` sealed
    into it when given. Mirrors what a real `credentials init` +
    `credentials set` (Task 8's sibling commands) would leave behind."""
    write_store_config(directory, f'[profile.{name}]\nbackend_url = "{backend_url}"\n')
    if credentials is not None:
        store.init_store(password, directory=directory)
        store.set_credentials(name, credentials, password, directory=directory)


# ---------------------------------------------------------------------------
# A real child process, driven entirely by environment variables it reads
# itself -- never by argv, so it works identically as `exec`'s command or as
# `$SHELL` (which `shell` invokes with no arguments at all).
# ---------------------------------------------------------------------------

_STUB = """\
#!/usr/bin/env python3
import json, os, sys

dump_env_to = os.environ.get("STUB_DUMP_ENV_TO")
if dump_env_to:
    with open(dump_env_to, "w") as f:
        json.dump(dict(os.environ), f)

dump_argv_to = os.environ.get("STUB_DUMP_ARGV_TO")
if dump_argv_to:
    with open(dump_argv_to, "w") as f:
        json.dump(sys.argv[1:], f)

sig = os.environ.get("STUB_SIGNAL")
if sig:
    os.kill(os.getpid(), int(sig))

sys.exit(int(os.environ.get("STUB_EXIT_CODE", "0")))
"""


@pytest.fixture
def stub(tmp_path) -> Path:
    path = tmp_path / "stub.py"
    path.write_text(_STUB)
    path.chmod(0o755)
    return path


def fake_pulumi(monkeypatch, executable: Path) -> None:
    """Make `shutil.which("pulumi")` resolve to `executable` -- `cmd_login`'s
    only way of finding the real binary, and the seam every `login` test
    uses to run the stub in its place instead."""
    monkeypatch.setattr(
        session.shutil, "which", lambda name: str(executable) if name == "pulumi" else None
    )


# ---------------------------------------------------------------------------
# `_compose_backend_url` -- verbatim pass-through, and composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "s3://placeholder-bucket/placeholder-prefix",
        "gs://placeholder-bucket",
        "azblob://placeholder-container",
        "file:///placeholder/path",
        "s3://placeholder?endpoint=placeholder.invalid&region=placeholder-region",
        "some-future-scheme://placeholder;weird=chars&more=1",
    ],
)
def test_backend_url_passes_through_byte_for_byte(url):
    """No normalisation, re-encoding or reordering -- whatever a profile
    names is exactly what `login`/`exec`/`shell` use."""
    profile = Profile(name="p", backend_url=url)
    assert session._compose_backend_url(profile) == url


def test_component_form_with_bucket_only():
    profile = Profile(name="p", bucket="placeholder-bucket")
    assert session._compose_backend_url(profile) == "s3://placeholder-bucket"


def test_component_form_with_bucket_and_prefix():
    profile = Profile(name="p", bucket="placeholder-bucket", prefix="placeholder-prefix")
    assert (
        session._compose_backend_url(profile)
        == "s3://placeholder-bucket/placeholder-prefix"
    )


def test_component_form_with_region_only_adds_a_region_query_param():
    profile = Profile(name="p", bucket="placeholder-bucket", region="placeholder-region")
    url = session._compose_backend_url(profile)
    assert url == "s3://placeholder-bucket?region=placeholder-region"


def test_component_form_with_endpoint_forces_path_style():
    """A custom endpoint is exactly the case virtual-hosted-style addressing
    does not work for, so `s3ForcePathStyle` is added alongside it."""
    profile = Profile(name="p", bucket="placeholder-bucket", endpoint="placeholder.invalid")
    url = session._compose_backend_url(profile)
    assert "endpoint=placeholder.invalid" in url
    assert "s3ForcePathStyle=true" in url


def test_component_form_without_an_endpoint_never_adds_path_style():
    profile = Profile(name="p", bucket="placeholder-bucket", region="placeholder-region")
    assert "s3ForcePathStyle" not in session._compose_backend_url(profile)


def test_component_form_combines_every_field():
    profile = Profile(
        name="p",
        bucket="placeholder-bucket",
        prefix="placeholder-prefix",
        region="placeholder-region",
        endpoint="placeholder.invalid",
    )
    url = session._compose_backend_url(profile)
    assert url.startswith("s3://placeholder-bucket/placeholder-prefix?")
    assert "region=placeholder-region" in url
    assert "endpoint=placeholder.invalid" in url
    assert "s3ForcePathStyle=true" in url


def test_component_form_percent_encodes_special_characters():
    profile = Profile(name="p", bucket="a bucket/weird", prefix="a prefix?with=chars")
    url = session._compose_backend_url(profile)
    # The bucket's own '/' must be encoded (it is not a path separator here);
    # the prefix's '/' would be, but this prefix has none.
    assert "a%20bucket%2Fweird" in url
    assert "a%20prefix%3Fwith%3Dchars" in url


def test_an_empty_prefix_region_or_endpoint_is_treated_as_absent():
    profile = Profile(name="p", bucket="placeholder-bucket", prefix="", region="", endpoint="")
    assert session._compose_backend_url(profile) == "s3://placeholder-bucket"


def test_a_profile_with_no_backend_at_all_fails_closed_naming_the_profile():
    """Unreachable through `store._build_profile`, which already refuses to
    construct such a `Profile` -- covers a hand-built one instead, so this
    function never raises a bare `AttributeError`."""
    profile = Profile(name="broken-profile")
    with pytest.raises(SessionError, match="broken-profile"):
        session._compose_backend_url(profile)


# ---------------------------------------------------------------------------
# Reading Pulumi's current backend, and the guard built on it
# ---------------------------------------------------------------------------


def test_pulumi_home_defaults_when_the_env_var_is_absent(monkeypatch):
    monkeypatch.delenv("PULUMI_HOME", raising=False)
    assert session._pulumi_home() == Path.home() / ".pulumi"


def test_pulumi_home_honours_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("PULUMI_HOME", str(tmp_path / "custom"))
    assert session._pulumi_home() == tmp_path / "custom"


def test_current_backend_is_none_when_never_logged_in():
    assert session._current_backend() is None


def test_current_backend_reads_the_current_field(monkeypatch, tmp_path):
    home = tmp_path / "ph"
    monkeypatch.setenv("PULUMI_HOME", str(home))
    home.mkdir()
    (home / "credentials.json").write_text(json.dumps({"current": "file:///seen"}))
    assert session._current_backend() == "file:///seen"


def test_current_backend_is_none_when_the_field_is_absent_from_valid_json(
    monkeypatch, tmp_path
):
    """A credentials file with no `current` entry yet is the same legitimate
    "never logged in" state as no file at all -- not corruption."""
    home = tmp_path / "ph"
    monkeypatch.setenv("PULUMI_HOME", str(home))
    home.mkdir()
    (home / "credentials.json").write_text(json.dumps({"accounts": {}}))
    assert session._current_backend() is None


def test_current_backend_raises_on_malformed_json(monkeypatch, tmp_path):
    home = tmp_path / "ph"
    monkeypatch.setenv("PULUMI_HOME", str(home))
    home.mkdir()
    (home / "credentials.json").write_text("{not json")
    with pytest.raises(SessionError):
        session._current_backend()


def test_current_backend_raises_when_the_document_is_not_an_object(monkeypatch, tmp_path):
    home = tmp_path / "ph"
    monkeypatch.setenv("PULUMI_HOME", str(home))
    home.mkdir()
    (home / "credentials.json").write_text("[1, 2, 3]")
    with pytest.raises(SessionError):
        session._current_backend()


def test_current_backend_raises_when_current_is_not_a_string(monkeypatch, tmp_path):
    home = tmp_path / "ph"
    monkeypatch.setenv("PULUMI_HOME", str(home))
    home.mkdir()
    (home / "credentials.json").write_text(json.dumps({"current": 5}))
    with pytest.raises(SessionError):
        session._current_backend()


def test_current_backend_raises_on_an_unreadable_file_rather_than_treating_it_as_absent(
    monkeypatch, tmp_path
):
    """Fail closed: distinguishing "genuinely never logged in" (a missing
    file) from "cannot tell" (any other read failure) is what stops the guard
    itself from becoming a fail-open path."""
    home = tmp_path / "ph"
    home.mkdir()
    monkeypatch.setenv("PULUMI_HOME", str(home))
    (home / "credentials.json").write_text("{}")

    def boom(self, *a, **k):
        raise PermissionError("simulated: cannot read")

    monkeypatch.setattr(Path, "read_bytes", boom)
    with pytest.raises(SessionError):
        session._current_backend()


def test_guard_passes_when_never_logged_in_at_all(monkeypatch):
    monkeypatch.setattr(session, "_current_backend", lambda: None)
    session._check_backend_guard("file:///anything")  # must not raise


def test_guard_passes_on_an_exact_match(monkeypatch):
    monkeypatch.setattr(session, "_current_backend", lambda: "file:///same")
    session._check_backend_guard("file:///same")  # must not raise


def test_guard_refuses_a_mismatch_naming_both(monkeypatch):
    monkeypatch.setattr(session, "_current_backend", lambda: "file:///other")
    with pytest.raises(SessionError) as raised:
        session._check_backend_guard("file:///mine")
    message = str(raised.value)
    assert "file:///mine" in message
    assert "file:///other" in message


def test_guard_message_redacts_userinfo_in_both_urls(monkeypatch):
    """The guard's "naming both" message must never print a password that
    happened to be embedded in a backend URL (`postgres://user:pass@host/db`
    is a documented Pulumi backend form)."""
    monkeypatch.setattr(
        session, "_current_backend", lambda: "postgres://u:current-secret@host/db"
    )
    with pytest.raises(SessionError) as raised:
        session._check_backend_guard("postgres://u:resolved-secret@host/db")
    message = str(raised.value)
    assert "current-secret" not in message
    assert "resolved-secret" not in message
    assert "<redacted>@host/db" in message


def test_exec_refuses_against_a_real_disagreeing_credentials_file_on_disk(
    cli_store, monkeypatch, tmp_path, stub, capsys
):
    """The guard's two halves -- reading a real file, and comparing against
    it -- proven together, not each in isolation against a mock of the
    other."""
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    # `no_real_pulumi_state` (autouse) already points PULUMI_HOME here.
    home = Path(os.environ["PULUMI_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "credentials.json").write_text(
        json.dumps({"current": "file:///a-real-disagreement"})
    )
    marker_file = tmp_path / "ran"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(marker_file))

    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 2
    err = capsys.readouterr().err
    assert "file:///staging-backend" in err
    assert "file:///a-real-disagreement" in err
    assert not marker_file.exists()


# ---------------------------------------------------------------------------
# `_redact_url`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "s3://placeholder-bucket/placeholder-prefix",
        "file:///placeholder/path",
        "gs://placeholder-bucket",
    ],
)
def test_redact_url_leaves_a_userinfo_free_url_unchanged(url):
    assert session._redact_url(url) == url


def test_redact_url_replaces_userinfo_with_a_placeholder():
    url = "postgres://dbuser:super-secret@db.invalid:5432/state"
    redacted = session._redact_url(url)
    assert "super-secret" not in redacted
    assert "dbuser" not in redacted
    assert redacted == "postgres://<redacted>@db.invalid:5432/state"


def test_redact_url_handles_a_username_with_no_password():
    url = "postgres://dbuser@db.invalid:5432/state"
    assert session._redact_url(url) == "postgres://<redacted>@db.invalid:5432/state"


def test_redact_url_only_touches_the_first_at_sign():
    """A later `@` -- in a path or query string, say -- is not userinfo and
    must survive."""
    url = "postgres://dbuser:secret@db.invalid:5432/state?note=a@b"
    redacted = session._redact_url(url)
    assert "secret" not in redacted
    assert redacted.endswith("?note=a@b")


# ---------------------------------------------------------------------------
# `_current_backend` honouring an already-exported `PULUMI_BACKEND_URL`
# ---------------------------------------------------------------------------


def test_current_backend_prefers_an_exported_env_var_over_the_persisted_file(
    monkeypatch, tmp_path
):
    """Pulumi's own CLI honours `PULUMI_BACKEND_URL` over a persisted login,
    so a value already exported into the caller's shell is what a stray
    `pulumi` command would actually use -- and is what the guard must catch,
    even when the persisted file says something else entirely."""
    home = tmp_path / "ph"
    home.mkdir()
    monkeypatch.setenv("PULUMI_HOME", str(home))
    (home / "credentials.json").write_text(json.dumps({"current": "file:///from-the-file"}))
    monkeypatch.setenv(PULUMI_BACKEND_URL, "file:///from-the-env-var")
    assert session._current_backend() == "file:///from-the-env-var"


def test_current_backend_falls_back_to_the_file_when_the_env_var_is_absent(
    monkeypatch, tmp_path
):
    home = tmp_path / "ph"
    home.mkdir()
    monkeypatch.setenv("PULUMI_HOME", str(home))
    (home / "credentials.json").write_text(json.dumps({"current": "file:///from-the-file"}))
    monkeypatch.delenv(PULUMI_BACKEND_URL, raising=False)
    assert session._current_backend() == "file:///from-the-file"


def test_current_backend_treats_an_empty_env_var_as_absent(monkeypatch, tmp_path):
    home = tmp_path / "ph"
    home.mkdir()
    monkeypatch.setenv("PULUMI_HOME", str(home))
    (home / "credentials.json").write_text(json.dumps({"current": "file:///from-the-file"}))
    monkeypatch.setenv(PULUMI_BACKEND_URL, "")
    assert session._current_backend() == "file:///from-the-file"


# ---------------------------------------------------------------------------
# The store password: env var, interactive prompt, and the tty refusal --
# never `sys.stdin`.
# ---------------------------------------------------------------------------


class _PoisonedStdin:
    """Raises if anything reads from it. `_read_store_password` must never
    touch `sys.stdin` on any branch -- it is the child's, for `exec`/`shell`."""

    def read(self, *a, **k):
        raise AssertionError("_read_store_password touched sys.stdin")

    def readline(self, *a, **k):
        raise AssertionError("_read_store_password touched sys.stdin")


@pytest.fixture(autouse=True)
def poisoned_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _PoisonedStdin())


def test_password_from_env_var_is_used_without_prompting(monkeypatch):
    monkeypatch.setenv(ENV_STORE_PASSWORD, "from-the-environment")

    def boom(*a, **k):
        raise AssertionError("getpass.getpass must not be called when the env var is set")

    monkeypatch.setattr(session.getpass, "getpass", boom)
    assert session._read_store_password() == "from-the-environment"


def test_password_prompts_interactively_when_a_tty_is_available(monkeypatch):
    monkeypatch.delenv(ENV_STORE_PASSWORD, raising=False)
    monkeypatch.setattr(session, "_tty_available", lambda: True)
    monkeypatch.setattr(session.getpass, "getpass", lambda prompt="": "typed-password")
    assert session._read_store_password() == "typed-password"


def test_password_refuses_rather_than_falling_back_to_stdin_when_no_tty(monkeypatch):
    monkeypatch.delenv(ENV_STORE_PASSWORD, raising=False)
    monkeypatch.setattr(session, "_tty_available", lambda: False)

    def boom(*a, **k):
        raise AssertionError("getpass.getpass must not run without a tty")

    monkeypatch.setattr(session.getpass, "getpass", boom)
    with pytest.raises(SessionError, match=ENV_STORE_PASSWORD):
        session._read_store_password()


# ---------------------------------------------------------------------------
# Required credential names, and building the child environment
# ---------------------------------------------------------------------------


def test_all_three_names_present_is_accepted():
    session._require_credential_names(dict(FULL_CREDENTIALS), "p")  # must not raise


def test_a_missing_name_is_named_but_no_value_is():
    partial = dict(FULL_CREDENTIALS)
    del partial[AWS_SECRET_ACCESS_KEY]
    with pytest.raises(SessionError) as raised:
        session._require_credential_names(partial, "p")
    message = str(raised.value)
    assert AWS_SECRET_ACCESS_KEY in message
    for marker in ALL_MARKERS:
        assert marker not in message


def test_every_missing_name_is_named():
    with pytest.raises(SessionError) as raised:
        session._require_credential_names({}, "p")
    message = str(raised.value)
    for name in (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, PULUMI_CONFIG_PASSPHRASE):
        assert name in message


def test_extra_names_beyond_the_three_required_are_accepted():
    extra = dict(FULL_CREDENTIALS)
    extra["SOME_OTHER_NAME"] = "whatever"
    session._require_credential_names(extra, "p")  # must not raise


def test_child_env_overlays_exactly_four_names_on_a_copy_of_the_parent():
    parent_before = dict(os.environ)
    env = session._child_env(FULL_CREDENTIALS, "file:///backend")

    assert dict(os.environ) == parent_before  # the real environment is untouched

    parent_keys = set(parent_before)
    child_keys = set(env)
    assert child_keys - parent_keys == {
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        PULUMI_CONFIG_PASSPHRASE,
        PULUMI_BACKEND_URL,
    }
    assert parent_keys - child_keys == set()  # nothing dropped
    for key in parent_keys:
        assert env[key] == parent_before[key]  # nothing else changed
    assert env[AWS_ACCESS_KEY_ID] == MARKER_KEY
    assert env[AWS_SECRET_ACCESS_KEY] == MARKER_SECRET
    assert env[PULUMI_CONFIG_PASSPHRASE] == MARKER_PASSPHRASE
    assert env[PULUMI_BACKEND_URL] == "file:///backend"


# ---------------------------------------------------------------------------
# Exit status translation, including a signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [0, 1, 2, 42, 255])
def test_exit_status_passes_a_normal_code_through_unchanged(code):
    assert session._exit_status(code) == code


@pytest.mark.parametrize(
    "returncode,expected",
    [(-signal.SIGTERM, 128 + signal.SIGTERM), (-signal.SIGKILL, 128 + signal.SIGKILL)],
)
def test_exit_status_translates_a_negative_signal_return_code(returncode, expected):
    assert session._exit_status(returncode) == expected


# ---------------------------------------------------------------------------
# `_run_child` against a real subprocess
# ---------------------------------------------------------------------------


def test_run_child_propagates_a_nonzero_exit_code(stub):
    code = session._run_child([str(stub)], {**os.environ, "STUB_EXIT_CODE": "7"})
    assert code == 7


def test_run_child_propagates_a_signal_terminated_child(stub):
    code = session._run_child(
        [str(stub)], {**os.environ, "STUB_SIGNAL": str(int(signal.SIGTERM))}
    )
    assert code == 128 + signal.SIGTERM


def test_run_child_reports_a_missing_executable_without_a_traceback(tmp_path, capsys):
    missing = tmp_path / "does-not-exist"
    code = session._run_child([str(missing)])
    assert code == 2
    assert "does-not-exist" in capsys.readouterr().err


def test_run_child_passes_env_none_through_to_inherit_unchanged(monkeypatch, stub):
    monkeypatch.setenv("STACKWARD_TEST_MARKER_VAR", "seen")
    dump = None

    def fake_run(command, env=None):
        nonlocal dump
        dump = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(session.subprocess, "run", fake_run)
    session._run_child([str(stub)])
    assert dump is None


def test_run_child_catches_a_keyboard_interrupt_and_returns_the_sigint_status(monkeypatch):
    """`fail_closed` cannot catch this -- `KeyboardInterrupt` inherits from
    `BaseException`, not `Exception` -- so `_run_child` must, or a `Ctrl-C`
    during `stackward exec -- pulumi up` prints a raw traceback."""

    def fake_run(command, env=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(session.subprocess, "run", fake_run)
    assert session._run_child(["irrelevant"]) == 128 + signal.SIGINT


def test_exec_prints_no_traceback_on_a_keyboard_interrupt(
    cli_store, monkeypatch, stub, capsys
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)

    def fake_run(command, env=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(session.subprocess, "run", fake_run)
    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 128 + signal.SIGINT
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "KeyboardInterrupt" not in captured.err


# ---------------------------------------------------------------------------
# `_resolve_profile` -- every precedence branch, including the repo-config
# tier this task is responsible for wiring.
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path) -> Path:
    """A directory tree `find_repo_config` will search from, bounded by a
    `.git` marker -- no real git repository is needed for that function."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    return root


def test_explicit_profile_resolves_regardless_of_other_sources(store_directory):
    """`--profile` at the wiring level: `select_profile` itself already
    proves (in `test_store.py`) that an explicit value wins over a competing
    `repo_profile` string; `_resolve_profile` goes one step further and never
    even calls `_repo_profile()` when `explicit` is given (see
    `test_an_explicit_profile_bypasses_a_broken_repo_config`, which proves
    that specifically) -- so there is no competing repository profile for
    this test to write, and doing so would only look like it was proving
    something it no longer does."""
    write_store_config(store_directory, '[profile.chosen]\nbackend_url = "file:///chosen"\n')
    profile = session._resolve_profile("chosen", directory=store_directory)
    assert profile.name == "chosen"


def test_env_var_wins_over_repo_and_default(store_directory, repo, monkeypatch):
    write_store_config(
        store_directory,
        'default_profile = "fallback"\n'
        '[profile.fallback]\nbackend_url = "file:///fallback"\n'
        '[profile.from_repo]\nbackend_url = "file:///from-repo"\n'
        '[profile.from_env]\nbackend_url = "file:///from-env"\n',
    )
    (repo / ".stackward.toml").write_text('profile = "from_repo"\n')
    monkeypatch.chdir(repo)
    monkeypatch.setenv(store.ENV_PROFILE, "from_env")
    profile = session._resolve_profile(None, directory=store_directory)
    assert profile.name == "from_env"


def test_the_repo_profile_tier_is_wired_and_beats_the_default(
    store_directory, repo, monkeypatch
):
    """The tier Task 6's own report flagged as dead code: `select_profile`
    has always accepted `repo_profile`, but nothing supplied it until this
    module read `.stackward.toml` itself. This is the normal case -- a
    repository's stacks live in one backend -- so it must win over
    `default_profile`, not merely be reachable."""
    write_store_config(
        store_directory,
        'default_profile = "fallback"\n'
        '[profile.fallback]\nbackend_url = "file:///fallback"\n'
        '[profile.from_repo]\nbackend_url = "file:///from-repo"\n',
    )
    (repo / ".stackward.toml").write_text('profile = "from_repo"\n')
    monkeypatch.chdir(repo)
    monkeypatch.delenv(store.ENV_PROFILE, raising=False)
    profile = session._resolve_profile(None, directory=store_directory)
    assert profile.name == "from_repo"


def test_default_profile_is_the_last_resort(store_directory, repo, monkeypatch):
    write_store_config(
        store_directory,
        'default_profile = "fallback"\n[profile.fallback]\nbackend_url = "file:///fallback"\n',
    )
    monkeypatch.chdir(repo)  # no .stackward.toml here at all
    monkeypatch.delenv(store.ENV_PROFILE, raising=False)
    profile = session._resolve_profile(None, directory=store_directory)
    assert profile.name == "fallback"


def test_no_profile_anywhere_is_a_profile_error(store_directory, repo, monkeypatch):
    write_store_config(store_directory, '[profile.unrelated]\nbackend_url = "file:///x"\n')
    monkeypatch.chdir(repo)
    monkeypatch.delenv(store.ENV_PROFILE, raising=False)
    with pytest.raises(ProfileError):
        session._resolve_profile(None, directory=store_directory)


def test_absent_repo_config_falls_through_cleanly_to_the_default(
    store_directory, repo, monkeypatch
):
    """No `.stackward.toml` at all is a normal state, not an error -- the
    repo tier simply does not speak, and selection moves on."""
    write_store_config(
        store_directory,
        'default_profile = "fallback"\n[profile.fallback]\nbackend_url = "file:///fallback"\n',
    )
    monkeypatch.chdir(repo)
    monkeypatch.delenv(store.ENV_PROFILE, raising=False)
    assert session._repo_profile() is None
    profile = session._resolve_profile(None, directory=store_directory)
    assert profile.name == "fallback"


def test_an_explicit_profile_bypasses_a_broken_repo_config(
    store_directory, repo, monkeypatch
):
    """`--profile` is self-sufficient: it must not require an unrelated,
    broken `.stackward.toml` elsewhere in the repository to be fixed first."""
    write_store_config(store_directory, '[profile.chosen]\nbackend_url = "file:///chosen"\n')
    (repo / ".stackward.toml").write_text("[profile\nbroken = ")
    monkeypatch.chdir(repo)
    profile = session._resolve_profile("chosen", directory=store_directory)
    assert profile.name == "chosen"


def test_a_malformed_repo_config_is_a_config_error_not_a_silent_miss(repo, monkeypatch):
    """A `.stackward.toml` that fails to parse must not be treated the same
    as no file at all -- that would turn a broken repo policy into a silently
    missing one."""
    (repo / ".stackward.toml").write_text("[profile\nbroken = ")
    monkeypatch.chdir(repo)
    from stackward.config import ConfigError

    with pytest.raises(ConfigError):
        session._repo_profile()


def test_a_malformed_repo_config_is_reported_by_login_as_a_could_not_run_error(
    cli_store, repo, monkeypatch, capsys
):
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    (repo / ".stackward.toml").write_text("[profile\nbroken = ")
    monkeypatch.chdir(repo)

    code = main(["login"])

    assert code == 2
    assert "invalid TOML" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `_prepare_env` -- the full pipeline `exec`/`shell` share, at the unit level
# ---------------------------------------------------------------------------


def test_prepare_env_happy_path_builds_the_expected_environment(store_directory, monkeypatch):
    seed_profile(
        store_directory,
        "staging",
        backend_url="file:///staging-backend",
        credentials=FULL_CREDENTIALS,
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    env = session._prepare_env("staging", directory=store_directory)
    assert env[AWS_ACCESS_KEY_ID] == MARKER_KEY
    assert env[AWS_SECRET_ACCESS_KEY] == MARKER_SECRET
    assert env[PULUMI_CONFIG_PASSPHRASE] == MARKER_PASSPHRASE
    assert env[PULUMI_BACKEND_URL] == "file:///staging-backend"


def test_prepare_env_checks_the_backend_guard_before_reading_a_password(
    store_directory, monkeypatch
):
    """Ordering matters: a mismatch must be reported without ever prompting
    for -- or requiring -- a password."""
    seed_profile(
        store_directory,
        "staging",
        backend_url="file:///staging-backend",
        credentials=FULL_CREDENTIALS,
    )
    monkeypatch.setattr(session, "_current_backend", lambda: "file:///somewhere-else")

    def boom():
        raise AssertionError("password must not be read when the guard refuses")

    monkeypatch.setattr(session, "_read_store_password", boom)
    with pytest.raises(SessionError, match="backend mismatch"):
        session._prepare_env("staging", directory=store_directory)


def test_prepare_env_allows_a_matching_backend(store_directory, monkeypatch):
    seed_profile(
        store_directory,
        "staging",
        backend_url="file:///staging-backend",
        credentials=FULL_CREDENTIALS,
    )
    monkeypatch.setattr(session, "_current_backend", lambda: "file:///staging-backend")
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    env = session._prepare_env("staging", directory=store_directory)
    assert env[PULUMI_BACKEND_URL] == "file:///staging-backend"


def test_prepare_env_wrong_password_raises_and_leaks_nothing(store_directory, monkeypatch):
    seed_profile(
        store_directory,
        "staging",
        backend_url="file:///staging-backend",
        credentials=FULL_CREDENTIALS,
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, WRONG_PASSWORD)
    with pytest.raises(StoreError) as raised:
        session._prepare_env("staging", directory=store_directory)
    message = str(raised.value)
    assert WRONG_PASSWORD not in message
    for marker in ALL_MARKERS:
        assert marker not in message


def test_prepare_env_missing_credential_name_raises_naming_it(store_directory, monkeypatch):
    incomplete = {AWS_ACCESS_KEY_ID: MARKER_KEY, AWS_SECRET_ACCESS_KEY: MARKER_SECRET}
    seed_profile(
        store_directory, "staging", backend_url="file:///staging-backend", credentials=incomplete
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    with pytest.raises(SessionError, match=PULUMI_CONFIG_PASSPHRASE):
        session._prepare_env("staging", directory=store_directory)


def test_prepare_env_profile_with_no_credentials_set_raises(store_directory, monkeypatch):
    """A profile in `config` with no envelope selects but cannot resolve --
    R3 from Task 6's store, exercised through this module's own pipeline."""
    seed_profile(store_directory, "staging", backend_url="file:///staging-backend")
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    with pytest.raises(StoreError):
        session._prepare_env("staging", directory=store_directory)


# ---------------------------------------------------------------------------
# Full CLI wiring: `main([...])`, real subprocesses, `capfd`-level secrecy.
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_store(tmp_path, monkeypatch) -> Path:
    """Points the real config-home lookup (`XDG_CONFIG_HOME`) at an isolated
    directory, the way `cli.config_home` is actually consulted by every
    `cmd_*` entry point -- as opposed to the `directory=` seam the unit tests
    above use directly."""
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home / "stackward"


def test_exec_with_no_command_is_a_usage_error(monkeypatch, capsys):
    assert main(["exec"]) == 2
    assert "requires a command" in capsys.readouterr().err


def test_exec_with_only_a_bare_separator_is_a_usage_error(capsys):
    assert main(["exec", "--"]) == 2
    assert "requires a command" in capsys.readouterr().err


def test_exec_usage_error_is_reported_before_any_profile_resolution(monkeypatch, capsys):
    """If the usage check ran after resolution instead of before it, this
    would still exit 2 -- but via `fail_closed`'s generic catch-all around
    the poisoned `_resolve_profile` below, with a *different* message. The
    message assertion, not just the exit code, is what makes this
    discriminate between the two orderings."""
    monkeypatch.setattr(
        session,
        "_resolve_profile",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("profile resolution must not run before the usage check")
        ),
    )
    assert main(["exec"]) == 2
    err = capsys.readouterr().err
    assert "requires a command" in err
    assert "AssertionError" not in err


def test_shell_with_no_shell_env_var_errors(monkeypatch, capsys):
    monkeypatch.delenv("SHELL", raising=False)
    assert main(["shell"]) == 2
    assert "SHELL" in capsys.readouterr().err


def test_exec_runs_the_child_with_exactly_the_expected_environment_delta(
    cli_store, monkeypatch, tmp_path, stub, capfd
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    dump = tmp_path / "env.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump))
    parent_before = dict(os.environ)

    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 0
    child_env = json.loads(dump.read_text())
    parent_keys = set(parent_before)
    child_keys = set(child_env)
    assert child_keys - parent_keys == {
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        PULUMI_CONFIG_PASSPHRASE,
        PULUMI_BACKEND_URL,
    }
    # The one name that must be *dropped*, not merely left alone: the store
    # master password unlocks every profile, not just this one, and must
    # never reach caller-supplied code.
    assert parent_keys - child_keys == {ENV_STORE_PASSWORD}
    assert ENV_STORE_PASSWORD not in child_env
    for key in parent_keys - {"STUB_DUMP_ENV_TO", ENV_STORE_PASSWORD}:
        assert child_env[key] == parent_before[key]
    assert child_env[AWS_ACCESS_KEY_ID] == MARKER_KEY
    assert child_env[AWS_SECRET_ACCESS_KEY] == MARKER_SECRET
    assert child_env[PULUMI_CONFIG_PASSPHRASE] == MARKER_PASSPHRASE
    assert child_env[PULUMI_BACKEND_URL] == "file:///staging-backend"
    assert PASSWORD not in child_env.values()

    captured = capfd.readouterr()
    for marker in (*ALL_MARKERS, PASSWORD):
        assert marker not in captured.out
        assert marker not in captured.err


def test_exec_passes_extra_flags_through_to_the_child_argv(
    cli_store, monkeypatch, tmp_path, stub
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    dump = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(dump))

    code = main(["exec", "--profile", "staging", "--", str(stub), "up", "--yes"])

    assert code == 0
    assert json.loads(dump.read_text()) == ["up", "--yes"]


def test_exec_dash_dash_help_runs_the_child_rather_than_printing_stackwards_help(
    cli_store, monkeypatch, tmp_path, stub, capsys
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    dump = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(dump))

    code = main(["exec", "--profile", "staging", "--", str(stub), "--help"])

    assert code == 0
    assert json.loads(dump.read_text()) == ["--help"]
    assert "usage:" not in capsys.readouterr().out


def test_a_second_separator_survives_and_reaches_the_child(
    cli_store, monkeypatch, tmp_path, stub
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    dump = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(dump))

    code = main(["exec", "--profile", "staging", "--", str(stub), "--", "pulumi"])

    assert code == 0
    assert json.loads(dump.read_text()) == ["--", "pulumi"]


def test_exec_propagates_a_nonzero_exit_code_from_a_real_child(
    cli_store, monkeypatch, stub, capfd
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    monkeypatch.setenv("STUB_EXIT_CODE", "3")

    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 3
    captured = capfd.readouterr()
    for marker in ALL_MARKERS:
        assert marker not in captured.out
        assert marker not in captured.err


def test_exec_propagates_a_signal_terminated_child(cli_store, monkeypatch, stub, capfd):
    """The one path where decrypted credential values are genuinely live in
    this process's memory at the moment the child dies -- the case a sweep
    over only early-failing paths would miss entirely."""
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    monkeypatch.setenv("STUB_SIGNAL", str(int(signal.SIGTERM)))

    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 128 + signal.SIGTERM
    captured = capfd.readouterr()
    for marker in ALL_MARKERS:
        assert marker not in captured.out
        assert marker not in captured.err


def test_exec_refuses_a_backend_mismatch_and_never_runs_the_child(
    cli_store, monkeypatch, tmp_path, stub, capsys
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    marker_file = tmp_path / "ran"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(marker_file))
    monkeypatch.setattr(session, "_current_backend", lambda: "file:///a-different-backend")

    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 2
    err = capsys.readouterr().err
    assert "file:///staging-backend" in err
    assert "file:///a-different-backend" in err
    assert not marker_file.exists()


def test_exec_with_wrong_password_refuses_and_never_runs_the_child(
    cli_store, monkeypatch, tmp_path, stub, capsys
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, WRONG_PASSWORD)
    marker_file = tmp_path / "ran"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(marker_file))

    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 2
    captured = capsys.readouterr()
    assert not marker_file.exists()
    for marker in (WRONG_PASSWORD, *ALL_MARKERS):
        assert marker not in captured.out
        assert marker not in captured.err


def test_exec_with_no_password_available_refuses_and_never_runs_the_child(
    cli_store, monkeypatch, tmp_path, stub, capsys
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.delenv(ENV_STORE_PASSWORD, raising=False)
    monkeypatch.setattr(session, "_tty_available", lambda: False)
    marker_file = tmp_path / "ran"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(marker_file))

    code = main(["exec", "--profile", "staging", "--", str(stub)])

    assert code == 2
    assert not marker_file.exists()
    captured = capsys.readouterr()
    # Must be *this* module's own refusal, naming the escape hatch -- not
    # `fail_closed`'s generic catch-all, which an unguarded `getpass` call
    # could also route through by raising some other exception once it hits
    # the poisoned stdin in this sandbox (no controlling terminal is ever
    # available while running under pytest).
    assert ENV_STORE_PASSWORD in captured.err
    assert "RuntimeError" not in captured.err
    assert "AssertionError" not in captured.err
    for marker in ALL_MARKERS:
        assert marker not in captured.out
        assert marker not in captured.err


def test_exec_reports_an_unexpected_exception_via_fail_closed(
    cli_store, monkeypatch, stub, capsys
):
    """Proves `@fail_closed` is actually wired on `cmd_exec`: an
    unanticipated exception must exit 2 with a type-only message, never a
    traceback and never Python's own default exit code of 1."""
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)

    def boom(*a, **k):
        raise RuntimeError("unanticipated")

    monkeypatch.setattr(session, "_compose_backend_url", boom)
    code = main(["exec", "--profile", "staging", "--", str(stub)])
    assert code == 2
    err = capsys.readouterr().err
    assert "RuntimeError" in err
    assert "unanticipated" not in err


def test_exec_uses_the_wired_repo_profile_tier_end_to_end(
    cli_store, monkeypatch, tmp_path, stub
):
    seed_profile(
        cli_store, "from_repo", backend_url="file:///from-repo-backend", credentials=FULL_CREDENTIALS
    )
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".stackward.toml").write_text('profile = "from_repo"\n')
    monkeypatch.chdir(repo)
    monkeypatch.delenv(store.ENV_PROFILE, raising=False)
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    dump = tmp_path / "env.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump))

    code = main(["exec", "--", str(stub)])

    assert code == 0
    child_env = json.loads(dump.read_text())
    assert child_env[PULUMI_BACKEND_URL] == "file:///from-repo-backend"


def test_shell_runs_stub_as_the_shell_with_the_expected_environment_delta(
    cli_store, monkeypatch, tmp_path, stub
):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv("SHELL", str(stub))
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    dump = tmp_path / "env.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump))

    code = main(["shell", "--profile", "staging"])

    assert code == 0
    child_env = json.loads(dump.read_text())
    assert child_env[PULUMI_BACKEND_URL] == "file:///staging-backend"
    assert child_env[AWS_ACCESS_KEY_ID] == MARKER_KEY


def test_shell_propagates_a_signal_terminated_child(cli_store, monkeypatch, stub, capfd):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv("SHELL", str(stub))
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    monkeypatch.setenv("STUB_SIGNAL", str(int(signal.SIGTERM)))

    code = main(["shell", "--profile", "staging"])

    assert code == 128 + signal.SIGTERM
    captured = capfd.readouterr()
    for marker in ALL_MARKERS:
        assert marker not in captured.out
        assert marker not in captured.err


def test_shell_backend_mismatch_refuses(cli_store, monkeypatch, tmp_path, stub, capsys):
    seed_profile(
        cli_store, "staging", backend_url="file:///staging-backend", credentials=FULL_CREDENTIALS
    )
    monkeypatch.setenv("SHELL", str(stub))
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    monkeypatch.setattr(session, "_current_backend", lambda: "file:///elsewhere")

    code = main(["shell", "--profile", "staging"])

    assert code == 2
    err = capsys.readouterr().err
    assert "file:///staging-backend" in err
    assert "file:///elsewhere" in err


# ---------------------------------------------------------------------------
# `login`
# ---------------------------------------------------------------------------


def test_login_runs_pulumi_login_with_the_url_via_env_not_argv(
    cli_store, monkeypatch, tmp_path, stub
):
    """The URL reaches `pulumi login` as `PULUMI_BACKEND_URL`, never as a
    positional argument -- some backend forms (`postgres://user:pass@host/db`)
    can carry a plaintext password in the URL itself, and argv is visible in
    `ps` for the life of the call."""
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    argv_dump = tmp_path / "argv.json"
    env_dump = tmp_path / "env.json"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(env_dump))
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 0
    assert json.loads(argv_dump.read_text()) == ["login"]
    child_env = json.loads(env_dump.read_text())
    assert child_env[PULUMI_BACKEND_URL] == "file:///staging-backend"


def test_login_never_opens_the_credentials_store(cli_store, monkeypatch, tmp_path, stub):
    """A profile can be logged into before its credentials have ever been
    set -- `store.py`'s own R3, exercised here: no `credentials` file exists
    at all, and `login` must still succeed."""
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    assert not (cli_store / store.CREDENTIALS_FILENAME).exists()
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 0
    assert not (cli_store / store.CREDENTIALS_FILENAME).exists()


def test_login_composes_the_component_form(cli_store, monkeypatch, tmp_path, stub):
    write_store_config(
        cli_store,
        '[profile.staging]\nbucket = "placeholder-bucket"\nregion = "placeholder-region"\n',
    )
    dump = tmp_path / "env.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump))
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 0
    child_env = json.loads(dump.read_text())
    assert child_env[PULUMI_BACKEND_URL] == "s3://placeholder-bucket?region=placeholder-region"


def test_login_scrubs_the_store_password_from_pulumis_environment(
    cli_store, monkeypatch, tmp_path, stub
):
    """`login` never needs the store password -- it never opens the
    credentials store -- but it must not hand one along to `pulumi` anyway,
    for the same reason `exec`/`shell` scrub it."""
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    dump = tmp_path / "env.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump))
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 0
    child_env = json.loads(dump.read_text())
    assert ENV_STORE_PASSWORD not in child_env
    assert PASSWORD not in child_env.values()


def test_login_never_puts_a_userinfo_bearing_url_in_argv_or_stderr(
    cli_store, monkeypatch, tmp_path, stub, capsys
):
    """A `postgres://user:password@host/db` profile -- a form
    `pulumi login --help` documents -- must never put its password where
    `ps` or a printed message could show it."""
    secret_url = "postgres://dbuser:super-secret-password@db.invalid:5432/state"
    seed_profile(cli_store, "staging", backend_url=secret_url)
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 0
    argv = json.loads(argv_dump.read_text())
    assert argv == ["login"]
    for text in argv:
        assert "super-secret-password" not in text
    captured = capsys.readouterr()
    assert "super-secret-password" not in captured.out
    assert "super-secret-password" not in captured.err


def test_login_reports_a_missing_pulumi_executable(cli_store, monkeypatch, capsys):
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    monkeypatch.setattr(session.shutil, "which", lambda name: None)

    code = main(["login", "--profile", "staging"])

    assert code == 2
    assert "pulumi" in capsys.readouterr().err


def test_login_reports_profile_resolution_failure(cli_store, capsys):
    code = main(["login", "--profile", "nonexistent"])
    assert code == 2
    assert "nonexistent" in capsys.readouterr().err


def test_login_propagates_a_nonzero_exit_code(cli_store, monkeypatch, stub):
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    monkeypatch.setenv("STUB_EXIT_CODE", "9")
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 9


def test_login_propagates_a_signal_terminated_child(cli_store, monkeypatch, stub):
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    monkeypatch.setenv("STUB_SIGNAL", str(int(signal.SIGTERM)))
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 128 + signal.SIGTERM


def test_login_is_not_subject_to_the_backend_guard(cli_store, monkeypatch, tmp_path, stub):
    """`login` is what *sets* the backend Pulumi is pointed at -- gating it on
    matching that same state would make switching profiles impossible."""
    seed_profile(cli_store, "staging", backend_url="file:///staging-backend")
    monkeypatch.setattr(session, "_current_backend", lambda: "file:///a-totally-different-place")
    fake_pulumi(monkeypatch, stub)

    code = main(["login", "--profile", "staging"])

    assert code == 0
