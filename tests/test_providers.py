"""Tests for the provider interface (Task 11).

**The centerpiece is the contract test.** The whole point of this task is
that swapping `provider = "..."` changes nothing a caller can observe —
`test_dotenv_and_a_fake_remote_produce_identical_dry_run_outcomes`,
`test_dotenv_and_a_fake_remote_produce_identical_pulumi_invocations` and
`test_file_and_a_fake_remote_produce_identical_child_environments` seed a
`dotenv`/`file` provider and a fake **remote** provider (`providers.command`,
driven by a real subprocess script — never a Python object standing in for
"remote", which would prove nothing about the shell-out requirement) with
the same values and assert the real `set-secrets`/`exec` entry points behave
identically: same printed outcomes, the same `pulumi` argv *and* stdin
byte-for-byte, the same injected environment.

**The gate-path invariant is enforced, not merely stated.**
`test_the_check_config_entry_point_never_loads_a_provider_module` (and its
stronger sibling that runs the command through `cli.main`) checks
`sys.modules` in a subprocess — this test process has already imported half
the tool by the time either test runs, so asserting against its own
`sys.modules` would prove nothing.

**No test ever prints a credential value.** Every marker below is a
placeholder distinctive enough that an accidental substring match in
captured output would be implausible (GC4); assertions are on absence, on
exception types, or on values compared to a known constant, never on a
value's appearance in output being treated as proof of anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from stackward import store
from stackward.cli import main
from stackward.commands import session, set_secrets
from stackward.commands.session import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    CREDENTIAL_NAMES,
    ENV_STORE_PASSWORD,
    PULUMI_BACKEND_URL,
    PULUMI_CONFIG_PASSPHRASE,
    SessionError,
)
from stackward.commands.set_secrets import SetSecretsError
from stackward.config import find_repo_config, load_config
from stackward.providers import CredentialStore, ProviderError, SecretSource
from stackward.providers.command import CommandCredentialStore, CommandSecretSource
from stackward.providers.dotenv import DotenvSecretSource
from stackward.providers.env import EnvCredentialStore, EnvSecretSource
from stackward.providers.file import FileCredentialStore
from stackward.store import Profile, StoreError

PASSWORD = "the store password"
WRONG_PASSWORD = "not the store password"

# Placeholder values only -- see GC4. Distinctive enough that an accidental
# substring match elsewhere in captured output would be implausible.
MARKER_KEY = "marker-access-key-1a2b3c4d"
MARKER_SECRET = "marker-secret-key-5e6f7a8b"
MARKER_PASSPHRASE = "marker-passphrase-9c0d1e2f"
MARKER_DB_PASSWORD = "marker-db-password-1a2b3c4d"
MARKER_API_TOKEN = "marker-api-token-5e6f7a8b"

FULL_CREDENTIALS = {
    AWS_ACCESS_KEY_ID: MARKER_KEY,
    AWS_SECRET_ACCESS_KEY: MARKER_SECRET,
    PULUMI_CONFIG_PASSPHRASE: MARKER_PASSPHRASE,
}


@pytest.fixture(autouse=True)
def cheap_kdf(monkeypatch):
    """Lower the Argon2id cost for this module only -- see `test_store.py`'s
    identical fixture. `FileCredentialStore` performs real seal/open round
    trips."""
    from stackward import crypto

    monkeypatch.setattr(crypto, "ARGON2ID_MEMORY_KIB", 8192)
    monkeypatch.setattr(crypto, "ARGON2ID_ITERATIONS", 1)
    monkeypatch.setattr(crypto, "ARGON2ID_LANES", 1)


@pytest.fixture(autouse=True)
def no_real_pulumi_state(monkeypatch, tmp_path):
    """Point `PULUMI_HOME` at an empty, per-test directory, and clear
    `PULUMI_BACKEND_URL` -- see `test_session.py`'s identical fixture. No
    test in this file may read or depend on this machine's real Pulumi
    state."""
    monkeypatch.setenv("PULUMI_HOME", str(tmp_path / "pulumi-home"))
    monkeypatch.delenv(PULUMI_BACKEND_URL, raising=False)


# ---------------------------------------------------------------------------
# A real "remote" CLI, driven entirely by environment variables -- never a
# Python object standing in for "remote". `STUB_MODE` picks the wire shape:
# "secret" (one name in, one value or nothing on stdout -- `CommandSecretSource`)
# or "credentials" (one profile in, a JSON object out -- `CommandCredentialStore`).
# ---------------------------------------------------------------------------

_REMOTE_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys, time

    sleep_seconds = os.environ.get("STUB_SLEEP_SECONDS")
    if sleep_seconds:
        time.sleep(float(sleep_seconds))

    exit_code = int(os.environ.get("STUB_EXIT_CODE", "0"))
    if exit_code != 0:
        sys.exit(exit_code)

    mode = os.environ.get("STUB_MODE", "secret")
    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "credentials":
        values = json.loads(os.environ.get("STUB_CREDENTIALS", "{}"))
        sys.stdout.write(json.dumps(values))
    elif mode == "raw":
        sys.stdout.write(os.environ.get("STUB_RAW_OUTPUT", ""))
    else:
        values = json.loads(os.environ.get("STUB_VALUES", "{}"))
        sys.stdout.write(values.get(arg, ""))
    """
)


@pytest.fixture
def remote_stub(tmp_path) -> Path:
    path = tmp_path / "remote-stub.py"
    path.write_text(_REMOTE_STUB)
    path.chmod(0o755)
    return path


def remote_command(stub: Path) -> list[str]:
    """The `command` list a `[secrets.source]`/`[profile.<n>.credentials]`
    table would declare -- `sys.executable` first, since the stub is a
    Python script and not directly executable on every platform this suite
    might run on."""
    return [sys.executable, str(stub)]


# A minimal `pulumi` stand-in: records every invocation's argv and stdin,
# then reports success for a `set`. No `get` support -- no test in this file
# declares a `drift_pairs` entry, so `_check_drift` never calls one.
_PULUMI_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys

    argv = sys.argv[1:]
    dump_argv_to = os.environ.get("STUB_DUMP_ARGV_TO")
    if dump_argv_to:
        with open(dump_argv_to, "a") as f:
            f.write(json.dumps(argv) + "\\n")

    if "set" in argv:
        data = sys.stdin.buffer.read()
        dump_stdin_to = os.environ.get("STUB_DUMP_STDIN_TO")
        if dump_stdin_to:
            with open(dump_stdin_to, "ab") as f:
                f.write(data + b"\\n")
        sys.exit(0)

    sys.exit(1)
    """
)


@pytest.fixture
def pulumi_stub(tmp_path) -> Path:
    path = tmp_path / "pulumi-stub.py"
    path.write_text(_PULUMI_STUB)
    path.chmod(0o755)
    return path


def fake_pulumi(monkeypatch, executable: Path) -> None:
    monkeypatch.setattr(
        set_secrets.shutil, "which", lambda name: str(executable) if name == "pulumi" else None
    )


# A real child, dumping its own environment -- see `test_session.py`'s `_STUB`.
_ENV_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys

    dump_env_to = os.environ.get("STUB_DUMP_ENV_TO")
    if dump_env_to:
        with open(dump_env_to, "w") as f:
            json.dump(dict(os.environ), f)
    sys.exit(0)
    """
)


@pytest.fixture
def env_stub(tmp_path) -> Path:
    path = tmp_path / "env-stub.py"
    path.write_text(_ENV_STUB)
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# Repos and stores -- mirrors `test_set_secrets.py`/`test_session.py`.
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


@pytest.fixture
def store_directory(tmp_path) -> Path:
    return tmp_path / "stackward"


def write_store_config(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / store.CONFIG_FILENAME
    path.write_text(text)
    path.chmod(store.CONFIG_MODE)
    return path


# ---------------------------------------------------------------------------
# `FileCredentialStore` -- the thin adapter over `store.resolve_credentials`
# ---------------------------------------------------------------------------


def test_file_credential_store_resolves_exactly_what_the_store_holds(store_directory):
    store.init_store(PASSWORD, directory=store_directory)
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=store_directory)

    provider = FileCredentialStore(PASSWORD, directory=store_directory)
    assert provider.resolve("staging") == FULL_CREDENTIALS


def test_file_credential_store_propagates_store_error_for_a_wrong_password(store_directory):
    """Deliberately *not* wrapped in `ProviderError` -- see `providers.file`'s
    own docstring on why `StoreError` propagates unchanged."""
    store.init_store(PASSWORD, directory=store_directory)
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=store_directory)

    provider = FileCredentialStore(WRONG_PASSWORD, directory=store_directory)
    with pytest.raises(StoreError):
        provider.resolve("staging")


def test_file_credential_store_satisfies_the_credential_store_protocol(store_directory):
    store.init_store(PASSWORD, directory=store_directory)
    provider = FileCredentialStore(PASSWORD, directory=store_directory)
    assert isinstance(provider, CredentialStore)


# ---------------------------------------------------------------------------
# `EnvCredentialStore`
# ---------------------------------------------------------------------------


def test_env_credential_store_reads_only_the_requested_names():
    environ = {
        AWS_ACCESS_KEY_ID: MARKER_KEY,
        AWS_SECRET_ACCESS_KEY: MARKER_SECRET,
        "UNRELATED_VARIABLE": "should never appear",
    }
    provider = EnvCredentialStore(CREDENTIAL_NAMES, environ)
    resolved = provider.resolve("any-profile-name")
    assert resolved == {AWS_ACCESS_KEY_ID: MARKER_KEY, AWS_SECRET_ACCESS_KEY: MARKER_SECRET}
    assert "UNRELATED_VARIABLE" not in resolved


def test_env_credential_store_ignores_the_profile_argument():
    """A CI runner's environment is not partitioned by profile -- whichever
    profile name is asked for gets the same answer."""
    environ = {AWS_ACCESS_KEY_ID: MARKER_KEY}
    provider = EnvCredentialStore((AWS_ACCESS_KEY_ID,), environ)
    assert provider.resolve("staging") == provider.resolve("anything-else")


def test_env_credential_store_omits_a_name_it_does_not_have_rather_than_erroring():
    provider = EnvCredentialStore((AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), {})
    assert provider.resolve("staging") == {}


def test_env_credential_store_satisfies_the_credential_store_protocol():
    assert isinstance(EnvCredentialStore((), {}), CredentialStore)


# ---------------------------------------------------------------------------
# `EnvSecretSource`
# ---------------------------------------------------------------------------


def test_env_secret_source_distinguishes_unset_from_empty():
    """Presence, not truthiness -- see `SecretSource.resolve`'s own
    contract. `LocalSecretSource`'s environment tier makes the identical
    distinction; this provider must not weaken it."""
    provider = EnvSecretSource({"DB_PASSWORD": ""})
    assert provider.resolve("DB_PASSWORD") == ""
    assert provider.resolve("DB_PASSWORD") is not None
    assert provider.resolve("NEVER_SET") is None


def test_env_secret_source_never_consults_a_file(tmp_path):
    """`env` and `dotenv` must not be indistinguishable (the exact trap Task
    2's review caught for `sensitive_parents`): given the same environment
    and a file that *would* answer the name, `EnvSecretSource` still says
    "not set" while `DotenvSecretSource` (below) resolves it from the file.
    `EnvSecretSource` has no `files` constructor parameter at all -- there
    is no way to hand it the file even by mistake."""
    env_file = tmp_path / "secrets.env"
    env_file.write_text(f"DB_PASSWORD={MARKER_DB_PASSWORD}\n")
    environ: dict[str, str] = {}
    assert EnvSecretSource(environ).resolve("DB_PASSWORD") is None
    assert DotenvSecretSource(environ, [env_file]).resolve("DB_PASSWORD") == MARKER_DB_PASSWORD


def test_env_secret_source_satisfies_the_secret_source_protocol():
    assert isinstance(EnvSecretSource({}), SecretSource)


# ---------------------------------------------------------------------------
# `DotenvSecretSource` -- the adapter over `LocalSecretSource`
# ---------------------------------------------------------------------------


def test_dotenv_secret_source_prefers_the_environment_over_a_file(tmp_path):
    env_file = tmp_path / "secrets.env"
    env_file.write_text("DB_PASSWORD=from-the-file\n")
    env_file.chmod(0o600)
    provider = DotenvSecretSource({"DB_PASSWORD": MARKER_DB_PASSWORD}, [env_file])
    assert provider.resolve("DB_PASSWORD") == MARKER_DB_PASSWORD


def test_dotenv_secret_source_falls_back_to_a_file_when_unset_in_the_environment(tmp_path):
    env_file = tmp_path / "secrets.env"
    env_file.write_text(f"API_TOKEN={MARKER_API_TOKEN}\n")
    env_file.chmod(0o600)
    provider = DotenvSecretSource({}, [env_file])
    assert provider.resolve("API_TOKEN") == MARKER_API_TOKEN


def test_dotenv_secret_source_ignores_a_file_that_does_not_exist(tmp_path):
    missing = tmp_path / "does-not-exist.env"
    provider = DotenvSecretSource({}, [missing])
    assert provider.resolve("ANYTHING") is None


def test_dotenv_secret_source_returns_none_for_an_unresolved_name():
    provider = DotenvSecretSource({}, [])
    assert provider.resolve("NEVER_SET") is None


def test_dotenv_secret_source_satisfies_the_secret_source_protocol():
    assert isinstance(DotenvSecretSource({}, []), SecretSource)


# ---------------------------------------------------------------------------
# `CommandSecretSource` / `CommandCredentialStore` -- the generic remote shape
# ---------------------------------------------------------------------------


def test_command_secret_source_reads_the_value_from_a_real_subprocess(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_VALUES", json.dumps({"DB_PASSWORD": MARKER_DB_PASSWORD}))
    provider = CommandSecretSource(remote_command(remote_stub))
    assert provider.resolve("DB_PASSWORD") == MARKER_DB_PASSWORD


def test_command_secret_source_empty_output_means_not_set(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_VALUES", json.dumps({}))
    provider = CommandSecretSource(remote_command(remote_stub))
    assert provider.resolve("NEVER_PUBLISHED") is None


def test_command_secret_source_a_nonzero_exit_is_a_provider_error_not_none(
    monkeypatch, remote_stub
):
    """The failure this whole task's brief calls out: a broken remote must
    never be indistinguishable from "not set"."""
    monkeypatch.setenv("STUB_EXIT_CODE", "1")
    provider = CommandSecretSource(remote_command(remote_stub))
    with pytest.raises(ProviderError):
        provider.resolve("DB_PASSWORD")


def test_command_secret_source_a_timeout_is_a_provider_error(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_SLEEP_SECONDS", "5")
    provider = CommandSecretSource(remote_command(remote_stub), timeout=0.2)
    with pytest.raises(ProviderError):
        provider.resolve("DB_PASSWORD")


def test_command_secret_source_a_missing_executable_is_a_provider_error(tmp_path):
    provider = CommandSecretSource([str(tmp_path / "does-not-exist-anywhere")])
    with pytest.raises(ProviderError):
        provider.resolve("DB_PASSWORD")


def test_command_secret_source_refuses_an_empty_command():
    with pytest.raises(ProviderError):
        CommandSecretSource([])


def test_command_secret_source_never_prints_the_resolved_value(monkeypatch, remote_stub, capfd):
    """A fetched value is held in memory for the call that resolved it and
    handed back to the caller -- never printed by this provider itself."""
    monkeypatch.setenv("STUB_VALUES", json.dumps({"DB_PASSWORD": MARKER_DB_PASSWORD}))
    provider = CommandSecretSource(remote_command(remote_stub))
    provider.resolve("DB_PASSWORD")
    assert MARKER_DB_PASSWORD not in capfd.readouterr().out


def test_command_credential_store_parses_a_json_object_from_stdout(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_MODE", "credentials")
    monkeypatch.setenv("STUB_CREDENTIALS", json.dumps(FULL_CREDENTIALS))
    provider = CommandCredentialStore(remote_command(remote_stub))
    assert provider.resolve("staging") == FULL_CREDENTIALS


def test_command_credential_store_a_nonzero_exit_is_a_provider_error(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_MODE", "credentials")
    monkeypatch.setenv("STUB_EXIT_CODE", "1")
    provider = CommandCredentialStore(remote_command(remote_stub))
    with pytest.raises(ProviderError):
        provider.resolve("staging")


def test_command_credential_store_rejects_output_that_is_not_a_json_object(
    monkeypatch, remote_stub
):
    monkeypatch.setenv("STUB_MODE", "raw")
    monkeypatch.setenv("STUB_RAW_OUTPUT", "not json at all")
    provider = CommandCredentialStore(remote_command(remote_stub))
    with pytest.raises(ProviderError):
        provider.resolve("staging")


def test_command_credential_store_rejects_a_json_array(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_MODE", "raw")
    monkeypatch.setenv("STUB_RAW_OUTPUT", json.dumps(["not", "an", "object"]))
    provider = CommandCredentialStore(remote_command(remote_stub))
    with pytest.raises(ProviderError):
        provider.resolve("staging")


def test_command_providers_satisfy_their_protocols(remote_stub):
    assert isinstance(CommandSecretSource(remote_command(remote_stub)), SecretSource)
    assert isinstance(CommandCredentialStore(remote_command(remote_stub)), CredentialStore)


# ---------------------------------------------------------------------------
# Config wiring: `[profile.<name>.credentials]`
# ---------------------------------------------------------------------------


def test_a_profile_with_no_credentials_table_defaults_to_the_file_provider(store_directory):
    write_store_config(store_directory, '[profile.staging]\nbackend_url = "file:///x"\n')
    profile = store.load_store_config(store_directory).profiles["staging"]
    assert profile.credentials.get("provider", "file") == "file"


def test_a_profile_can_declare_a_non_default_credentials_provider(store_directory):
    write_store_config(
        store_directory,
        '[profile.staging]\nbackend_url = "file:///x"\n\n'
        '[profile.staging.credentials]\nprovider = "env"\n',
    )
    profile = store.load_store_config(store_directory).profiles["staging"]
    assert profile.credentials["provider"] == "env"


def test_a_non_table_credentials_value_is_refused(store_directory):
    write_store_config(
        store_directory,
        '[profile.staging]\nbackend_url = "file:///x"\ncredentials = "not a table"\n',
    )
    with pytest.raises(StoreError):
        store.load_store_config(store_directory)


def test_a_non_string_credentials_provider_is_refused(store_directory):
    write_store_config(
        store_directory,
        '[profile.staging]\nbackend_url = "file:///x"\n\n'
        "[profile.staging.credentials]\nprovider = 1\n",
    )
    with pytest.raises(StoreError):
        store.load_store_config(store_directory)


# ---------------------------------------------------------------------------
# `session._build_credential_store` -- provider dispatch
# ---------------------------------------------------------------------------


def test_build_credential_store_defaults_to_file(store_directory, monkeypatch):
    store.init_store(PASSWORD, directory=store_directory)
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=store_directory)
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    profile = Profile(name="staging")
    provider = session._build_credential_store(profile, directory=store_directory)
    assert provider.resolve("staging") == FULL_CREDENTIALS


def test_build_credential_store_selects_env(monkeypatch):
    monkeypatch.setenv(AWS_ACCESS_KEY_ID, MARKER_KEY)
    profile = Profile(name="staging", credentials={"provider": "env"})
    provider = session._build_credential_store(profile, directory=None)
    assert provider.resolve("staging")[AWS_ACCESS_KEY_ID] == MARKER_KEY


def test_build_credential_store_selects_command(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_MODE", "credentials")
    monkeypatch.setenv("STUB_CREDENTIALS", json.dumps(FULL_CREDENTIALS))
    profile = Profile(
        name="staging", credentials={"provider": "command", "command": remote_command(remote_stub)}
    )
    provider = session._build_credential_store(profile, directory=None)
    assert provider.resolve("staging") == FULL_CREDENTIALS


def test_build_credential_store_command_without_a_command_list_is_refused():
    profile = Profile(name="staging", credentials={"provider": "command"})
    with pytest.raises(SessionError, match="command"):
        session._build_credential_store(profile, directory=None)


def test_build_credential_store_unknown_provider_is_refused():
    profile = Profile(name="staging", credentials={"provider": "not-a-real-provider"})
    with pytest.raises(SessionError, match="not-a-real-provider"):
        session._build_credential_store(profile, directory=None)


# ---------------------------------------------------------------------------
# `set_secrets._build_secret_source` -- provider dispatch
# ---------------------------------------------------------------------------


def test_build_secret_source_defaults_to_dotenv(repo, monkeypatch):
    write_repo_config(repo, '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n')
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    config = load_config(find_repo_config(repo))
    source = set_secrets._build_secret_source(config, repo, None)
    assert source.resolve("DB_PASSWORD") == MARKER_DB_PASSWORD


def test_build_secret_source_env_ignores_a_configured_files_list(repo, monkeypatch):
    """The Task 2 trap this codebase already learned from: an `env`
    provider that happened to also honour `files` would be indistinguishable
    from `dotenv` in a test that never populates a file. This one does."""
    env_file = repo / "secrets.env"
    env_file.write_text(f"DB_PASSWORD={MARKER_DB_PASSWORD}\n")
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n\n'
        '[secrets.source]\nprovider = "env"\nfiles = ["secrets.env"]\n',
    )
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    config = load_config(find_repo_config(repo))
    source = set_secrets._build_secret_source(config, repo, None)
    assert source.resolve("DB_PASSWORD") is None


def test_build_secret_source_selects_command(repo, monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_VALUES", json.dumps({"DB_PASSWORD": MARKER_DB_PASSWORD}))
    command_toml = json.dumps(remote_command(remote_stub))
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n\n'
        f'[secrets.source]\nprovider = "command"\ncommand = {command_toml}\n',
    )
    config = load_config(find_repo_config(repo))
    source = set_secrets._build_secret_source(config, repo, None)
    assert source.resolve("DB_PASSWORD") == MARKER_DB_PASSWORD


def test_build_secret_source_command_without_a_command_list_is_refused(repo):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n\n'
        '[secrets.source]\nprovider = "command"\n',
    )
    config = load_config(find_repo_config(repo))
    with pytest.raises(SetSecretsError, match="command"):
        set_secrets._build_secret_source(config, repo, None)


def test_build_secret_source_unknown_provider_is_refused(repo):
    write_repo_config(
        repo,
        '[secrets."."]\nsecret = { "db.password" = "DB_PASSWORD" }\n\n'
        '[secrets.source]\nprovider = "not-a-real-provider"\n',
    )
    config = load_config(find_repo_config(repo))
    with pytest.raises(SetSecretsError, match="not-a-real-provider"):
        set_secrets._build_secret_source(config, repo, None)


# ---------------------------------------------------------------------------
# A provider failure is reported, never a silent empty value
# ---------------------------------------------------------------------------


def test_publish_entries_reports_a_failing_provider_as_failed_not_skipped(monkeypatch, remote_stub):
    monkeypatch.setenv("STUB_EXIT_CODE", "1")
    manifest = set_secrets.ProjectManifest(secret={"db.password": "DB_PASSWORD"})
    source = CommandSecretSource(remote_command(remote_stub))
    outcomes = set_secrets._publish_entries(
        manifest, source, pulumi=None, stack=None, dry_run=True, timeout=5.0
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "failed"
    assert outcomes[0].status != "skipped"


def test_check_drift_reports_a_failing_provider_as_a_warning_not_silence(
    monkeypatch, remote_stub, capsys
):
    """A provider failure resolving `bootstrap` must not fall into the same
    silent `continue` a legitimately-unresolved bootstrap value gets -- the
    three silent states `_check_drift`'s own docstring names are all
    absences, and this is not one. `pulumi` is a non-`None` placeholder: the
    provider raises before `_pulumi_config_get` would ever be reached, so no
    real `pulumi` binary is needed for this path to be exercised."""
    monkeypatch.setenv("STUB_EXIT_CODE", "1")
    manifest = set_secrets.ProjectManifest(
        secret={"db.password": "DB_PASSWORD"},
        drift_pairs=(
            set_secrets.DriftPair(bootstrap="DB_PASSWORD", managed="DB_PASSWORD_MANAGED"),
        ),
    )
    source = CommandSecretSource(remote_command(remote_stub))

    set_secrets._check_drift(manifest, source, pulumi="placeholder-pulumi", stack=None, timeout=5.0)

    err = capsys.readouterr().err
    assert "warning" in err
    assert "DB_PASSWORD" in err


def test_exec_reports_a_failing_command_credential_store_rather_than_running_with_nothing(
    tmp_path, monkeypatch, remote_stub, env_stub, capsys
):
    """A provider failure during credential resolution must stop `exec`
    outright -- never run the child with an empty or partial environment
    that a caller could mistake for "this profile simply has no
    credentials"."""
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    write_store_config(
        home / "stackward",
        '[profile.staging]\nbackend_url = "file:///staging-backend"\n\n'
        '[profile.staging.credentials]\nprovider = "command"\n'
        f"command = {json.dumps(remote_command(remote_stub))}\n",
    )
    monkeypatch.setenv("STUB_MODE", "credentials")
    monkeypatch.setenv("STUB_EXIT_CODE", "1")

    code = main(["exec", "--profile", "staging", "--", str(env_stub)])

    assert code == 2
    assert "error" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# The gate-path invariant
# ---------------------------------------------------------------------------


def test_the_check_config_entry_point_never_loads_a_provider_module():
    """The brief's own literal requirement: importing `check-config`'s
    module must never bring in a `stackward.providers*` module. Checked in a
    subprocess -- this test process has already imported half the tool via
    other test modules in the same session, so asserting against its own
    `sys.modules` would prove nothing. Enumerated by prefix, not a fixed
    tuple, so a provider module added later is covered without editing this
    test.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import stackward.commands.check_config; "
            "import stackward.commands.pre_commit; "
            "print(sorted(m for m in sys.modules if m == 'stackward.providers' "
            "or m.startswith('stackward.providers.')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "[]"


def test_running_check_config_through_the_real_cli_never_loads_a_provider_module(tmp_path):
    """The stronger form: `cli.py` imports `commands.session` and
    `commands.set_secrets` at its own module scope, unconditionally, so a
    module-scope provider import in *either* of those (rather than the
    function-local imports this task actually uses) would make a plain
    `stackward check-config` reach provider code even though the weaker,
    literal test above stays green. This runs the real dispatch path
    (`cli.main`) a `git commit` actually takes, not just a bare module
    import, so it would catch that regression where the weaker form cannot.
    """
    target = tmp_path / "config.yaml"
    target.write_text("key: value\n")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from stackward.cli import main; "
            f"main(['check-config', {str(target)!r}]); "
            "print(sorted(m for m in sys.modules if m == 'stackward.providers' "
            "or m.startswith('stackward.providers.')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip().splitlines()[-1] == "[]"


# ---------------------------------------------------------------------------
# The contract test: swapping the provider changes nothing for the caller
# ---------------------------------------------------------------------------

# The manifest body is identical across both providers under test -- only
# `[secrets.source]` differs -- so any difference in outcome or invocation
# can only come from the provider swap, never from a manifest difference.
_MANIFEST_BODY = (
    '[secrets."."]\n'
    'secret = { "db.password" = "DB_PASSWORD", "api.token" = "API_TOKEN" }\n'
    'plaintext = { "registry.user" = "REGISTRY_USER" }\n\n'
    '[secrets.".".required]\n'
    'paths = ["db.password"]\n\n'
)


@pytest.fixture
def dotenv_repo(tmp_path, monkeypatch) -> Path:
    """`db.password` resolved via the environment tier, `api.token` via the
    file tier, `registry.user` left unresolved -- exercising all three
    outcome states through `dotenv`'s layered resolution, which the `command`
    run below must match without any layering of its own."""
    root = tmp_path / "dotenv-repo"
    (root / ".git").mkdir(parents=True)
    env_file = root / "secrets.env"
    env_file.write_text(f"API_TOKEN={MARKER_API_TOKEN}\n")
    env_file.chmod(0o600)
    write_repo_config(root, _MANIFEST_BODY + '[secrets.source]\nfiles = ["secrets.env"]\n')
    monkeypatch.setenv("DB_PASSWORD", MARKER_DB_PASSWORD)
    monkeypatch.delenv("REGISTRY_USER", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)
    return root


@pytest.fixture
def command_repo(tmp_path, monkeypatch, remote_stub) -> Path:
    """The same three logical names, seeded with the same values, answered
    entirely by the fake remote CLI -- no file, no environment variable of
    the manifest's own names involved at all."""
    root = tmp_path / "command-repo"
    (root / ".git").mkdir(parents=True)
    command_toml = json.dumps(remote_command(remote_stub))
    write_repo_config(
        root,
        _MANIFEST_BODY
        + f'[secrets.source]\nprovider = "command"\ncommand = {command_toml}\n',
    )
    monkeypatch.setenv(
        "STUB_VALUES",
        json.dumps({"DB_PASSWORD": MARKER_DB_PASSWORD, "API_TOKEN": MARKER_API_TOKEN}),
    )
    return root


def test_dotenv_and_a_fake_remote_produce_identical_dry_run_outcomes(
    dotenv_repo, command_repo, monkeypatch, capfd
):
    """The first half of the contract: `set-secrets --dry-run`'s printed
    outcomes -- would-set, skip, and (via `required`) failed -- are
    byte-identical whether the values came from `dotenv`'s environment+file
    layering or from the fake remote, seeded with the same values."""
    monkeypatch.chdir(dotenv_repo)
    assert main(["set-secrets", "--dry-run"]) == 0
    dotenv_output = capfd.readouterr().out

    monkeypatch.chdir(command_repo)
    assert main(["set-secrets", "--dry-run"]) == 0
    command_output = capfd.readouterr().out

    assert dotenv_output == command_output
    assert "would set" in dotenv_output
    assert "skip" in dotenv_output
    for marker in (MARKER_DB_PASSWORD, MARKER_API_TOKEN):
        assert marker not in dotenv_output
        assert marker not in command_output


def test_dotenv_and_a_fake_remote_produce_identical_pulumi_invocations(
    dotenv_repo, command_repo, monkeypatch, tmp_path, pulumi_stub
):
    """The second half: without `--dry-run`, the real `pulumi config set`
    argv *and* stdin this module sends are byte-identical across both
    providers -- proving the same value, not merely the same flags, reached
    the child either way. `--dry-run` alone cannot prove this: it invokes no
    subprocess at all (`_publish_entries` returns before
    `_pulumi_config_set`), so this is the operation the brief's "identical
    subprocess invocations" actually requires.
    """
    fake_pulumi(monkeypatch, pulumi_stub)

    argv_a, stdin_a = tmp_path / "argv-a.log", tmp_path / "stdin-a.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_a))
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_a))
    monkeypatch.chdir(dotenv_repo)
    assert main(["set-secrets"]) == 0

    argv_b, stdin_b = tmp_path / "argv-b.log", tmp_path / "stdin-b.log"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_b))
    monkeypatch.setenv("STUB_DUMP_STDIN_TO", str(stdin_b))
    monkeypatch.chdir(command_repo)
    assert main(["set-secrets"]) == 0

    assert argv_a.read_bytes() == argv_b.read_bytes()
    assert stdin_a.read_bytes() == stdin_b.read_bytes()
    # Two `secret` entries resolved (`db.password`, `api.token`);
    # `registry.user` never resolved, so it never reaches `pulumi` at all.
    assert len(argv_a.read_text().splitlines()) == 2


def test_file_and_a_fake_remote_produce_identical_child_environments(
    tmp_path, monkeypatch, remote_stub, env_stub
):
    """The `CredentialStore` half of the contract: `exec` injects the exact
    same four names, with the exact same values, whether the profile it
    resolves uses `file` (the default), `command` (the fake remote) or
    `env` (CI's own no-store-at-all path) -- all three seeded with the same
    credentials. `env` gets no encrypted store at all: this is the path a CI
    runner actually takes, and it is exercised here through the real `exec`
    entry point, not only through `session._build_credential_store` in
    isolation."""
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    store_dir = home / "stackward"
    command_toml = json.dumps(remote_command(remote_stub))
    write_store_config(
        store_dir,
        '[profile.staging-file]\nbackend_url = "file:///staging-backend"\n\n'
        '[profile.staging-remote]\nbackend_url = "file:///staging-backend"\n\n'
        '[profile.staging-remote.credentials]\n'
        f'provider = "command"\ncommand = {command_toml}\n\n'
        '[profile.staging-env]\nbackend_url = "file:///staging-backend"\n\n'
        '[profile.staging-env.credentials]\nprovider = "env"\n',
    )
    store.init_store(PASSWORD, directory=store_dir)
    store.set_credentials("staging-file", FULL_CREDENTIALS, PASSWORD, directory=store_dir)
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    monkeypatch.setenv("STUB_MODE", "credentials")
    monkeypatch.setenv("STUB_CREDENTIALS", json.dumps(FULL_CREDENTIALS))
    # `staging-env` has no envelope at all -- these are the only place its
    # credentials come from, exactly as a CI runner's own secrets would be.
    for name, value in FULL_CREDENTIALS.items():
        monkeypatch.setenv(name, value)

    dump_a = tmp_path / "env-a.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump_a))
    assert main(["exec", "--profile", "staging-file", "--", str(env_stub)]) == 0
    env_a = json.loads(dump_a.read_text())

    dump_b = tmp_path / "env-b.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump_b))
    assert main(["exec", "--profile", "staging-remote", "--", str(env_stub)]) == 0
    env_b = json.loads(dump_b.read_text())

    dump_c = tmp_path / "env-c.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump_c))
    assert main(["exec", "--profile", "staging-env", "--", str(env_stub)]) == 0
    env_c = json.loads(dump_c.read_text())

    relevant = (*CREDENTIAL_NAMES, PULUMI_BACKEND_URL)
    expected = {key: env_a[key] for key in relevant}
    assert {key: env_b[key] for key in relevant} == expected
    assert {key: env_c[key] for key in relevant} == expected
    assert env_a[AWS_ACCESS_KEY_ID] == MARKER_KEY
