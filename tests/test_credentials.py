"""Tests for `credentials init|set|list|show|rotate`.

Three things this file is careful about, for the reasons `test_store.py` and
`test_session.py` state them:

*No assertion depends on a credential value being displayed.* Where a value
must round-trip it is compared against a constant; where it must not appear
the assertion is on absence, through `leakcheck.assert_no_leak`, which also
catches a truncated or re-encoded disclosure.

*The markers are opaque.* `assert_no_leak` fails on any eight-character run
of a value, and these commands legitimately print profile names, store paths
and the literal string `AWS_ACCESS_KEY_ID`. A marker built out of real words
collides with that and makes the check unusable — see `tests/leakcheck.py`.

*A "must not be called" control is a `BaseException`.* Every entry point here
is wrapped in `@fail_closed`, which turns any `Exception` into `return 2`, so
an `AssertionError` raised inside one is swallowed and the test then asserts
on an exit code that looks like an ordinary refusal. `_PromptViolation` below
inherits from `BaseException` instead, and is not `KeyboardInterrupt`, which
pytest would treat as "the human wants out" and use to abort the session.
"""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from leakcheck import assert_no_leak
from stackward import store
from stackward.cli import main
from stackward.commands import credentials, session
from stackward.commands.credentials import (
    CONFIRM_PASSWORD_PROMPT,
    NEW_PASSWORD_PROMPT,
    OLD_PASSWORD_PROMPT,
)
from stackward.commands.session import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    ENV_STORE_PASSWORD,
    PULUMI_BACKEND_URL,
    PULUMI_CONFIG_PASSPHRASE,
)

# Placeholder values only (GC4), and deliberately wordless: every one of
# these is passed to `assert_no_leak` against output that also contains the
# tool's own vocabulary.
PASSWORD = "Kx7Rq2Vn9Ld4Ts6Bw"
WRONG_PASSWORD = "Zj3Mp8Hc5Yf1Gd7Nr"
NEW_PASSWORD = "Qv6Wb4Xs2Tk9Pm3Ld"

MARKER_KEY = "Ab7Kd2Ns9Rv4Tq6Xw"
MARKER_SECRET = "Cf3Jm8Ph5Lz1Yb7Gd"
MARKER_PASSPHRASE = "Ew9Ur4Vt2Kn6Sx8Qc"
ALL_MARKERS = (MARKER_KEY, MARKER_SECRET, MARKER_PASSPHRASE)

FULL_CREDENTIALS = {
    AWS_ACCESS_KEY_ID: MARKER_KEY,
    AWS_SECRET_ACCESS_KEY: MARKER_SECRET,
    PULUMI_CONFIG_PASSPHRASE: MARKER_PASSPHRASE,
}

BACKEND_URL = "file:///placeholder-backend"


def env_text(values: dict[str, str]) -> str:
    return "".join(f"{name}={value}\n" for name, value in values.items())


class _PromptViolation(BaseException):
    """Raised when something asks for input a test did not expect it to ask
    for. See this module's docstring for why it is not an `Exception`."""


class _Terminal(io.StringIO):
    """Stands in for a `sys.stdin` attached to a terminal. `io.StringIO`
    already reports `isatty()` as `False`, which is the piped case, so only
    this direction needs a subclass."""

    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def cheap_kdf(monkeypatch):
    """Lower the Argon2id cost for this module only -- the same fixture
    `test_store.py` and `test_session.py` carry, for the same reason: these
    tests perform several real seal/open round trips each and none of them is
    about how expensive a guess is. `test_crypto.py` pins the shipped
    parameters."""
    from stackward import crypto

    monkeypatch.setattr(crypto, "ARGON2ID_MEMORY_KIB", 8192)
    monkeypatch.setattr(crypto, "ARGON2ID_ITERATIONS", 1)
    monkeypatch.setattr(crypto, "ARGON2ID_LANES", 1)


@pytest.fixture(autouse=True)
def no_ambient_state(monkeypatch, tmp_path):
    """Nothing about the store, the profile or the backend left to whoever's
    shell runs the suite. Each test opts back in to whatever it needs."""
    for name in (
        ENV_STORE_PASSWORD,
        store.ENV_PROFILE,
        PULUMI_BACKEND_URL,
        *session.CREDENTIAL_NAMES,
        *session.SCRUBBED_NAMES,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PULUMI_HOME", str(tmp_path / "pulumi-home"))


@pytest.fixture
def cli_store(tmp_path, monkeypatch) -> Path:
    """Point the real config-home lookup at an isolated directory -- the way
    `cli.config_home` is actually consulted by every `cmd_*` entry point,
    rather than through the `directory=` seam `store.py`'s own unit tests
    use."""
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home / "stackward"


@pytest.fixture
def configured(cli_store) -> Path:
    """A store directory whose `config` declares one profile, and no
    `credentials` file at all -- the state a user is in before
    `credentials init`."""
    cli_store.mkdir(parents=True, exist_ok=True)
    config = cli_store / store.CONFIG_FILENAME
    config.write_text(f'[profile.staging]\nbackend_url = "{BACKEND_URL}"\n')
    config.chmod(store.CONFIG_MODE)
    return cli_store


@pytest.fixture
def initialised(configured, monkeypatch) -> Path:
    """`configured`, plus a store initialised under `PASSWORD`.

    The variable is removed again once the store exists. Leaving it set
    would be an ambient answer to every later password read in the test, and
    the tests that most need it *not* to be set -- rotation, which refuses
    outright while it is -- are exactly the ones that would then be testing
    the fixture rather than the command. Each test opts back in.
    """
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    assert main(["credentials", "init"]) == 0
    monkeypatch.delenv(ENV_STORE_PASSWORD)
    return configured


def pipe(monkeypatch, text: str) -> None:
    """Make `sys.stdin` a pipe carrying `text`."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))


def terminal(monkeypatch, text: str = "") -> None:
    monkeypatch.setattr(sys, "stdin", _Terminal(text))


def answer_prompts(monkeypatch, answers: dict[str, str]) -> None:
    """Answer `getpass` by prompt text, and refuse any prompt not listed.

    Keying on the prompt is what lets one seam serve both the master
    password and the per-name value prompts, and what makes "it asked for
    something else" a failure rather than a silently wrong answer.
    """

    def fake_getpass(prompt: str = "") -> str:
        if prompt not in answers:
            raise _PromptViolation(f"unexpected prompt: {prompt!r}")
        return answers[prompt]

    monkeypatch.setattr(credentials.getpass, "getpass", fake_getpass)
    monkeypatch.setattr(session, "_tty_available", lambda: True)

    def refuse_input(*_args, **_kwargs):
        raise _PromptViolation("a prompt echoed: builtins.input was used")

    monkeypatch.setattr("builtins.input", refuse_input)


# ---------------------------------------------------------------------------
# `credentials init`
# ---------------------------------------------------------------------------


def test_init_creates_a_store_the_session_commands_can_open(configured, monkeypatch):
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)

    assert main(["credentials", "init"]) == 0

    path = configured / store.CREDENTIALS_FILENAME
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == store.CREDENTIALS_MODE
    assert stat.S_IMODE(configured.stat().st_mode) == store.DIR_MODE
    # Empty, and openable: the verifier is what proves the password took.
    assert store.store_profiles(configured) == []
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=configured)


def test_init_refuses_to_overwrite_an_existing_store(initialised, monkeypatch, capsys):
    """Overwriting would discard every sealed envelope irrecoverably, in
    response to a command someone could plausibly run twice."""
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)
    before = (initialised / store.CREDENTIALS_FILENAME).read_bytes()
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)

    assert main(["credentials", "init"]) == 2

    assert (initialised / store.CREDENTIALS_FILENAME).read_bytes() == before
    assert store.resolve_credentials("staging", PASSWORD, directory=initialised)
    assert "already exists" in capsys.readouterr().err


def test_init_asks_for_the_new_password_twice_and_refuses_a_mismatch(
    configured, monkeypatch, capsys
):
    """A typo in the password that seals a store is not recoverable by any
    means afterwards, because nothing else can open what it sealed. The
    second read is what catches it, and the refusal must leave no store
    behind -- a half-created one would be opened by neither answer."""
    answer_prompts(
        monkeypatch,
        {NEW_PASSWORD_PROMPT: PASSWORD, CONFIRM_PASSWORD_PROMPT: WRONG_PASSWORD},
    )

    assert main(["credentials", "init"]) == 2

    assert not (configured / store.CREDENTIALS_FILENAME).exists()
    assert "do not match" in capsys.readouterr().err


def test_init_prompts_without_echoing_and_seals_what_was_typed(
    configured, monkeypatch
):
    """The interactive path, end to end: `getpass` (never `input`, which
    echoes -- `answer_prompts` fails the test if `input` is used at all), and
    the store really opens under the typed password afterwards."""
    answer_prompts(
        monkeypatch,
        {NEW_PASSWORD_PROMPT: PASSWORD, CONFIRM_PASSWORD_PROMPT: PASSWORD},
    )

    assert main(["credentials", "init"]) == 0

    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=configured)
    assert (
        store.resolve_credentials("staging", PASSWORD, directory=configured)
        == FULL_CREDENTIALS
    )


def test_init_says_when_it_took_the_password_from_the_environment(
    configured, monkeypatch, capsys
):
    """A store sealed with an ambient variable the user has forgotten about
    is a store they cannot open tomorrow. The name is printed; the value
    never is."""
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)

    assert main(["credentials", "init"]) == 0

    out = capsys.readouterr().out
    assert ENV_STORE_PASSWORD in out
    assert_no_leak(out, PASSWORD, what="the store password")


# ---------------------------------------------------------------------------
# `credentials set`
# ---------------------------------------------------------------------------


def test_set_seals_values_read_from_stdin(initialised, monkeypatch):
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, env_text(FULL_CREDENTIALS))

    assert main(["credentials", "set", "--profile", "staging"]) == 0

    assert (
        store.resolve_credentials("staging", PASSWORD, directory=initialised)
        == FULL_CREDENTIALS
    )


def test_the_sealed_file_holds_none_of_the_values(initialised, monkeypatch):
    """A `set` that stringified or base64-encoded its input instead of
    sealing it would pass the round-trip test above and fail here."""
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, env_text(FULL_CREDENTIALS))
    assert main(["credentials", "set", "--profile", "staging"]) == 0

    written = (initialised / store.CREDENTIALS_FILENAME).read_text()
    for marker in ALL_MARKERS:
        assert_no_leak(written, marker, what="a sealed credential")
    assert_no_leak(written, PASSWORD, what="the store password")


def test_set_prints_no_value_of_its_own(initialised, monkeypatch, capfd):
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, env_text(FULL_CREDENTIALS))

    assert main(["credentials", "set", "--profile", "staging"]) == 0

    captured = capfd.readouterr()
    for marker in ALL_MARKERS:
        assert_no_leak(captured.out + captured.err, marker, what="a credential")


def test_set_takes_no_value_as_an_argument(initialised, monkeypatch):
    """Global Constraint 6, pinned at the parser rather than by inspection:
    a command line is visible in `ps` for the life of the call and stays in
    shell history afterwards, so there must be no positional and no
    `--value` for a value to be passed *through*. argparse rejects the
    attempt itself, before any command code runs.
    """
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    for argv in (
        ["credentials", "set", "--profile", "staging", f"{AWS_ACCESS_KEY_ID}={MARKER_KEY}"],
        ["credentials", "set", "--profile", "staging", "--value", MARKER_KEY],
    ):
        with pytest.raises(SystemExit) as raised:
            main(argv)
        assert raised.value.code == 2
    assert store.store_profiles(initialised) == []


def test_set_refuses_a_missing_required_name_and_writes_nothing(
    initialised, monkeypatch, capsys
):
    """The same refusal `exec` would make, made at the point the envelope is
    written instead of at the point it is needed -- and `set_credentials`
    *replaces* an envelope, so accepting a partial one would silently drop
    the names that were not supplied."""
    before = (initialised / store.CREDENTIALS_FILENAME).read_bytes()
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, env_text({AWS_ACCESS_KEY_ID: MARKER_KEY}))

    assert main(["credentials", "set", "--profile", "staging"]) == 2

    assert (initialised / store.CREDENTIALS_FILENAME).read_bytes() == before
    err = capsys.readouterr().err
    assert AWS_SECRET_ACCESS_KEY in err and PULUMI_CONFIG_PASSPHRASE in err
    assert_no_leak(err, MARKER_KEY, what="a credential")


def test_set_refuses_an_empty_value_for_a_required_name(
    initialised, monkeypatch, capsys
):
    """A placeholder line in a `.env` (`AWS_ACCESS_KEY_ID=`) seals perfectly
    cleanly and then fails to authenticate with nothing pointing back at
    this command, so it is refused here where the cause is still visible."""
    before = (initialised / store.CREDENTIALS_FILENAME).read_bytes()
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, env_text({**FULL_CREDENTIALS, AWS_ACCESS_KEY_ID: ""}))

    assert main(["credentials", "set", "--profile", "staging"]) == 2

    assert (initialised / store.CREDENTIALS_FILENAME).read_bytes() == before
    assert AWS_ACCESS_KEY_ID in capsys.readouterr().err


def test_set_refuses_a_malformed_line_without_echoing_it(
    initialised, monkeypatch, capsys
):
    """A line with no `=` is fail-closed, not silently skipped -- and the
    refusal names a line number, never the line, because the thing that
    failed to parse may be exactly what a value was pasted into."""
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, f"{env_text(FULL_CREDENTIALS)}{MARKER_SECRET}\n")

    assert main(["credentials", "set", "--profile", "staging"]) == 2

    err = capsys.readouterr().err
    assert "stdin:4" in err
    assert_no_leak(err, MARKER_SECRET, what="a malformed input line")


def test_set_refuses_empty_stdin_rather_than_sealing_nothing(
    initialised, monkeypatch, capsys
):
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, "\n  \n")

    assert main(["credentials", "set", "--profile", "staging"]) == 2

    assert "no credentials on stdin" in capsys.readouterr().err
    assert store.store_profiles(initialised) == []


def test_set_with_a_wrong_password_is_refused_by_the_verifier(
    initialised, monkeypatch, capsys
):
    """The store-level verifier is what makes this immediate: without it, an
    envelope sealed under a mistyped password would be written happily and
    fail weeks later, at first use, as unexplained damage."""
    before = (initialised / store.CREDENTIALS_FILENAME).read_bytes()
    monkeypatch.setenv(ENV_STORE_PASSWORD, WRONG_PASSWORD)
    pipe(monkeypatch, env_text(FULL_CREDENTIALS))

    assert main(["credentials", "set", "--profile", "staging"]) == 2

    assert (initialised / store.CREDENTIALS_FILENAME).read_bytes() == before
    err = capsys.readouterr().err
    assert "wrong password" in err
    assert_no_leak(err, WRONG_PASSWORD, what="the store password")


def test_set_replaces_the_whole_envelope(initialised, monkeypatch):
    """Documented behaviour, pinned: `store.set_credentials` writes what it
    is given rather than merging, which is why every required name has to be
    supplied on every run."""
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    first = {**FULL_CREDENTIALS, "EXTRA_NAME": "Nt4Bq8Wz2Ck6Hm9Rd"}
    pipe(monkeypatch, env_text(first))
    assert main(["credentials", "set", "--profile", "staging"]) == 0

    pipe(monkeypatch, env_text(FULL_CREDENTIALS))
    assert main(["credentials", "set", "--profile", "staging"]) == 0

    assert (
        store.resolve_credentials("staging", PASSWORD, directory=initialised)
        == FULL_CREDENTIALS
    )


def test_set_prompts_for_each_name_without_echo_when_stdin_is_a_terminal(
    initialised, monkeypatch
):
    """The interactive path. `answer_prompts` refuses `builtins.input`
    outright, so an implementation that echoed what was typed fails here
    rather than passing quietly."""
    terminal(monkeypatch)
    answer_prompts(
        monkeypatch,
        {
            f"{AWS_ACCESS_KEY_ID}: ": MARKER_KEY,
            f"{AWS_SECRET_ACCESS_KEY}: ": MARKER_SECRET,
            f"{PULUMI_CONFIG_PASSPHRASE}: ": MARKER_PASSPHRASE,
            session.DEFAULT_PASSWORD_PROMPT: PASSWORD,
        },
    )

    assert main(["credentials", "set", "--profile", "staging"]) == 0

    assert (
        store.resolve_credentials("staging", PASSWORD, directory=initialised)
        == FULL_CREDENTIALS
    )


def test_set_refuses_a_profile_that_has_no_table_in_config(
    initialised, monkeypatch, capsys
):
    """Profile selection is `session._resolve_profile`, unchanged: a name
    that no `[profile.<name>]` declares is an error rather than a silent
    fall-through, so credentials cannot be sealed under a typo that nothing
    will ever read."""
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    pipe(monkeypatch, env_text(FULL_CREDENTIALS))

    assert main(["credentials", "set", "--profile", "typo"]) == 2

    assert "typo" in capsys.readouterr().err
    assert store.store_profiles(initialised) == []


def test_set_honours_the_same_profile_precedence_as_exec(initialised, monkeypatch):
    """`STACKWARD_PROFILE` with no `--profile`: the tier below the flag. If
    this module selected profiles its own way, the two commands could seal
    and read different envelopes from the same environment."""
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    monkeypatch.setenv(store.ENV_PROFILE, "staging")
    pipe(monkeypatch, env_text(FULL_CREDENTIALS))

    assert main(["credentials", "set"]) == 0

    assert store.store_profiles(initialised) == ["staging"]


# ---------------------------------------------------------------------------
# `credentials list` and `credentials show`
# ---------------------------------------------------------------------------


def test_list_prints_the_profiles_that_have_credentials(
    initialised, monkeypatch, capsys
):
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)

    assert main(["credentials", "list"]) == 0

    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["staging"]
    for marker in ALL_MARKERS:
        assert_no_leak(captured.out + captured.err, marker, what="a credential")


def test_list_needs_no_password_at_all(initialised, monkeypatch):
    """`store_profiles` reads the document's keys and opens nothing, so a
    listing must not prompt -- and must not fail on a machine where no
    password is available."""

    def refuse(*_args, **_kwargs):
        raise _PromptViolation("listing asked for the store password")

    monkeypatch.setattr(session, "_read_store_password", refuse)
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)

    assert main(["credentials", "list"]) == 0


def test_list_before_init_names_the_command_that_fixes_it(configured, capsys):
    """Global Constraint 3: a refusal says what to do about it."""
    assert main(["credentials", "list"]) == 2
    assert "credentials init" in capsys.readouterr().err


def test_list_says_so_when_no_profile_has_credentials_yet(initialised, capsys):
    assert main(["credentials", "list"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "credentials set" in captured.err


def test_show_lists_names_and_never_values(initialised, monkeypatch, capfd):
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)

    assert main(["credentials", "show", "--profile", "staging"]) == 0

    captured = capfd.readouterr()
    for name in session.CREDENTIAL_NAMES:
        assert name in captured.out
    for marker in ALL_MARKERS:
        assert_no_leak(captured.out + captured.err, marker, what="a credential")


def test_show_reports_a_shortfall_without_failing(initialised, monkeypatch, capfd):
    """The question was answered correctly and the store is not in a state
    anything needs to refuse, so this exits 0 -- `exec` is what refuses an
    incomplete envelope, and already does."""
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)
    partial = {AWS_ACCESS_KEY_ID: MARKER_KEY}
    store.set_credentials("staging", partial, PASSWORD, directory=initialised)

    assert main(["credentials", "show", "--profile", "staging"]) == 0

    captured = capfd.readouterr()
    assert AWS_SECRET_ACCESS_KEY in captured.err
    assert_no_leak(captured.out + captured.err, MARKER_KEY, what="a credential")


def test_show_on_a_profile_with_no_envelope_says_so(initialised, monkeypatch, capsys):
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)

    assert main(["credentials", "show", "--profile", "staging"]) == 2

    assert "no credentials" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `credentials rotate`
# ---------------------------------------------------------------------------


def test_rotate_reseals_every_envelope_under_the_new_password(
    initialised, monkeypatch
):
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)
    store.set_credentials("other", FULL_CREDENTIALS, PASSWORD, directory=initialised)
    answer_prompts(
        monkeypatch,
        {
            OLD_PASSWORD_PROMPT: PASSWORD,
            NEW_PASSWORD_PROMPT: NEW_PASSWORD,
            CONFIRM_PASSWORD_PROMPT: NEW_PASSWORD,
        },
    )

    assert main(["credentials", "rotate"]) == 0

    for name in ("staging", "other"):
        assert (
            store.resolve_credentials(name, NEW_PASSWORD, directory=initialised)
            == FULL_CREDENTIALS
        )
        with pytest.raises(store.PasswordError):
            store.resolve_credentials(name, PASSWORD, directory=initialised)


def test_rotate_refuses_a_mismatched_confirmation_and_changes_nothing(
    initialised, monkeypatch, capsys
):
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)
    before = (initialised / store.CREDENTIALS_FILENAME).read_bytes()
    answer_prompts(
        monkeypatch,
        {
            OLD_PASSWORD_PROMPT: PASSWORD,
            NEW_PASSWORD_PROMPT: NEW_PASSWORD,
            CONFIRM_PASSWORD_PROMPT: WRONG_PASSWORD,
        },
    )

    assert main(["credentials", "rotate"]) == 2

    assert (initialised / store.CREDENTIALS_FILENAME).read_bytes() == before
    assert "do not match" in capsys.readouterr().err


def test_rotate_refuses_when_the_environment_variable_is_the_only_source(
    initialised, monkeypatch, capsys
):
    """`STACKWARD_PASSWORD` answers every read of the store password, so with
    it set the "old" and "new" passwords are the same string and the user was
    asked nothing. Reporting that as a bare sameness would describe something
    they never typed, so the refusal names the variable and the fix."""
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)
    before = (initialised / store.CREDENTIALS_FILENAME).read_bytes()
    monkeypatch.setenv(ENV_STORE_PASSWORD, PASSWORD)

    assert main(["credentials", "rotate"]) == 2

    assert (initialised / store.CREDENTIALS_FILENAME).read_bytes() == before
    err = capsys.readouterr().err
    assert ENV_STORE_PASSWORD in err
    assert "unset" in err


def test_rotate_that_cannot_reseal_one_envelope_changes_nothing(
    initialised, monkeypatch, capsys
):
    """All-or-nothing, surfaced honestly. A partial rotation would leave a
    store whose profiles need different passwords with nothing recording
    which is which -- the worst outcome available -- and reporting a failed
    rotation as success would be the second worst."""
    store.set_credentials("staging", FULL_CREDENTIALS, PASSWORD, directory=initialised)
    path = initialised / store.CREDENTIALS_FILENAME
    document = json.loads(path.read_text())
    document["profiles"]["damaged"] = document["profiles"]["staging"]
    path.write_text(json.dumps(document))
    before = path.read_bytes()

    answer_prompts(
        monkeypatch,
        {
            OLD_PASSWORD_PROMPT: PASSWORD,
            NEW_PASSWORD_PROMPT: NEW_PASSWORD,
            CONFIRM_PASSWORD_PROMPT: NEW_PASSWORD,
        },
    )

    assert main(["credentials", "rotate"]) == 2

    assert path.read_bytes() == before
    assert (
        store.resolve_credentials("staging", PASSWORD, directory=initialised)
        == FULL_CREDENTIALS
    )
    assert "nothing has been changed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Dispatch: registered, and off the gate path
# ---------------------------------------------------------------------------


_IMPORT_PROBE = """
import sys
from stackward.cli import build_parser
build_parser()
print("loaded", "stackward.commands.credentials" in sys.modules)
"""


def test_building_the_parser_does_not_import_the_credentials_module():
    """`credentials` imports `store` -> `crypto` -> `cryptography`, and
    `build_parser` runs on *every* invocation, `check-config` and
    `pre-commit` included. `tests/test_gate_isolation.py` asserts the
    property over the union of forbidden modules; this asserts the mechanism
    that keeps it true for this command specifically, so that a regression
    here is reported as what it is rather than as a module-list diff.

    A subprocess, because this test session has already imported the module
    directly at the top of this file.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "loaded False" in proc.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["credentials", "init"],
        ["credentials", "set"],
        ["credentials", "list"],
        ["credentials", "show"],
        ["credentials", "rotate"],
    ],
)
def test_every_subcommand_is_registered_and_dispatches(argv, cli_store, monkeypatch):
    """Wiring only. Each must reach its own entry point -- which then refuses,
    because `cli_store` points `XDG_CONFIG_HOME` at a directory holding no
    store and no profile -- rather than argparse's "invalid choice", which
    would raise `SystemExit` here instead of returning.

    `_tty_available` is forced false so that the two subcommands which read a
    password (`init`, `rotate`) refuse instead of prompting: without it this
    test would block on `/dev/tty` for anyone running the suite from a
    terminal, and pass only on machines that have none.
    """
    monkeypatch.setattr(session, "_tty_available", lambda: False)
    assert main(argv) == 2


def test_a_bare_credentials_command_prints_help_rather_than_guessing(capsys):
    assert main(["credentials"]) == 2
    assert "usage:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The migration the plan's Step 2 exit criterion names, end to end.
# ---------------------------------------------------------------------------


_STUB = """\
import json, os, sys
with open(os.environ["STUB_DUMP_ENV_TO"], "w") as handle:
    json.dump(dict(os.environ), handle)
"""


def test_a_dotenv_file_can_be_sealed_and_then_deleted_and_exec_still_works(
    configured, monkeypatch, tmp_path, capfd
):
    """The plan's Step 2 exit criterion, walked end to end.

    "`stackward exec -- pulumi ...` works with the repo `.env` **renamed**,
    then delete the `.env` bootstrap copies." Every part of that is asserted
    rather than assumed:

    * the values reach the child from the **store**, not from the
      environment -- the three credential names are deleted from this
      process's own environment first, and the assertion at the end is that
      no `STACKWARD_*` variable survived either, so the `env` provider and
      the master-password variable are both out of the picture;
    * no `.env` exists anywhere under `tmp_path` when `exec` runs, so the
      `dotenv` provider cannot be what answered;
    * the password is *typed*, not exported, which is the path a fresh user
      actually has;
    * and nothing leaks: the whole run's output is checked at
      file-descriptor level (`capfd`, not `capsys` -- a real child's writes
      do not pass through `sys.stdout`).

    A stub child rather than `pulumi`, for the reason `test_session.py`
    gives: what is under test is this tool's environment handling, not
    Pulumi's presence on the machine.
    """
    # 1. The bootstrap file a repository has today.
    bootstrap = tmp_path / "bootstrap-values"
    bootstrap.write_text(env_text(FULL_CREDENTIALS))

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    answer_prompts(
        monkeypatch,
        {
            NEW_PASSWORD_PROMPT: PASSWORD,
            CONFIRM_PASSWORD_PROMPT: PASSWORD,
            session.DEFAULT_PASSWORD_PROMPT: PASSWORD,
        },
    )

    # 2. Seal it, exactly as `stackward credentials set --profile staging
    #    < .env` would.
    assert main(["credentials", "init"]) == 0
    pipe(monkeypatch, bootstrap.read_text())
    assert main(["credentials", "set", "--profile", "staging"]) == 0

    # 3. Delete the bootstrap copy. Nothing on disk holds these values in
    #    plaintext from here on.
    bootstrap.unlink()
    assert list(tmp_path.rglob("*.env")) == []
    assert list(tmp_path.rglob(".env")) == []
    # Named, not swept by prefix. A `STACKWARD_*` prefix check would also
    # catch `STACKWARD_DENYLIST`, which this repository's own README tells
    # developers to export and `scripts/pre-push` reads -- so the test would
    # fail on the machine of anyone who followed the instructions, for a
    # reason with nothing to do with the store. These three names are what
    # the claim is actually about.
    assert ENV_STORE_PASSWORD not in os.environ
    assert store.ENV_PROFILE not in os.environ
    for name in session.CREDENTIAL_NAMES:
        assert name not in os.environ

    # 4. `exec` a child, and read back what it actually received.
    stub = tmp_path / "stub.py"
    stub.write_text(_STUB)
    dump = tmp_path / "child-env.json"
    monkeypatch.setenv("STUB_DUMP_ENV_TO", str(dump))

    assert main(["exec", "--profile", "staging", "--", sys.executable, str(stub)]) == 0

    child_env = json.loads(dump.read_text())
    assert child_env[AWS_ACCESS_KEY_ID] == MARKER_KEY
    assert child_env[AWS_SECRET_ACCESS_KEY] == MARKER_SECRET
    assert child_env[PULUMI_CONFIG_PASSPHRASE] == MARKER_PASSPHRASE
    assert child_env[PULUMI_BACKEND_URL] == BACKEND_URL

    captured = capfd.readouterr()
    for marker in (*ALL_MARKERS, PASSWORD):
        assert_no_leak(captured.out + captured.err, marker, what="a credential")
