"""Tests for the credential store: profiles, file handling, and the store API.

Two things these tests are careful about.

*They never assert on a credential value being present in output.* A test that
proved a value was printed would itself print it on failure. Where a value must
not appear, the assertion is on absence; where a value must round-trip, it is
compared to a constant, not displayed.

*Several are written to fail against the plausible wrong implementation rather
than merely to exercise the right one.* `--profile` naming a missing profile
must not fall through to `default_profile`; rotation must re-seal an envelope
whose config table has been deleted; an atomic write that fails must leave both
the original file and the directory as they were. Each of those is a bug a
passing "happy path" suite would not notice.

The Argon2id cost is lowered here by an autouse fixture — these tests are about
store semantics, not about how expensive a guess is, and the shipped parameters
are pinned separately in `tests/test_crypto.py`.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import stat
import tomllib

import pytest

from leakcheck import assert_no_leak
from stackward import crypto, store
from stackward.store import (
    CONFIG_MODE,
    CREDENTIALS_MODE,
    DIR_MODE,
    ENV_PROFILE,
    PasswordError,
    Profile,
    ProfileError,
    StoreConfig,
    StoreError,
    atomic_write,
    init_store,
    load_store_config,
    resolve_credentials,
    rotate_password,
    select_profile,
    set_credentials,
    store_profiles,
)

PASSWORD = "the store password"
NEW_PASSWORD = "the rotated store password"
WRONG_PASSWORD = "not the store password"

# Placeholder credential names and values. The names are arbitrary here on
# purpose: the store does not know which variables a backend needs, and naming
# real ones would put a vendor's vocabulary into a module that has none.
MARKER = "marker-credential-value-8f2c41"
CREDENTIALS = {"NAME_ONE": MARKER, "NAME_TWO": "second-placeholder-value"}


@pytest.fixture(autouse=True)
def cheap_kdf(monkeypatch):
    """Lower the Argon2id cost for this module only.

    At the shipped parameters every seal costs about 60 ms, and this file
    performs several hundred. The properties under test here — precedence,
    atomicity, all-or-nothing rotation — do not depend on the cost, and
    `test_crypto.py::test_the_shipped_cost_parameters_are_the_documented_ones`
    is what stops this fixture hiding a production weakening.
    """
    monkeypatch.setattr(crypto, "ARGON2ID_MEMORY_KIB", 8192)
    monkeypatch.setattr(crypto, "ARGON2ID_ITERATIONS", 1)
    monkeypatch.setattr(crypto, "ARGON2ID_LANES", 1)


@pytest.fixture
def store_directory(tmp_path):
    """An isolated store directory. Every store function takes `directory`, so
    no test depends on the real `XDG_CONFIG_HOME` or writes outside `tmp_path`."""
    return tmp_path / "stackward"


def write_config(store_directory, text: str):
    store_directory.mkdir(parents=True, exist_ok=True)
    path = store_directory / "config"
    path.write_text(text)
    # At the required mode, so that these tests do not trip the permission
    # warning as a side effect of whatever umask the suite runs under. A
    # literal rather than `CONFIG_MODE`: this helper is used by the test that
    # pins the call site's constant, which cannot pin anything if the helper
    # moves with it.
    os.chmod(path, 0o600)
    return path


def config_with(*names: str, default: str | None = None) -> str:
    lines = [] if default is None else [f'default_profile = "{default}"', ""]
    for name in names:
        lines += [f"[profile.{name}]", 'backend_url = "file:///placeholder"', ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# `config`: backend identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "s3://placeholder-bucket/placeholder-prefix",
        "gs://placeholder-bucket",
        "azblob://placeholder-container",
        "file:///placeholder/path",
        "s3://placeholder/prefix?endpoint=placeholder.invalid&region=placeholder",
        "some-future-scheme://placeholder",
    ],
)
def test_a_backend_url_is_carried_through_byte_for_byte(store_directory, url):
    """Pass-through, not parsing. Any normalising or reassembling this module
    did would show up here as a changed string, and would be the thing that
    stopped an unanticipated scheme from working."""
    write_config(store_directory, f"[profile.one]\nbackend_url = {json.dumps(url)}\n")
    assert load_store_config(store_directory).profiles["one"].backend_url == url


def test_the_component_form_is_carried_as_structured_data(store_directory):
    write_config(
        store_directory,
        "[profile.one]\n"
        'bucket = "placeholder-bucket"\n'
        'prefix = "placeholder/prefix"\n'
        'endpoint = "https://placeholder.invalid"\n'
        'region = "placeholder-region"\n',
    )
    profile = load_store_config(store_directory).profiles["one"]
    assert profile.backend_url is None
    assert profile.bucket == "placeholder-bucket"
    assert profile.prefix == "placeholder/prefix"
    assert profile.endpoint == "https://placeholder.invalid"
    assert profile.region == "placeholder-region"


def test_a_profile_carrying_both_forms_is_refused(store_directory):
    """Two answers to "which backend?" with no rule for choosing between them."""
    write_config(
        store_directory,
        '[profile.one]\nbackend_url = "file:///placeholder"\nbucket = "placeholder"\n',
    )
    with pytest.raises(StoreError):
        load_store_config(store_directory)


def test_a_profile_carrying_neither_form_is_refused(store_directory):
    """There is no default backend to fall back on, so an empty profile is an
    error and not an inherited one."""
    write_config(store_directory, "[profile.one]\n")
    with pytest.raises(StoreError):
        load_store_config(store_directory)


def test_the_component_form_without_a_bucket_is_refused(store_directory):
    write_config(store_directory, '[profile.one]\nregion = "placeholder-region"\n')
    with pytest.raises(StoreError):
        load_store_config(store_directory)


def test_an_unknown_profile_key_is_refused_rather_than_ignored(store_directory):
    write_config(
        store_directory,
        '[profile.one]\nbackend_url = "file:///placeholder"\nbucket_name = "typo"\n',
    )
    with pytest.raises(StoreError):
        load_store_config(store_directory)


def test_a_default_profile_with_no_table_is_refused(store_directory):
    write_config(store_directory, 'default_profile = "missing"\n' + config_with("one"))
    with pytest.raises(StoreError):
        load_store_config(store_directory)


def test_an_absent_config_is_empty_rather_than_an_error(store_directory):
    """Not having set the tool up yet is a normal state. It is `select_profile`
    that reports it, because only it can say what to do about it."""
    assert load_store_config(store_directory) == StoreConfig()


def test_an_invalid_config_raises_rather_than_reading_as_absent(store_directory):
    """A typo must not degrade into "no profiles", which would turn a broken
    backend into a silently missing one."""
    write_config(store_directory, "[profile.one\nbackend_url = ")
    with pytest.raises(StoreError):
        load_store_config(store_directory)


def test_an_invalid_config_is_reported_by_position_and_never_by_quoting_it(
    store_directory,
):
    """`config` can hold a credential, so its parse error must not quote it.

    This call site used to interpolate `tomllib`'s own message in full, on
    the stated grounds that "`config` holds backend identity and never a
    credential". Both halves of that were wrong. A `backend_url` of the
    documented `postgres://user:password@host/db` form is a credential in
    `config`; and `tomllib` does read document text back -- most of its
    messages are a fixed description plus a coordinate, but a duplicated
    table declaration names the key, which is the shape written below
    (`Cannot declare ('...',) twice`).

    The coordinate is asserted too: dropping the message entirely would also
    satisfy a bare no-leak check, and the position is what makes the error
    actionable.
    """
    secret = "supersecret-marker-9f1e2d"
    write_config(
        store_directory,
        f'[{secret}]\nbackend_url = "file:///placeholder"\n[{secret}]\n',
    )
    with pytest.raises(StoreError) as raised:
        load_store_config(store_directory)

    message = str(raised.value)
    assert_no_leak(message, secret, what="a name written into `config`")
    assert "invalid TOML" in message
    assert "line 3" in message


def test_a_config_value_is_never_quoted_back_by_a_syntax_error(store_directory):
    """The same rule for a credential in the *value* position -- the shape a
    real `postgres://user:password@host/db` backend URL takes."""
    secret = "supersecret-marker-4b7c0a"
    write_config(
        store_directory,
        f'[profile.one]\nbackend_url = "postgres://u:{secret}@h.invalid/db"\n[broken\n',
    )
    with pytest.raises(StoreError) as raised:
        load_store_config(store_directory)
    assert_no_leak(str(raised.value), secret, what="a backend URL password")


def test_a_credential_at_the_end_of_the_document_is_not_quoted_back(store_directory):
    """The other coordinate `tomllib` produces, `(at end of document)`.

    `_toml_position` matches two message shapes and every other test here
    exercises only the first, which would leave the second silently
    unrecognised -- and an unrecognised shape yields no coordinate at all,
    so the branch is worth a test of its own. An unterminated string is the
    natural way to reach it, and is also the shape a half-pasted backend URL
    actually takes.
    """
    secret = "supersecret-marker-2e8d6c"
    write_config(store_directory, f'backend_url = "postgres://u:{secret}@h.invalid/db')
    with pytest.raises(StoreError) as raised:
        load_store_config(store_directory)

    message = str(raised.value)
    assert_no_leak(message, secret, what="a backend URL password")
    assert "invalid TOML" in message
    assert "end of document" in message


def test_a_toml_message_with_no_coordinate_loses_the_coordinate_not_the_secrecy(
    store_directory, monkeypatch
):
    """The fallback: an unrecognised message shape must degrade to no
    position, never to quoting the message.

    `TOMLDecodeError`'s text is not an API and this project supports three
    Python versions, so `_toml_position` has to cope with a message it
    cannot parse. The safe degradation is losing the diagnostic; the unsafe
    one is falling back to the whole message, which is exactly what this
    call site used to do unconditionally.
    """
    secret = "supersecret-marker-5a3f1b"
    write_config(store_directory, '[profile.one]\nbucket = "b"\n')

    def raise_unparseable(*_args, **_kwargs):
        raise tomllib.TOMLDecodeError(f"a shape from some future release: {secret}")

    monkeypatch.setattr(tomllib, "load", raise_unparseable)
    with pytest.raises(StoreError) as raised:
        load_store_config(store_directory)

    message = str(raised.value)
    assert_no_leak(message, secret, what="the parser's own message")
    assert message.endswith("invalid TOML")


def test_an_unknown_top_level_key_is_refused_naming_it(store_directory):
    """The profile-level analogue of this is tested; the top-level one was
    not, so `_build_store_config`'s own refusal never ran at all. An
    unrecognised top-level key must not be ignored: a misspelled
    `default_profile` silently dropped would make the store fall through to
    "no profile selected" with nothing pointing at the typo."""
    write_config(
        store_directory, 'defualt_profile = "one"\n[profile.one]\nbucket = "b"\n'
    )
    with pytest.raises(StoreError, match="defualt_profile"):
        load_store_config(store_directory)


# ---------------------------------------------------------------------------
# Profile selection precedence
# ---------------------------------------------------------------------------


@pytest.fixture
def four_profiles(store_directory):
    write_config(
        store_directory, config_with("chosen", "from_env", "from_repo", "fallback")
    )
    return load_store_config(store_directory)


def test_the_explicit_profile_wins_over_every_other_source(four_profiles):
    selected = select_profile(
        "chosen",
        config=four_profiles,
        repo_profile="from_repo",
        environ={ENV_PROFILE: "from_env"},
    )
    assert selected.name == "chosen"


def test_the_environment_wins_when_no_explicit_profile_was_given(four_profiles):
    selected = select_profile(
        config=four_profiles,
        repo_profile="from_repo",
        environ={ENV_PROFILE: "from_env"},
    )
    assert selected.name == "from_env"


def test_the_repository_config_wins_over_the_default_profile(store_directory):
    write_config(store_directory, config_with("from_repo", "fallback", default="fallback"))
    selected = select_profile(
        config=load_store_config(store_directory), repo_profile="from_repo", environ={}
    )
    assert selected.name == "from_repo"


def test_the_default_profile_is_the_last_resort(store_directory):
    write_config(store_directory, config_with("fallback", default="fallback"))
    selected = select_profile(config=load_store_config(store_directory), environ={})
    assert selected.name == "fallback"


def test_no_profile_anywhere_is_an_error_and_never_a_guess(four_profiles):
    with pytest.raises(ProfileError):
        select_profile(config=four_profiles, environ={})


def test_a_named_profile_that_does_not_exist_errors_instead_of_using_the_default(
    store_directory,
):
    """The case that separates "first source that *speaks* wins" from "first
    source that *resolves* wins". Falling through here would run the command
    against a backend nobody asked for — the worst thing this tool could do
    quietly — and a loop that skipped invalid sources would pass every other
    precedence test in this file.
    """
    write_config(store_directory, config_with("fallback", default="fallback"))
    config = load_store_config(store_directory)
    for explicit, environ, repo_profile in (
        ("absent", {}, None),
        (None, {ENV_PROFILE: "absent"}, None),
        (None, {}, "absent"),
    ):
        with pytest.raises(ProfileError) as raised:
            select_profile(
                explicit, config=config, environ=environ, repo_profile=repo_profile
            )
        assert "absent" in str(raised.value)
        assert "fallback" not in str(raised.value)


def test_an_empty_environment_variable_is_an_error_rather_than_a_miss(store_directory):
    """A variable set to the empty string is a broken variable, not an absent
    one. Skipping it would silently substitute the default."""
    write_config(store_directory, config_with("fallback", default="fallback"))
    with pytest.raises(ProfileError):
        select_profile(
            config=load_store_config(store_directory), environ={ENV_PROFILE: ""}
        )


def test_a_profile_name_that_would_collide_with_the_verifier_is_rejected(store_directory):
    """The verifier's AAD is `stackward:verifier`. Excluding ':' from the name
    charset is the mechanism that keeps the two AAD namespaces disjoint, so a
    profile envelope can never be interchangeable with the verifier's."""
    with pytest.raises(ProfileError):
        store.validate_profile_name("stackward:verifier", "test")


@pytest.mark.parametrize(
    "name",
    [
        "with space",
        "-leading-dash",
        "with/slash",
        "with:colon",
        # TOML reads a dot in a table header as nesting, so `[profile.with.dot]`
        # declares a table inside a table rather than this profile.
        "with.dot",
        "",
        "wîth-å",
    ],
)
def test_an_unspellable_profile_name_is_rejected(name):
    with pytest.raises(ProfileError):
        store.validate_profile_name(name, "test")


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


def test_the_store_directory_is_created_private_whatever_the_umask(store_directory):
    """0o700 as a literal, deliberately -- never `DIR_MODE`.

    Asserting `== DIR_MODE` compares the module against itself: the
    expectation moves with the constant, so `DIR_MODE = 0o777` produces a
    world-writable store directory and this test still passes. The number is
    the requirement, so the number is what is written here.
    """
    previous = os.umask(0)
    try:
        store.ensure_store_dir(store_directory)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(store_directory.stat().st_mode) == 0o700


def test_the_credentials_file_is_written_private_whatever_the_umask(store_directory):
    """0o600 as a literal -- see the sibling test above on why not
    `CREDENTIALS_MODE`. `CREDENTIALS_MODE = 0o606` leaves the credentials
    file world-*writable* and passes an `== CREDENTIALS_MODE` assertion."""
    previous = os.umask(0)
    try:
        init_store(PASSWORD, directory=store_directory)
    finally:
        os.umask(previous)
    mode = stat.S_IMODE((store_directory / "credentials").stat().st_mode)
    assert mode == 0o600


def test_the_config_file_mode_the_store_expects_is_owner_only(store_directory, capsys):
    """The mode `config` is *held to*, asserted where the module actually
    applies it rather than against the constant it applies.

    Nothing in this module writes `config` -- it is hand-edited -- so the
    only place `CONFIG_MODE` has an effect is the permissiveness warning
    `load_store_config` raises, and that is therefore where it has to be
    pinned. A unit test of `warn_if_permissive(path, 0o600)` proves the
    function; it says nothing about which constant the call site passes,
    and `CONFIG_MODE = 0o644` leaves that unit test green.

    0644 is the specific mode that matters: `config` can hold a credential
    (a `backend_url` of the documented `postgres://user:password@host/db`
    form), so a world-readable one has to be reported.
    """
    write_config(store_directory, '[profile.one]\nbackend_url = "file:///placeholder"\n')
    os.chmod(store_directory / "config", 0o644)

    load_store_config(store_directory)

    captured = capsys.readouterr()
    assert "0644" in captured.err
    assert str(store_directory / "config") in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_the_mode_is_set_before_the_rename_not_after(store_directory, monkeypatch, mode):
    """Setting the mode after the rename leaves a window in which the finished
    file is visible at its real path with whatever mode it was created with.
    This records the mode of the temporary file at the moment of the rename.

    Two modes are checked because only one of them discriminates: `mkstemp`
    already creates at 0600, so at that mode a `chmod` after the rename would
    look identical and this test would prove nothing. 0644 is what makes the
    ordering observable -- which is why it is written as a literal and not as
    `CONFIG_MODE`, whose value is now also 0600. Parametrising over the
    module's two constants would have quietly turned this into two copies of
    the case that proves nothing the moment `CONFIG_MODE` changed.
    """
    store_directory.mkdir(parents=True)
    real_replace = os.replace
    observed: list[int] = []

    def spy(src, dst):
        observed.append(stat.S_IMODE(os.stat(src).st_mode))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    atomic_write(store_directory / "credentials", b"content", mode)
    assert observed == [mode]


def test_a_failed_write_leaves_the_original_file_intact(store_directory, monkeypatch):
    target = store_directory / "credentials"
    store_directory.mkdir(parents=True)
    target.write_bytes(b"the original content")

    def boom(*_args, **_kwargs):
        raise OSError("induced failure")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        atomic_write(target, b"the replacement content", CREDENTIALS_MODE)
    assert target.read_bytes() == b"the original content"


def test_a_failed_rename_leaves_the_original_file_intact(store_directory, monkeypatch):
    target = store_directory / "credentials"
    store_directory.mkdir(parents=True)
    target.write_bytes(b"the original content")

    def boom(*_args, **_kwargs):
        raise OSError("induced failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(target, b"the replacement content", CREDENTIALS_MODE)
    assert target.read_bytes() == b"the original content"


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_a_failed_write_leaves_no_temporary_file_behind(
    store_directory, monkeypatch, failing
):
    """Debris in the store directory would be a sealed credentials file nobody
    is tracking, kept at whatever mode `mkstemp` produced."""
    target = store_directory / "credentials"
    store_directory.mkdir(parents=True)
    target.write_bytes(b"the original content")

    def boom(*_args, **_kwargs):
        raise OSError("induced failure")

    monkeypatch.setattr(os, failing, boom)
    with pytest.raises(OSError):
        atomic_write(target, b"the replacement content", CREDENTIALS_MODE)
    assert [path.name for path in store_directory.iterdir()] == ["credentials"]


@pytest.mark.parametrize(
    ("filename", "allowed", "mode", "expected"),
    [
        ("credentials", 0o600, 0o600, False),
        ("credentials", 0o600, 0o640, True),
        ("credentials", 0o600, 0o644, True),
        ("config", 0o600, 0o600, False),
        ("config", 0o600, 0o644, True),
        ("config", 0o600, 0o666, True),
        # More restrictive than allowed is not "more permissive".
        ("credentials", 0o644, 0o600, False),
    ],
)
def test_a_more_permissive_mode_than_required_is_warned_about(
    store_directory, capsys, filename, allowed, mode, expected
):
    """`allowed` is a literal on both halves, deliberately.

    Passing `allowed=CREDENTIALS_MODE` made every row compare the module
    against itself: widen the constant and the expectation widens with it,
    so the parametrisation kept passing while describing a different, more
    permissive policy than the one written down here.
    """
    store_directory.mkdir(parents=True)
    path = store_directory / filename
    path.write_text("")
    os.chmod(path, mode)
    store.warn_if_permissive(path, allowed)
    captured = capsys.readouterr()
    assert (str(path) in captured.err) is expected
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Sealing and resolving
# ---------------------------------------------------------------------------


@pytest.fixture
def initialised(store_directory):
    init_store(PASSWORD, directory=store_directory)
    set_credentials("one", CREDENTIALS, PASSWORD, directory=store_directory)
    return store_directory


def test_credentials_round_trip_through_the_store(initialised):
    assert resolve_credentials("one", PASSWORD, directory=initialised) == CREDENTIALS


def test_the_credentials_file_holds_none_of_the_credential_bytes(initialised):
    """A store that stringified or base64-encoded its values instead of sealing
    them would pass the round-trip test above and fail here."""
    raw = (initialised / "credentials").read_bytes()
    assert MARKER.encode("ascii") not in raw
    assert MARKER.encode("ascii").hex().encode("ascii") not in raw
    assert base64.b64encode(MARKER.encode("ascii")) not in raw
    assert PASSWORD.encode("ascii") not in raw


def test_a_wrong_password_is_reported_as_a_wrong_password(initialised):
    """What the store-level verifier buys: the failure names the actual problem
    instead of surfacing as an unopenable profile."""
    with pytest.raises(PasswordError):
        resolve_credentials("one", WRONG_PASSWORD, directory=initialised)


def test_a_wrong_password_is_refused_before_any_profile_envelope_is_touched(
    initialised, monkeypatch
):
    """The verifier must be the first thing opened, or a mistyped password
    would be indistinguishable from a damaged store."""
    set_credentials("two", CREDENTIALS, PASSWORD, directory=initialised)
    with pytest.raises(PasswordError):
        set_credentials("three", CREDENTIALS, WRONG_PASSWORD, directory=initialised)
    assert store_profiles(initialised) == ["one", "two"]


def test_an_envelope_moved_between_profiles_does_not_open(initialised):
    """The store-level consequence of the AAD binding: an attacker (or a
    well-meaning hand edit) who copies one profile's envelope over another's
    gets a refusal, not that profile's credentials under a different name."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["profiles"]["two"] = document["profiles"]["one"]
    path.write_text(json.dumps(document))

    assert resolve_credentials("one", PASSWORD, directory=initialised) == CREDENTIALS
    with pytest.raises(StoreError) as raised:
        resolve_credentials("two", PASSWORD, directory=initialised)
    assert not isinstance(raised.value, PasswordError)
    assert MARKER not in str(raised.value)


def test_a_profile_with_no_envelope_is_an_error_not_an_empty_mapping(initialised):
    """Fail closed. Returning `{}` would let a caller run a command with no
    credentials in the environment and blame the backend for the result."""
    with pytest.raises(ProfileError):
        resolve_credentials("absent", PASSWORD, directory=initialised)


def test_a_profile_in_config_without_credentials_selects_but_does_not_resolve(
    store_directory,
):
    """How the two files relate when a profile is in one and not the other.
    `config` alone is enough to know which backend a profile names — which is
    all `login` needs — so selection succeeds; resolving credentials that were
    never sealed is an error naming the profile."""
    write_config(store_directory, config_with("one"))
    init_store(PASSWORD, directory=store_directory)

    config = load_store_config(store_directory)
    assert select_profile("one", config=config, environ={}).name == "one"
    with pytest.raises(ProfileError) as raised:
        resolve_credentials("one", PASSWORD, directory=store_directory)
    assert "one" in str(raised.value)


def test_resolving_before_the_store_exists_says_so(store_directory):
    with pytest.raises(StoreError):
        resolve_credentials("one", PASSWORD, directory=store_directory)


def test_initialising_over_an_existing_store_is_refused(initialised):
    """`init` twice is a plausible mistake; discarding every sealed envelope in
    response to it is not a recoverable one."""
    with pytest.raises(StoreError):
        init_store(PASSWORD, directory=initialised)
    assert resolve_credentials("one", PASSWORD, directory=initialised) == CREDENTIALS


@pytest.mark.parametrize(
    "payload",
    [
        {"NAME": 42},  # a value that is not a string
        {"NAME": [MARKER]},
        [MARKER],  # not a mapping at all
        MARKER,
        # A mapping built the wrong way round, so the credential is the *key*.
        # This is what makes naming the offending entry unsafe: the assumption
        # that keys are harmless variable names is not one this branch can make.
        {MARKER: 42},
    ],
)
def test_a_malformed_decrypted_payload_is_refused_without_echoing_it(
    initialised, payload
):
    """The only branch in the module where decrypted values are in scope when an
    exception is raised — every other failure happens before anything is opened,
    so an absence assertion there is trivially true. This one is not: the
    payload is sealed under the right password and the right AAD, so it decrypts
    successfully and *then* fails validation. Includes a mapping whose key is
    the credential, which is why the message names neither half of an entry.
    """
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["profiles"]["malformed"] = crypto.seal_json(payload, PASSWORD, "malformed")
    path.write_text(json.dumps(document))

    with pytest.raises(StoreError) as raised:
        resolve_credentials("malformed", PASSWORD, directory=initialised)
    assert MARKER not in str(raised.value)
    assert "malformed" in str(raised.value)


def test_a_damaged_store_file_is_refused_rather_than_read_as_empty(initialised):
    (initialised / "credentials").write_text("{ not json")
    with pytest.raises(StoreError):
        resolve_credentials("one", PASSWORD, directory=initialised)


def test_an_unsupported_store_version_is_refused(initialised):
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["v"] = 99
    path.write_text(json.dumps(document))
    with pytest.raises(StoreError):
        resolve_credentials("one", PASSWORD, directory=initialised)


def test_a_store_with_no_verifier_is_refused_rather_than_opened(initialised):
    """The verifier is what makes a mistyped password fail immediately
    instead of weeks later, so a store that has none cannot be treated as
    "no check to run" -- that would silently restore the failure mode the
    verifier exists to remove. None of `_check_password`'s three refusals
    had ever executed."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    del document[store.VERIFIER_KEY]
    path.write_text(json.dumps(document))

    with pytest.raises(StoreError, match="verifier"):
        resolve_credentials("one", PASSWORD, directory=initialised)


def test_an_unusable_verifier_envelope_is_a_store_error_not_a_wrong_password(
    initialised,
):
    """A structurally broken verifier envelope is damage, and must be
    reported as damage: telling a user "wrong password" for a store their
    password is fine for sends them to retype it forever."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document[store.VERIFIER_KEY] = {"v": 1, "kdf": "not an object"}
    path.write_text(json.dumps(document))

    with pytest.raises(StoreError) as raised:
        resolve_credentials("one", PASSWORD, directory=initialised)
    assert not isinstance(raised.value, PasswordError)
    assert "unusable" in str(raised.value)


def test_a_verifier_that_opens_to_the_wrong_plaintext_is_refused(initialised):
    """The known-plaintext half of the check, which the GCM tag alone does
    not provide: an envelope sealed under the right password and the right
    AAD but carrying different content still is not this tool's verifier."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document[store.VERIFIER_KEY] = crypto.seal_json(
        "not the verifier plaintext", PASSWORD, store.VERIFIER_AAD
    )
    path.write_text(json.dumps(document))

    with pytest.raises(StoreError, match="did not match"):
        resolve_credentials("one", PASSWORD, directory=initialised)


def test_the_temporary_file_is_created_in_the_target_directory(
    store_directory, monkeypatch
):
    """`atomic_write`'s atomicity rests entirely on the rename being within
    one filesystem, which rests entirely on `mkstemp(dir=...)`. Nothing
    pinned it: dropping `dir=directory` puts the temporary file under
    `$TMPDIR`, and `os.replace` across filesystems raises `OSError` -- or,
    worse, on a setup where `$TMPDIR` happens to share the filesystem,
    succeeds while writing sealed credentials into a world-traversable
    directory first.

    Recorded at the moment of the rename, which is the only point the
    temporary file is guaranteed to still exist.
    """
    store_directory.mkdir(parents=True)
    real_replace = os.replace
    observed: list[pathlib.Path] = []

    def spy(src, dst):
        observed.append(pathlib.Path(src).parent)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    atomic_write(store_directory / "credentials", b"content", 0o600)
    assert observed == [store_directory]


# ---------------------------------------------------------------------------
# Password rotation
# ---------------------------------------------------------------------------


def test_rotation_reseals_every_envelope(initialised):
    set_credentials(
        "two", {"NAME": "another-placeholder"}, PASSWORD, directory=initialised
    )
    rotate_password(PASSWORD, NEW_PASSWORD, directory=initialised)

    assert resolve_credentials("one", NEW_PASSWORD, directory=initialised) == CREDENTIALS
    assert resolve_credentials("two", NEW_PASSWORD, directory=initialised) == {
        "NAME": "another-placeholder"
    }
    with pytest.raises(PasswordError):
        resolve_credentials("one", PASSWORD, directory=initialised)


def test_rotation_draws_fresh_salts_rather_than_reusing_the_old_ones(initialised):
    before = json.loads((initialised / "credentials").read_bytes())
    rotate_password(PASSWORD, NEW_PASSWORD, directory=initialised)
    after = json.loads((initialised / "credentials").read_bytes())
    for section in (["verifier"], ["profiles", "one"]):
        old, new = before, after
        for key in section:
            old, new = old[key], new[key]
        assert old["salt"] != new["salt"]
        assert old["nonce"] != new["nonce"]


def test_rotation_reseals_an_envelope_whose_config_table_is_gone(store_directory):
    """Rotation iterates the credentials file, not the profile list in `config`.
    An implementation that walked `config` instead would silently and
    irrecoverably destroy this envelope, and every other test here would pass."""
    write_config(store_directory, config_with("one"))
    init_store(PASSWORD, directory=store_directory)
    set_credentials("one", CREDENTIALS, PASSWORD, directory=store_directory)
    set_credentials(
        "orphan", {"NAME": "orphan-placeholder"}, PASSWORD, directory=store_directory
    )

    rotate_password(PASSWORD, NEW_PASSWORD, directory=store_directory)

    assert store_profiles(store_directory) == ["one", "orphan"]
    assert resolve_credentials("orphan", NEW_PASSWORD, directory=store_directory) == {
        "NAME": "orphan-placeholder"
    }


def test_rotation_with_the_wrong_current_password_changes_nothing(initialised):
    before = (initialised / "credentials").read_bytes()
    with pytest.raises(PasswordError):
        rotate_password(WRONG_PASSWORD, NEW_PASSWORD, directory=initialised)
    assert (initialised / "credentials").read_bytes() == before


def test_rotation_is_all_or_nothing_when_one_reseal_fails(initialised, monkeypatch):
    """Induce a failure partway through and prove the store is untouched: the
    same bytes, both envelopes still opening under the *old* password, neither
    under the new one, and no debris beside the file. A partial rotation would
    leave a store whose profiles need different passwords with nothing recording
    which is which.
    """
    set_credentials(
        "two", {"NAME": "another-placeholder"}, PASSWORD, directory=initialised
    )
    before = (initialised / "credentials").read_bytes()

    real_seal_json = store.seal_json

    def failing_seal_json(payload, password, aad):
        # Keyed on the profile, not on a call count. An implementation that
        # wrote after each re-seal would already have persisted "one" by the
        # time this fires, so the byte-comparison below catches it — a
        # count-based trigger happened to fire before the first such write and
        # let that implementation pass.
        if aad == "two":
            raise RuntimeError("induced failure partway through the rotation")
        return real_seal_json(payload, password, aad)

    monkeypatch.setattr(store, "seal_json", failing_seal_json)
    with pytest.raises(RuntimeError):
        rotate_password(PASSWORD, NEW_PASSWORD, directory=initialised)
    monkeypatch.undo()

    assert (initialised / "credentials").read_bytes() == before
    assert resolve_credentials("one", PASSWORD, directory=initialised) == CREDENTIALS
    assert resolve_credentials("two", PASSWORD, directory=initialised) == {
        "NAME": "another-placeholder"
    }
    with pytest.raises(PasswordError):
        resolve_credentials("one", NEW_PASSWORD, directory=initialised)
    assert sorted(path.name for path in initialised.iterdir()) == ["credentials"]


def test_rotation_touches_the_file_exactly_once(initialised, monkeypatch):
    """The mechanism behind all-or-nothing, asserted directly: build every
    re-sealed envelope in memory, then write once. An implementation that saved
    after each profile would be all-or-nothing only by luck about where a
    failure landed, and the induced-failure test above can only observe the
    failures it induces."""
    set_credentials(
        "two", {"NAME": "another-placeholder"}, PASSWORD, directory=initialised
    )
    real_atomic_write = store.atomic_write
    writes: list[str] = []

    def counting_atomic_write(path, data, mode):
        writes.append(path.name)
        return real_atomic_write(path, data, mode)

    monkeypatch.setattr(store, "atomic_write", counting_atomic_write)
    rotate_password(PASSWORD, NEW_PASSWORD, directory=initialised)
    assert writes == ["credentials"]


def test_rotation_stops_rather_than_dropping_an_envelope_it_cannot_open(initialised):
    """An envelope that will not open must abort the rotation. Skipping it would
    turn a damaged envelope into a deleted one."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["profiles"]["damaged"] = {**document["profiles"]["one"], "ct": "AAAA"}
    path.write_text(json.dumps(document))
    before = path.read_bytes()

    with pytest.raises(StoreError):
        rotate_password(PASSWORD, NEW_PASSWORD, directory=initialised)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Location, and what never leaks
# ---------------------------------------------------------------------------


def test_the_store_lives_under_the_xdg_config_home(tmp_path, monkeypatch):
    """`store_dir` defers to `cli.config_home` rather than re-deriving the XDG
    base, so there is one answer to where this tool's files live."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert store.store_dir() == tmp_path / "stackward"
    assert store.credentials_path() == tmp_path / "stackward" / "credentials"
    assert store.config_path() == tmp_path / "stackward" / "config"


def test_no_error_message_from_any_failing_path_contains_a_credential(
    initialised, capsys
):
    """One assertion over every failure this module can produce. Any new branch
    that interpolated a value would have to be added here to be seen — but a
    branch that leaked one would already be caught by the sweep below."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["profiles"]["moved"] = document["profiles"]["one"]
    path.write_text(json.dumps(document))

    messages: list[str] = []
    for call in (
        lambda: resolve_credentials("one", WRONG_PASSWORD, directory=initialised),
        lambda: resolve_credentials("absent", PASSWORD, directory=initialised),
        lambda: resolve_credentials("moved", PASSWORD, directory=initialised),
        lambda: rotate_password(WRONG_PASSWORD, NEW_PASSWORD, directory=initialised),
        lambda: init_store(PASSWORD, directory=initialised),
        lambda: set_credentials(
            "four", CREDENTIALS, WRONG_PASSWORD, directory=initialised
        ),
    ):
        with pytest.raises(StoreError) as raised:
            call()
        messages.append(str(raised.value))

    captured = capsys.readouterr()
    for text in [*messages, captured.out, captured.err]:
        assert MARKER not in text
        assert PASSWORD not in text
        assert WRONG_PASSWORD not in text


def test_a_profile_dataclass_reports_which_form_it_carries():
    assert not Profile(name="one", backend_url="file:///placeholder").has_components
    assert Profile(name="one", bucket="placeholder-bucket").has_components


# ---------------------------------------------------------------------------
# An absent password, per entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("password", ["", "   ", "\t\n"])
def test_init_store_refuses_an_absent_password_instead_of_sealing_with_nothing(
    store_directory, password
):
    """Argon2id derives a usable key from `b""`, so an unguarded `init_store("")`
    writes a well-formed store that opens with no password at all — no error, no
    warning, and no way to tell it from a protected one."""
    with pytest.raises(PasswordError):
        init_store(password, directory=store_directory)
    assert not (store_directory / "credentials").exists()


@pytest.mark.parametrize("password", ["", "   ", "\t\n"])
def test_set_credentials_refuses_an_absent_password(initialised, password):
    with pytest.raises(PasswordError):
        set_credentials("two", CREDENTIALS, password, directory=initialised)
    assert store_profiles(initialised) == ["one"]


@pytest.mark.parametrize("password", ["", "   ", "\t\n"])
def test_resolve_credentials_refuses_an_absent_password(initialised, password):
    with pytest.raises(PasswordError):
        resolve_credentials("one", password, directory=initialised)


@pytest.mark.parametrize("password", ["", "   ", "\t\n"])
def test_rotation_refuses_an_absent_new_password_and_changes_nothing(
    initialised, password
):
    """The irreversible one: rotating *to* an empty password would leave a store
    that opens with nothing, with the old password gone."""
    before = (initialised / "credentials").read_bytes()
    with pytest.raises(PasswordError):
        rotate_password(PASSWORD, password, directory=initialised)
    assert (initialised / "credentials").read_bytes() == before
    assert resolve_credentials("one", PASSWORD, directory=initialised) == CREDENTIALS


@pytest.mark.parametrize("password", ["", "   ", "\t\n"])
def test_rotation_refuses_an_absent_current_password(initialised, password):
    with pytest.raises(PasswordError):
        rotate_password(password, NEW_PASSWORD, directory=initialised)


def test_an_absent_password_is_not_reported_as_a_damaged_store(initialised):
    """`EmptyPasswordError` is a `CryptoError`, so the broad clause in
    `_check_password` would otherwise recode "no password" as "the verifier is
    unusable" — a message that sends the reader after the wrong problem."""
    with pytest.raises(PasswordError) as raised:
        resolve_credentials("one", "", directory=initialised)
    assert "required" in str(raised.value)


def test_rotation_keeps_top_level_fields_it_does_not_own(initialised):
    """Rotation replaces the verifier and the profiles and leaves the rest of
    the document alone. Rebuilding it from a literal would silently drop any
    other field, and would leave the two write paths disagreeing about what a
    document is — `set_credentials` preserves the whole thing."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["a_field_a_later_version_added"] = {"kept": True}
    path.write_text(json.dumps(document))

    rotate_password(PASSWORD, NEW_PASSWORD, directory=initialised)

    after = json.loads(path.read_bytes())
    assert after["a_field_a_later_version_added"] == {"kept": True}
    assert resolve_credentials("one", NEW_PASSWORD, directory=initialised) == CREDENTIALS
