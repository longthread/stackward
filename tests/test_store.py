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
import stat

import pytest

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
def home(tmp_path):
    """An isolated store directory. Every store function takes `home`, so no
    test depends on the real `XDG_CONFIG_HOME` or writes outside `tmp_path`."""
    return tmp_path / "stackward"


def write_config(home, text: str):
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config"
    path.write_text(text)
    # At the required mode, so that these tests do not trip the permission
    # warning as a side effect of whatever umask the suite runs under. The
    # warning has its own tests below.
    os.chmod(path, CONFIG_MODE)
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
def test_a_backend_url_is_carried_through_byte_for_byte(home, url):
    """Pass-through, not parsing. Any normalising or reassembling this module
    did would show up here as a changed string, and would be the thing that
    stopped an unanticipated scheme from working."""
    write_config(home, f"[profile.one]\nbackend_url = {json.dumps(url)}\n")
    assert load_store_config(home).profiles["one"].backend_url == url


def test_the_component_form_is_carried_as_structured_data(home):
    write_config(
        home,
        "[profile.one]\n"
        'bucket = "placeholder-bucket"\n'
        'prefix = "placeholder/prefix"\n'
        'endpoint = "https://placeholder.invalid"\n'
        'region = "placeholder-region"\n',
    )
    profile = load_store_config(home).profiles["one"]
    assert profile.backend_url is None
    assert profile.bucket == "placeholder-bucket"
    assert profile.prefix == "placeholder/prefix"
    assert profile.endpoint == "https://placeholder.invalid"
    assert profile.region == "placeholder-region"


def test_a_profile_carrying_both_forms_is_refused(home):
    """Two answers to "which backend?" with no rule for choosing between them."""
    write_config(
        home,
        '[profile.one]\nbackend_url = "file:///placeholder"\nbucket = "placeholder"\n',
    )
    with pytest.raises(StoreError):
        load_store_config(home)


def test_a_profile_carrying_neither_form_is_refused(home):
    """There is no default backend to fall back on, so an empty profile is an
    error and not an inherited one."""
    write_config(home, "[profile.one]\n")
    with pytest.raises(StoreError):
        load_store_config(home)


def test_the_component_form_without_a_bucket_is_refused(home):
    write_config(home, '[profile.one]\nregion = "placeholder-region"\n')
    with pytest.raises(StoreError):
        load_store_config(home)


def test_an_unknown_profile_key_is_refused_rather_than_ignored(home):
    write_config(
        home,
        '[profile.one]\nbackend_url = "file:///placeholder"\nbucket_name = "typo"\n',
    )
    with pytest.raises(StoreError):
        load_store_config(home)


def test_a_default_profile_with_no_table_is_refused(home):
    write_config(home, 'default_profile = "missing"\n' + config_with("one"))
    with pytest.raises(StoreError):
        load_store_config(home)


def test_an_absent_config_is_empty_rather_than_an_error(home):
    """Not having set the tool up yet is a normal state. It is `select_profile`
    that reports it, because only it can say what to do about it."""
    assert load_store_config(home) == StoreConfig()


def test_an_invalid_config_raises_rather_than_reading_as_absent(home):
    """A typo must not degrade into "no profiles", which would turn a broken
    backend into a silently missing one."""
    write_config(home, "[profile.one\nbackend_url = ")
    with pytest.raises(StoreError):
        load_store_config(home)


# ---------------------------------------------------------------------------
# Profile selection precedence
# ---------------------------------------------------------------------------


@pytest.fixture
def four_profiles(home):
    write_config(home, config_with("chosen", "from_env", "from_repo", "fallback"))
    return load_store_config(home)


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


def test_the_repository_config_wins_over_the_default_profile(home):
    write_config(home, config_with("from_repo", "fallback", default="fallback"))
    selected = select_profile(
        config=load_store_config(home), repo_profile="from_repo", environ={}
    )
    assert selected.name == "from_repo"


def test_the_default_profile_is_the_last_resort(home):
    write_config(home, config_with("fallback", default="fallback"))
    selected = select_profile(config=load_store_config(home), environ={})
    assert selected.name == "fallback"


def test_no_profile_anywhere_is_an_error_and_never_a_guess(four_profiles):
    with pytest.raises(ProfileError):
        select_profile(config=four_profiles, environ={})


def test_a_named_profile_that_does_not_exist_errors_instead_of_using_the_default(home):
    """The case that separates "first source that *speaks* wins" from "first
    source that *resolves* wins". Falling through here would run the command
    against a backend nobody asked for — the worst thing this tool could do
    quietly — and a loop that skipped invalid sources would pass every other
    precedence test in this file.
    """
    write_config(home, config_with("fallback", default="fallback"))
    config = load_store_config(home)
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


def test_an_empty_environment_variable_is_an_error_rather_than_a_miss(home):
    """A variable set to the empty string is a broken variable, not an absent
    one. Skipping it would silently substitute the default."""
    write_config(home, config_with("fallback", default="fallback"))
    with pytest.raises(ProfileError):
        select_profile(config=load_store_config(home), environ={ENV_PROFILE: ""})


def test_a_profile_name_that_would_collide_with_the_verifier_is_rejected(home):
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


def test_the_store_directory_is_created_private_whatever_the_umask(home):
    previous = os.umask(0)
    try:
        store.ensure_store_dir(home)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(home.stat().st_mode) == DIR_MODE


def test_the_credentials_file_is_written_private_whatever_the_umask(home):
    previous = os.umask(0)
    try:
        init_store(PASSWORD, home=home)
    finally:
        os.umask(previous)
    assert stat.S_IMODE((home / "credentials").stat().st_mode) == CREDENTIALS_MODE


@pytest.mark.parametrize("mode", [CREDENTIALS_MODE, CONFIG_MODE])
def test_the_mode_is_set_before_the_rename_not_after(home, monkeypatch, mode):
    """Setting the mode after the rename leaves a window in which the finished
    file is visible at its real path with whatever mode it was created with.
    This records the mode of the temporary file at the moment of the rename.

    Both modes are checked because only one of them discriminates: `mkstemp`
    already creates at 0600, so for the credentials file a mode set after the
    rename would look identical and this test would prove nothing. 0644 is what
    makes the ordering observable.
    """
    home.mkdir(parents=True)
    real_replace = os.replace
    observed: list[int] = []

    def spy(src, dst):
        observed.append(stat.S_IMODE(os.stat(src).st_mode))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    atomic_write(home / "credentials", b"content", mode)
    assert observed == [mode]


def test_a_failed_write_leaves_the_original_file_intact(home, monkeypatch):
    target = home / "credentials"
    home.mkdir(parents=True)
    target.write_bytes(b"the original content")

    def boom(*_args, **_kwargs):
        raise OSError("induced failure")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        atomic_write(target, b"the replacement content", CREDENTIALS_MODE)
    assert target.read_bytes() == b"the original content"


def test_a_failed_rename_leaves_the_original_file_intact(home, monkeypatch):
    target = home / "credentials"
    home.mkdir(parents=True)
    target.write_bytes(b"the original content")

    def boom(*_args, **_kwargs):
        raise OSError("induced failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write(target, b"the replacement content", CREDENTIALS_MODE)
    assert target.read_bytes() == b"the original content"


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_a_failed_write_leaves_no_temporary_file_behind(home, monkeypatch, failing):
    """Debris in the store directory would be a sealed credentials file nobody
    is tracking, kept at whatever mode `mkstemp` produced."""
    target = home / "credentials"
    home.mkdir(parents=True)
    target.write_bytes(b"the original content")

    def boom(*_args, **_kwargs):
        raise OSError("induced failure")

    monkeypatch.setattr(os, failing, boom)
    with pytest.raises(OSError):
        atomic_write(target, b"the replacement content", CREDENTIALS_MODE)
    assert [path.name for path in home.iterdir()] == ["credentials"]


@pytest.mark.parametrize(
    ("filename", "allowed", "mode", "expected"),
    [
        ("credentials", CREDENTIALS_MODE, 0o600, False),
        ("credentials", CREDENTIALS_MODE, 0o640, True),
        ("credentials", CREDENTIALS_MODE, 0o644, True),
        ("config", CONFIG_MODE, 0o644, False),
        ("config", CONFIG_MODE, 0o600, False),
        ("config", CONFIG_MODE, 0o666, True),
    ],
)
def test_a_more_permissive_mode_than_required_is_warned_about(
    home, capsys, filename, allowed, mode, expected
):
    home.mkdir(parents=True)
    path = home / filename
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
def initialised(home):
    init_store(PASSWORD, home=home)
    set_credentials("one", CREDENTIALS, PASSWORD, home=home)
    return home


def test_credentials_round_trip_through_the_store(initialised):
    assert resolve_credentials("one", PASSWORD, home=initialised) == CREDENTIALS


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
        resolve_credentials("one", WRONG_PASSWORD, home=initialised)


def test_a_wrong_password_is_refused_before_any_profile_envelope_is_touched(
    initialised, monkeypatch
):
    """The verifier must be the first thing opened, or a mistyped password
    would be indistinguishable from a damaged store."""
    set_credentials("two", CREDENTIALS, PASSWORD, home=initialised)
    with pytest.raises(PasswordError):
        set_credentials("three", CREDENTIALS, WRONG_PASSWORD, home=initialised)
    assert store_profiles(initialised) == ["one", "two"]


def test_an_envelope_moved_between_profiles_does_not_open(initialised):
    """The store-level consequence of the AAD binding: an attacker (or a
    well-meaning hand edit) who copies one profile's envelope over another's
    gets a refusal, not that profile's credentials under a different name."""
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["profiles"]["two"] = document["profiles"]["one"]
    path.write_text(json.dumps(document))

    assert resolve_credentials("one", PASSWORD, home=initialised) == CREDENTIALS
    with pytest.raises(StoreError) as raised:
        resolve_credentials("two", PASSWORD, home=initialised)
    assert not isinstance(raised.value, PasswordError)
    assert MARKER not in str(raised.value)


def test_a_profile_with_no_envelope_is_an_error_not_an_empty_mapping(initialised):
    """Fail closed. Returning `{}` would let a caller run a command with no
    credentials in the environment and blame the backend for the result."""
    with pytest.raises(ProfileError):
        resolve_credentials("absent", PASSWORD, home=initialised)


def test_a_profile_in_config_without_credentials_selects_but_does_not_resolve(home):
    """How the two files relate when a profile is in one and not the other.
    `config` alone is enough to know which backend a profile names — which is
    all `login` needs — so selection succeeds; resolving credentials that were
    never sealed is an error naming the profile."""
    write_config(home, config_with("one"))
    init_store(PASSWORD, home=home)

    assert select_profile("one", config=load_store_config(home), environ={}).name == "one"
    with pytest.raises(ProfileError) as raised:
        resolve_credentials("one", PASSWORD, home=home)
    assert "one" in str(raised.value)


def test_resolving_before_the_store_exists_says_so(home):
    with pytest.raises(StoreError):
        resolve_credentials("one", PASSWORD, home=home)


def test_initialising_over_an_existing_store_is_refused(initialised):
    """`init` twice is a plausible mistake; discarding every sealed envelope in
    response to it is not a recoverable one."""
    with pytest.raises(StoreError):
        init_store(PASSWORD, home=initialised)
    assert resolve_credentials("one", PASSWORD, home=initialised) == CREDENTIALS


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
        resolve_credentials("malformed", PASSWORD, home=initialised)
    assert MARKER not in str(raised.value)
    assert "malformed" in str(raised.value)


def test_a_damaged_store_file_is_refused_rather_than_read_as_empty(initialised):
    (initialised / "credentials").write_text("{ not json")
    with pytest.raises(StoreError):
        resolve_credentials("one", PASSWORD, home=initialised)


def test_an_unsupported_store_version_is_refused(initialised):
    path = initialised / "credentials"
    document = json.loads(path.read_bytes())
    document["v"] = 99
    path.write_text(json.dumps(document))
    with pytest.raises(StoreError):
        resolve_credentials("one", PASSWORD, home=initialised)


# ---------------------------------------------------------------------------
# Password rotation
# ---------------------------------------------------------------------------


def test_rotation_reseals_every_envelope(initialised):
    set_credentials("two", {"NAME": "another-placeholder"}, PASSWORD, home=initialised)
    rotate_password(PASSWORD, NEW_PASSWORD, home=initialised)

    assert resolve_credentials("one", NEW_PASSWORD, home=initialised) == CREDENTIALS
    assert resolve_credentials("two", NEW_PASSWORD, home=initialised) == {
        "NAME": "another-placeholder"
    }
    with pytest.raises(PasswordError):
        resolve_credentials("one", PASSWORD, home=initialised)


def test_rotation_draws_fresh_salts_rather_than_reusing_the_old_ones(initialised):
    before = json.loads((initialised / "credentials").read_bytes())
    rotate_password(PASSWORD, NEW_PASSWORD, home=initialised)
    after = json.loads((initialised / "credentials").read_bytes())
    for section in (["verifier"], ["profiles", "one"]):
        old, new = before, after
        for key in section:
            old, new = old[key], new[key]
        assert old["salt"] != new["salt"]
        assert old["nonce"] != new["nonce"]


def test_rotation_reseals_an_envelope_whose_config_table_is_gone(home):
    """Rotation iterates the credentials file, not the profile list in `config`.
    An implementation that walked `config` instead would silently and
    irrecoverably destroy this envelope, and every other test here would pass."""
    write_config(home, config_with("one"))
    init_store(PASSWORD, home=home)
    set_credentials("one", CREDENTIALS, PASSWORD, home=home)
    set_credentials("orphan", {"NAME": "orphan-placeholder"}, PASSWORD, home=home)

    rotate_password(PASSWORD, NEW_PASSWORD, home=home)

    assert store_profiles(home) == ["one", "orphan"]
    assert resolve_credentials("orphan", NEW_PASSWORD, home=home) == {
        "NAME": "orphan-placeholder"
    }


def test_rotation_with_the_wrong_current_password_changes_nothing(initialised):
    before = (initialised / "credentials").read_bytes()
    with pytest.raises(PasswordError):
        rotate_password(WRONG_PASSWORD, NEW_PASSWORD, home=initialised)
    assert (initialised / "credentials").read_bytes() == before


def test_rotation_is_all_or_nothing_when_one_reseal_fails(initialised, monkeypatch):
    """Induce a failure partway through and prove the store is untouched: the
    same bytes, both envelopes still opening under the *old* password, neither
    under the new one, and no debris beside the file. A partial rotation would
    leave a store whose profiles need different passwords with nothing recording
    which is which.
    """
    set_credentials("two", {"NAME": "another-placeholder"}, PASSWORD, home=initialised)
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
        rotate_password(PASSWORD, NEW_PASSWORD, home=initialised)
    monkeypatch.undo()

    assert (initialised / "credentials").read_bytes() == before
    assert resolve_credentials("one", PASSWORD, home=initialised) == CREDENTIALS
    assert resolve_credentials("two", PASSWORD, home=initialised) == {
        "NAME": "another-placeholder"
    }
    with pytest.raises(PasswordError):
        resolve_credentials("one", NEW_PASSWORD, home=initialised)
    assert sorted(path.name for path in initialised.iterdir()) == ["credentials"]


def test_rotation_touches_the_file_exactly_once(initialised, monkeypatch):
    """The mechanism behind all-or-nothing, asserted directly: build every
    re-sealed envelope in memory, then write once. An implementation that saved
    after each profile would be all-or-nothing only by luck about where a
    failure landed, and the induced-failure test above can only observe the
    failures it induces."""
    set_credentials("two", {"NAME": "another-placeholder"}, PASSWORD, home=initialised)
    real_atomic_write = store.atomic_write
    writes: list[str] = []

    def counting_atomic_write(path, data, mode):
        writes.append(path.name)
        return real_atomic_write(path, data, mode)

    monkeypatch.setattr(store, "atomic_write", counting_atomic_write)
    rotate_password(PASSWORD, NEW_PASSWORD, home=initialised)
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
        rotate_password(PASSWORD, NEW_PASSWORD, home=initialised)
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
        lambda: resolve_credentials("one", WRONG_PASSWORD, home=initialised),
        lambda: resolve_credentials("absent", PASSWORD, home=initialised),
        lambda: resolve_credentials("moved", PASSWORD, home=initialised),
        lambda: rotate_password(WRONG_PASSWORD, NEW_PASSWORD, home=initialised),
        lambda: init_store(PASSWORD, home=initialised),
        lambda: set_credentials("four", CREDENTIALS, WRONG_PASSWORD, home=initialised),
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
