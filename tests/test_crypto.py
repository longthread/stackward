"""Tests for the sealed envelope.

Each test is named for the property it proves, and several are written
specifically so that they cannot pass for the wrong reason. The three that
matter most:

- a wrong password must fail *through the GCM tag*, so the test asserts on the
  underlying `InvalidTag` and not merely on "something was raised" — a length
  check or a plaintext comparison would satisfy the weaker assertion while
  leaving unauthenticated bytes reachable;
- an envelope must not open under a different AAD, which is the only evidence
  that the AAD is genuinely bound rather than passed and ignored;
- the ciphertext must contain none of the plaintext, asserted against a known
  marker value's bytes in three encodings, so that "we base64-encoded it and
  called it sealed" fails loudly.

No test prints a password or a plaintext on any path, including a failing one:
assertions are on absence and on exception types, never on captured secret text.
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm

from leakcheck import assert_no_leak

from stackward import crypto
from stackward.crypto import (
    ARGON2ID_ITERATIONS,
    ARGON2ID_LANES,
    ARGON2ID_MEMORY_KIB,
    ENVELOPE_VERSION,
    KDF_NAME,
    KEY_BYTES,
    NONCE_BYTES,
    SALT_BYTES,
    CryptoError,
    DecryptionError,
    EmptyPasswordError,
    EnvelopeError,
    describe_value,
    seal,
    seal_json,
    unseal,
    unseal_json,
)

PASSWORD = "correct password for these tests"
OTHER_PASSWORD = "a different password entirely"
AAD = "profile-one"
OTHER_AAD = "profile-two"

# A value chosen to be findable: if any part of it survives into the envelope,
# the absence assertions below will say so.
MARKER = b"marker-value-never-to-appear-in-ciphertext"


def test_seal_and_unseal_round_trip_returns_the_exact_plaintext():
    envelope = seal(MARKER, PASSWORD, AAD)
    assert unseal(envelope, PASSWORD, AAD) == MARKER


def test_a_wrong_password_fails_through_the_gcm_tag_rather_than_returning_garbage():
    """The failure must come from authentication, not from a check this module
    performs after the fact — otherwise unauthenticated plaintext exists
    somewhere in the call, however briefly."""
    envelope = seal(MARKER, PASSWORD, AAD)
    with pytest.raises(DecryptionError) as raised:
        unseal(envelope, OTHER_PASSWORD, AAD)
    assert isinstance(raised.value.__cause__, InvalidTag)


def test_an_envelope_sealed_under_one_profile_does_not_open_under_another():
    """The AAD is what stops an envelope being moved between profiles. A build
    that accepted the `aad` argument and never passed it to AES-GCM would pass
    the round-trip test above and fail here."""
    envelope = seal(MARKER, PASSWORD, AAD)
    with pytest.raises(DecryptionError) as raised:
        unseal(envelope, PASSWORD, OTHER_AAD)
    assert isinstance(raised.value.__cause__, InvalidTag)


def test_the_envelope_contains_none_of_the_plaintext():
    envelope = json.dumps(seal(MARKER, PASSWORD, AAD)).encode("utf-8")
    assert MARKER not in envelope
    assert base64.b64encode(MARKER) not in envelope
    assert MARKER.hex().encode("ascii") not in envelope


def test_the_envelope_contains_none_of_the_password():
    envelope = json.dumps(seal(MARKER, PASSWORD, AAD)).encode("utf-8")
    assert PASSWORD.encode("utf-8") not in envelope


def test_every_seal_draws_a_fresh_salt_and_nonce():
    """Sealing the same plaintext twice must produce three different fields.
    Equal ciphertexts would mean a reused nonce, which for GCM is not a
    weakness but a break."""
    first = seal(MARKER, PASSWORD, AAD)
    second = seal(MARKER, PASSWORD, AAD)
    assert first["salt"] != second["salt"]
    assert first["nonce"] != second["nonce"]
    assert first["ct"] != second["ct"]


def test_a_modified_ciphertext_is_refused():
    envelope = seal(MARKER, PASSWORD, AAD)
    raw = bytearray(base64.b64decode(envelope["ct"]))
    raw[0] ^= 0x01
    envelope["ct"] = base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(DecryptionError) as raised:
        unseal(envelope, PASSWORD, AAD)
    assert isinstance(raised.value.__cause__, InvalidTag)


def test_the_envelope_has_exactly_the_documented_fields():
    envelope = seal(MARKER, PASSWORD, AAD)
    assert set(envelope) == {"v", "kdf", "salt", "nonce", "ct"}
    assert envelope["v"] == ENVELOPE_VERSION
    assert envelope["kdf"]["name"] == KDF_NAME
    for field in ("salt", "nonce", "ct"):
        assert isinstance(envelope[field], str)
        base64.b64decode(envelope[field], validate=True)


def test_the_shipped_cost_parameters_are_the_documented_ones():
    """Pinned deliberately. `tests/test_store.py` lowers these for speed, so
    without this test a production weakening would go unnoticed."""
    assert (ARGON2ID_MEMORY_KIB, ARGON2ID_ITERATIONS, ARGON2ID_LANES) == (65536, 3, 4)
    assert (KEY_BYTES, SALT_BYTES, NONCE_BYTES) == (32, 16, 12)


def test_the_envelope_records_its_parameters_so_a_later_default_change_still_opens_it(
    monkeypatch,
):
    """The reason `kdf` carries parameters rather than only a name: raising the
    cost for new envelopes must not orphan every envelope already on disk."""
    envelope = seal(MARKER, PASSWORD, AAD)
    monkeypatch.setattr(crypto, "ARGON2ID_MEMORY_KIB", ARGON2ID_MEMORY_KIB * 2)
    monkeypatch.setattr(crypto, "ARGON2ID_ITERATIONS", ARGON2ID_ITERATIONS + 1)
    assert unseal(envelope, PASSWORD, AAD) == MARKER
    assert seal(MARKER, PASSWORD, AAD)["kdf"]["memory_kib"] == ARGON2ID_MEMORY_KIB * 2


def test_a_password_is_normalised_so_the_same_characters_open_the_same_envelope():
    """The same text entered on two platforms can arrive as two different byte
    sequences — precomposed or decomposed. Without NFC normalisation the store
    would open only on the machine that created it, and the failure would look
    like a forgotten password."""
    # Written as escapes rather than as literal characters: an editor or a
    # tool that normalised this source file would otherwise quietly turn
    # this into a test of nothing.
    precomposed = "passw\u00f6rd"  # o-with-diaeresis, one code point
    decomposed = "passwo\u0308rd"  # o, then a combining diaeresis
    assert precomposed != decomposed
    envelope = seal(MARKER, precomposed, AAD)
    assert unseal(envelope, decomposed, AAD) == MARKER


# ---------------------------------------------------------------------------
# Malformed input. `credentials` is hand-editable, so every one of these is
# reachable, and each must produce this module's own error rather than the
# library's — a caller catching `CryptoError` must not also have to enumerate
# `ValueError`, `OverflowError`, `TypeError` and `MemoryError` to stay closed.
# ---------------------------------------------------------------------------


def test_an_unsupported_envelope_version_is_refused():
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope["v"] = ENVELOPE_VERSION + 1
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


def test_an_unknown_kdf_is_refused():
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope["kdf"]["name"] = "something-else"
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


# ---------------------------------------------------------------------------
# `describe_value` -- the guard on the two header fields whose *contents* get
# printed back. Untested until now: `return repr(value)` for every input
# passed the entire suite, because the two tests that reach the branch
# (`test_an_unsupported_envelope_version_is_refused` and
# `test_an_unknown_kdf_is_refused`) assert only that something was raised.
# ---------------------------------------------------------------------------

# Opaque on purpose -- see `tests/leakcheck.py`. A marker containing a real
# word would share an eight-character run with the words these very messages
# print ("version", "unsupported"), and the run check would fire on output
# that disclosed nothing.
PASTED_INTO_A_HEADER_FIELD = "Zq7Xv4Rm2Kt9Lp5Nc8Wd6Hb3Jf"


@pytest.mark.parametrize(
    "value",
    [
        "some string",
        PASTED_INTO_A_HEADER_FIELD,
        ["a", "list"],
        {"a": "dict"},
        b"some bytes",
    ],
    # Named, so that a value never reaches a test id: pytest builds ids out
    # of the parameters themselves, and a failure report is output like any
    # other (GC4).
    ids=["str", "pasted-credential", "list", "dict", "bytes"],
)
def test_describe_value_reports_a_type_and_never_the_value(value):
    described = describe_value(value)
    assert described == f"<{type(value).__name__}>"
    assert str(value) not in described


@pytest.mark.parametrize("value", [None, 0, 1, -7, 1.5, True])
def test_describe_value_shows_what_cannot_carry_a_credential(value):
    """A number, a bool and `None` are shown, because the message is far more
    useful naming the version it actually found than reporting `<int>` -- and
    none of the three can be a pasted credential. `True` is included
    deliberately: `bool` is an `int` subclass, so it takes the shown branch,
    and a future rewrite that tightened the check to `type(value) is int`
    would silently start reporting `<bool>`."""
    assert describe_value(value) == repr(value)


def test_a_credential_pasted_into_the_version_field_is_not_read_back():
    """`v` is hand-editable and this message is printed. A user who pasted a
    credential one field too high must not have it echoed by the error that
    caught the mistake."""
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope["v"] = PASTED_INTO_A_HEADER_FIELD

    with pytest.raises(EnvelopeError) as raised:
        unseal(envelope, PASSWORD, AAD)

    message = str(raised.value)
    assert_no_leak(message, PASTED_INTO_A_HEADER_FIELD, what="a pasted header value")
    assert "<str>" in message


def test_a_credential_pasted_into_the_kdf_name_is_not_read_back():
    """The second call site, which the version test above cannot cover: the
    two are separate `describe_value` calls and a fix applied to one only
    would leave the other echoing."""
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope["kdf"]["name"] = PASTED_INTO_A_HEADER_FIELD

    with pytest.raises(EnvelopeError) as raised:
        unseal(envelope, PASSWORD, AAD)

    message = str(raised.value)
    assert_no_leak(message, PASTED_INTO_A_HEADER_FIELD, what="a pasted header value")
    assert "<str>" in message


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # Rejected by _MAX_MEMORY_KIB before it reaches the library. Note this
        # bound has a fallback: were it gone, `_derive` would still map the
        # library's MemoryError. The `iterations` ceiling below does not.
        ("memory_kib", 2**31),
        ("memory_kib", 1),
        ("memory_kib", 0),
        ("memory_kib", -1),
        ("memory_kib", "65536"),
        ("memory_kib", True),
        ("memory_kib", 65536.0),
        ("iterations", 0),
        ("iterations", -1),
        # The case the bounds exist for: without the ceiling this does not
        # raise, it runs — the test hangs rather than failing.
        ("iterations", 10**9),
        ("lanes", 0),
        ("lanes", 2**40),
        ("lanes", None),
    ],
)
def test_a_hand_edited_kdf_parameter_is_refused_before_it_reaches_the_library(key, value):
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope["kdf"][key] = value
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


@pytest.mark.parametrize("field", ["salt", "nonce", "ct"])
def test_a_field_that_is_not_valid_base64_is_refused(field):
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope[field] = "not base64!!"
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


@pytest.mark.parametrize("field", ["salt", "nonce", "ct", "kdf", "v"])
def test_a_missing_field_is_refused(field):
    envelope = seal(MARKER, PASSWORD, AAD)
    del envelope[field]
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


def test_a_nonce_of_the_wrong_length_is_refused():
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope["nonce"] = base64.b64encode(b"\x00" * (NONCE_BYTES + 1)).decode("ascii")
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


def test_a_salt_shorter_than_the_floor_is_refused():
    envelope = seal(MARKER, PASSWORD, AAD)
    envelope["salt"] = base64.b64encode(b"\x00" * 4).decode("ascii")
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


@pytest.mark.parametrize("envelope", ["a string", 17, None, ["a", "list"]])
def test_something_that_is_not_an_envelope_at_all_is_refused(envelope):
    with pytest.raises(EnvelopeError):
        unseal(envelope, PASSWORD, AAD)


# ---------------------------------------------------------------------------
# The JSON layer
# ---------------------------------------------------------------------------


def test_seal_json_round_trips_a_mapping():
    payload = {"NAME_ONE": "value-one", "NAME_TWO": "value-two"}
    envelope = seal_json(payload, PASSWORD, AAD)
    assert unseal_json(envelope, PASSWORD, AAD) == payload


def test_seal_json_does_not_depend_on_the_order_a_mapping_was_built_in():
    first = seal_json({"A": "1", "B": "2"}, PASSWORD, AAD)
    second = seal_json({"B": "2", "A": "1"}, PASSWORD, AAD)
    assert len(base64.b64decode(first["ct"])) == len(base64.b64decode(second["ct"]))
    assert unseal_json(first, PASSWORD, AAD) == unseal_json(second, PASSWORD, AAD)


def test_authenticated_plaintext_that_is_not_json_is_reported_rather_than_crashing():
    envelope = seal(b"\xff\xfe not json at all", PASSWORD, AAD)
    with pytest.raises(EnvelopeError):
        unseal_json(envelope, PASSWORD, AAD)


def test_unseal_json_does_not_parse_anything_the_tag_rejected():
    """Ordering property: JSON parsing must sit behind authentication, never in
    front of it. A wrong password produces a decryption error, never a parse
    error, because nothing is parsed at all."""
    envelope = seal_json({"NAME": "value"}, PASSWORD, AAD)
    with pytest.raises(DecryptionError):
        unseal_json(envelope, OTHER_PASSWORD, AAD)


def test_no_error_message_repeats_the_password_or_the_plaintext():
    envelope = seal(MARKER, PASSWORD, AAD)
    messages: list[str] = []
    for broken in (
        {**envelope, "v": 99},
        {**envelope, "salt": "!!"},
        {**envelope, "kdf": {**envelope["kdf"], "memory_kib": 2**31}},
        envelope,
    ):
        try:
            unseal(broken, OTHER_PASSWORD, OTHER_AAD)
        except Exception as exc:  # noqa: BLE001 - the messages are the subject
            messages.append(str(exc))
    assert messages
    for message in messages:
        assert MARKER.decode("ascii") not in message
        assert PASSWORD not in message
        assert OTHER_PASSWORD not in message


# ---------------------------------------------------------------------------
# The password itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("password", ["", " ", "\t", "\n", "   \t\n  "])
def test_an_absent_password_is_refused_rather_than_deriving_a_key_from_nothing(password):
    """Argon2id derives a perfectly good 32-byte key from `b""` — verified
    against the pinned library — so without this check an empty password seals
    a real store that opens with no password at all, indistinguishably from one
    that was protected. Strength is the prompting command's business; emptiness
    is not, because "" cannot be told apart from "nothing was supplied".
    """
    with pytest.raises(EmptyPasswordError):
        seal(MARKER, password, AAD)
    envelope = seal(MARKER, PASSWORD, AAD)
    with pytest.raises(EmptyPasswordError):
        unseal(envelope, password, AAD)


def test_an_absent_password_is_refused_as_a_crypto_error():
    """`EmptyPasswordError` must sit under `CryptoError`, or `store.py`'s
    `except CryptoError` boundaries would not see it at all."""
    assert issubclass(EmptyPasswordError, CryptoError)


def test_a_password_is_refused_but_never_trimmed():
    """Whitespace-only is refused; a password with significant surrounding
    whitespace is honoured exactly as typed. Stripping would silently change
    what the user entered into something else."""
    padded = "  " + PASSWORD + "  "
    envelope = seal(MARKER, padded, AAD)
    assert unseal(envelope, padded, AAD) == MARKER
    with pytest.raises(DecryptionError):
        unseal(envelope, PASSWORD, AAD)


def test_a_backend_that_cannot_do_argon2id_fails_closed_as_a_crypto_error(monkeypatch):
    """Argon2id needs an OpenSSL 3.2+ backend; on an older one `cryptography`
    raises `UnsupportedAlgorithm`, which inherits straight from `Exception`.
    Uncaught it would sail past every `except CryptoError` in `store.py` as a
    raw traceback, on the one class of machine where nothing about the store
    works. Simulated, because this build's backend does support it.
    """

    def unsupported(*_args, **_kwargs):
        raise UnsupportedAlgorithm("no Argon2id in this backend")

    monkeypatch.setattr(crypto, "Argon2id", unsupported)
    with pytest.raises(CryptoError):
        seal(MARKER, PASSWORD, AAD)


# ---------------------------------------------------------------------------
# A lone surrogate — reachable once a provider feeds this module a password
# read out of `os.environ`, which Python decodes with `surrogateescape`
# rather than raising. See `EncodingError`.
# ---------------------------------------------------------------------------

# One low surrogate `os.environ` would actually produce for a byte that was
# never valid UTF-8 (`os.fsdecode(b"\x80")`, on a POSIX filesystem encoding) —
# not an arbitrary codepoint picked for the shape alone.
SURROGATE_PASSWORD = "before\udc80after"


def test_a_lone_surrogate_password_fails_as_a_crypto_error_not_a_bare_one():
    """`_normalise`'s `str.encode("utf-8")` is not JSON-escaped the way a
    `seal_json` payload is, so a password containing a lone surrogate must
    raise `EncodingError` (a `CryptoError`), not let Python's own
    `UnicodeEncodeError` escape past every `except CryptoError` boundary in
    `store.py`."""
    with pytest.raises(crypto.EncodingError):
        seal(MARKER, SURROGATE_PASSWORD, AAD)


def test_encoding_error_is_a_crypto_error():
    """`pytest.raises(crypto.EncodingError)` above proves the *specific*
    type; this proves the property those tests actually rely on for
    `store.py`'s `except CryptoError` boundaries to see it at all -- the
    same distinction `test_an_absent_password_is_refused_as_a_crypto_error`
    draws for `EmptyPasswordError`."""
    assert issubclass(crypto.EncodingError, CryptoError)


def test_a_lone_surrogate_password_fails_closed_on_unseal_too():
    """`_derive` — and therefore `_normalise` — sits under `unseal` as well
    as `seal`; the guard must hold on the read path, not only the write
    path a review first flagged this on."""
    envelope = seal(MARKER, PASSWORD, AAD)
    with pytest.raises(crypto.EncodingError):
        unseal(envelope, SURROGATE_PASSWORD, AAD)


def test_a_lone_surrogate_password_without_the_guard_would_raise_a_bare_unicode_error():
    """Proves the failure this guard exists for is real, not hypothetical:
    the exact same encode, done the way `_normalise` used to do it, raises
    Python's own `UnicodeEncodeError` — outside `CryptoError` entirely."""
    import unicodedata

    with pytest.raises(UnicodeEncodeError):
        unicodedata.normalize("NFC", SURROGATE_PASSWORD).encode("utf-8")


def test_seal_json_payload_with_a_lone_surrogate_round_trips_without_raising():
    """The opposite finding, stated as a test: `json.dumps`'s default
    `ensure_ascii=True` escapes a lone surrogate in the *payload* to plain
    ASCII before it is ever encoded, so `seal_json` does not raise on one —
    unlike the password case above. Documented here so a future change to
    `seal_json`'s `ensure_ascii`/`separators` that reintroduced the failure
    would be caught by `test_a_lone_surrogate_payload_value_fails_as_a_crypto_error`
    below, and so this module's own claim about where the bug does and does
    not live is verified, not merely asserted in a docstring."""
    envelope = seal_json({"NAME": SURROGATE_PASSWORD}, PASSWORD, AAD)
    assert unseal_json(envelope, PASSWORD, AAD) == {"NAME": SURROGATE_PASSWORD}


def test_a_lone_surrogate_payload_value_fails_as_a_crypto_error(monkeypatch):
    """`seal_json`'s own defensive guard, exercised directly: forces the
    `ensure_ascii` escaping that normally protects this call to fail, so the
    `except UnicodeEncodeError` inside `seal_json` itself is proven live
    rather than merely present in the source."""

    class _Unencodable(str):
        def encode(self, *_args, **_kwargs):
            raise UnicodeEncodeError("utf-8", self, 0, 1, "simulated")

    def fake_dumps(*_args, **_kwargs):
        return _Unencodable("{}")

    monkeypatch.setattr(crypto.json, "dumps", fake_dumps)
    with pytest.raises(crypto.EncodingError):
        seal_json({"NAME": "value"}, PASSWORD, AAD)


def test_a_lone_surrogate_aad_fails_as_a_crypto_error():
    """The same guard on `aad.encode("utf-8")` in `seal`/`unseal` — no
    current caller passes an AAD sourced from the environment, but the encode
    is identical in shape to the password's, and identically reachable."""
    with pytest.raises(crypto.EncodingError):
        seal(MARKER, PASSWORD, SURROGATE_PASSWORD)
