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
from cryptography.exceptions import InvalidTag

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
    DecryptionError,
    EnvelopeError,
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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("memory_kib", 2**31),  # `cryptography` answers this with a MemoryError
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
