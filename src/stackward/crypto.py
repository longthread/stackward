"""The sealed envelope: Argon2id + AES-256-GCM, and nothing else.

This module knows how to turn bytes and a password into an envelope and back
again. It does not know what a profile is, where the store lives, or how a
password was obtained — `store.py` owns all of that. The split is deliberate:
the crypto here is the part that is hard to get right and easy to get subtly
wrong, so it is kept small enough to read in one sitting and testable without
touching a filesystem.

**Why Argon2id and not PBKDF2.** This store interoperates with nothing. It is
read only by this tool, so it is free to use the modern memory-hard KDF rather
than a legacy one. `commands/check_passphrase.py` deliberately uses PBKDF2 with
1,000,000 iterations instead — not as an inconsistency, but because that code
must reproduce *Pulumi's* envelope to verify a stack passphrase, which is a
different job with a fixed, externally-imposed format. Do not unify them.

**The envelope.**

    {"v": 1,
     "kdf": {"name": "argon2id", "memory_kib": …, "iterations": …, "lanes": …},
     "salt": "<base64>", "nonce": "<base64>", "ct": "<base64>"}

Two readings of the brief's "fields `v`, `kdf`, `salt`, `nonce`, `ct`, each
base64" are possible. The one taken here base64-encodes the three *byte-valued*
fields — the ones JSON cannot carry natively — and leaves `v` an integer and
`kdf` a structured object. Base64-encoding a version number defeats the purpose
of having one: the first thing a reader of a broken store needs is to see which
format version it is looking at, and `IjEi` does not tell them. `kdf` carries
its parameters rather than only a name so that a future change to the cost
parameters can still open envelopes written today; expressing that as a nested
object rather than as additional top-level fields keeps the envelope to the five
fields the brief names.

**AAD.** Every seal binds an associated-data string, which `store.py` sets to
the profile name. AES-GCM authenticates it without encrypting it, so an envelope
lifted out of one profile's slot and dropped into another's fails to open even
under the correct password. That is a property of the tag, not of a check this
code performs, which is what makes it hard to accidentally remove.

**Passwords are NFC-normalised before encoding.** A password containing a
non-ASCII character can be delivered as either a precomposed or a decomposed
sequence depending on the platform and input method — the same characters,
different UTF-8 bytes, a different derived key, and a store that opens on the
machine it was created on and nowhere else. Normalising costs one line and must
be decided before any store exists: adding it later would break every password
that worked without it.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import unicodedata
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

ENVELOPE_VERSION = 1
KDF_NAME = "argon2id"

# RFC 9106's second recommended option (t=3, m=64 MiB, p=4), which measures at
# roughly 60 ms here — small enough to sit in front of an interactive command,
# large enough to make an offline guessing attack on a stolen file expensive.
# Written into every new envelope; envelopes already on disk are opened with
# whatever parameters they carry.
ARGON2ID_MEMORY_KIB = 65536
ARGON2ID_ITERATIONS = 3
ARGON2ID_LANES = 4

KEY_BYTES = 32  # AES-256
SALT_BYTES = 16  # RFC 9106 §4
NONCE_BYTES = 12  # the GCM nonce length everything agrees on

# Sanity bounds on parameters read back from a file, which is untrusted input
# even when it is the user's own: the file is hand-editable and this code must
# not turn a typo into a crash or a hang.
#
# These are NOT a security control, and describing them as downgrade protection
# would be wrong: changing a parameter in the file changes the derived key, so
# the GCM tag fails regardless, and the exposure this store addresses (a stolen
# disk, backup or tarball) does not include an attacker with write access to a
# live config directory.
#
# What earns them their place is the `iterations` ceiling specifically. An
# absurd `memory_kib` is survivable without a bound — `cryptography` answers it
# with a `MemoryError`, which `_derive` below maps like any other library
# rejection. An absurd `iterations` raises nothing at all: Argon2id simply runs,
# for as long as it is asked to, so a mistyped digit becomes a command that
# never returns and cannot be told apart from a hung tool. There is no exception
# to catch for that one, which is why it has to be refused before the derivation
# starts rather than handled after it.
_MIN_MEMORY_KIB = 8192
_MAX_MEMORY_KIB = 1048576  # 1 GiB
_MIN_ITERATIONS = 1
_MAX_ITERATIONS = 16
_MIN_LANES = 1
_MAX_LANES = 16
_MIN_SALT_BYTES = 8  # `cryptography`'s own floor
_MAX_SALT_BYTES = 1024


class CryptoError(Exception):
    """Base for anything this module refuses to do. Never carries a plaintext,
    a password, or any part of either — callers print these."""


class EnvelopeError(CryptoError):
    """The envelope is not something this module can attempt to open: a missing
    or misshapen field, an unknown version or KDF, a parameter outside the
    sanity bounds, undecodable base64.

    Distinct from `DecryptionError` on purpose. This one means "the input is
    not an envelope"; that one means "it is an envelope and it did not open",
    which is the branch a caller reports as a wrong password."""


class DecryptionError(CryptoError):
    """The GCM tag did not verify: a wrong password, a wrong AAD (an envelope
    moved between profiles), or a modified ciphertext — indistinguishable by
    construction, and deliberately so.

    Always raised `from` the underlying `InvalidTag`, so a test can prove the
    failure came from the tag rather than from a length check or a comparison
    this module performed itself."""


def _normalise(password: str) -> bytes:
    return unicodedata.normalize("NFC", password).encode("utf-8")


def describe_value(value: Any) -> str:
    """Name a value read out of the store without echoing it.

    Envelope fields are hand-editable, and this module's messages are printed.
    A number or `None` in a header field cannot carry a credential, so it is
    shown; anything else is reported by type only. A user who pasted a
    credential into `v` or `kdf.name` by mistake would otherwise have it read
    back at them by the error that caught the mistake — the same rule
    `commands/check_config.describe_yaml_error` follows for the same reason.

    Public (not `_`-prefixed) specifically so `store.py` can reuse it for the
    version field of the credentials document itself, rather than growing a
    second copy of the same judgement that would have to be kept in step
    with this one by hand.
    """
    if value is None or isinstance(value, (int, float)):
        return repr(value)
    return f"<{type(value).__name__}>"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise EnvelopeError(f"envelope field {field!r} must be a base64 string")
    try:
        # `validate=True` so that stray characters are rejected rather than
        # silently discarded, which would let two distinct texts decode to the
        # same bytes and make the envelope's encoding non-canonical.
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EnvelopeError(f"envelope field {field!r} is not valid base64") from exc


def _require_int(data: dict[str, Any], key: str, low: int, high: int) -> int:
    value = data.get(key)
    # `bool` is an `int` subclass; `True` would otherwise read as the integer 1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise EnvelopeError(f"kdf parameter {key!r} must be an integer")
    if not low <= value <= high:
        raise EnvelopeError(f"kdf parameter {key!r} must be between {low} and {high}")
    return value


def _derive(password: str, salt: bytes, params: dict[str, int]) -> bytes:
    """Argon2id, with the parameters the envelope carries.

    Everything reaching here has already passed the sanity bounds above, so a
    failure from `cryptography` means this module's own validation and the
    library's disagree. Map it to `EnvelopeError` rather than letting a raw
    `ValueError`, `OverflowError` or `MemoryError` escape: a caller catching
    `CryptoError` must not have to enumerate the library's exception types to
    stay fail-closed.
    """
    try:
        return Argon2id(
            salt=salt,
            length=KEY_BYTES,
            iterations=params["iterations"],
            lanes=params["lanes"],
            memory_cost=params["memory_kib"],
        ).derive(_normalise(password))
    except (ValueError, OverflowError, MemoryError, TypeError) as exc:
        raise EnvelopeError(
            f"key derivation rejected the envelope's parameters ({type(exc).__name__})"
        ) from exc


def seal(plaintext: bytes, password: str, aad: str) -> dict[str, Any]:
    """Encrypt `plaintext` under `password`, binding `aad` into the tag.

    A fresh salt and a fresh nonce are drawn from `os.urandom` on every call and
    never reused. That is what makes it safe to seal the same plaintext under
    the same password repeatedly — as password rotation does across every
    envelope in the store — without leaking that the plaintexts are equal, and
    it is why nonce reuse (catastrophic for GCM) cannot arise here: no nonce is
    ever derived from anything, so there is no scheme to get wrong.
    """
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    params = {
        "memory_kib": ARGON2ID_MEMORY_KIB,
        "iterations": ARGON2ID_ITERATIONS,
        "lanes": ARGON2ID_LANES,
    }
    key = _derive(password, salt, params)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad.encode("utf-8"))
    return {
        "v": ENVELOPE_VERSION,
        "kdf": {"name": KDF_NAME, **params},
        "salt": _b64(salt),
        "nonce": _b64(nonce),
        "ct": _b64(ciphertext),
    }


def unseal(envelope: Any, password: str, aad: str) -> bytes:
    """Open an envelope, or raise.

    Raises `EnvelopeError` when the input is not a well-formed envelope this
    version understands, and `DecryptionError` when it is but the tag does not
    verify. There is no third outcome: this function never returns bytes it did
    not authenticate, so a wrong password cannot yield garbage for a caller to
    misinterpret as a credential.
    """
    if not isinstance(envelope, dict):
        raise EnvelopeError("envelope must be a JSON object")

    version = envelope.get("v")
    if version != ENVELOPE_VERSION:
        raise EnvelopeError(
            f"unsupported envelope version {describe_value(version)} "
            f"(this build writes and reads {ENVELOPE_VERSION})"
        )

    kdf = envelope.get("kdf")
    if not isinstance(kdf, dict):
        raise EnvelopeError("envelope field 'kdf' must be a JSON object")
    if kdf.get("name") != KDF_NAME:
        raise EnvelopeError(
            f"unsupported kdf {describe_value(kdf.get('name'))} (expected {KDF_NAME!r})"
        )
    params = {
        "memory_kib": _require_int(kdf, "memory_kib", _MIN_MEMORY_KIB, _MAX_MEMORY_KIB),
        "iterations": _require_int(kdf, "iterations", _MIN_ITERATIONS, _MAX_ITERATIONS),
        "lanes": _require_int(kdf, "lanes", _MIN_LANES, _MAX_LANES),
    }

    salt = _unb64(envelope.get("salt"), "salt")
    if not _MIN_SALT_BYTES <= len(salt) <= _MAX_SALT_BYTES:
        raise EnvelopeError(
            f"envelope salt must be between {_MIN_SALT_BYTES} and "
            f"{_MAX_SALT_BYTES} bytes, got {len(salt)}"
        )
    nonce = _unb64(envelope.get("nonce"), "nonce")
    if len(nonce) != NONCE_BYTES:
        raise EnvelopeError(
            f"envelope nonce must be {NONCE_BYTES} bytes, got {len(nonce)}"
        )
    ciphertext = _unb64(envelope.get("ct"), "ct")

    key = _derive(password, salt, params)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad.encode("utf-8"))
    except InvalidTag as exc:
        raise DecryptionError(
            "could not decrypt: wrong password, or the sealed data does not "
            "belong here"
        ) from exc


def seal_json(payload: Any, password: str, aad: str) -> dict[str, Any]:
    """`seal` over a JSON-serialisable value.

    `sort_keys` so that the plaintext depends on the mapping's content and not
    on the order a dict happened to be built in — the length of the ciphertext
    already leaks the size of the payload, and there is no reason to let its
    construction order leak too.
    """
    return seal(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        password,
        aad,
    )


def unseal_json(envelope: Any, password: str, aad: str) -> Any:
    """`unseal`, then parse the authenticated plaintext as JSON.

    Parsing only ever happens on bytes GCM has already authenticated, so the
    JSON parser is never exposed to attacker-chosen input — an ordering that
    matters and is easy to reverse by accident.
    """
    plaintext = unseal(envelope, password, aad)
    try:
        return json.loads(plaintext)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Reachable only for an envelope this tool did not write, since the tag
        # already proved it was sealed under this password and AAD. The message
        # names the failure and nothing about the plaintext.
        raise EnvelopeError(
            f"decrypted content is not valid JSON ({type(exc).__name__})"
        ) from exc
