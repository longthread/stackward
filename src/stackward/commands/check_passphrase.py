"""`check-passphrase`: prove a passphrase actually decrypts a stack.

**Why this exists.** `pulumi config` (the list form) renders every secret as
`[secret]` **without decrypting**, and exits 0 under any passphrase at all,
including a flatly wrong one — `set_secrets.py`'s own `_pulumi_config_get`
docstring makes the same point about why drift detection cannot use it
either. There is no `pulumi` subcommand that answers "is this passphrase
right" directly. The only way to actually prove one is to reproduce the
decryption Pulumi itself would perform against the stack's own exported
state, which is what this module does. It is a verifier, not an alternative
convenience wrapper around some other `pulumi` command — if an easier
`pulumi` invocation could answer this question, this file would not exist.

**Where the ciphertext comes from.** `pulumi stack export --stack <name>`
prints the stack's deployment state as JSON. For a stack using Pulumi's
built-in passphrase-based secrets provider (the default for a self-managed,
non-Service backend), that document carries, at
`deployment.secrets_providers.state.salt`, a string of the form

    v1:<base64 salt>:v1:<base64 nonce>:<base64 ciphertext+tag>

which decodes to an 8-byte salt, a 12-byte AES-GCM nonce, and a ciphertext
whose last 16 bytes are the GCM authentication tag. Decrypting that
ciphertext (no associated data) under the key PBKDF2 derives from the
candidate passphrase and that salt yields the literal plaintext `b"pulumi"`
if and only if the passphrase is the one the stack was created with. That
literal string is a canary Pulumi itself writes for exactly this purpose —
this module does not invent the check, it reproduces it.

Every field name and byte length above (`v1:`-prefixed, standard/padded
base64, 8-byte salt, 12-byte nonce, no AAD, PBKDF2-HMAC-SHA256 at 1,000,000
iterations, a 32-byte key, plaintext `b"pulumi"`) was confirmed empirically
against a real `pulumi stack export` from a throwaway local-backend stack
created for this purpose, decrypted successfully with the passphrase that
created it and rejected with a different one — not taken on faith from
memory or from reading Pulumi's source. `tests/test_check_passphrase.py`
freezes that exact salt string as a fixture for the same reason: a round
trip against this module's own encryption would prove only self-consistency,
never that the format matches Pulumi's.

**Why PBKDF2 here and Argon2id in `crypto.py`.** This is deliberate, not an
oversight — see the parameter definitions below for the reason, and
`crypto.py`'s own module docstring, which names this module for the same
reason from the other side. Do not unify them: this module is reproducing a
fixed, externally-imposed format it does not control, and has no freedom to
pick a stronger KDF even though one exists elsewhere in this codebase.

**Fail-closed, not three-valued-but-secretly-two.** A stack is reported as
exactly one of `accepted`, `rejected`, or an error — never conflated. A
malformed salt string, a `pulumi stack export` that fails or times out, a
document missing `deployment`, `secrets_providers`, `state`, or `salt` (for
instance because the stack uses a cloud KMS-based secrets provider instead
of a passphrase, which carries no `salt` field at all), or output that is
not valid JSON are all errors — this module cannot tell whether the
passphrase would have worked, and reporting "rejected" for any of them would
be a false negative wearing the shape of a real answer. This is what "fail
closed" concretely means for this command, not merely a slogan: every one of
those conditions raises `CheckPassphraseError`, which the per-stack loop in
`cmd_check_passphrase` reports as `error`, never as `accepted` or `rejected`.

**The passphrase is never a command-line argument.** It is read as one line
from `sys.stdin` by default — the expected shape is a pipe, e.g. a value
drawn from a password manager or a candidate list — falling back to an
echo-free `getpass.getpass` prompt when `sys.stdin` is a terminal (nothing
piped in, a human at a keyboard). This is the mirror image of
`session.py`'s `_read_store_password`, not a reuse of its exact mechanism,
and deliberately so: `session.py` must never read `sys.stdin` because that
descriptor belongs to the child process it is about to run (`pulumi up`,
etc.) and reading even one byte of it here would be consumed before the
child ever saw it. This command spawns no interactive child — `pulumi stack
export` never reads from its own stdin — so `sys.stdin` is this command's to
use, and checking `sys.stdin.isatty()` directly (rather than `session.py`'s
`/dev/tty`-opened-independently check) is the correct question to ask here:
it is the exact fallback condition the task names, and it decides "was
something piped in" using the same descriptor that will actually be read.

**The fingerprint identifies a candidate, not the store.** It is the first
16 hex characters of SHA-256 over the exact UTF-8 bytes handed to PBKDF2 —
not NFC-normalised, unlike `crypto.py`'s stored-password handling, because
there is nothing here to keep byte-for-byte reproducible across sessions;
the fingerprint only has to agree with what was actually derived *this run*,
so it is computed from the identical bytes rather than a canonicalised
form that could disagree with them.

**Exit codes are 0 or 2, never 1** — `check-config`'s docstring and
`set_secrets.py`'s both reserve exit code 1 exclusively for "a credential
was found," a claim this command does not make. 0 means every named stack
accepted; 2 means at least one did not (rejected, or could not be checked).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import getpass
import hashlib
import json
import shutil
import subprocess
import sys
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .check_config import fail_closed

# Pulumi's own KDF, reproduced exactly -- NOT `crypto.py`'s Argon2id.
#
# This is not an inconsistency to "fix": `crypto.py`'s store interoperates
# with nothing, so it is free to use the modern memory-hard KDF. This module
# is reproducing *Pulumi's* envelope format in order to verify a stack's own
# ciphertext, which is a fixed, externally-imposed scheme this module has no
# freedom to change -- PBKDF2-HMAC-SHA256 at 1,000,000 iterations, deriving
# a 32-byte AES-256 key, confirmed against a real `pulumi stack export` (see
# the module docstring). Changing either number here would make this module
# derive a key Pulumi itself does not, and every real stack would reject
# even its own correct passphrase.
PBKDF2_ITERATIONS = 1_000_000
KEY_BYTES = 32

# Byte lengths Pulumi's own encoding fixes, confirmed against a real export
# (see the module docstring) rather than assumed:
#   - the salt is `make([]byte, 8)` on the Go side;
#   - the nonce is AES-GCM's own standard 12-byte size;
#   - the ciphertext carries a 16-byte GCM tag appended, so it can never be
#     shorter than that.
# These are enforced explicitly, before any decryption is attempted, because
# `cryptography`'s `AESGCM.decrypt` raises `InvalidTag` -- not a `ValueError`
# -- for a too-short ciphertext or for a nonce of an unexpected-but-still-
# valid length. Without these checks a malformed salt string would be
# misreported as "rejected" (a wrong passphrase) instead of the error it
# actually is, which is exactly what fail-closed forbids: see the module
# docstring's "Fail-closed" section.
_SALT_BYTES = 8
_NONCE_BYTES = 12
_MIN_CIPHERTEXT_BYTES = 16

# The literal plaintext Pulumi encrypts as its own passphrase canary.
PULUMI_TEST_PLAINTEXT = b"pulumi"

# How long a single `pulumi stack export` is allowed to run before this
# module gives up on that one stack and reports it as an error -- a module
# level constant (not a function default) so a test can lower it with
# `monkeypatch.setattr`, matching `set_secrets.DEFAULT_TIMEOUT_SECONDS`.
DEFAULT_TIMEOUT_SECONDS = 30.0


class CheckPassphraseError(Exception):
    """A stack could not be checked at all: a malformed salt string, a
    `pulumi stack export` that failed, timed out, or returned something
    that is not parseable JSON, or an exported document missing
    `deployment`, `secrets_providers`, `state`, or `salt`.

    Never means "the passphrase was wrong" -- that is `check_salt` returning
    `False`, a distinct outcome the per-stack loop never conflates with this
    one. Never carries the passphrase or any part of it: every message here
    names a stack, a byte length, a `pulumi` exit code, or a JSON field --
    never a credential value.
    """


def fingerprint(passphrase: str) -> str:
    """First 16 hex characters of SHA-256 over `passphrase`'s UTF-8 bytes --
    the same bytes `check_salt` hands to PBKDF2, not an NFC-normalised or
    otherwise adjusted form (contrast `crypto.py`'s stored passwords, which
    are normalised because they must stay reproducible across sessions on
    different platforms; this value only has to match what was actually
    derived in this one run).

    Lets two candidate passphrases be told apart in a log safely: the
    fingerprint reveals nothing about the passphrase itself (SHA-256 is not
    invertible, and this is a truncated fingerprint of a hash, never the
    hash used as a key or a proof of knowledge), but two different
    passphrases fingerprint differently with overwhelming probability, and
    the same passphrase always fingerprints the same way.
    """
    return hashlib.sha256(passphrase.encode("utf-8")).hexdigest()[:16]


def _derive_key(passphrase: bytes, salt: bytes) -> bytes:
    """PBKDF2-HMAC-SHA256, Pulumi's own parameters -- see `PBKDF2_ITERATIONS`
    for why these numbers and not `crypto.py`'s Argon2id ones."""
    return PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    ).derive(passphrase)


def _parse_salt(salt_field: str) -> tuple[bytes, bytes, bytes]:
    """Decode `v1:<b64 salt>:v1:<b64 nonce>:<b64 ciphertext>` into
    `(salt, nonce, ciphertext)`, or raise `CheckPassphraseError` naming what
    is wrong -- never a raw `ValueError` or `binascii.Error`, and never a
    guess at bytes that were not actually there.

    Base64 never contains `:`, so a plain `str.split(":")` producing exactly
    five parts is a safe, unambiguous parse of this format -- it is also
    exactly how Pulumi's own two-part construction (`v1:<salt>` joined with
    `v1:<nonce>:<ciphertext>`) round-trips through one string.

    The byte-length checks after decoding are not optional hardening: see
    `_SALT_BYTES`'s comment for why skipping them would let a malformed
    salt masquerade as a wrong passphrase instead of the error it is.
    """
    parts = salt_field.split(":")
    if len(parts) != 5 or parts[0] != "v1" or parts[2] != "v1":
        raise CheckPassphraseError(
            "malformed secrets provider salt (expected "
            "'v1:<salt>:v1:<nonce>:<ciphertext>')"
        )
    try:
        salt = base64.b64decode(parts[1], validate=True)
        nonce = base64.b64decode(parts[3], validate=True)
        ciphertext = base64.b64decode(parts[4], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CheckPassphraseError(
            "malformed secrets provider salt: invalid base64"
        ) from exc

    if len(salt) != _SALT_BYTES:
        raise CheckPassphraseError(
            f"malformed secrets provider salt: expected a {_SALT_BYTES}-byte "
            f"salt, got {len(salt)}"
        )
    if len(nonce) != _NONCE_BYTES:
        raise CheckPassphraseError(
            f"malformed secrets provider salt: expected a {_NONCE_BYTES}-byte "
            f"nonce, got {len(nonce)}"
        )
    if len(ciphertext) < _MIN_CIPHERTEXT_BYTES:
        raise CheckPassphraseError(
            "malformed secrets provider salt: ciphertext shorter than the "
            f"{_MIN_CIPHERTEXT_BYTES}-byte GCM tag ({len(ciphertext)} bytes)"
        )
    return salt, nonce, ciphertext


def check_salt(salt_field: str, passphrase: str) -> bool:
    """`True` if `passphrase` opens `salt_field` (Pulumi's own encoded
    secrets-provider state), `False` if it is well-formed but a *different*
    passphrase opened it.

    Raises `CheckPassphraseError` when `salt_field` itself is not something
    this function can attempt to open at all -- never returns `False` for
    that case, which is what would make a malformed export indistinguishable
    from a genuinely wrong passphrase in the per-stack report.
    """
    if not isinstance(salt_field, str):
        raise CheckPassphraseError("secrets provider salt must be a string")
    salt, nonce, ciphertext = _parse_salt(salt_field)
    key = _derive_key(passphrase.encode("utf-8"), salt)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except InvalidTag:
        return False
    return plaintext == PULUMI_TEST_PLAINTEXT


def _read_passphrase() -> str:
    """The candidate passphrase: one line from `sys.stdin` by default, or an
    echo-free `getpass` prompt when `sys.stdin` is a terminal.

    See the module docstring's "The passphrase is never a command-line
    argument" section for why the branch condition is `sys.stdin.isatty()`
    itself, rather than `session.py`'s independent `/dev/tty`-opened check:
    this command, unlike `session.py`, is free to read `sys.stdin`, so the
    literal question -- was something piped into *this* descriptor -- is the
    right one to ask.

    Raises `CheckPassphraseError` when stdin is not a terminal and reaches
    EOF with nothing on it at all, rather than silently proceeding with an
    empty string a caller almost certainly did not intend. An explicit empty
    line (as opposed to no line at all) is not treated the same way: `""` is
    a passphrase a caller can deliberately test, however unlikely to open
    anything.
    """
    if sys.stdin.isatty():
        return getpass.getpass("passphrase to test: ")
    line = sys.stdin.readline()
    if not line:
        raise CheckPassphraseError("no passphrase provided on stdin")
    return line.rstrip("\r\n")


def _pulumi_stack_export(pulumi: str, stack: str, *, timeout: float) -> dict[str, Any]:
    """`pulumi stack export --stack <stack>`, parsed as a JSON object.

    `stdin=subprocess.DEVNULL`: this process has already read the
    passphrase from its own `sys.stdin` (or nothing, on the `getpass`
    branch), and `pulumi` must never inherit either any leftover piped bytes
    meant for this process, or -- on the interactive branch -- this
    process's controlling terminal, which would let a `pulumi` that decided
    to prompt for something hang against a human who is not expecting it.

    Output is captured and parsed as bytes, not decoded via `text=True`:
    `json.loads` accepts bytes directly, and doing it this way turns
    undecodable output into this function's own `CheckPassphraseError`
    rather than a `UnicodeDecodeError` that `fail_closed` would report with
    a generic message instead of a stack-specific one.

    Raises `CheckPassphraseError` for a timeout, a missing `pulumi` binary,
    a non-zero exit, or output that does not parse as a JSON object --
    never includes `pulumi`'s own stdout or stderr in the message, which is
    not this module's data to vouch for as free of anything sensitive.
    """
    try:
        completed = subprocess.run(
            [pulumi, "stack", "export", "--stack", stack],
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            capture_output=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise CheckPassphraseError("pulumi stack export timed out") from exc
    except OSError as exc:
        raise CheckPassphraseError(f"cannot run pulumi: {exc}") from exc

    if completed.returncode != 0:
        raise CheckPassphraseError(
            f"pulumi stack export exited {completed.returncode}"
        )
    try:
        document = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckPassphraseError(
            "pulumi stack export did not return valid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise CheckPassphraseError("exported stack state is not a JSON object")
    return document


def _extract_salt(document: dict[str, Any]) -> str:
    """`deployment.secrets_providers.state.salt` out of a parsed export, or
    `CheckPassphraseError` naming exactly which level is missing or the
    wrong shape.

    A stack using a non-passphrase secrets provider (a cloud KMS URL, or the
    Pulumi Service's own managed encryption) has no `salt` field at all --
    see the module docstring's "Fail-closed" section for why that is an
    error for this command, on the same footing as any other malformed or
    incomplete export, and never a silent "rejected" or "accepted".
    """
    deployment = document.get("deployment")
    if not isinstance(deployment, dict):
        raise CheckPassphraseError("exported state has no 'deployment' object")
    providers = deployment.get("secrets_providers")
    if not isinstance(providers, dict):
        raise CheckPassphraseError(
            "exported state has no 'secrets_providers' object -- this stack "
            "may not use a passphrase-based secrets provider"
        )
    state = providers.get("state")
    if not isinstance(state, dict):
        raise CheckPassphraseError(
            "exported state has no 'secrets_providers.state' object"
        )
    salt = state.get("salt")
    if not isinstance(salt, str):
        raise CheckPassphraseError(
            "exported state has no 'secrets_providers.state.salt' string"
        )
    return salt


def _check_stack(pulumi: str | None, stack: str, passphrase: str, *, timeout: float) -> bool:
    """`True`/`False` for `stack`, or `CheckPassphraseError` naming why it
    could not be checked at all -- a missing `pulumi` binary included, on
    the same footing as every other could-not-run condition this function
    can raise."""
    if pulumi is None:
        raise CheckPassphraseError("pulumi executable not found on PATH")
    document = _pulumi_stack_export(pulumi, stack, timeout=timeout)
    salt_field = _extract_salt(document)
    return check_salt(salt_field, passphrase)


@fail_closed
def cmd_check_passphrase(args: argparse.Namespace) -> int:
    """Entry point for `stackward check-passphrase STACK...`.

    Prints the passphrase's fingerprint once, then one `accepted` /
    `rejected` / `error: <reason>` line per named stack -- a could-not-check
    stack does not abort the run, so a real answer on one stack is never
    hidden behind a problem on another (the same reasoning
    `cmd_check_config` applies to its own per-file loop).

    Exit 0 only if every named stack accepted; 2 otherwise, whether that is
    because a stack rejected the passphrase or because a stack could not be
    checked at all -- this command reserves exit 1 for nothing, since
    `check-config` and `set_secrets` both already reserve it exclusively
    for "a credential was found," a claim this command never makes.
    """
    try:
        passphrase = _read_passphrase()
    except CheckPassphraseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"passphrase fingerprint: {fingerprint(passphrase)}")

    pulumi = shutil.which("pulumi")
    all_accepted = True
    for stack in args.stacks:
        try:
            accepted = _check_stack(
                pulumi, stack, passphrase, timeout=DEFAULT_TIMEOUT_SECONDS
            )
        except CheckPassphraseError as exc:
            print(f"{stack}: error: {exc}", file=sys.stderr)
            all_accepted = False
            continue
        except Exception as exc:  # noqa: BLE001 - fail closed on anything unanticipated
            print(
                f"{stack}: error: could not check ({type(exc).__name__})",
                file=sys.stderr,
            )
            all_accepted = False
            continue
        print(f"{stack}: {'accepted' if accepted else 'rejected'}")
        if not accepted:
            all_accepted = False

    return 0 if all_accepted else 2
