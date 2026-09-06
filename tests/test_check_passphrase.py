"""Tests for `check-passphrase`.

**The interoperability proof is a frozen, real `pulumi`-generated vector, not
a round trip against this module's own code.** `REAL_PULUMI_SALT_FIELD`
below was captured once from an actual `pulumi stack export`, against a
throwaway local (`file://`) backend created solely to produce it -- a
placeholder project, stack, and passphrase, per GC4, none of it real. A test
that only encrypted and then decrypted with this module's own functions
would prove nothing beyond internal self-consistency: if this module's
understanding of the byte layout, the base64 variant, or the KDF parameters
were wrong in some way that happened to be wrong *consistently*, such a test
would still pass. `test_check_salt_accepts_a_real_pulumi_generated_vector`
below cannot pass that way -- it can only pass if this module's decryption
genuinely matches what Pulumi itself produced.

A second, independently-written PBKDF2+AES-GCM helper
(`_encrypt_pulumi_style`) is used for the malformed-input and
wrong-passphrase fixtures a fixed real vector cannot conveniently produce on
demand -- it calls `cryptography` primitives directly and never reuses this
module's own `_derive_key`/`check_salt`, so it does not share a bug with the
code it is testing.

**No test prints the passphrase, at the Python level or the real child
process's own stdout/stderr.** Assertions about absence are made against
`capsys`-captured output for pure-Python paths and `capfd`-captured output
(file-descriptor level, covering the real stub `pulumi` child process's own
writes) for anything that goes through `main()`.

**Several tests exist to fail against a plausible wrong implementation, not
merely to exercise a right one** -- most importantly the too-short-ciphertext
and wrong-length-nonce tests: `cryptography`'s `AESGCM.decrypt` raises
`InvalidTag`, not `ValueError`, for both, so an implementation that skips the
explicit byte-length checks would silently report "rejected" instead of the
error this is required to be. Each is written to fail on such an
implementation and to pass only once the explicit checks are present.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes

from leakcheck import assert_no_leak
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from stackward.cli import main
from stackward.commands import check_passphrase
from stackward.commands.check_passphrase import (
    CheckPassphraseError,
    check_salt,
    fingerprint,
    _extract_salt,
    _read_passphrase,
)

# Placeholder values only -- see GC4. Distinctive enough that an accidental
# substring match elsewhere in captured output would be implausible, and
# deliberately free of any word this command itself prints: the leak checks
# below look for any eight-character run of these values, and a placeholder
# spelled `marker-passphrase-...` collides with the command's own
# `passphrase fingerprint:` line on the run `passphra`. A placeholder that
# embeds the tool's vocabulary makes a partial-disclosure check unusable,
# which is how such a check ends up deleted.
MARKER_PASSPHRASE = "marker-candidate-7c8d9e0f"
WRONG_PASSPHRASE = "decoy-input-1a2b3c4d5e6f"

# ---------------------------------------------------------------------------
# A real vector, captured once from an actual `pulumi stack export` against
# a throwaway local backend -- see the module docstring. Reproduced here:
#
#   $ pulumi login file:///tmp/scratch-backend
#   $ pulumi stack init placeholder-stack --non-interactive
#   $ PULUMI_CONFIG_PASSPHRASE=placeholder-passphrase-marker \
#       pulumi config set --secret placeholder.key placeholder-value
#   $ pulumi stack export
#   -> deployment.secrets_providers.state.salt
# ---------------------------------------------------------------------------
REAL_PULUMI_SALT_FIELD = "v1:qs/zMEaMbWE=:v1:iMv8Ne7bNxnv6rkP:vGQfE+/D8kOTOH8eT1DTPj68sddAoA=="
REAL_PULUMI_PASSPHRASE = "placeholder-passphrase-marker"


def assert_no_disclosure(text: str, value: str, *, what: str) -> None:
    """`assert_no_leak`, plus the one case it cannot see on its own here: a
    **truncated** re-encoding of the value.

    `assert_no_leak` checks each whole encoding (so `value.hex()` in full is
    caught) and every eight-character run of the *raw* value. This command
    prints a fixed-length value derived from the passphrase, so the shape
    that matters is neither: `passphrase.encode("utf-8").hex()[:16]` is a
    16-character *prefix* of the hex encoding, so the whole-encoding check
    never matches it, and it shares no raw run with the passphrase because
    it is hex. Verified rather than assumed -- that exact defective
    implementation passes a bare `assert_no_leak(fingerprint(p), p)`.

    Handing the hex encoding in as the *value* is what closes it: the
    run-based half of the check then operates on hex runs, and a truncated
    hex encoding of any length above eight characters is caught.
    """
    assert_no_leak(text, value, what=what)
    assert_no_leak(text, value.encode("utf-8").hex(), what=f"{what}, hex-encoded")


def _encrypt_pulumi_style(passphrase: str, plaintext: bytes = b"pulumi") -> str:
    """Build a `v1:<salt>:v1:<nonce>:<ciphertext>` string the same way
    Pulumi's own passphrase secrets provider does -- see the module
    docstring for why this is written independently of
    `check_passphrase._derive_key`/`check_salt` rather than reusing them.
    """
    salt = os.urandom(8)
    nonce = os.urandom(12)
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=1_000_000
    ).derive(passphrase.encode("utf-8"))
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")  # noqa: E731
    return f"v1:{b64(salt)}:v1:{b64(nonce)}:{b64(ciphertext)}"


def _with_field(salt_field: str, index: int, raw_bytes: bytes) -> str:
    """`salt_field` with the base64 field at `index` (1=salt, 3=nonce,
    4=ciphertext) replaced by `raw_bytes`, re-encoded -- for building
    deliberately malformed variants of an otherwise well-formed vector."""
    parts = salt_field.split(":")
    parts[index] = base64.b64encode(raw_bytes).decode("ascii")
    return ":".join(parts)


# ---------------------------------------------------------------------------
# The interoperability proof.
# ---------------------------------------------------------------------------


def test_check_salt_accepts_a_real_pulumi_generated_vector():
    """The one test that cannot pass on self-consistency alone -- see the
    module docstring."""
    assert check_salt(REAL_PULUMI_SALT_FIELD, REAL_PULUMI_PASSPHRASE) is True


def test_check_salt_rejects_a_wrong_passphrase_against_the_real_vector():
    assert check_salt(REAL_PULUMI_SALT_FIELD, "not the right passphrase") is False


# ---------------------------------------------------------------------------
# The synthesised-salt proof the brief names explicitly, independent of
# whether a real `pulumi` binary is available to generate one.
# ---------------------------------------------------------------------------


def test_check_salt_accepts_a_synthesised_salt_the_known_passphrase_opens():
    salt_field = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    assert check_salt(salt_field, MARKER_PASSPHRASE) is True


def test_check_salt_rejects_a_wrong_passphrase_against_a_synthesised_salt():
    salt_field = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    assert check_salt(salt_field, WRONG_PASSPHRASE) is False


# ---------------------------------------------------------------------------
# Fail-closed: malformed input is an error, never "rejected" or "accepted".
# ---------------------------------------------------------------------------


def test_a_salt_field_missing_the_v1_prefix_errors():
    with pytest.raises(CheckPassphraseError):
        check_salt("bogus:AAAA:v1:BBBB:CCCC", MARKER_PASSPHRASE)


def test_a_salt_field_with_the_wrong_number_of_parts_errors():
    with pytest.raises(CheckPassphraseError):
        check_salt("v1:AAAA:v1:BBBB", MARKER_PASSPHRASE)


def test_a_salt_field_with_invalid_base64_errors():
    with pytest.raises(CheckPassphraseError):
        check_salt("v1:not-valid-base64!!!:v1:BBBB:CCCC", MARKER_PASSPHRASE)


def test_a_non_string_salt_field_errors_rather_than_crashing():
    with pytest.raises(CheckPassphraseError):
        check_salt(12345, MARKER_PASSPHRASE)  # type: ignore[arg-type]


def test_a_wrong_length_salt_errors_rather_than_being_silently_derived_from():
    base = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    mutated = _with_field(base, 1, b"\x00" * 4)  # real salt is 8 bytes
    with pytest.raises(CheckPassphraseError):
        check_salt(mutated, MARKER_PASSPHRASE)


def test_a_wrong_length_nonce_errors_rather_than_being_reported_as_rejected():
    """`AESGCM.decrypt` tolerates a 16-byte nonce without raising -- it just
    fails the tag -- so this can only pass if `check_salt` validates the
    nonce length itself before ever calling `AESGCM.decrypt`. An
    implementation missing that check would report this case as "rejected"
    (a wrong passphrase) instead of the malformed-input error it actually
    is, which fail-closed forbids."""
    base = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    mutated = _with_field(base, 3, b"\x00" * 16)  # real nonce is 12 bytes
    with pytest.raises(CheckPassphraseError):
        check_salt(mutated, MARKER_PASSPHRASE)


def test_a_ciphertext_shorter_than_the_gcm_tag_errors_rather_than_being_reported_as_rejected():
    """`AESGCM.decrypt` raises `InvalidTag` -- not a distinguishable error --
    for a ciphertext too short to even hold a tag, so the length must be
    validated before decryption is attempted at all. Without that check this
    case is indistinguishable from a wrong passphrase and would be reported
    as "rejected" instead of the malformed-input error it actually is."""
    base = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    mutated = _with_field(base, 4, b"\x00" * 5)  # shorter than the 16-byte tag
    with pytest.raises(CheckPassphraseError):
        check_salt(mutated, MARKER_PASSPHRASE)


# ---------------------------------------------------------------------------
# The fingerprint.
# ---------------------------------------------------------------------------


# SHA-256 of the exact UTF-8 bytes, truncated to 16 hex characters, written
# out as literals. Computed once with `hashlib` at a Python prompt and pasted;
# deliberately not recomputed inside the test from `hashlib.sha256(...)`, which
# would only prove this file and `check_passphrase` agree on how to spell the
# same call rather than pinning the digest to a known value.
FINGERPRINT_OF_X = "2d711642b726b044"
FINGERPRINT_OF_MARKER = "7e742b5d41ffb360"


def test_fingerprint_is_the_truncated_sha256_of_the_passphrase_bytes():
    """The digest itself, pinned against known values.

    Nothing pinned it before. "16 hex characters", "stable", "differs for
    different inputs" and "does not contain the passphrase" are jointly
    satisfied by `passphrase.encode("utf-8").hex()[:16]` -- which is not a
    digest at all, but the passphrase's own first eight bytes, and which
    `cmd_check_passphrase` prints to stdout on every run. Every one of those
    four tests passed against it. This is the assertion that does not.
    """
    assert fingerprint("x") == FINGERPRINT_OF_X
    assert fingerprint(MARKER_PASSPHRASE) == FINGERPRINT_OF_MARKER


def test_fingerprint_is_16_hex_characters():
    fp = fingerprint(MARKER_PASSPHRASE)
    assert len(fp) == 16
    int(fp, 16)  # raises ValueError if it is not hex


def test_fingerprint_is_over_the_exact_bytes_and_not_a_normalised_form():
    """The module's stated rule -- the fingerprint covers the identical bytes
    handed to PBKDF2, unlike `crypto.py`'s stored passwords, which are NFC
    normalised because they must reproduce across sessions. Two spellings of
    the same grapheme derive two different keys, so they must fingerprint
    differently; a fingerprint that normalised would report one candidate
    for two that behave differently, which is the one thing it exists to
    prevent."""
    composed = "caf\u00e9"  # e-acute as a single code point
    decomposed = "cafe\u0301"  # 'e' plus a combining acute
    assert composed != decomposed
    assert fingerprint(composed) != fingerprint(decomposed)


def test_fingerprint_is_stable_for_the_same_passphrase():
    assert fingerprint(MARKER_PASSPHRASE) == fingerprint(MARKER_PASSPHRASE)


def test_fingerprint_differs_for_different_passphrases():
    """The whole point of a fingerprint is telling two candidates apart --
    a constant or a collision-prone implementation would defeat that even
    while satisfying "16 hex characters" and "stable"."""
    assert fingerprint(MARKER_PASSPHRASE) != fingerprint(WRONG_PASSPHRASE)


def test_fingerprint_never_discloses_the_passphrase_whole_or_in_part():
    """`MARKER_PASSPHRASE not in fingerprint(...)` was a tautology: the
    fingerprint is pinned to 16 characters by the test above and the
    passphrase is 26, so no implementation could ever fail it.
    `assert_no_leak` is what actually discriminates -- it checks the hex
    encoding and every eight-character run, which is exactly the shape a
    `passphrase.encode("utf-8").hex()[:16]` "fingerprint" takes."""
    assert_no_disclosure(
        fingerprint(MARKER_PASSPHRASE), MARKER_PASSPHRASE, what="the passphrase"
    )


# ---------------------------------------------------------------------------
# Reading the passphrase -- stdin by default, `getpass` when stdin is a tty,
# never a command-line argument (there is no code path that could accept
# one -- `build_parser` gives `check-passphrase` no such option, and this
# section proves `_read_passphrase` never touches anything but `sys.stdin`
# or `getpass.getpass`).
# ---------------------------------------------------------------------------


class _FakeStdin:
    """A minimal stand-in for `sys.stdin`: `isatty()` reports a fixed value,
    and `readline()` returns each of `lines` in turn, then `""` (EOF)."""

    def __init__(self, isatty: bool, lines: list[str] | None = None):
        self._isatty = isatty
        self._lines = list(lines or [])

    def isatty(self) -> bool:
        return self._isatty

    def readline(self, *a, **k) -> str:
        return self._lines.pop(0) if self._lines else ""


class _PoisonedReadline:
    """`isatty()` reports a real terminal; `readline()` raises if called --
    proves `_read_passphrase` never reads `sys.stdin` on the tty branch."""

    def isatty(self) -> bool:
        return True

    def readline(self, *a, **k):
        raise AssertionError("_read_passphrase touched stdin on the tty branch")


def test_read_passphrase_reads_a_line_from_piped_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(isatty=False, lines=[f"{MARKER_PASSPHRASE}\n"]))

    def boom(*a, **k):
        raise AssertionError("getpass.getpass must not run when stdin is not a tty")

    monkeypatch.setattr(check_passphrase.getpass, "getpass", boom)
    assert _read_passphrase() == MARKER_PASSPHRASE


def test_read_passphrase_strips_only_the_line_terminator(monkeypatch):
    """Trailing whitespace that is not the line terminator is preserved --
    trimming it would silently change what the caller actually piped in."""
    monkeypatch.setattr(
        sys, "stdin", _FakeStdin(isatty=False, lines=[f"{MARKER_PASSPHRASE} \r\n"])
    )
    assert _read_passphrase() == f"{MARKER_PASSPHRASE} "


def test_read_passphrase_raises_on_empty_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(isatty=False, lines=[]))
    with pytest.raises(CheckPassphraseError):
        _read_passphrase()


def test_read_passphrase_falls_back_to_getpass_when_stdin_is_a_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _PoisonedReadline())
    monkeypatch.setattr(check_passphrase.getpass, "getpass", lambda prompt="": MARKER_PASSPHRASE)
    assert _read_passphrase() == MARKER_PASSPHRASE


def test_read_passphrase_never_reads_stdin_on_the_tty_branch(monkeypatch):
    """`_PoisonedReadline` itself asserts this, but this test names the
    property directly: the interactive branch must go through `getpass`
    only, with no fallback read of `sys.stdin`."""
    monkeypatch.setattr(sys, "stdin", _PoisonedReadline())
    monkeypatch.setattr(check_passphrase.getpass, "getpass", lambda prompt="": "typed")
    assert _read_passphrase() == "typed"  # would have raised via readline() otherwise


# ---------------------------------------------------------------------------
# `_extract_salt`: the pure, subprocess-free half of reading an export.
# ---------------------------------------------------------------------------


def test_extract_salt_reads_the_nested_field():
    document = {
        "deployment": {"secrets_providers": {"state": {"salt": REAL_PULUMI_SALT_FIELD}}}
    }
    assert _extract_salt(document) == REAL_PULUMI_SALT_FIELD


def test_extract_salt_errors_when_deployment_is_missing():
    with pytest.raises(CheckPassphraseError):
        _extract_salt({})


def test_extract_salt_errors_when_secrets_providers_is_missing():
    """The case of a stack using a non-passphrase secrets provider (a cloud
    KMS URL, or the Pulumi Service's own managed encryption): no `salt`
    field exists to check at all, which fail-closed treats as an error for
    this stack, never a silent "rejected" or "accepted" -- see the module
    docstring."""
    with pytest.raises(CheckPassphraseError):
        _extract_salt({"deployment": {}})


def test_extract_salt_errors_when_state_is_missing():
    with pytest.raises(CheckPassphraseError):
        _extract_salt({"deployment": {"secrets_providers": {}}})


def test_extract_salt_errors_when_salt_is_missing():
    with pytest.raises(CheckPassphraseError):
        _extract_salt({"deployment": {"secrets_providers": {"state": {}}}})


def test_extract_salt_errors_when_salt_is_not_a_string():
    with pytest.raises(CheckPassphraseError):
        _extract_salt(
            {"deployment": {"secrets_providers": {"state": {"salt": 12345}}}}
        )


# ---------------------------------------------------------------------------
# End-to-end, through `main()`, against a real stub `pulumi` process --
# never a mocked `subprocess.run` -- so a leak through the real child's own
# stdout/stderr, or through its actual `argv`, would be caught the same way
# `test_set_secrets.py`'s stdin/argv test is.
# ---------------------------------------------------------------------------

_PULUMI_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys, time

    argv = sys.argv[1:]

    dump_argv_to = os.environ.get("STUB_DUMP_ARGV_TO")
    if dump_argv_to:
        with open(dump_argv_to, "a") as f:
            f.write(json.dumps(argv) + "\\n")


    def arg_after(flag):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return None


    stack = arg_after("--stack")

    slow_stack = os.environ.get("STUB_SLOW_STACK")
    if slow_stack and stack == slow_stack:
        time.sleep(float(os.environ.get("STUB_SLEEP_SECONDS", "5")))

    exit_codes = json.loads(os.environ.get("STUB_EXIT_CODES", "{}"))
    if stack in exit_codes:
        sys.stderr.write(os.environ.get("STUB_ECHO", ""))
        sys.exit(int(exit_codes[stack]))

    exports = json.loads(os.environ.get("STUB_EXPORTS", "{}"))
    sys.stdout.write(exports.get(stack, os.environ.get("STUB_DEFAULT_EXPORT", "{}")))
    sys.exit(0)
    """
)


@pytest.fixture
def stub_pulumi(tmp_path) -> Path:
    path = tmp_path / "pulumi-stub.py"
    path.write_text(_PULUMI_STUB)
    path.chmod(0o755)
    return path


def fake_pulumi(monkeypatch, executable: Path | None) -> None:
    monkeypatch.setattr(
        check_passphrase.shutil,
        "which",
        lambda name: str(executable) if (executable and name == "pulumi") else None,
    )


def feed_stdin(monkeypatch, passphrase: str) -> None:
    monkeypatch.setattr(
        check_passphrase.sys, "stdin", _FakeStdin(isatty=False, lines=[f"{passphrase}\n"])
    )


@pytest.fixture(autouse=True)
def fast_timeout(monkeypatch):
    """The real default (30s) would make the timeout test below take thirty
    real seconds for no benefit."""
    monkeypatch.setattr(check_passphrase, "DEFAULT_TIMEOUT_SECONDS", 0.3)


def _export_with_salt(salt_field: str) -> str:
    return json.dumps(
        {"deployment": {"secrets_providers": {"type": "passphrase", "state": {"salt": salt_field}}}}
    )


def test_a_correct_passphrase_is_accepted_end_to_end(stub_pulumi, monkeypatch, capsys):
    salt_field = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    monkeypatch.setenv("STUB_EXPORTS", json.dumps({"stack-a": _export_with_salt(salt_field)}))
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-a"]) == 0
    out = capsys.readouterr().out
    assert "stack-a: accepted" in out
    assert "passphrase fingerprint:" in out


def test_a_wrong_passphrase_is_rejected_and_exits_non_zero(stub_pulumi, monkeypatch, capsys):
    salt_field = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    monkeypatch.setenv("STUB_EXPORTS", json.dumps({"stack-a": _export_with_salt(salt_field)}))
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, WRONG_PASSPHRASE)

    assert main(["check-passphrase", "stack-a"]) == 2
    assert "stack-a: rejected" in capsys.readouterr().out


def test_every_named_stack_must_accept_for_exit_zero(stub_pulumi, monkeypatch, capsys):
    good = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    bad = _encrypt_pulumi_style(WRONG_PASSPHRASE)
    monkeypatch.setenv(
        "STUB_EXPORTS",
        json.dumps(
            {"stack-a": _export_with_salt(good), "stack-b": _export_with_salt(bad)}
        ),
    )
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-a", "stack-b"]) == 2
    out = capsys.readouterr().out
    assert "stack-a: accepted" in out
    assert "stack-b: rejected" in out


def test_all_stacks_accepted_is_the_only_way_to_exit_zero(stub_pulumi, monkeypatch, capsys):
    good = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    monkeypatch.setenv(
        "STUB_EXPORTS",
        json.dumps(
            {"stack-a": _export_with_salt(good), "stack-b": _export_with_salt(good)}
        ),
    )
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-a", "stack-b"]) == 0
    out = capsys.readouterr().out
    assert "stack-a: accepted" in out
    assert "stack-b: accepted" in out


def test_a_missing_pulumi_binary_is_reported_as_error_not_rejected(monkeypatch, capsys):
    fake_pulumi(monkeypatch, None)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-a"]) == 2
    err = capsys.readouterr().err
    assert "stack-a: error:" in err
    assert "rejected" not in err


def test_a_nonzero_pulumi_exit_is_reported_as_error_not_rejected(stub_pulumi, monkeypatch, capsys):
    monkeypatch.setenv("STUB_EXIT_CODES", json.dumps({"stack-a": 1}))
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-a"]) == 2
    captured = capsys.readouterr()
    assert "stack-a: error:" in captured.err
    assert "stack-a: rejected" not in captured.out


def test_a_stack_lacking_a_secrets_provider_is_reported_as_error(stub_pulumi, monkeypatch, capsys):
    monkeypatch.setenv("STUB_EXPORTS", json.dumps({"stack-a": json.dumps({"deployment": {}})}))
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-a"]) == 2
    assert "stack-a: error:" in capsys.readouterr().err


def test_invalid_json_from_pulumi_is_reported_as_error(stub_pulumi, monkeypatch, capsys):
    monkeypatch.setenv("STUB_EXPORTS", json.dumps({"stack-a": "not json at all"}))
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-a"]) == 2
    assert "stack-a: error:" in capsys.readouterr().err


def test_a_pulumi_timeout_is_reported_as_error_without_aborting_other_stacks(
    stub_pulumi, monkeypatch, capsys
):
    good = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    monkeypatch.setenv("STUB_SLOW_STACK", "stack-slow")
    monkeypatch.setenv("STUB_EXPORTS", json.dumps({"stack-fast": _export_with_salt(good)}))
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    assert main(["check-passphrase", "stack-slow", "stack-fast"]) == 2
    captured = capsys.readouterr()
    assert "stack-slow: error:" in captured.err
    assert "stack-fast: accepted" in captured.out


class _ControlViolation(BaseException):
    """Raised by a control that must never run. `BaseException`, not
    `Exception`, and that is the whole point of it.

    `cmd_check_passphrase` is `@fail_closed` *and* its per-stack loop has
    its own `except Exception`, so a control raised as an `AssertionError`
    inside either is swallowed into `all_accepted = False` and the command
    returns 2 -- the same 2 the test below expects for the right reason.
    The assertion then holds whether the early refusal exists or not, which
    is not a test. `tests/test_credentials.py`'s `_PromptViolation` and
    `tests/test_gate_isolation.py`'s `GateViolation` are the same device for
    the same reason. Not `KeyboardInterrupt`, which pytest treats as a
    request to abort the whole session.
    """


def test_no_passphrase_provided_on_stdin_errors_before_any_stack_is_checked(monkeypatch):
    """The refusal must happen *before* the loop, not be absorbed by it.

    Both halves of that are load-bearing: without a passphrase there is
    nothing to check, and a per-stack "could not check" would report the
    same exit code while having run `pulumi` against every named stack.
    Only a control that `fail_closed` cannot swallow can tell the two
    apart -- see `_ControlViolation`.
    """
    monkeypatch.setattr(check_passphrase.sys, "stdin", _FakeStdin(isatty=False, lines=[]))

    def boom(*a, **k):
        raise _ControlViolation("no stack should be checked without a passphrase")

    monkeypatch.setattr(check_passphrase, "_check_stack", boom)
    assert main(["check-passphrase", "stack-a"]) == 2


# ---------------------------------------------------------------------------
# The passphrase never appears anywhere in output -- Python-level `print`s
# (capsys) or the real stub child process's own writes (capfd) -- across
# every reachable per-stack outcome, and never reaches the child's `argv`.
# ---------------------------------------------------------------------------


def test_the_passphrase_never_appears_in_output_on_any_outcome(stub_pulumi, monkeypatch, capfd):
    good = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    monkeypatch.setenv(
        "STUB_EXPORTS",
        json.dumps({"stack-accept": _export_with_salt(good), "stack-error": "not json"}),
    )
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)
    main(["check-passphrase", "stack-accept", "stack-error"])

    combined = "".join(capfd.readouterr())
    # `assert_no_disclosure`, not `not in`: this command *prints* a value
    # derived from the passphrase on every run (`passphrase fingerprint:
    # ...`), so a whole-string check here is exactly the assertion a partial
    # or re-encoded disclosure walks straight through -- and did.
    assert_no_disclosure(combined, MARKER_PASSPHRASE, what="the passphrase")

    # A rejected outcome too, in a separate run against a fresh capture.
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, WRONG_PASSPHRASE)
    main(["check-passphrase", "stack-accept"])
    combined_reject = "".join(capfd.readouterr())
    assert_no_disclosure(combined_reject, MARKER_PASSPHRASE, what="the passphrase")
    assert_no_disclosure(
        combined_reject, WRONG_PASSPHRASE, what="the wrong passphrase"
    )


def test_the_passphrase_never_reaches_pulumis_argv(stub_pulumi, monkeypatch, tmp_path):
    good = _encrypt_pulumi_style(MARKER_PASSPHRASE)
    monkeypatch.setenv("STUB_EXPORTS", json.dumps({"stack-a": _export_with_salt(good)}))
    argv_dump = tmp_path / "argv.jsonl"
    monkeypatch.setenv("STUB_DUMP_ARGV_TO", str(argv_dump))
    fake_pulumi(monkeypatch, stub_pulumi)
    feed_stdin(monkeypatch, MARKER_PASSPHRASE)

    main(["check-passphrase", "stack-a"])

    assert_no_disclosure(
        argv_dump.read_text(), MARKER_PASSPHRASE, what="the passphrase"
    )
