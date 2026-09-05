"""One assertion for "this text does not disclose that credential".

Every leak test on this branch was written as `assert value not in text`.
The final review showed what that idiom cannot see: `{value!r}` is caught
(repr embeds the whole string), but `value[:8]` in an error message, or the
value base64-encoded into a payload, passes a whole-string `in` check while
disclosing the credential just as completely. A partial disclosure is still
a disclosure -- an attacker who learns the first eight characters of a
passphrase has had the search space cut by orders of magnitude.

So the check is: the value, any run of it long enough to be identifying,
and the encodings it plausibly passes through on its way to an output
stream.

**Choose opaque markers.** The partial-run check compares against the text
a command legitimately prints, and a marker built out of real words will
collide with it. `marker-db-password-1a2b3c4d` shares the run "password"
with the config path `db.password` that the same test asserts is reported,
so the helper fires on output that discloses nothing. That is the helper
working, not a bug in it -- a credential really can be recovered a run at a
time, and the checker cannot know which run was a coincidence. Give the
marker no substring in common with any path, flag or message under test,
and put the reason in the variable name rather than in the value.
"""

from __future__ import annotations

import base64
import urllib.parse

# Shorter than this and a "leak" is more likely a coincidental collision
# with ordinary output than a disclosure -- a 4-character run of a
# hex-ish credential appears in unrelated text often enough to make the
# assertion flaky, which is how a leak check gets deleted.
MIN_IDENTIFYING_RUN = 8

# A *truncated* encoding is still a disclosure, and the whole-string check
# above cannot see one: `value.encode("utf-8").hex()[:16]` contains neither
# the raw value nor the complete hex encoding, yet `bytes.fromhex` turns it
# straight back into the first eight characters of the credential. That is
# not hypothetical -- it is the exact shape of the passphrase-fingerprint
# defect this branch shipped and the tautological test that hid it.
#
# So the run check applies to the byte-aligned encodings too, at run
# lengths that each recover about the same amount of plaintext as
# MIN_IDENTIFYING_RUN does. Runs are NOT checked for the URL-quoted forms:
# percent-encoding is variable-width, so a run boundary can land mid-escape
# and the recovered text is not what the run implies.
RUN_LENGTHS = {
    "raw": MIN_IDENTIFYING_RUN,
    "hex": MIN_IDENTIFYING_RUN * 2,          # 2 hex characters per byte
    "base64": ((MIN_IDENTIFYING_RUN + 2) // 3) * 4,   # 4 characters per 3 bytes
}


def _encodings(value: str) -> dict[str, str]:
    """The forms a credential can reach an output stream in.

    Not exhaustive -- it cannot be. These are the ones this codebase
    actually produces: raw, what `json`/`repr` emit, what a URL carrying
    userinfo emits, and what a serialised envelope emits.
    """
    raw = value.encode("utf-8")
    return {
        "raw": value,
        "base64": base64.b64encode(raw).decode("ascii"),
        "base64-unpadded": base64.b64encode(raw).decode("ascii").rstrip("="),
        "hex": raw.hex(),
        "urlquote": urllib.parse.quote(value, safe=""),
        "urlquote-plus": urllib.parse.quote_plus(value),
    }


def assert_no_leak(text: str, value: str, *, what: str = "value") -> None:
    """Fail if `text` discloses `value` whole, in part, or re-encoded.

    `what` names the thing in the failure message. The message never
    contains `value` itself -- a leak check that prints the credential to
    prove the credential leaked would be its own bug (Global Constraint 4).
    It reports the encoding and the offset, which is enough to find it.
    """
    if not value:
        raise ValueError(
            "assert_no_leak called with an empty value: it would pass "
            "against any text and prove nothing"
        )

    for name, encoded in _encodings(value).items():
        if encoded and encoded in text:
            raise AssertionError(
                f"{what} leaked into output, {name}-encoded, in full "
                f"(at offset {text.index(encoded)})"
            )

    # Partial disclosure: an identifying-length run of the value, or of an
    # encoding a reader can decode a fragment of.
    encodings = _encodings(value)
    for name, run_length in RUN_LENGTHS.items():
        candidate = encodings[name]
        for start in range(0, max(1, len(candidate) - run_length + 1)):
            run = candidate[start : start + run_length]
            if len(run) < run_length:
                break
            if run in text:
                raise AssertionError(
                    f"{what} partially leaked into output: a {run_length}-"
                    f"character run of its {name} form, starting at index "
                    f"{start}, appears at offset {text.index(run)}"
                )
