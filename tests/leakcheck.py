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
"""

from __future__ import annotations

import base64
import urllib.parse

# Shorter than this and a "leak" is more likely a coincidental collision
# with ordinary output than a disclosure -- a 4-character run of a
# hex-ish credential appears in unrelated text often enough to make the
# assertion flaky, which is how a leak check gets deleted.
MIN_IDENTIFYING_RUN = 8


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

    # Partial disclosure: any identifying-length run of the raw value.
    # Only the raw form -- a run of a base64 encoding is not recoverable
    # without its alignment, and checking it produces false positives.
    for start in range(0, max(1, len(value) - MIN_IDENTIFYING_RUN + 1)):
        run = value[start : start + MIN_IDENTIFYING_RUN]
        if len(run) < MIN_IDENTIFYING_RUN:
            break
        if run in text:
            raise AssertionError(
                f"{what} partially leaked into output: a "
                f"{MIN_IDENTIFYING_RUN}-character run starting at index "
                f"{start} of the value appears at offset {text.index(run)}"
            )
