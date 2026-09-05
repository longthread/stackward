"""The one config-path grammar, shared by findings, manifests and matching.

Pulumi's `--path` syntax is dotted segments (`a.b`), `[n]` for a list index,
and `["..."]` for a key that cannot be written bare — one containing `.`, `[`,
`]` or `"`, or the empty string. Everything downstream (the finding a scan
prints, the key a manifest stores, the walk that matches config data against
policy) needs to agree on exactly this representation; if the gate and the
matcher each grew their own path formatter, a key that round-trips through one
and not the other would silently stop matching. So there is exactly one
`parse`/`render` pair, here, and nothing else in the tool writes a path by
hand.

The one property that has to hold, in both directions:

    render(parse(text)) == text     for every canonical text
    parse(render(segments)) == segments   for every segment list

`parse` only ever produces an `int` from an *unquoted* `[n]` — a dotted or
quoted segment is always `str`, even when it looks like a number. That is
what keeps `["1"]` (the string key `"1"`) and `[1]` (list index `1`) distinct
on both sides of the round trip: `render` never has to guess, and `parse`
never has to coerce.
"""

from __future__ import annotations

# Characters that cannot appear in a bare or dotted segment. A key containing
# any of these, or the empty string, must be written `["..."]`.
_NEEDS_QUOTING = frozenset('.[]"')


def _needs_quoting(key: str) -> bool:
    return key == "" or any(char in _NEEDS_QUOTING for char in key)


def render(segments: list[str | int]) -> str:
    """The inverse of `parse`: bracket-quote a segment only when required.

    A digit-only string (the dict key `"1"`) is rendered bare/dotted, not
    quoted — `parse` never treats a dotted segment as an index, so quoting
    buys nothing and `render(parse(x)) == x` would otherwise fail for the
    canonical text `a.1`.
    """
    if not segments:
        raise ValueError("invalid path: segment list is empty")

    parts: list[str] = []
    for index, segment in enumerate(segments):
        if isinstance(segment, bool):
            # bool is an int subclass; letting one through would silently
            # render `True` as the list index `[1]`.
            raise ValueError(
                f"invalid path segment at index {index}: bool is not a valid segment"
            )
        if isinstance(segment, int):
            if segment < 0:
                raise ValueError(
                    f"invalid path segment at index {index}: list indices cannot be negative"
                )
            parts.append(f"[{segment}]")
        elif _needs_quoting(segment):
            parts.append('["{}"]'.format(segment.replace('"', '\\"')))
        elif index == 0:
            parts.append(segment)
        else:
            parts.append(f".{segment}")
    return "".join(parts)


def _scan_bare_key(text: str, pos: int) -> tuple[str, int]:
    """Consume a bare/dotted key starting at `pos`, stopping before `.` or `[`."""
    start = pos
    while pos < len(text) and text[pos] not in ".[":
        if text[pos] in '"]':
            raise ValueError(
                f"malformed path at position {pos}: unexpected character in key"
            )
        pos += 1
    key = text[start:pos]
    if not key:
        raise ValueError(f"malformed path at position {start}: empty key")
    return key, pos


def _scan_bracket(text: str, pos: int) -> tuple[str | int, int]:
    """Consume a `[...]` segment starting at the `[` and return it with the
    position just past the closing `]`."""
    start = pos
    pos += 1  # past '['
    if pos < len(text) and text[pos] == '"':
        pos += 1
        content_start = pos
        chars: list[str] = []
        closed = False
        while pos < len(text):
            char = text[pos]
            if char == "\\" and pos + 1 < len(text) and text[pos + 1] == '"':
                chars.append('"')
                pos += 2
                continue
            if char == '"':
                closed = True
                pos += 1
                break
            chars.append(char)
            pos += 1
        if not closed:
            raise ValueError(
                f"malformed path at position {content_start}: unterminated quoted key"
            )
        if pos >= len(text) or text[pos] != "]":
            raise ValueError(
                f"malformed path at position {pos}: expected ']' after quoted key"
            )
        return "".join(chars), pos + 1

    digits_start = pos
    while pos < len(text) and text[pos] != "]":
        pos += 1
    if pos >= len(text):
        raise ValueError(f"malformed path at position {start}: unterminated '[' bracket")
    content = text[digits_start:pos]
    if not content or any(char not in "0123456789" for char in content):
        raise ValueError(f"malformed path at position {digits_start}: invalid list index")
    return int(content), pos + 1


def parse(text: str) -> list[str | int]:
    """Parse Pulumi `--path` syntax into segments, list indices as `int`.

    Raises `ValueError` naming the offending position for malformed input.
    The message never echoes the surrounding text — only the position and
    what was wrong with it.
    """
    if not text:
        raise ValueError("malformed path at position 0: empty path")

    segments: list[str | int] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char == ".":
            if not segments:
                raise ValueError(
                    f"malformed path at position {pos}: path cannot start with '.'"
                )
            key, pos = _scan_bare_key(text, pos + 1)
            segments.append(key)
        elif char == "[":
            segment, pos = _scan_bracket(text, pos)
            segments.append(segment)
        else:
            if segments:
                # A bare character right after a previous segment, with no
                # '.' or '[' separator — e.g. the 'b' in "a[0]b".
                raise ValueError(
                    f"malformed path at position {pos}: expected '.' or '[' before next segment"
                )
            key, pos = _scan_bare_key(text, pos)
            segments.append(key)
    return segments
