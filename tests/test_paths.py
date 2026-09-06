"""Tests for the config-path grammar: findings, manifests and matching all
depend on `parse`/`render` agreeing with each other in both directions."""

from __future__ import annotations

import pytest

from stackward.paths import parse, render

# Spelled out via concatenation rather than string literals with `\\` runs,
# so the number of actual backslash characters in each fixture is unambiguous
# at the call site.
BACKSLASH = "\\"
TWO_BACKSLASHES = BACKSLASH * 2

# Canonical texts: render(parse(x)) must reproduce x exactly. Every entry
# here is already in the form `render` would itself produce.
CANONICAL_TEXTS = [
    "a.b.c",
    "a[0]",
    "a.b[3].c",
    'a["b.c"]',
    'a["b[c]"]',
    'a["b\\"c"]',
    'a[""]',
    "a.1",
    "[0]",
    "a[0][1]",
    '["a\\\\b"]',  # a literal backslash, doubled
]

# Segment lists: parse(render(y)) must reproduce y exactly. Includes forms
# `render` would not itself choose (e.g. a quoted digit key), which is fine —
# the round trip only requires equality, not that the text be canonical.
SEGMENT_LISTS: list[list[str | int]] = [
    ["a", "b", "c"],
    ["a", 0],
    ["a", "b", 3, "c"],
    ["a", "b.c"],
    ["a", "b[c]"],
    ['a', 'b"c'],
    ["a", ""],
    ["a", "1"],
    [0],
    ["1"],
    [1],
    ["a", 0, "b"],  # dot must be restored after a bracket
    ["a", 0, 1],  # two indices in a row, no dot between brackets
    ["a", 0, "b.c"],  # a quoted bracket right after another bracket
    ["a" + BACKSLASH + "b"],  # one backslash, no other reason to quote
    ["a." + BACKSLASH],  # quoted for the dot, then ends in one backslash
    ["a." + TWO_BACKSLASHES],  # quoted for the dot, then ends in two backslashes
]


def test_parses_a_plain_dotted_path():
    assert parse("a.b.c") == ["a", "b", "c"]


def test_parses_a_list_index_as_int():
    segments = parse("a[0]")
    assert segments == ["a", 0]
    assert type(segments[1]) is int


def test_key_containing_a_dot_round_trips_through_bracket_form():
    segments = ["a", "b.c"]
    text = render(segments)
    assert text == 'a["b.c"]'
    assert parse(text) == segments


def test_key_containing_a_bracket_round_trips_through_bracket_form():
    segments = ["a", "b[c]"]
    text = render(segments)
    assert text == 'a["b[c]"]'
    assert parse(text) == segments


def test_key_containing_a_quote_is_escaped_as_backslash_quote():
    segments = ["a", 'b"c']
    text = render(segments)
    assert text == 'a["b\\"c"]'
    assert parse(text) == segments


def test_empty_string_key_round_trips_through_bracket_form():
    segments = ["a", ""]
    text = render(segments)
    assert text == 'a[""]'
    assert parse(text) == segments


def test_quoted_string_digit_key_and_unquoted_list_index_do_not_collapse_on_parse():
    """`["1"]` is the string "1"; `[1]` is the index 1 — parsing must keep
    them apart, not just the segment lists' equality but their types too."""
    string_segments = parse('a["1"]')
    index_segments = parse("a[1]")

    assert string_segments == ["a", "1"]
    assert type(string_segments[1]) is str

    assert index_segments == ["a", 1]
    assert type(index_segments[1]) is int

    assert string_segments != index_segments


def test_string_digit_key_and_list_index_do_not_collapse_on_render():
    """The same distinction, the other way: rendering a string "1" must not
    produce the same text as rendering the index 1."""
    string_text = render(["a", "1"])
    index_text = render(["a", 1])

    assert string_text == "a.1"
    assert index_text == "a[1]"
    assert string_text != index_text

    # And each parses back to the segment it came from, not the other one.
    assert parse(string_text) == ["a", "1"]
    assert parse(index_text) == ["a", 1]


@pytest.mark.parametrize("text", CANONICAL_TEXTS)
def test_render_of_parse_reproduces_canonical_text(text):
    assert render(parse(text)) == text


@pytest.mark.parametrize("segments", SEGMENT_LISTS)
def test_parse_of_render_reproduces_segment_list(segments):
    assert parse(render(segments)) == segments


def test_dot_is_restored_after_a_bracket_for_the_next_bare_segment():
    assert render(["a", 0, "b"]) == "a[0].b"


def test_no_dot_between_two_consecutive_indices():
    assert render(["a", 0, 1]) == "a[0][1]"


def test_no_dot_before_a_quoted_bracket_that_follows_another_bracket():
    assert render(["a", 0, "b.c"]) == 'a[0]["b.c"]'


def test_unterminated_bracket_raises_value_error_naming_the_position():
    with pytest.raises(ValueError) as exc_info:
        parse("a[0")
    message = str(exc_info.value)
    assert "position 1" in message  # the unclosed '['
    assert "a[0" not in message


def test_empty_key_between_dots_raises_value_error_naming_the_position():
    with pytest.raises(ValueError) as exc_info:
        parse("a..b")
    message = str(exc_info.value)
    assert "position 2" in message
    assert "a..b" not in message


def test_non_digit_list_index_raises_value_error_naming_the_position():
    with pytest.raises(ValueError) as exc_info:
        parse("a[x]")
    message = str(exc_info.value)
    assert "position 2" in message
    assert "a[x]" not in message


def test_render_rejects_bool_segment_rather_than_silently_treating_it_as_an_index():
    """bool is an int subclass; letting one through would make `render([True])`
    silently produce `[1]`."""
    with pytest.raises(ValueError):
        render(["a", True])


def test_render_rejects_negative_index():
    with pytest.raises(ValueError):
        render(["a", -1])


def test_render_rejects_empty_segment_list():
    with pytest.raises(ValueError):
        render([])


def test_backslash_alone_is_quoted_and_escaped_as_double_backslash():
    """A key with no dot/bracket/quote but containing `\\` still has to be
    quoted — `\\` is only safe to leave bare if it never means anything
    special, and it does mean something special once it appears inside a
    quoted key elsewhere."""
    segments = ["a" + BACKSLASH + "b"]
    text = render(segments)
    assert text == '["a\\\\b"]'
    assert parse(text) == segments


def test_quoted_key_ending_in_one_literal_backslash_round_trips():
    """Regression: escaping only `"` (never `\\`) let a trailing backslash
    merge with the closing delimiter's quote on decode, so a key quoted for
    an unrelated reason (here, the dot) that also ends in `\\` failed to
    round-trip — `parse(render(y))` raised instead of reproducing `y`."""
    segments = ["a." + BACKSLASH]
    assert parse(render(segments)) == segments


def test_quoted_key_ending_in_two_literal_backslashes_round_trips():
    segments = ["a." + TWO_BACKSLASHES]
    assert parse(render(segments)) == segments


def test_leading_zero_in_list_index_is_rejected_as_malformed():
    """`render` never produces "01" for index 1; tolerating it on parse would
    give the same index two silently-equivalent spellings."""
    with pytest.raises(ValueError) as exc_info:
        parse("a[01]")
    message = str(exc_info.value)
    assert "position 2" in message
    assert "a[01]" not in message


def test_invalid_escape_sequence_raises_value_error_naming_the_position():
    """A `\\` inside a quoted key that isn't followed by `"` or `\\` is
    malformed, not a literal backslash silently passed through."""
    with pytest.raises(ValueError) as exc_info:
        parse('a["b\\nc"]')
    message = str(exc_info.value)
    assert "position 4" in message  # the stray backslash itself
    assert "b\\nc" not in message


# ---------------------------------------------------------------------------
# The grammar's refusals. Every branch below is a `ValueError` that no other
# test in this file reaches, and `set_secrets.parse_project_manifest` wraps
# `parse` in its own `try`, so none of them can ever surface as a traceback a
# reader would notice. A refusal that has never executed is where a fail-open
# hides: the branch could have been deleted, or could raise something the
# caller does not catch, and the suite would stay green either way.
#
# Each case asserts the position as well as the wording, because the position
# is what makes a refusal actionable, and asserts the offending text is absent
# from the message, which is the promise `parse`'s own docstring makes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "position", "fragment"),
    [
        # A path may not begin with the separator: `.a` names no first segment.
        (".a", 0, "cannot start with"),
        # A bare key may not contain a character that only means something
        # inside the bracket form -- here the closing bracket itself.
        ("a]b", 1, "unexpected character in key"),
        # A quoted key whose closing `"` never arrives.
        ('a["b', 3, "unterminated quoted key"),
        # A quoted key that closes, followed by anything other than `]`.
        ('a["b"c', 5, "expected ']' after quoted key"),
        # A bare character resuming straight after a bracket, with no
        # separator -- the `b` in `a[0]b`.
        ("a[0]b", 4, "expected '.' or '['"),
    ],
)
def test_a_malformed_path_is_refused_naming_the_position(text, position, fragment):
    with pytest.raises(ValueError) as exc_info:
        parse(text)
    message = str(exc_info.value)
    assert f"position {position}" in message
    assert fragment in message
    assert text not in message


def test_the_empty_path_is_refused():
    """Its own test rather than a parametrised case: there is no offending
    text to assert the absence of, and `""` is a substring of every string,
    so the shared `text not in message` assertion would be vacuous for it."""
    with pytest.raises(ValueError) as exc_info:
        parse("")
    message = str(exc_info.value)
    assert "position 0" in message
    assert "empty path" in message
