"""Tests for the heuristic net (`find_plaintext_credentials`) and the
`check-config` command built on top of it.

This is the gate itself: everything else in the project is scaffolding
around this check being correct. Each subtle rule in the brief — what
"encrypted" and "empty" mean, exactly — gets its own test, named for the
wrong implementation it would catch.
"""

from __future__ import annotations

import pytest
import yaml

from stackward.cli import main
from stackward.commands import check_config as check_config_module
from stackward.commands.check_config import CheckError, fail_closed, scan_file
from stackward.config import CheckConfig
from stackward.nets.heuristic import DocumentError, find_plaintext_credentials

# Characters that flow through PyYAML's `%r` interpolation into a scanner
# or parser message. `_YAML_TRIGGER_CHARACTERS`'s point is exactly that it
# is a *set*, not one instance: the apostrophe is the one character whose
# `repr()` switches delimiter (`"'"`, not `'''`), and a redaction pattern
# that only matched `'...'` passed every test built from the others.
_YAML_TRIGGER_CHARACTERS = [
    pytest.param("'", id="apostrophe"),
    pytest.param('"', id="double_quote"),
    pytest.param("\\", id="backslash"),
    pytest.param("\n", id="newline"),
    pytest.param("\x01", id="non_printable"),
]


def _marked_yaml_error(problem: str) -> yaml.MarkedYAMLError:
    """A `MarkedYAMLError` shaped like the ones PyYAML's scanner/parser
    actually raise, with a fixed, recognisable line/column so a test can
    assert position survives redaction alongside the interpolated
    character not surviving it."""
    mark = yaml.Mark("<unicode string>", 0, 4, 17, None, None)
    return yaml.MarkedYAMLError(problem=problem, problem_mark=mark)

# ---------------------------------------------------------------------------
# find_plaintext_credentials — the content-level net.
# ---------------------------------------------------------------------------


def test_clean_config_has_no_findings():
    document = {
        "name": "myproject",
        "runtime": "nodejs",
        "config": {"aws:region": "us-east-1"},
    }
    assert find_plaintext_credentials(document, CheckConfig()) == []


def test_plaintext_value_flagged_by_key_match():
    """The leaf's own key matches `sensitive_keys` (case-insensitive
    substring) — no parent policy needed."""
    document = {"config": {"myproject:dbPassword": "hunter2"}}
    assert find_plaintext_credentials(document, CheckConfig()) == [
        "config.myproject:dbPassword"
    ]


def test_plaintext_value_flagged_by_sensitive_parent():
    """The leaf's own key ("DB_USER") matches no built-in sensitive key —
    only a declared `sensitive_parents` ancestor makes it a finding."""
    check = CheckConfig(sensitive_parents=frozenset({"environment_variables"}))
    document = {"environment_variables": {"DB_USER": "admin"}}
    assert find_plaintext_credentials(document, check) == ["environment_variables.DB_USER"]


def test_encrypted_leaf_passes():
    """A leaf made sensitive only by its `sensitive_parents` ancestor, whose
    parent mapping is exactly `{"secure": ...}`, must not be flagged."""
    check = CheckConfig(sensitive_parents=frozenset({"environment_variables"}))
    document = {"environment_variables": {"DB_PASS": {"secure": "v1:AAAA"}}}
    assert find_plaintext_credentials(document, check) == []


def test_malformed_secure_wrapper_with_extra_sibling_is_flagged():
    """`{"secure": ..., "other": ...}` is NOT the one-key encryption shape —
    both children must be flagged. A `path.endswith(".secure")` check would
    wrongly exclude the first of the two; asserting it is present is the
    point of this test."""
    check = CheckConfig(sensitive_parents=frozenset({"environment_variables"}))
    document = {
        "environment_variables": {
            "DB_PASS": {"secure": "v1:AAAA", "other": "leaked-plaintext"}
        }
    }
    findings = find_plaintext_credentials(document, check)
    assert findings == [
        "environment_variables.DB_PASS.other",
        "environment_variables.DB_PASS.secure",
    ]


# ---------------------------------------------------------------------------
# Ruling: a sensitive key propagates to its subtree. Without this, the
# malformed-envelope case above is unreachable under the default policy —
# see the module docstring for why.
# ---------------------------------------------------------------------------


def test_encrypted_leaf_passes_under_default_policy_via_key_propagation():
    """`apiToken` matches `sensitive_keys` directly and propagates to its
    subtree; the leaf `apiToken.secure` inherits that sensitivity, but its
    parent mapping is exactly `{"secure": ...}` — still exempt. The
    exemption keeps working through propagation, exactly as it did through
    `sensitive_parents` above."""
    document = {"apiToken": {"secure": "v1:AAAA"}}
    assert find_plaintext_credentials(document, CheckConfig()) == []


def test_malformed_secure_wrapper_flagged_under_default_policy():
    """Regression for the propagation ruling: under the shipped default
    policy (no `sensitive_parents` declared, which is every repository's
    starting policy), this returned `[]` before propagation — neither
    "secure" nor "other" matches a built-in key, and no ancestor was
    declared a sensitive parent, so `is_encrypted`'s parent-shape check
    never even ran. A sensitive key now makes its whole subtree sensitive,
    so both leaves are correctly flagged."""
    document = {"apiToken": {"secure": "v1:AAAA", "other": "leaked-plaintext"}}
    findings = find_plaintext_credentials(document, CheckConfig())
    assert findings == ["apiToken.other", "apiToken.secure"]


def test_sensitive_key_propagates_through_several_levels_of_nesting():
    document = {"credential": {"a": {"b": {"c": "leaked-plaintext"}}}}
    assert find_plaintext_credentials(document, CheckConfig()) == ["credential.a.b.c"]


def test_empty_value_under_a_sensitive_key_still_passes():
    """Empty beats propagation: a sensitive key's subtree still exempts
    `None`/`""`/`True`/`False` leaves, same as a directly-matched key
    would."""
    document = {"credential": {"nested": {"value": None}}}
    assert find_plaintext_credentials(document, CheckConfig()) == []


def test_propagation_does_not_leak_to_a_sibling_non_sensitive_mapping():
    """A sensitive key's subtree is sensitive; an unrelated sibling
    subtree, whose own keys match nothing, is not — propagation flows
    down a branch, never sideways."""
    document = {
        "credential": {"a": "leaked-plaintext"},
        "unrelated": {"b": "not-flagged-by-this-policy"},
    }
    assert find_plaintext_credentials(document, CheckConfig()) == ["credential.a"]


def test_propagation_does_not_leak_to_a_sibling_non_sensitive_mapping_reversed_order():
    """Same document as above with insertion order reversed. `_walk`
    computes each key's contribution fresh from the `ancestor_sensitive`
    value it was called with, never by mutating a variable shared across
    loop iterations — so this must pass regardless of which sibling a dict
    happens to iterate first. A bug that leaked sensitivity across sibling
    branches via shared mutable state, instead of a fresh value per
    branch, could otherwise hide behind dict ordering."""
    document = {
        "unrelated": {"b": "not-flagged-by-this-policy"},
        "credential": {"a": "leaked-plaintext"},
    }
    assert find_plaintext_credentials(document, CheckConfig()) == ["credential.a"]


def test_self_referential_mapping_terminates_instead_of_recursing_forever():
    """A YAML anchor/alias pair like `credential: &x\\n  b: *x\\n` produces
    a dict that contains itself. Without a cycle guard, `_walk` recurses
    forever — each cycle produces a longer, distinct rendered path, so a
    depth cap would only delay the crash, not avoid it."""
    cyclic: dict = {"b": None}
    cyclic["b"] = cyclic
    document = {"credential": cyclic}
    assert find_plaintext_credentials(document, CheckConfig()) == []


def test_self_referential_sequence_terminates_instead_of_recursing_forever():
    """The same cycle, one level further down inside a list rather than a
    dict."""
    cyclic: list = ["placeholder"]
    cyclic[0] = cyclic
    document = {"credential": cyclic}
    assert find_plaintext_credentials(document, CheckConfig()) == []


def test_shared_non_cyclic_object_is_scanned_at_each_occurrence():
    """Two YAML anchors pointing at the same (non-cyclic) mapping from two
    unrelated branches is not a cycle — the cycle guard is scoped to the
    current descent path, not "every id ever seen", so both occurrences
    must still be scanned at their own distinct path."""
    shared = {"password": "leaked-plaintext"}
    document = {"copy1": shared, "copy2": shared}
    assert find_plaintext_credentials(document, CheckConfig()) == [
        "copy1.password",
        "copy2.password",
    ]


def test_none_and_empty_string_pass():
    document = {"password": None, "token": ""}
    assert find_plaintext_credentials(document, CheckConfig()) == []


def test_true_and_false_pass():
    """Identity, not the `value in (None, "", False, True)` containment
    test this rule replaces: booleans are excluded regardless."""
    document = {"password": True, "token": False}
    assert find_plaintext_credentials(document, CheckConfig()) == []


def test_integers_zero_one_and_larger_are_flagged():
    """`0 == False` and `1 == True` in Python, so a containment test for
    emptiness wrongly swallows these — they must still be findings."""
    document = {"password": 0, "token": 1, "secret": 42}
    assert find_plaintext_credentials(document, CheckConfig()) == [
        "password",
        "secret",
        "token",
    ]


def test_dict_key_containing_dot_is_reported_with_bracket_quoting():
    """The path is produced by `paths.render`, never by joining segments
    with `.` — a key containing a literal `.` must come back bracket-quoted,
    not run together with its parent."""
    document = {"config": {"my.password": "hunter2"}}
    assert find_plaintext_credentials(document, CheckConfig()) == ['config["my.password"]']


def test_allowed_reference_path_passes():
    """A path listed in `allowed_references` names another secret rather
    than holding one, and is never a finding even though it would otherwise
    qualify."""
    check = CheckConfig(allowed_references=['config["my.password"]'])
    document = {"config": {"my.password": "SOME_OTHER_SECRET_NAME"}}
    assert find_plaintext_credentials(document, check) == []


def test_list_document_raises():
    with pytest.raises(DocumentError):
        find_plaintext_credentials(["not", "a", "mapping"], CheckConfig())


def test_scalar_document_raises():
    with pytest.raises(DocumentError):
        find_plaintext_credentials("just a string", CheckConfig())


def test_empty_document_raises():
    """An empty YAML file parses to `None` — still not a mapping, still an
    error rather than "no findings"."""
    with pytest.raises(DocumentError):
        find_plaintext_credentials(None, CheckConfig())


# ---------------------------------------------------------------------------
# fail_closed — the outer boundary shared by check-config and pre-commit.
# ---------------------------------------------------------------------------


def test_fail_closed_converts_an_unhandled_exception_to_exit_2(capsys):
    """The property `fail_closed` exists for, isolated from either
    command: an exception the wrapped function does not itself catch
    becomes exit 2, never left to propagate (which would exit 1 by
    Python's own default — the code reserved for "a credential was
    found")."""

    @fail_closed
    def boom(_args):
        raise OSError("simulated failure unrelated to any file's content")

    assert boom(None) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "OSError" in captured.err
    # Only the exception's type name is reported, never its own text —
    # the same reasoning `describe_yaml_error` uses for never trusting
    # `str(exc)`: an unanticipated exception's message is not something
    # this boundary can vouch for as free of file content.
    assert "simulated failure unrelated to any file's content" not in captured.err


def test_fail_closed_does_not_interfere_with_a_normal_return(capsys):
    """A wrapped function that returns normally is unaffected — the
    decorator only ever intervenes on an exception."""

    @fail_closed
    def clean(_args):
        print("normal output")
        return 0

    assert clean(None) == 0
    assert "normal output" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scan_file / check-config — the file-path wrapper and the CLI it backs.
# ---------------------------------------------------------------------------


def write_yaml(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_scan_file_returns_no_findings_for_a_clean_file(tmp_path):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "name: myproject\nruntime: nodejs\n")
    assert scan_file(path, CheckConfig(), None) == []


def test_scan_file_finds_a_plaintext_credential(tmp_path):
    path = write_yaml(
        tmp_path, "Pulumi.dev.yaml", "config:\n  myproject:dbPassword: hunter2\n"
    )
    assert scan_file(path, CheckConfig(), None) == ["config.myproject:dbPassword"]


def test_scan_file_raises_check_error_on_invalid_yaml(tmp_path):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "key: [unclosed\n")
    with pytest.raises(CheckError):
        scan_file(path, CheckConfig(), None)


def test_invalid_yaml_error_never_prints_a_credential_shaped_value(tmp_path):
    """`MarkedYAMLError.__str__` includes a source-line snippet via PyYAML's
    `Mark.get_snippet()` — an unescaped colon in a value (entirely
    plausible for a real password) produces a parse error whose default
    message would otherwise embed that value verbatim. This is the one
    file in the whole system expected to contain a credential, so the
    reported message must never include it."""
    path = write_yaml(
        tmp_path,
        "Pulumi.leak.yaml",
        "config:\n  myproject:dbPassword: hunter2: not-valid-yaml\n",
    )
    with pytest.raises(CheckError) as exc_info:
        scan_file(path, CheckConfig(), None)
    assert "hunter2" not in str(exc_info.value)


def test_invalid_yaml_bad_escape_error_redacts_the_leaked_character(tmp_path):
    """`exc.problem` is not unconditionally free of document content:
    PyYAML's scanner interpolates one raw character via `%r` into
    "found unknown escape character %r" for a bad backslash escape. A
    value like `"hunter\\2ok"` (a plausible fragment of a real password)
    would otherwise leak the digit `2` into the error message. The error
    kind and position must survive redaction even though the character
    does not."""
    path = write_yaml(tmp_path, "Pulumi.leak.yaml", 'password: "hunter\\2ok"\n')
    with pytest.raises(CheckError) as exc_info:
        scan_file(path, CheckConfig(), None)
    message = str(exc_info.value)
    assert "'2'" not in message
    assert "hunter" not in message
    assert "<redacted>" in message
    assert "found unknown escape character" in message
    assert "line 1" in message
    assert "column 19" in message


def test_invalid_yaml_reserved_leading_character_error_redacts_the_leaked_character(
    tmp_path,
):
    """The same `%r` leak, from a different PyYAML message: a character
    that cannot start any token — a backtick right before what looks like
    a plaintext credential — interpolates that character into
    "found character %r that cannot start any token"."""
    path = write_yaml(tmp_path, "Pulumi.leak.yaml", "password: `s3cr3tPass99\n")
    with pytest.raises(CheckError) as exc_info:
        scan_file(path, CheckConfig(), None)
    message = str(exc_info.value)
    assert "'`'" not in message
    assert "s3cr3tPass99" not in message
    assert "<redacted>" in message
    assert "found character" in message
    assert "that cannot start any token" in message
    assert "line 1" in message
    assert "column 11" in message


def test_scan_file_redacts_an_apostrophe_that_flips_reprs_delimiter(tmp_path):
    """Regression for the specific gap the two tests above could not
    catch: when the character PyYAML interpolates via `%r` is itself an
    apostrophe, Python's `repr()` switches to double quotes (`"'"`, not
    `'''`) — a redaction pattern that only matched `'...'` left exactly
    this one character unredacted, end to end through the real CLI path,
    not just a synthetic exception."""
    path = write_yaml(tmp_path, "Pulumi.leak.yaml", "password: \"s3cr3t\\'value\"\n")
    with pytest.raises(CheckError) as exc_info:
        scan_file(path, CheckConfig(), None)
    message = str(exc_info.value)
    assert "'" not in message
    assert "s3cr3t" not in message
    assert "value" not in message
    assert "<redacted>" in message
    assert "found unknown escape character" in message


@pytest.mark.parametrize("trigger_char", _YAML_TRIGGER_CHARACTERS)
def test_yaml_error_redaction_removes_any_percent_r_escape_character(trigger_char):
    """The property, not an instance: redaction must hold for *any*
    character PyYAML's scanner interpolates via `%r` into "found unknown
    escape character %r" — not just the one or two a hand-picked
    reproduction happens to trigger. Two instance-pinned tests (`'2'`,
    `` '`' ``) both passed while this exact class of gap — the apostrophe
    flipping `repr()`'s delimiter to `"'"` — was still open, which is
    precisely what a test quantified over characters, rather than fixed to
    one, is for."""
    exc = _marked_yaml_error(f"found unknown escape character {trigger_char!r}")
    message = check_config_module.describe_yaml_error(exc)
    assert trigger_char not in message
    assert "<redacted>" in message
    assert "found unknown escape character" in message
    assert "line 5" in message
    assert "column 18" in message


@pytest.mark.parametrize("trigger_char", _YAML_TRIGGER_CHARACTERS)
def test_yaml_error_redaction_removes_any_percent_r_reserved_character(trigger_char):
    """The same property, for the other PyYAML message family that
    interpolates a raw character: "found character %r that cannot start
    any token"."""
    problem = f"found character {trigger_char!r} that cannot start any token"
    exc = _marked_yaml_error(problem)
    message = check_config_module.describe_yaml_error(exc)
    assert trigger_char not in message
    assert "<redacted>" in message
    assert "found character" in message
    assert "that cannot start any token" in message
    assert "line 5" in message
    assert "column 18" in message


def test_scan_file_raises_check_error_on_non_mapping_document(tmp_path):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "- a\n- b\n")
    with pytest.raises(CheckError):
        scan_file(path, CheckConfig(), None)


def test_scan_file_raises_check_error_on_missing_file(tmp_path):
    with pytest.raises(CheckError):
        scan_file(tmp_path / "does-not-exist.yaml", CheckConfig(), None)


def test_check_config_exits_0_when_clean(tmp_path, capsys):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "name: myproject\nruntime: nodejs\n")
    assert main(["check-config", str(path)]) == 0
    assert capsys.readouterr().out == ""


def test_check_config_exits_1_and_prints_the_finding_format(tmp_path, capsys):
    path = write_yaml(
        tmp_path, "Pulumi.dev.yaml", "config:\n  myproject:dbPassword: hunter2\n"
    )
    assert main(["check-config", str(path)]) == 1
    out = capsys.readouterr().out
    assert out == f"{path}: plaintext credential at 'config.myproject:dbPassword'\n"
    assert "hunter2" not in out


def test_check_config_findings_print_sorted(tmp_path, capsys):
    """Insertion order in the file is zebra-then-apple; output must be
    sorted regardless."""
    path = write_yaml(
        tmp_path,
        "Pulumi.dev.yaml",
        "zebra_password: hunter2\napple_token: abc123\n",
    )
    assert main(["check-config", str(path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"{path}: plaintext credential at 'apple_token'",
        f"{path}: plaintext credential at 'zebra_password'",
    ]


def test_check_config_exits_2_on_invalid_yaml(tmp_path, capsys):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "key: [unclosed\n")
    assert main(["check-config", str(path)]) == 2
    assert capsys.readouterr().out == ""


def test_check_config_exits_2_on_non_mapping_document(tmp_path):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "- a\n- b\n")
    assert main(["check-config", str(path)]) == 2


def test_check_config_exits_2_with_no_files_given():
    """argparse's own usage error for a missing required FILE — exit 2,
    same code as any other "could not run" reason, never 1."""
    with pytest.raises(SystemExit) as exit_info:
        main(["check-config"])
    assert exit_info.value.code == 2


def test_check_config_exits_2_on_non_utf8_file(tmp_path, capsys):
    """`Path.read_text()` raises `UnicodeDecodeError` for a file that is
    not valid UTF-8 — not an `OSError`, and previously uncaught, which
    would exit 1 (Python's default for an uncaught exception): the code
    reserved exclusively for "a credential was found". A file this command
    never finished reading is a could-not-run failure, not a finding."""
    path = tmp_path / "Pulumi.dev.yaml"
    path.write_bytes(b"password: \xff\xfe not valid utf8\n")
    assert main(["check-config", str(path)]) == 2
    assert capsys.readouterr().out == ""


def test_check_config_exits_2_on_unanticipated_scan_error(tmp_path, monkeypatch, capsys):
    """Any exception a per-file scan raises other than `CheckError` must
    still map to exit 2, not propagate and exit 1 (Python's default for an
    uncaught exception) — regardless of what raised it. Simulated here
    with a directly-injected failure rather than relying on a specific
    trigger, so this test does not duplicate the self-referential-YAML or
    non-UTF8 regressions covered elsewhere."""
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "name: myproject\n")

    def boom(_path, _check):
        raise RuntimeError("unanticipated failure, not a CheckError")

    monkeypatch.setattr("stackward.commands.check_config.scan_file", boom)
    assert main(["check-config", str(path)]) == 2
    assert capsys.readouterr().out == ""


def test_check_config_exits_2_when_policy_loading_raises_an_unexpected_error(
    tmp_path, monkeypatch, capsys
):
    """A bare exception from policy loading -- not `ConfigError`, which
    `cmd_check_config`'s own `except ConfigError` already handles -- must
    not escape uncaught. Concretely: `find_repo_config` walks upward
    through parent directories with plain `.is_file()`/`.exists()` calls
    and no try/except of its own, so a `PermissionError` on a
    non-traversable parent would surface as a bare `OSError`. Without the
    `@fail_closed` decorator on `cmd_check_config`, that would propagate
    out of `main()` entirely and exit 1 by Python's own default --
    misreporting "credential found" for an invocation that never got as
    far as loading policy, let alone scanning a file."""
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "name: myproject\n")

    def boom():
        raise OSError("simulated permission error walking parent directories")

    monkeypatch.setattr("stackward.commands.check_config.find_repo_config", boom)
    assert main(["check-config", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "OSError" in captured.err


def test_check_config_prints_a_real_finding_even_when_another_file_errors(
    tmp_path, capsys
):
    """A parse error on one file must never hide a genuine finding on
    another: the exit code (2, since a check-could-not-run error always
    wins) carries the could-not-run signal, but the finding still prints."""
    good = write_yaml(tmp_path, "good.yaml", 'password: "hunter2"\n')
    broken = write_yaml(tmp_path, "broken.yaml", "key: [unclosed\n")
    assert main(["check-config", str(good), str(broken)]) == 2
    out = capsys.readouterr().out
    assert out == f"{good}: plaintext credential at 'password'\n"
