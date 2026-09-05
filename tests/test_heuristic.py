"""Tests for the heuristic net (`find_plaintext_credentials`) and the
`check-config` command built on top of it.

This is the gate itself: everything else in the project is scaffolding
around this check being correct. Each subtle rule in the brief — what
"encrypted" and "empty" mean, exactly — gets its own test, named for the
wrong implementation it would catch.
"""

from __future__ import annotations

import pytest

from stackward.cli import main
from stackward.commands.check_config import CheckError, scan_file
from stackward.config import CheckConfig
from stackward.nets.heuristic import DocumentError, find_plaintext_credentials

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
    declared a sensitive parent, so `_is_encrypted`'s parent-shape check
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
# scan_file / check-config — the file-path wrapper and the CLI it backs.
# ---------------------------------------------------------------------------


def write_yaml(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_scan_file_returns_no_findings_for_a_clean_file(tmp_path):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "name: myproject\nruntime: nodejs\n")
    assert scan_file(path, CheckConfig()) == []


def test_scan_file_finds_a_plaintext_credential(tmp_path):
    path = write_yaml(
        tmp_path, "Pulumi.dev.yaml", "config:\n  myproject:dbPassword: hunter2\n"
    )
    assert scan_file(path, CheckConfig()) == ["config.myproject:dbPassword"]


def test_scan_file_raises_check_error_on_invalid_yaml(tmp_path):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "key: [unclosed\n")
    with pytest.raises(CheckError):
        scan_file(path, CheckConfig())


def test_scan_file_raises_check_error_on_non_mapping_document(tmp_path):
    path = write_yaml(tmp_path, "Pulumi.dev.yaml", "- a\n- b\n")
    with pytest.raises(CheckError):
        scan_file(path, CheckConfig())


def test_scan_file_raises_check_error_on_missing_file(tmp_path):
    with pytest.raises(CheckError):
        scan_file(tmp_path / "does-not-exist.yaml", CheckConfig())


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
