"""Tests for `pre-commit` -- the gate run as a git hook body over staged
content.

Every test here uses a real temporary git repository, created with `git
init`. This command's entire reason to exist is a handful of git behaviours
that are easy to get subtly wrong (a rename-plus-edit staging as `R`, an
unmerged path vanishing from a plain `--diff-filter=ACMR` listing, a
non-ASCII or space-containing path being quoted without `-z`) -- mocking
git would hide exactly the bugs this suite exists to catch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from stackward.cli import main
from stackward.commands import pre_commit as pre_commit_module

# ---------------------------------------------------------------------------
# Real-git-repository fixtures and helpers.
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real, minimal git repository with one initial commit -- so every
    `git diff --cached` below has a HEAD to diff against."""
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "commit", "-q", "--allow-empty", "-m", "init")
    return path


def write(repo: Path, name: str, content: str) -> Path:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def write_bytes(repo: Path, name: str, content: bytes) -> Path:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def stage(repo: Path, *names: str) -> None:
    _git(repo, "add", *names)


def run_pre_commit(repo: Path, monkeypatch) -> int:
    monkeypatch.chdir(repo)
    return main(["pre-commit"])


def numbered_yaml(count: int = 200) -> str:
    """A YAML mapping large enough that git's default rename-similarity
    threshold (50%) still classifies a one-line edit as a rename, not a
    delete-plus-add."""
    return "\n".join(f"key{i}: value{i}" for i in range(count)) + "\n"


# ---------------------------------------------------------------------------
# Clean and simple credential-finding cases.
# ---------------------------------------------------------------------------


def test_clean_stack_config_exits_0_with_no_output(repo, monkeypatch, capsys):
    write(repo, "Pulumi.dev.yaml", "name: myproject\nruntime: nodejs\n")
    stage(repo, "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_plaintext_credential_exits_1_with_finding_and_remediation(
    repo, monkeypatch, capsys
):
    write(repo, "Pulumi.dev.yaml", "config:\n  myproject:dbPassword: hunter2\n")
    stage(repo, "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "Pulumi.dev.yaml: plaintext credential at 'config.myproject:dbPassword'" in out
    assert "pulumi config set --secret --path 'config.myproject:dbPassword'" in out
    assert "--no-verify" in out
    # The value itself must never appear anywhere in the output.
    assert "hunter2" not in out


# ---------------------------------------------------------------------------
# R matters: a rename plus an edit stages as `R`, not `A`/`M`.
# ---------------------------------------------------------------------------


def test_rename_plus_edit_staged_as_r_is_caught(repo, monkeypatch, capsys):
    """The exact bug `--diff-filter=ACMR` (not `ACM`) exists to prevent:
    renaming a stack config to a new name while also introducing a
    credential stages as a single `R` change, not `A`+`D` or `M`. Omitting
    `R` from the filter would let this change bypass the gate entirely."""
    write(repo, "Pulumi.old.yaml", numbered_yaml())
    stage(repo, "Pulumi.old.yaml")
    _git(repo, "commit", "-q", "-m", "add old config")

    _git(repo, "mv", "Pulumi.old.yaml", "Pulumi.dev.yaml")
    edited = (repo / "Pulumi.dev.yaml").read_text().replace(
        "key5: value5", "password: hunter2"
    )
    write(repo, "Pulumi.dev.yaml", edited)
    stage(repo, "Pulumi.dev.yaml")

    # Prove the premise: git really did record this as a rename, not a
    # plain modify or an add+delete pair. Without this assertion the test
    # could pass for the wrong reason if git's rename heuristic ever
    # stopped firing here.
    status = _git(repo, "diff", "--cached", "--name-status").stdout
    assert status.startswith("R")

    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "Pulumi.dev.yaml: plaintext credential at 'password'" in out


def test_diff_filter_without_r_would_have_missed_the_renamed_file(repo, monkeypatch):
    """Direct proof of why `R` is in the filter: with it excluded, the
    renamed-and-edited path is invisible to the staged-file listing this
    command builds everything else on top of."""
    write(repo, "Pulumi.old.yaml", numbered_yaml())
    stage(repo, "Pulumi.old.yaml")
    _git(repo, "commit", "-q", "-m", "add old config")
    _git(repo, "mv", "Pulumi.old.yaml", "Pulumi.dev.yaml")
    edited = (repo / "Pulumi.dev.yaml").read_text().replace(
        "key5: value5", "password: hunter2"
    )
    write(repo, "Pulumi.dev.yaml", edited)
    stage(repo, "Pulumi.dev.yaml")

    monkeypatch.chdir(repo)
    assert pre_commit_module._staged_paths("ACM") == []
    assert pre_commit_module._staged_paths("ACMR") == ["Pulumi.dev.yaml"]


# ---------------------------------------------------------------------------
# T matters too: converting an already-tracked stack config into a
# symlink stages as a type-change (`T`), which `ACMR` alone excludes just
# as it excludes an unmerged path -- ruled during review as a genuine
# fail-open gap in the brief's own literal `--diff-filter=ACMR`, closed by
# adding `T`.
# ---------------------------------------------------------------------------


def test_tracked_config_replaced_by_a_symlink_exits_2_not_0(repo, monkeypatch, capsys):
    """A symlink's staged "content" (via `git show ":<path>"`) is its
    target path, not YAML -- so once the type-change is actually seen by
    the scan, it fails closed (exit 2) rather than passing as clean or
    silently vanishing from the listing entirely."""
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")
    _git(repo, "commit", "-q", "-m", "add config")

    (repo / "Pulumi.dev.yaml").unlink()
    (repo / "Pulumi.dev.yaml").symlink_to("/etc/hostname")
    stage(repo, "Pulumi.dev.yaml")

    # Prove the premise: git really did record this as a type-change, not
    # a plain modify.
    status = _git(repo, "diff", "--cached", "--name-status").stdout
    assert status.startswith("T")

    assert run_pre_commit(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Pulumi.dev.yaml" in captured.err


def test_diff_filter_without_t_would_have_missed_the_symlinked_config(repo, monkeypatch):
    """Direct proof of why `T` is in the filter: with it excluded, a
    tracked stack config converted to a symlink is invisible to the
    staged-file listing -- exactly the way an unmerged path is invisible
    to a filter that omits `U` -- so the commit would exit 0 instead of
    ever reaching the scan that would refuse it."""
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")
    _git(repo, "commit", "-q", "-m", "add config")

    (repo / "Pulumi.dev.yaml").unlink()
    (repo / "Pulumi.dev.yaml").symlink_to("/etc/hostname")
    stage(repo, "Pulumi.dev.yaml")

    monkeypatch.chdir(repo)
    assert pre_commit_module._staged_paths("ACMR") == []
    assert pre_commit_module._staged_paths("ACMRT") == ["Pulumi.dev.yaml"]


# ---------------------------------------------------------------------------
# Pulumi state exports: refused outright, before any content check.
# ---------------------------------------------------------------------------


def test_staged_state_json_is_refused_before_any_scan(repo, monkeypatch, capsys):
    write(repo, "state.json", '{"deployment": {"resources": []}}')
    stage(repo, "state.json")
    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "state.json" in out
    assert "refused" in out
    assert "--no-verify" in out


def test_staged_stack_export_json_suffix_is_refused(repo, monkeypatch, capsys):
    write(repo, "backups/mydeploy.stack-export.json", "{}")
    stage(repo, "backups/mydeploy.stack-export.json")
    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "backups/mydeploy.stack-export.json" in out


def test_state_export_refusal_wins_over_a_broken_pulumi_config(repo, monkeypatch, capsys):
    """"Before any other check": a state export staged alongside a stack
    config that would otherwise fail to parse must still exit 1 (refused),
    never 2 (as the parse failure alone would) -- the state-export rule is
    the very first gate, not one gate among several evaluated together."""
    write(repo, "state.json", "{}")
    write(repo, "Pulumi.dev.yaml", "key: [unclosed\n")
    stage(repo, "state.json", "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 1
    assert "state.json" in capsys.readouterr().out


def test_state_export_refusal_needs_no_valid_repo_config(repo, monkeypatch, capsys):
    """The outright refusal happens before `.stackward.toml` is even
    loaded: a state export staged in a repository whose policy file is
    broken must still exit 1 (refused), never 2 (as the broken config
    alone would if a real credential scan needed to run). The broken file
    only needs to be on disk -- `find_repo_config` reads the working tree,
    not the index -- so it is deliberately not staged here."""
    write(repo, ".stackward.toml", "this is not valid toml [[[")
    write(repo, "state.json", "{}")
    stage(repo, "state.json")
    assert run_pre_commit(repo, monkeypatch) == 1
    assert "state.json" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Matching `Pulumi.<stack>.yaml` by prefix/suffix, not a character class.
# ---------------------------------------------------------------------------


def test_project_file_pulumi_yaml_is_not_a_stack_config(repo, monkeypatch, capsys):
    """`Pulumi.yaml` (no stack name) is the *project* file, not a stack
    config -- a naive `startswith("Pulumi.") and endswith(".yaml")` also
    matches it, because the two literals overlap on the shared `.`."""
    write(repo, "Pulumi.yaml", "name: myproject\npassword: hunter2\n")
    stage(repo, "Pulumi.yaml")
    assert run_pre_commit(repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_stack_name_containing_dots_is_matched(repo, monkeypatch, capsys):
    write(repo, "Pulumi.prod.eu.yaml", "password: hunter2\n")
    stage(repo, "Pulumi.prod.eu.yaml")
    assert run_pre_commit(repo, monkeypatch) == 1
    assert "Pulumi.prod.eu.yaml: plaintext credential at 'password'" in capsys.readouterr().out


@pytest.mark.parametrize(
    "basename",
    [
        "Pulumi.dev.yaml",
        "Pulumi.prod.eu.yaml",
        # Stack name is a single "." -- "Pulumi." + "." + ".yaml".
        "Pulumi...yaml",
    ],
)
def test_is_pulumi_stack_config_matches_a_non_empty_stack(basename):
    assert pre_commit_module._is_pulumi_stack_config(basename) is True


@pytest.mark.parametrize(
    "basename",
    [
        "Pulumi.yaml",
        "NotPulumi.dev.yaml",
        "Pulumi.dev.yml",
        "readme.md",
        # "Pulumi." + "" + ".yaml": the prefix and suffix abut with no
        # stack name between them at all -- the empty-stack case, same as
        # "Pulumi.yaml" but spelled with the shared "." written out twice
        # instead of once.
        "Pulumi..yaml",
    ],
)
def test_is_pulumi_stack_config_rejects_non_stack_names(basename):
    assert pre_commit_module._is_pulumi_stack_config(basename) is False


@pytest.mark.parametrize(
    "path", ["state.json", "backups/state.json", "x.stack-export.json"]
)
def test_is_state_export_matches(path):
    assert pre_commit_module._is_state_export(path) is True


@pytest.mark.parametrize(
    "path", ["state.json.bak", "notstate.json", "Pulumi.dev.yaml"]
)
def test_is_state_export_rejects_lookalikes(path):
    assert pre_commit_module._is_state_export(path) is False


# ---------------------------------------------------------------------------
# Staged content, not the working tree.
# ---------------------------------------------------------------------------


def test_staged_credential_is_caught_even_after_the_working_tree_is_cleaned(
    repo, monkeypatch, capsys
):
    """The defining behaviour of this command: it reads `git show
    ":<path>"`, so an edit made *after* staging (here, one that removes the
    credential from the working tree entirely) must not let the
    already-staged credential through."""
    write(repo, "Pulumi.dev.yaml", "config:\n  myproject:dbPassword: hunter2\n")
    stage(repo, "Pulumi.dev.yaml")
    # Overwrite the working tree with a clean version, without staging it.
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")

    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "plaintext credential at 'config.myproject:dbPassword'" in out


def test_unstaged_credential_in_the_working_tree_does_not_block(repo, monkeypatch, capsys):
    """The inverse of the test above: a credential sitting only in the
    working tree, never staged, must not fail a commit of the clean staged
    content -- this command checks the index, not the working tree."""
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")
    # Introduce a credential in the working tree without staging it.
    write(repo, "Pulumi.dev.yaml", "password: hunter2\n")

    assert run_pre_commit(repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Paths with a space or non-ASCII characters, via -z / NUL splitting.
# ---------------------------------------------------------------------------


def test_filename_containing_a_space_is_checked(repo, monkeypatch, capsys):
    write(repo, "Pulumi.my stack.yaml", "password: hunter2\n")
    stage(repo, "Pulumi.my stack.yaml")
    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "Pulumi.my stack.yaml: plaintext credential at 'password'" in out


def test_filename_containing_non_ascii_is_checked(repo, monkeypatch, capsys):
    write(repo, "Pulumi.tést.yaml", "password: hunter2\n")
    stage(repo, "Pulumi.tést.yaml")
    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "Pulumi.tést.yaml: plaintext credential at 'password'" in out


# ---------------------------------------------------------------------------
# Unmerged index entries: an error, not a pass.
# ---------------------------------------------------------------------------


def test_unmerged_index_entry_exits_2_even_with_an_unrelated_clean_file_staged(
    repo, monkeypatch, capsys
):
    """A real merge conflict on a file unrelated to any Pulumi config
    still refuses the commit outright: git itself would never let a
    `git commit` reach this hook while any path is unmerged, and this
    command's own listing (`--diff-filter=ACMR`) silently excludes
    unmerged paths, so without a dedicated check, a clean Pulumi config
    staged alongside an unresolved conflict elsewhere would wrongly read
    as clean."""
    write(repo, "f.txt", "a\n")
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "f.txt", "Pulumi.dev.yaml")
    _git(repo, "commit", "-q", "-m", "base")

    _git(repo, "checkout", "-q", "-b", "other")
    write(repo, "f.txt", "from-other-branch\n")
    _git(repo, "commit", "-q", "-am", "change on other")

    _git(repo, "checkout", "-q", "-")
    write(repo, "f.txt", "from-main-branch\n")
    _git(repo, "commit", "-q", "-am", "change on main")

    subprocess.run(
        ["git", "merge", "other"], cwd=repo, capture_output=True, text=True
    )
    status = _git(repo, "status", "--porcelain").stdout
    assert "UU f.txt" in status

    assert run_pre_commit(repo, monkeypatch) == 2
    err = capsys.readouterr().err
    assert "f.txt" in err
    assert "unmerged" in err.lower() or "conflict" in err.lower()


def test_diff_filter_acmr_excludes_unmerged_paths(repo, monkeypatch):
    """The assumption `cmd_pre_commit`'s unmerged check depends on: a
    plain `--diff-filter=ACMR` listing does not surface an unmerged path
    at all (git reports it as bare status `U`), so relying on that listing
    alone -- without the dedicated `U` check -- would silently pass over
    it rather than error."""
    write(repo, "f.txt", "a\n")
    stage(repo, "f.txt")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "other")
    write(repo, "f.txt", "from-other\n")
    _git(repo, "commit", "-q", "-am", "c1")
    _git(repo, "checkout", "-q", "-")
    write(repo, "f.txt", "from-main\n")
    _git(repo, "commit", "-q", "-am", "c2")
    subprocess.run(["git", "merge", "other"], cwd=repo, capture_output=True, text=True)

    monkeypatch.chdir(repo)
    assert pre_commit_module._staged_paths("ACMR") == []
    assert pre_commit_module._staged_paths("U") == ["f.txt"]


# ---------------------------------------------------------------------------
# Could-not-run conditions: exit 2, never a silent pass or exit 1.
# ---------------------------------------------------------------------------


def test_invalid_yaml_exits_2_and_never_leaks_the_credential(repo, monkeypatch, capsys):
    write(repo, "Pulumi.dev.yaml", 'password: "hunter2: not-valid-yaml\n')
    stage(repo, "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 2
    err = capsys.readouterr().err
    assert "hunter2" not in err


def test_non_mapping_document_exits_2(repo, monkeypatch, capsys):
    write(repo, "Pulumi.dev.yaml", "- a\n- b\n")
    stage(repo, "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Pulumi.dev.yaml" in captured.err


def test_non_utf8_staged_content_exits_2(repo, monkeypatch, capsys):
    write_bytes(repo, "Pulumi.dev.yaml", b"password: \xff\xfe not valid utf8\n")
    stage(repo, "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Pulumi.dev.yaml" in captured.err


def test_unanticipated_scan_exception_exits_2_not_1(repo, monkeypatch, capsys):
    """Python's default exit code for an uncaught exception is 1 -- the
    code reserved exclusively for "a credential was found". A bug
    somewhere unrelated to a real finding must still map to 2."""
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")

    def boom(_path, _check, _net):
        raise RuntimeError("unanticipated failure, not a CheckError")

    monkeypatch.setattr(pre_commit_module, "_scan_staged_config", boom)
    assert run_pre_commit(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Pulumi.dev.yaml" in captured.err
    assert "RuntimeError" in captured.err


def test_policy_load_failure_exits_2_not_1(repo, monkeypatch, capsys):
    """A bare exception from policy loading -- not `ConfigError`, which
    `_load_check_policy`'s own caller already handles -- must not escape
    `cmd_pre_commit` uncaught. Concretely: `find_repo_config` walks
    upward through parent directories with plain `.is_file()`/`.exists()`
    calls and no try/except of its own, so a `PermissionError` on a
    non-traversable parent would surface as a bare `OSError`. Without the
    `@fail_closed` decorator on `cmd_pre_commit`, that would propagate
    out of `main()` entirely and exit 1 by Python's own default --
    misreporting "credential found" for an invocation that never got as
    far as loading policy, let alone scanning a file."""
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")

    def boom():
        raise OSError("simulated permission error walking parent directories")

    monkeypatch.setattr(pre_commit_module, "find_repo_config", boom)
    assert run_pre_commit(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "OSError" in captured.err


def test_a_real_finding_still_prints_even_when_another_file_errors(
    repo, monkeypatch, capsys
):
    write(repo, "Pulumi.dev.yaml", 'password: "hunter2"\n')
    write(repo, "Pulumi.broken.yaml", "key: [unclosed\n")
    stage(repo, "Pulumi.dev.yaml", "Pulumi.broken.yaml")
    assert run_pre_commit(repo, monkeypatch) == 2
    out = capsys.readouterr().out
    assert "Pulumi.dev.yaml: plaintext credential at 'password'" in out


def test_git_show_failure_exits_2(repo, monkeypatch, capsys):
    """A `git show` failure that is not "unmerged" -- e.g. the object
    somehow being gone -- must still be a could-not-run error, converted
    by `_run_git` itself, not left to propagate as an uncaught
    `CalledProcessError`.

    The failure injected here is a genuine non-zero exit from a real
    subprocess call (`subprocess.run` itself is replaced, not `_run_git`),
    so this exercises `_run_git`'s own conversion of a failing `git show`
    into `CheckError` -- unlike asserting on a pre-raised `CheckError`
    directly, which would prove nothing about that conversion.
    """
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")

    real_run = subprocess.run

    def failing_run(args, **kwargs):
        if args[:2] == ["git", "show"]:
            return subprocess.CompletedProcess(
                args, returncode=128, stdout=b"", stderr=b"fatal: simulated failure\n"
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_run)
    assert run_pre_commit(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "simulated failure" in captured.err


def test_pre_commit_is_registered_and_dispatches(repo, monkeypatch):
    """Wiring only -- the rules `pre-commit` enforces are covered above."""
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 0


def test_unexpected_argument_is_a_usage_error_exiting_2():
    """`pre-commit` takes no arguments -- it always operates on the current
    repository's index. argparse's own usage-error exit code (2) is the
    same code reserved for every other could-not-run reason, never 1."""
    with pytest.raises(SystemExit) as exit_info:
        main(["pre-commit", "unexpected-argument"])
    assert exit_info.value.code == 2
