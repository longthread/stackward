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


# The smallest policy that satisfies the gate: an explicit `model_net`, which
# is the whole minimum, because it is the one key whose default silently
# drops a net.
MINIMAL_POLICY = '[check]\nmodel_net = "none"\n'


@pytest.fixture
def bare_repo(tmp_path: Path, isolated_git: Path) -> Path:
    """A real git repository with one empty commit and **no** policy file.

    What every repository looked like before `.stackward.toml` was a
    refusal; kept as its own fixture so the tests that assert the refusal
    cannot be quietly rescued by a fixture that starts declaring one.
    """
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "commit", "-q", "--allow-empty", "-m", "init")
    return path


@pytest.fixture
def repo(bare_repo: Path) -> Path:
    """`bare_repo` with a minimal `.stackward.toml` **committed**.

    Committed, not merely written: this command reads its policy out of the
    index exactly as it reads the content it scans, so a policy sitting only
    in the working tree is not a policy as far as the gate is concerned and
    the whole command refuses (exit 2) before reaching whatever the test was
    actually about. Every test below that is not itself about policy
    discovery therefore needs one in the index.
    """
    write(bare_repo, ".stackward.toml", MINIMAL_POLICY)
    stage(bare_repo, ".stackward.toml")
    _git(bare_repo, "commit", "-q", "-m", "declare policy")
    return bare_repo


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
    alone would if a real credential scan needed to run).

    The broken config is *staged* here. It has to be: this command reads
    its policy from the index, so an unstaged edit would be invisible to it
    and the test would pass without the ordering it claims to check ever
    being exercised. Staging it makes the config genuinely unloadable for
    this invocation, which is exactly the condition the state-export rule
    must not depend on.
    """
    write(repo, ".stackward.toml", "this is not valid toml [[[")
    write(repo, "state.json", "{}")
    stage(repo, ".stackward.toml", "state.json")
    # Premise: this config really would sink the run if policy were loaded
    # first -- proved by the companion test below, which stages the same
    # broken file without a state export and gets exit 2.
    assert run_pre_commit(repo, monkeypatch) == 1
    assert "state.json" in capsys.readouterr().out


def test_a_broken_staged_config_alone_exits_2(repo, monkeypatch, capsys):
    """The companion that makes the ordering test above mean something: the
    same broken `.stackward.toml`, staged, with no state export to short-
    circuit the run, is a could-not-run refusal.

    Mutating `cmd_pre_commit`'s policy guard to swallow the error and carry
    on with `CheckConfig()` defaults survived the whole suite before this
    test existed -- a repository could ship a typo in its policy and be
    scanned by the weaker net without anyone being told.
    """
    write(repo, ".stackward.toml", "this is not valid toml [[[")
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, ".stackward.toml", "Pulumi.dev.yaml")
    assert run_pre_commit(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid TOML" in captured.err
    # Named the way a person reproduces it: `git show ":.stackward.toml"`.
    assert ":.stackward.toml" in captured.err


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
    """A bare exception from policy loading -- neither `ConfigError` nor
    `CheckError`, which `cmd_pre_commit`'s own guard already handles --
    must not escape uncaught. Without the `@fail_closed` decorator on
    `cmd_pre_commit`, it would propagate out of `main()` entirely and exit
    1 by Python's own default -- misreporting "credential found" for an
    invocation that never got as far as loading policy, let alone scanning
    a file."""
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")

    def boom():
        raise OSError("simulated failure inside policy discovery")

    monkeypatch.setattr(pre_commit_module, "_index_config_path", boom)
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


# ---------------------------------------------------------------------------
# The policy comes from the index too, not the working tree.
#
# The gate read *content* from the index and *rules* from the working tree,
# which is a fail-open with a working exploit: relax the rules without
# staging the relaxation and the credential commits under an excuse that is
# not itself committed. Each test below pins one half of the fix, and each
# has a companion proving the knob it uses is a knob at all -- otherwise
# "the relaxation did nothing" would be indistinguishable from "the gate
# ignores this key entirely".
# ---------------------------------------------------------------------------

CREDENTIAL_DOCUMENT = "config:\n  myproject:dbPassword: hunter2\n"
CREDENTIAL_PATH = "config.myproject:dbPassword"


def test_a_worktree_only_relaxation_does_not_let_a_staged_credential_through(
    repo, monkeypatch, capsys
):
    """The exploit, reproduced. A staged credential, then a working-tree
    edit to `.stackward.toml` adding an `allowed_references` entry for it,
    staging nothing. The commit being made carries the credential and not
    the relaxation, so the relaxation must not apply to it."""
    write(repo, "Pulumi.dev.yaml", CREDENTIAL_DOCUMENT)
    stage(repo, "Pulumi.dev.yaml")
    write(
        repo,
        ".stackward.toml",
        MINIMAL_POLICY + f'allowed_references = ["{CREDENTIAL_PATH}"]\n',
    )

    # Premise: the relaxation really is only in the working tree.
    assert " M .stackward.toml" in _git(repo, "status", "--porcelain").stdout

    assert run_pre_commit(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert f"plaintext credential at '{CREDENTIAL_PATH}'" in out
    assert "hunter2" not in out


def test_the_same_relaxation_staged_does_take_effect(repo, monkeypatch, capsys):
    """The companion. Without it, the test above would pass just as well if
    `allowed_references` were ignored altogether -- and then it would be
    proving nothing about *where* the policy was read from."""
    write(repo, "Pulumi.dev.yaml", CREDENTIAL_DOCUMENT)
    write(
        repo,
        ".stackward.toml",
        MINIMAL_POLICY + f'allowed_references = ["{CREDENTIAL_PATH}"]\n',
    )
    stage(repo, "Pulumi.dev.yaml", ".stackward.toml")

    assert run_pre_commit(repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_a_worktree_only_sensitive_parents_shrink_does_not_weaken_the_gate(
    repo, monkeypatch, capsys
):
    """The second of the three knobs that weaken from the working tree.
    `sensitive_parents` REPLACES rather than extends (`config._replace`), so
    a worktree edit can shrink what the index declared -- here from a
    category that flags the leaf to one that does not."""
    write(repo, ".stackward.toml", MINIMAL_POLICY + 'sensitive_parents = ["vars"]\n')
    write(repo, "Pulumi.dev.yaml", "vars:\n  DB_USER: admin\n")
    stage(repo, ".stackward.toml", "Pulumi.dev.yaml")
    _git(repo, "commit", "-q", "-m", "declare a sensitive parent")

    write(repo, "Pulumi.dev.yaml", "vars:\n  DB_USER: admin\n# changed\n")
    stage(repo, "Pulumi.dev.yaml")
    # Shrink the declared category in the working tree only.
    write(repo, ".stackward.toml", MINIMAL_POLICY + 'sensitive_parents = ["other"]\n')

    assert run_pre_commit(repo, monkeypatch) == 1
    assert "plaintext credential at 'vars.DB_USER'" in capsys.readouterr().out


def test_the_same_shrink_staged_does_take_effect(repo, monkeypatch, capsys):
    """Companion to the test above, for the same reason as the pair before
    it: proves `sensitive_parents` is a knob, so that "still flagged" above
    can only mean "read from the index"."""
    write(repo, ".stackward.toml", MINIMAL_POLICY + 'sensitive_parents = ["other"]\n')
    write(repo, "Pulumi.dev.yaml", "vars:\n  DB_USER: admin\n")
    stage(repo, ".stackward.toml", "Pulumi.dev.yaml")

    assert run_pre_commit(repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_a_policy_deleted_from_the_working_tree_still_applies(repo, monkeypatch, capsys):
    """The same rule from the other direction: deleting `.stackward.toml`
    from the working tree without staging the deletion cannot switch the
    gate off. A worktree-reading implementation would find no config at all
    here and (before this change) scan under defaults, or (after it) refuse
    -- either way, the wrong answer for a commit whose index still declares
    a policy."""
    write(repo, ".stackward.toml", MINIMAL_POLICY + 'allowed_references = ["a.b"]\n')
    stage(repo, ".stackward.toml")
    _git(repo, "commit", "-q", "-m", "declare policy with a reference")

    write(repo, "Pulumi.dev.yaml", "config:\n  a:\n    b: not-a-credential\n")
    stage(repo, "Pulumi.dev.yaml")
    (repo / ".stackward.toml").unlink()

    assert run_pre_commit(repo, monkeypatch) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# A missing policy file is a refusal, not a skip (Global Constraint 3).
# ---------------------------------------------------------------------------


def test_no_policy_in_the_index_is_a_refusal_naming_the_file(
    bare_repo, monkeypatch, capsys
):
    """Defaulting to `CheckConfig()` would make `model_net = "none"` -- a
    mode the plan requires to be *declared* -- a silent fallback, and would
    scan a repository under rules nobody chose. Exit 2, and the message has
    to name the minimal file, or the refusal is not actionable."""
    write(bare_repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(bare_repo, "Pulumi.dev.yaml")

    assert run_pre_commit(bare_repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert ".stackward.toml" in captured.err
    assert "model_net" in captured.err


def test_a_staged_credential_is_refused_not_reported_when_there_is_no_policy(
    bare_repo, monkeypatch, capsys
):
    """Exit 2, not 1, even though the heuristic net alone would have found
    this leaf. The command does not know what the repository considers
    sensitive, so it cannot claim a complete answer -- reporting one net's
    findings and exiting 1 would announce exactly that."""
    write(bare_repo, "Pulumi.dev.yaml", CREDENTIAL_DOCUMENT)
    stage(bare_repo, "Pulumi.dev.yaml")

    assert run_pre_commit(bare_repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hunter2" not in captured.err


def test_a_policy_only_in_the_working_tree_is_not_a_policy(
    bare_repo, monkeypatch, capsys
):
    """Written but never staged. The same rule this command applies to
    every stack config it scans, applied to the file that says how to scan
    them: if it is not in the index, it is not part of this commit."""
    write(bare_repo, ".stackward.toml", MINIMAL_POLICY)
    write(bare_repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(bare_repo, "Pulumi.dev.yaml")

    assert run_pre_commit(bare_repo, monkeypatch) == 2
    err = capsys.readouterr().err
    assert ".stackward.toml" in err
    # The fix is nameable: stage it.
    assert "git add" in err


def test_staging_the_policy_is_enough_it_need_not_be_committed(
    bare_repo, monkeypatch, capsys
):
    """The index, not HEAD. A repository declaring its policy in the very
    commit it is making must pass -- otherwise the first commit that adds a
    policy could never be made."""
    write(bare_repo, ".stackward.toml", MINIMAL_POLICY)
    write(bare_repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(bare_repo, ".stackward.toml", "Pulumi.dev.yaml")

    assert run_pre_commit(bare_repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Policy discovery walks upward, in the index, the way `find_repo_config`
# walks upward in the working tree.
# ---------------------------------------------------------------------------


def test_the_root_policy_applies_when_invoked_from_a_subdirectory(
    repo, monkeypatch, capsys
):
    """`git ls-files`' pathspecs resolve against the caller's directory, not
    the repository root, so a discovery that forgot `:(top)` would look for
    `deploy/.stackward.toml` from inside `deploy/` and find nothing."""
    write(repo, "deploy/Pulumi.dev.yaml", CREDENTIAL_DOCUMENT)
    stage(repo, "deploy/Pulumi.dev.yaml")

    assert run_pre_commit(repo / "deploy", monkeypatch) == 1
    assert f"plaintext credential at '{CREDENTIAL_PATH}'" in capsys.readouterr().out


def test_the_nearest_policy_wins_over_the_repository_root_one(
    repo, monkeypatch, capsys
):
    """Same precedence as `config.find_repo_config`'s working-tree walk: the
    file beside the caller beats the one at the root. Here the nested policy
    allows the reference the root policy does not."""
    write(
        repo,
        "deploy/.stackward.toml",
        MINIMAL_POLICY + f'allowed_references = ["{CREDENTIAL_PATH}"]\n',
    )
    write(repo, "deploy/Pulumi.dev.yaml", CREDENTIAL_DOCUMENT)
    stage(repo, "deploy/.stackward.toml", "deploy/Pulumi.dev.yaml")

    # From the root, the root policy applies and the leaf is a finding.
    assert run_pre_commit(repo, monkeypatch) == 1
    capsys.readouterr()
    # From the subdirectory, the nearer policy applies.
    assert run_pre_commit(repo / "deploy", monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_discovery_survives_a_directory_named_with_glob_characters(
    repo, monkeypatch, capsys
):
    """A directory name is not something this tool gets to constrain, and
    `[`, `*` and `?` are all pathspec metacharacters.

    Note what this does *not* prove: `:(literal)` is not what saves it.
    Verified directly against git -- a pathspec is compared literally before
    wildmatch is tried, so `cache[1]*/.stackward.toml` matches its own
    directory under `:(literal)`, `:(glob)` and no magic alike. What makes
    the metacharacters harmless is that `_index_config_path` accepts a
    candidate only by exact string equality, never by taking whatever git
    printed -- and that is what this goes red for (proved by replacing the
    match loop with `sorted(present)[0]`)."""
    subdir = "cache[1]*"
    write(
        repo,
        f"{subdir}/.stackward.toml",
        MINIMAL_POLICY + f'allowed_references = ["{CREDENTIAL_PATH}"]\n',
    )
    write(repo, f"{subdir}/Pulumi.dev.yaml", CREDENTIAL_DOCUMENT)
    stage(repo, f"{subdir}/.stackward.toml", f"{subdir}/Pulumi.dev.yaml")

    assert run_pre_commit(repo / subdir, monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_discovery_survives_a_non_ascii_directory_name(repo, monkeypatch, capsys):
    """`git rev-parse --show-prefix` is what says where the caller is
    standing, and its bytes are the path's own -- not `core.quotePath`
    escapes. A discovery that mis-decoded it would silently fall back to the
    repository root's policy, which here would report a finding."""
    subdir = "déploy"
    write(
        repo,
        f"{subdir}/.stackward.toml",
        MINIMAL_POLICY + f'allowed_references = ["{CREDENTIAL_PATH}"]\n',
    )
    write(repo, f"{subdir}/Pulumi.dev.yaml", CREDENTIAL_DOCUMENT)
    stage(repo, f"{subdir}/.stackward.toml", f"{subdir}/Pulumi.dev.yaml")

    assert run_pre_commit(repo / subdir, monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_git_runs_under_a_fixed_locale(repo, monkeypatch):
    """Every git invocation gets `LC_ALL=C`, with the environment merged
    rather than replaced.

    Nothing in this module branches on git's message text, so this is
    hardening rather than a live fix here -- but git's diagnostics are
    gettext-marked, and a message quoted verbatim into a `CheckError` should
    read the same in a bug report as it did on the machine that hit it.

    Scope, so this does not become a trap for whoever changes the fixture
    next: the loop covers every git invocation the command makes, and it
    holds only because `repo` declares `model_net = "none"`. Under
    `model_net = "artifact"`, `nets.model.run_git` also runs, and it sets no
    locale -- a second call site with the same gap, in a file this wave does
    not own. `commands.install_hooks._is_tracked` is a third, and there it
    is a live bug rather than hardening: it branches on git's *translated*
    strings, so under a non-English locale it raises and `hooks install`
    exits 2 in every repository.

    The merge half matters more than the override: git sets `GIT_INDEX_FILE`
    and `GIT_DIR` for a hook process, and passing a bare `env={"LC_ALL":
    "C"}` would make this command read a different index than the commit it
    is gating -- a fail-open that no output assertion elsewhere would catch.
    """
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    stage(repo, "Pulumi.dev.yaml")

    # Recording starts only now: this file's own `_git` helper shells out to
    # git too, and it is not what is under test.
    monkeypatch.setenv("STACKWARD_TEST_MARKER", "inherited")
    seen: list[dict[str, str] | None] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        if args and args[0] == "git":
            seen.append(kwargs.get("env"))
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    assert run_pre_commit(repo, monkeypatch) == 0

    assert seen, "no git invocation was recorded"
    for env in seen:
        assert env is not None, "a git invocation inherited the ambient locale"
        assert env.get("LC_ALL") == "C"
        assert env.get("STACKWARD_TEST_MARKER") == "inherited"
