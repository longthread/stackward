"""Tests for the model net: the declared-secrets graph, its generator, and
the freshness check that decides whether the committed one may be trusted.

Three layers, deliberately separate:

- **the matcher**, driven by hand-written graphs. These need neither pydantic
  nor git, and they are where the semantics that make the net worth having
  are pinned: prefix coverage, structural (not glob) key handling, and
  matching at unbounded depth through a recursive model.
- **the generator**, run for real under this interpreter against synthetic
  pydantic models written into a temporary repository. Nothing is mocked:
  the marks are read from real `model_fields`, the MRO walk crosses real
  module files, and the blob ids come from a real index. A mocked generator
  would prove the artifact's shape and nothing about whether a mark in a
  repository actually reaches it.
- **the freshness check**, on real git repositories, because "read the index,
  not the working tree" is a claim about git and cannot be tested without it.

Every model here is synthetic and named for its role in the test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from stackward.bootstrap import generator_source
from stackward.cli import main
from stackward.commands.sync_declared import SyncError, build_artifact, serialise
from stackward.config import CheckConfig
from stackward.nets.heuristic import DocumentError
from stackward.nets.model import (
    ARTIFACT_PATH,
    ARTIFACT_VERSION,
    DictOf,
    ListOf,
    ModelNetError,
    ModelRef,
    find_declared_credentials,
    parse_net,
)

# ---------------------------------------------------------------------------
# Matcher fixtures: graphs written by hand, so a matcher bug cannot be masked
# by a generator bug that happens to cancel it out.
# ---------------------------------------------------------------------------


def build_net(models: dict, roots: dict):
    """A `ModelNet` through the real artifact parser, not around it.

    Constructing `ModelNet` directly would skip every validation the parser
    performs, and the parser is what a malformed committed artifact meets
    first.
    """
    net, _sources = parse_net(
        {
            "version": ARTIFACT_VERSION,
            "models": models,
            "roots": roots,
            "sources": {"declared.py": "0" * 40},
        }
    )
    return net


def match(document, net, allowed: list[str] | None = None) -> list[str]:
    return find_declared_credentials(
        document, net, CheckConfig(allowed_references=allowed or [])
    )


SELF_REFERENTIAL = build_net(
    models={
        "declared:Node": {
            "token": {"secret": True},
            "plain": {},
            "children": {
                "child": {
                    "kind": "list",
                    "item": {"kind": "model", "model": "declared:Node"},
                }
            },
        }
    },
    roots={"app:app": "declared:Node"},
)

MUTUALLY_RECURSIVE = build_net(
    models={
        "declared:Even": {
            "odd": {"child": {"kind": "model", "model": "declared:Odd"}},
        },
        "declared:Odd": {
            "token": {"secret": True},
            "even": {"child": {"kind": "model", "model": "declared:Even"}},
        },
    },
    roots={"app:app": "declared:Even"},
)

MARKED_CONTAINER = build_net(
    models={"declared:Root": {"peer": {"secret": True}, "open": {}}},
    roots={"app:app": "declared:Root"},
)

MAP_OF_MODELS = build_net(
    models={
        "declared:Root": {
            "entries": {
                "child": {
                    "kind": "dict",
                    "value": {"kind": "model", "model": "declared:Entry"},
                }
            }
        },
        "declared:Entry": {"token": {"secret": True}},
    },
    roots={"app:app": "declared:Root"},
)


# ---------------------------------------------------------------------------
# The matcher: the semantics a flat pattern list cannot express.
# ---------------------------------------------------------------------------


def test_a_self_referential_model_matches_a_mark_at_depth():
    """The counterexample the graph form exists for.

    A generator emitting flat patterns needs a cycle guard, so it emits
    `token` and stops — while the data it describes is unbounded. Here the
    third level down must be found, which no finite pattern list produced
    from this model would contain.
    """
    document = {
        "config": {
            "app:app": {
                "token": "level0",
                "children": [
                    {"token": "level1", "children": [{"token": "level2"}]},
                ],
            }
        }
    }
    assert match(document, SELF_REFERENTIAL) == [
        "config.app:app.children[0].children[0].token",
        "config.app:app.children[0].token",
        "config.app:app.token",
    ]


def test_mutual_recursion_matches_through_both_models():
    """Neither model refers to itself; each refers to the other. The walk has
    to follow the cycle as far as the *data* goes and no further."""
    document = {
        "config": {
            "app:app": {
                "odd": {
                    "token": "first",
                    "even": {"odd": {"token": "second"}},
                }
            }
        }
    }
    assert match(document, MUTUALLY_RECURSIVE) == [
        "config.app:app.odd.even.odd.token",
        "config.app:app.odd.token",
    ]


def test_a_marked_container_covers_every_descendant():
    """Prefix, not exact: a mark covers everything beneath it whatever the
    field's annotation was, and however deeply the data nests."""
    document = {
        "config": {
            "app:app": {
                "peer": {"url": "u", "nested": {"k": "v", "list": ["a", "b"]}},
                "open": "not-declared",
            }
        }
    }
    assert match(document, MARKED_CONTAINER) == [
        "config.app:app.peer.nested.k",
        "config.app:app.peer.nested.list[0]",
        "config.app:app.peer.nested.list[1]",
        "config.app:app.peer.url",
    ]


def test_a_dict_key_containing_a_dot_is_reported_with_bracket_quoting():
    """Structural, not string-glob.

    The key `a.b` is one segment. A matcher that joined segments into a
    dotted string and matched patterns against it would read this as two
    segments and lose the leaf; the rendered finding is asserted in full
    because the rendering is exactly what such an implementation gets wrong.
    """
    document = {"config": {"app:app": {"entries": {"a.b": {"token": "value"}}}}}
    assert match(document, MAP_OF_MODELS) == ['config.app:app.entries["a.b"].token']


def test_a_dict_key_containing_a_bracket_is_reported_with_bracket_quoting():
    """The same for `[`, which a glob translating `*` to `[^.\\[]+` also
    loses."""
    document = {"config": {"app:app": {"entries": {"a[0]": {"token": "value"}}}}}
    assert match(document, MAP_OF_MODELS) == ['config.app:app.entries["a[0]"].token']


def test_a_mapping_key_is_never_matched_against_a_field_name():
    """Under a `dict[str, Model]` the keys are data, not field names.

    The key here is literally `token`, the name of the marked field one level
    further down. A matcher that consulted the model at a mapping position
    would treat it as that field and cover its whole subtree — so the value
    beneath it carries an *undeclared* leaf, which is what makes the
    assertion discriminate rather than merely hold.
    """
    document = {
        "config": {"app:app": {"entries": {"token": {"undeclared": "ordinary"}}}}
    }
    assert match(document, MAP_OF_MODELS) == []


def test_an_undeclared_field_is_not_reported():
    document = {"config": {"app:app": {"plain": "ordinary-value"}}}
    assert match(document, SELF_REFERENTIAL) == []


def test_a_namespace_with_no_declared_root_is_left_to_the_heuristic_net():
    document = {"config": {"other:other": {"token": "value"}}}
    assert match(document, SELF_REFERENTIAL) == []


def test_a_declared_field_holding_an_encryption_envelope_is_not_reported():
    """`{"secure": ...}` is Pulumi's encrypted form. Reporting it would make
    the net fire on every value it had already succeeded in protecting."""
    document = {"config": {"app:app": {"token": {"secure": "v1:ciphertext"}}}}
    assert match(document, SELF_REFERENTIAL) == []


def test_a_malformed_encryption_envelope_is_still_reported():
    """A sibling key beside `secure` means the mapping is not the envelope,
    and nothing tells a reader the sibling was not the value that leaked."""
    document = {
        "config": {"app:app": {"token": {"secure": "v1:ciphertext", "other": "leak"}}}
    }
    assert match(document, SELF_REFERENTIAL) == [
        "config.app:app.token.other",
        "config.app:app.token.secure",
    ]


@pytest.mark.parametrize("value", [None, "", True, False])
def test_an_empty_declared_field_is_not_reported(value):
    document = {"config": {"app:app": {"token": value}}}
    assert match(document, SELF_REFERENTIAL) == []


@pytest.mark.parametrize("value", [0, 1])
def test_the_integers_zero_and_one_are_not_treated_as_empty(value):
    """`value in (None, "", False, True)` would exclude these, because
    `0 == False` and `1 == True`."""
    document = {"config": {"app:app": {"token": value}}}
    assert match(document, SELF_REFERENTIAL) == ["config.app:app.token"]


def test_an_allowed_reference_is_not_reported():
    document = {"config": {"app:app": {"token": "names-another-secret"}}}
    assert match(document, SELF_REFERENTIAL, ["config.app:app.token"]) == []


def test_a_self_referencing_document_does_not_recurse_forever():
    """A YAML anchor/alias pair can make a mapping contain itself. The walk
    must stop, and must still report the credential sitting beside the back
    edge rather than abandoning the branch."""
    node: dict = {"token": "found-me"}
    node["children"] = [node]
    document = {"config": {"app:app": node}}
    assert match(document, SELF_REFERENTIAL) == ["config.app:app.token"]


def test_a_document_that_is_not_a_mapping_is_refused():
    with pytest.raises(DocumentError):
        match(["not", "a", "mapping"], SELF_REFERENTIAL)


def test_a_document_without_a_config_section_matches_nothing():
    assert match({"name": "project"}, SELF_REFERENTIAL) == []


def test_a_scalar_where_a_model_was_declared_matches_nothing():
    assert match({"config": {"app:app": "scalar"}}, SELF_REFERENTIAL) == []


# ---------------------------------------------------------------------------
# The artifact parser: a committed file this tool cannot fully account for is
# a refusal, never a partial walk.
# ---------------------------------------------------------------------------


def parse_should_fail(payload) -> str:
    with pytest.raises(ModelNetError) as excinfo:
        parse_net(payload)
    return str(excinfo.value)


def test_an_unsupported_artifact_version_is_refused():
    message = parse_should_fail(
        {"version": ARTIFACT_VERSION + 1, "models": {}, "roots": {}, "sources": {"a": "b"}}
    )
    assert "unsupported version" in message
    assert "sync-declared-secrets" in message


def test_an_artifact_naming_an_undefined_model_is_refused():
    assert "undefined model" in parse_should_fail(
        {
            "version": ARTIFACT_VERSION,
            "models": {},
            "roots": {"app:app": "declared:Missing"},
            "sources": {"a": "b"},
        }
    )


def test_an_artifact_with_an_unknown_child_kind_is_refused():
    assert "unknown child kind" in parse_should_fail(
        {
            "version": ARTIFACT_VERSION,
            "models": {"declared:Root": {"f": {"child": {"kind": "tuple"}}}},
            "roots": {},
            "sources": {"a": "b"},
        }
    )


def test_an_artifact_with_no_sources_is_refused():
    """An artifact recording no sources can never be found stale, so it would
    be trusted forever."""
    assert "freshness cannot be verified" in parse_should_fail(
        {"version": ARTIFACT_VERSION, "models": {}, "roots": {}, "sources": {}}
    )


def test_a_parsed_graph_keeps_the_child_node_shapes():
    SELF = {"kind": "model", "model": "declared:Root"}
    net = build_net(
        models={
            "declared:Root": {
                "a": {"child": SELF},
                "b": {"child": {"kind": "list", "item": SELF}},
                "c": {"child": {"kind": "dict", "value": SELF}},
            }
        },
        roots={},
    )
    fields = net.models["declared:Root"]
    assert fields["a"].child == ModelRef("declared:Root")
    assert fields["b"].child == ListOf(ModelRef("declared:Root"))
    assert fields["c"].child == DictOf(ModelRef("declared:Root"))


# ---------------------------------------------------------------------------
# Real repositories, real models, real git.
# ---------------------------------------------------------------------------


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    git(path, "commit", "-q", "--allow-empty", "-m", "init")
    return path


def write(repo: Path, name: str, content: str) -> Path:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content))
    return target


def declare(repo: Path, models: str = 'declared:Root', extra: str = "") -> None:
    """Write a `.stackward.toml` naming this interpreter and one root model."""
    write(
        repo,
        ".stackward.toml",
        f"""
        python = "{sys.executable}"

        [check]
        model_net = "artifact"
        stack_models = {{ "app:app" = "{models}" }}
        {extra}
        """,
    )


def sync(repo: Path, monkeypatch) -> int:
    monkeypatch.chdir(repo)
    return main(["sync-declared-secrets"])


def check(repo: Path, monkeypatch, name: str = "Pulumi.dev.yaml") -> int:
    monkeypatch.chdir(repo)
    return main(["check-config", name])


def artifact(repo: Path) -> dict:
    return json.loads((repo / ARTIFACT_PATH).read_text())


def sync_and_commit(repo: Path, monkeypatch, capsys) -> dict:
    assert sync(repo, monkeypatch) == 0
    capsys.readouterr()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "declare")
    return artifact(repo)


RECURSIVE_MODEL = """
    from __future__ import annotations

    from pydantic import BaseModel, Field


    class Root(BaseModel):
        token: str = Field(default="", json_schema_extra={"secret": True})
        plain: str = ""
        children: list[Root] = []
"""

# The same model with one more field. Used to change a *source* file after the
# artifact was written, which is what the freshness check has to notice.
RECURSIVE_MODEL_EDITED = RECURSIVE_MODEL + '        added_later: str = ""\n'

# A marked field whose name matches none of the heuristic net's built-in
# patterns (password, passphrase, token, secret, apikey, api_key, jwt,
# credential, private_key, access_key_id, secret_access_key). The model net is
# the only thing that can name this one, which is what the pair of tests using
# it exists to demonstrate.
HEURISTIC_BLIND_MODEL = """
    from __future__ import annotations

    from pydantic import BaseModel, Field


    class Root(BaseModel):
        binding: str = Field(default="", json_schema_extra={"secret": True})
        plain: str = ""
        children: list[Root] = []
"""

# A credential the heuristic net cannot reach: an unremarkable key name, at a
# depth no finite pattern list produced from the model would enumerate.
BLIND_DOCUMENT = (
    "config:\n"
    "  app:app:\n"
    "    children:\n"
    "      - children:\n"
    "          - binding: deeply-nested\n"
)


# ---------------------------------------------------------------------------
# Generation: what reaches the artifact, and what is refused rather than
# silently omitted from it.
# ---------------------------------------------------------------------------


def test_a_self_referential_model_generates_a_graph_not_a_pattern_list(
    repo, monkeypatch, capsys
):
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)

    fields = written["models"]["declared:Root"]
    assert fields["token"] == {"secret": True}
    # The reference points back at the same model, which is what makes the
    # unbounded case expressible in a finite file.
    assert fields["children"]["child"] == {
        "kind": "list",
        "item": {"kind": "model", "model": "declared:Root"},
    }


def test_a_mark_inherited_from_a_base_class_in_another_module_is_detected(
    repo, monkeypatch, capsys
):
    write(
        repo,
        "ancestry.py",
        """
        from pydantic import BaseModel, Field


        class Ancestor(BaseModel):
            handed_down: str = Field(default="", json_schema_extra={"secret": True})
        """,
    )
    write(
        repo,
        "declared.py",
        """
        from ancestry import Ancestor


        class Root(Ancestor):
            own: str = ""
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"]["handed_down"] == {"secret": True}


def test_a_base_class_in_another_module_is_recorded_as_a_source(
    repo, monkeypatch, capsys
):
    """`getsourcefile(Derived)` names only the subclass's own file, so the
    freshness check must walk the MRO. Without that, removing the mark from
    the base class would never register as a change."""
    write(
        repo,
        "ancestry.py",
        """
        from pydantic import BaseModel, Field


        class Ancestor(BaseModel):
            handed_down: str = Field(default="", json_schema_extra={"secret": True})
        """,
    )
    write(
        repo,
        "declared.py",
        """
        from ancestry import Ancestor


        class Root(Ancestor):
            own: str = ""
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert "ancestry.py" in written["sources"]


def test_a_mark_carried_by_an_annotated_alias_defined_elsewhere_is_detected(
    repo, monkeypatch, capsys
):
    write(
        repo,
        "conventions.py",
        """
        from typing import Annotated

        from pydantic import Field

        Guarded = Annotated[str, Field(json_schema_extra={"secret": True})]
        """,
    )
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel

        from conventions import Guarded


        class Root(BaseModel):
            borrowed: Guarded = ""
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"]["borrowed"] == {"secret": True}


def test_the_module_defining_an_annotated_alias_is_recorded_as_a_source(
    repo, monkeypatch, capsys
):
    """pydantic resolves the alias's mark onto the field and discards the
    wrapper, so nothing on the field says where the alias was written. The
    alias module is found by identity instead — and if it were not recorded,
    editing the mark out of it would leave the artifact silently wrong."""
    write(
        repo,
        "conventions.py",
        """
        from typing import Annotated

        from pydantic import Field

        Guarded = Annotated[str, Field(json_schema_extra={"secret": True})]
        """,
    )
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel

        from conventions import Guarded


        class Root(BaseModel):
            borrowed: Guarded = ""
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert "conventions.py" in written["sources"]


def test_the_artifact_never_records_itself_as_a_source(repo, monkeypatch, capsys):
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert ARTIFACT_PATH not in written["sources"]


def test_regenerating_unchanged_models_produces_identical_bytes(
    repo, monkeypatch, capsys
):
    """The sound closure for staleness is regenerating in CI and failing on
    any diff, which a generator with unstable output would turn into noise."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 0
    first = (repo / ARTIFACT_PATH).read_bytes()
    assert sync(repo, monkeypatch) == 0
    capsys.readouterr()
    assert (repo / ARTIFACT_PATH).read_bytes() == first


def test_the_artifact_is_serialised_with_sorted_keys():
    """Byte-determinism across *environments*, which regenerating twice on one
    machine cannot demonstrate: dictionary insertion order here follows the
    order classes happened to be walked in, and `sort_keys` is the only thing
    that makes the committed bytes independent of it.
    """
    text = serialise(
        {
            "version": ARTIFACT_VERSION,
            "sources": {"zebra.py": "b" * 40, "alpha.py": "a" * 40},
            "roots": {},
            "models": {},
        }
    ).decode("utf-8")
    assert text.index('"models"') < text.index('"roots"') < text.index('"sources"')
    assert text.index("alpha.py") < text.index("zebra.py")


def test_the_artifact_ends_with_exactly_one_newline(repo, monkeypatch, capsys):
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 0
    capsys.readouterr()
    assert (repo / ARTIFACT_PATH).read_bytes().endswith(b"}\n")


UNWALKABLE = {
    "union_of_two_models": "peer: Other | Third = Other()",
    "abstract_sequence_of_models": "peer: Sequence[Other] = []",
    "abstract_mapping_of_models": "peer: Mapping[str, Other] = {}",
    "bare_dict": "peer: dict = {}",
    "any": "peer: Any = None",
    "set_of_models": "peer: set[Other] = set()",
}


@pytest.mark.parametrize("name", sorted(UNWALKABLE))
def test_an_unwalkable_annotation_is_refused_rather_than_silently_skipped(
    name, repo, monkeypatch, capsys
):
    """Emitting nothing for an annotation this tool cannot walk is the
    fail-open the whole design exists to prevent: the artifact would look
    complete while covering less than the repository declared."""
    write(
        repo,
        "declared.py",
        f"""
        from typing import Any, Mapping, Sequence

        from pydantic import BaseModel, Field


        class Other(BaseModel):
            token: str = Field(default="", json_schema_extra={{"secret": True}})


        class Third(BaseModel):
            token: str = Field(default="", json_schema_extra={{"secret": True}})


        class Root(BaseModel):
            {UNWALKABLE[name]}
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert not (repo / ARTIFACT_PATH).exists()
    assert "declared:Root.peer" in capsys.readouterr().err


def test_a_marked_field_may_carry_an_annotation_that_cannot_be_walked(
    repo, monkeypatch, capsys
):
    """Coverage is prefix-based, so a marked field's annotation stops
    mattering — refusing one would refuse a declaration that is already
    complete."""
    write(
        repo,
        "declared.py",
        """
        from typing import Any

        from pydantic import BaseModel, Field


        class Root(BaseModel):
            anything: Any = Field(default=None, json_schema_extra={"secret": True})
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"]["anything"] == {"secret": True}


def test_an_annotation_that_cannot_hold_a_model_is_a_leaf(repo, monkeypatch, capsys):
    """`Sequence[str]` is refused only when it could carry a model. A
    container of scalars has nothing beneath it to declare, so it is not a
    refusal — otherwise the rule would be unusable."""
    write(
        repo,
        "declared.py",
        """
        from typing import Literal, Sequence

        from pydantic import BaseModel


        class Root(BaseModel):
            names: Sequence[str] = []
            mode: Literal["a", "b"] = "a"
            either: str | int = ""
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"] == {}


def test_a_graph_the_matcher_could_not_read_is_refused_before_it_is_written(
    repo, monkeypatch, capsys
):
    """The generator and the matcher are two halves of one schema running in
    two processes. Reading the artifact back through the matcher's own parser
    before writing it turns a drift between them into a failure where it was
    introduced, rather than a refusal in someone else's repository later."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    monkeypatch.chdir(repo)
    with pytest.raises(SyncError) as excinfo:
        build_artifact(
            repo,
            {
                "roots": {"app:app": "declared:NeverDefined"},
                "models": {},
                "files": [str(repo / "declared.py")],
                "external": [],
            },
        )
    assert "not readable" in str(excinfo.value)
    assert "undefined model" in str(excinfo.value)


def external_result(repo: Path) -> dict:
    """A generator result whose marks came partly from outside the repository
    — a base model or a marking alias imported from a dependency."""
    return {
        "roots": {},
        "models": {},
        "files": [str(repo / "declared.py")],
        "external": ["outside.package.conventions"],
    }


def test_an_out_of_repo_contributor_is_pinned_by_the_dependency_lockfile(repo):
    """`git show` cannot reach site-packages, so the lockfile that decides
    which version is installed is the closest thing to that module's identity
    the index holds."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(repo, "uv.lock", "# a lockfile\n")
    git(repo, "add", "-A")

    sources = build_artifact(repo, external_result(repo))["sources"]
    assert "uv.lock" in sources
    # Recorded by blob id, exactly like every other source.
    expected = git(repo, "rev-parse", ":uv.lock").stdout.strip()
    assert sources["uv.lock"] == expected


def test_every_tracked_lockfile_is_recorded_not_only_the_first(repo):
    """A repository can carry two. Pinning one while ignoring the other would
    leave a real dependency change invisible."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(repo, "uv.lock", "# a lockfile\n")
    write(repo, "requirements.txt", "somepackage==1.0\n")
    git(repo, "add", "-A")

    sources = build_artifact(repo, external_result(repo))["sources"]
    assert "uv.lock" in sources
    assert "requirements.txt" in sources


def test_an_untracked_lockfile_does_not_pin_an_out_of_repo_contributor(repo):
    """A lockfile on disk but not in the index has no blob id to compare
    against later, so it pins nothing and must not be treated as though it
    did."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(repo, "uv.lock", "# a lockfile\n")
    git(repo, "add", "declared.py")  # the lockfile is deliberately left out

    with pytest.raises(SyncError) as excinfo:
        build_artifact(repo, external_result(repo))
    assert "outside.package.conventions" in str(excinfo.value)


def test_an_out_of_repo_contributor_with_no_lockfile_is_refused(repo):
    """There would be nothing at all pinning that module's identity, so the
    artifact could never be found stale on account of it."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    git(repo, "add", "-A")

    with pytest.raises(SyncError) as excinfo:
        build_artifact(repo, external_result(repo))
    message = str(excinfo.value)
    assert "outside.package.conventions" in message
    assert "uv.lock" in message  # the message names what it looked for


def test_no_lockfile_is_recorded_when_nothing_came_from_outside_the_repo(repo):
    """The lockfile is evidence about a dependency, and a repository whose
    marks are all its own has no dependency to pin."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(repo, "uv.lock", "# a lockfile\n")
    git(repo, "add", "-A")

    sources = build_artifact(
        repo,
        {"roots": {}, "models": {}, "files": [str(repo / "declared.py")], "external": []},
    )["sources"]
    assert sources == {"declared.py": sources["declared.py"]}


def test_an_untracked_contributing_file_is_refused(repo, monkeypatch, capsys):
    """A file git does not track has no blob id, so its marks could change
    forever without the freshness check seeing anything."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", ".stackward.toml")
    assert sync(repo, monkeypatch) == 2
    assert "not tracked by git" in capsys.readouterr().err


def test_a_non_string_validation_alias_is_refused(repo, monkeypatch, capsys):
    """Without a determinable data key there is no position at which to apply
    coverage, so a mark on such a field would cover nothing at all."""
    write(
        repo,
        "declared.py",
        """
        from pydantic import AliasChoices, BaseModel, Field


        class Root(BaseModel):
            token: str = Field(
                default="",
                validation_alias=AliasChoices("token", "tok"),
                json_schema_extra={"secret": True},
            )
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert "validation_alias" in capsys.readouterr().err


def test_a_field_is_matched_under_its_string_alias(repo, monkeypatch, capsys):
    """Keying only by field name would miss an aliased field silently, which
    is a fail-open; the field is emitted under every key it could appear
    under."""
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel, Field


        class Root(BaseModel):
            token: str = Field(
                default="", alias="apiToken", json_schema_extra={"secret": True}
            )
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"]["apiToken"] == {"secret": True}
    assert written["models"]["declared:Root"]["token"] == {"secret": True}


def test_a_callable_json_schema_extra_is_refused(repo, monkeypatch, capsys):
    """Its result depends on a schema-generation context that does not exist
    here, so "cannot determine" must not become "not marked"."""
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel, Field


        def annotate(schema: dict) -> None:
            schema["secret"] = True


        class Root(BaseModel):
            token: str = Field(default="", json_schema_extra=annotate)
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert "callable" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `declared_paths_fn`: the escape hatch for a non-standard marking convention.
# ---------------------------------------------------------------------------


def test_a_declared_paths_fn_replaces_the_marking_convention(
    repo, monkeypatch, capsys
):
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel


        class Root(BaseModel):
            handled: str = ""
            ordinary: str = ""
        """,
    )
    write(
        repo,
        "convention.py",
        """
        def marks(model):
            return [name for name in model.model_fields if name == "handled"]
        """,
    )
    declare(repo, extra='declared_paths_fn = "convention:marks"')
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"] == {"handled": {"secret": True}}
    # The hook decides the marks, so its own module is a source too.
    assert "convention.py" in written["sources"]


def test_a_declared_paths_fn_returning_a_path_is_refused(repo, monkeypatch, capsys):
    """A hook returning flat paths would reintroduce the enumeration the
    graph replaced, so it is refused rather than interpreted."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(
        repo,
        "convention.py",
        """
        def marks(model):
            return ["children[0].token"]
        """,
    )
    declare(repo, extra='declared_paths_fn = "convention:marks"')
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert "field names, not paths" in capsys.readouterr().err


def test_a_declared_paths_fn_naming_an_unknown_field_is_refused(
    repo, monkeypatch, capsys
):
    """A typo that marked nothing would be indistinguishable from a field
    deliberately left unmarked."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(
        repo,
        "convention.py",
        """
        def marks(model):
            return ["toekn"]
        """,
    )
    declare(repo, extra='declared_paths_fn = "convention:marks"')
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert "not a field of this model" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The gate: what `check-config` does with the artifact it finds.
# ---------------------------------------------------------------------------


def prepare_gate(repo: Path, monkeypatch, capsys, config: str) -> None:
    """A repository whose models are declared, generated, and committed."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)
    write(repo, "Pulumi.dev.yaml", config)


def test_the_heuristic_net_alone_cannot_name_this_credential(
    repo, monkeypatch, capsys
):
    """Half of a pair, and the half that makes the other half mean something.

    With the model net switched off, this document is clean as far as the
    gate is concerned: `binding` matches no built-in key pattern. If this
    ever starts failing, the companion test below has stopped proving that
    the model net is what found it.
    """
    write(repo, ".stackward.toml", '[check]\nmodel_net = "none"\n')
    write(repo, "Pulumi.dev.yaml", BLIND_DOCUMENT)
    git(repo, "add", "-A")
    assert check(repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


def test_a_declared_secret_the_heuristic_cannot_name_is_found(
    repo, monkeypatch, capsys
):
    """The whole point of the second net: an unremarkable key name, at a depth
    no finite pattern list produced from this model would enumerate."""
    write(repo, "declared.py", HEURISTIC_BLIND_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)
    write(repo, "Pulumi.dev.yaml", BLIND_DOCUMENT)

    assert check(repo, monkeypatch) == 1
    out = capsys.readouterr().out
    assert (
        "Pulumi.dev.yaml: plaintext credential at "
        "'config.app:app.children[0].children[0].binding'" in out
    )
    assert "deeply-nested" not in out


def test_a_leaf_both_nets_name_is_reported_once(repo, monkeypatch, capsys):
    """`token` matches a built-in heuristic pattern *and* carries a mark. Two
    nets finding one leaf is one problem, not two."""
    prepare_gate(
        repo, monkeypatch, capsys, "config:\n  app:app:\n    token: value\n"
    )
    assert check(repo, monkeypatch) == 1
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "config.app:app.token" in line
    ]
    assert len(lines) == 1


def test_a_clean_config_under_a_declared_model_exits_0(repo, monkeypatch, capsys):
    prepare_gate(
        repo, monkeypatch, capsys, "config:\n  app:app:\n    plain: ordinary\n"
    )
    assert check(repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Freshness.
# ---------------------------------------------------------------------------


def test_a_stale_artifact_exits_2_not_1(repo, monkeypatch, capsys):
    """Exit 1 means "a credential was found". A refusal to run is a different
    answer and must not be filed under that one."""
    prepare_gate(
        repo, monkeypatch, capsys, "config:\n  app:app:\n    token: value\n"
    )
    write(repo, "declared.py", RECURSIVE_MODEL_EDITED)
    git(repo, "add", "declared.py")

    assert check(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert "declared.py has changed" in captured.err
    assert "sync-declared-secrets" in captured.err
    # And no finding was printed: the command does not know what was declared.
    assert captured.out == ""


def test_an_unstaged_regeneration_does_not_rescue_a_stale_artifact(
    repo, monkeypatch, capsys
):
    """The artifact is read from the index, not the working tree — otherwise
    a regeneration nobody staged would let the stale one that is actually
    about to be committed pass."""
    prepare_gate(
        repo, monkeypatch, capsys, "config:\n  app:app:\n    token: value\n"
    )
    write(repo, "declared.py", RECURSIVE_MODEL_EDITED)
    git(repo, "add", "declared.py")
    assert sync(repo, monkeypatch) == 0  # working tree only; never staged
    capsys.readouterr()

    assert check(repo, monkeypatch) == 2
    assert "sync-declared-secrets" in capsys.readouterr().err


def test_a_staged_regeneration_clears_the_staleness(repo, monkeypatch, capsys):
    """The other direction, so the test above is not passing for the trivial
    reason that nothing ever clears."""
    prepare_gate(
        repo, monkeypatch, capsys, "config:\n  app:app:\n    token: value\n"
    )
    write(repo, "declared.py", RECURSIVE_MODEL_EDITED)
    git(repo, "add", "declared.py")
    assert sync(repo, monkeypatch) == 0
    git(repo, "add", ARTIFACT_PATH)
    capsys.readouterr()

    assert check(repo, monkeypatch) == 1


def test_a_missing_artifact_under_artifact_mode_exits_2(repo, monkeypatch, capsys):
    """`"artifact"` never degrades to running without the net."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    plain: ordinary\n")
    git(repo, "add", "-A")

    assert check(repo, monkeypatch) == 2
    error = capsys.readouterr().err
    assert "is not in the git index" in error
    assert "sync-declared-secrets" in error


def test_a_malformed_artifact_in_the_index_exits_2(repo, monkeypatch, capsys):
    declare(repo)
    write(repo, ARTIFACT_PATH, "{not json")
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    plain: ordinary\n")
    git(repo, "add", "-A")

    assert check(repo, monkeypatch) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_none_mode_skips_cleanly_with_no_artifact_present(repo, monkeypatch, capsys):
    """A declared mode, not a fallback: nothing is read, so the absence of the
    artifact directory entirely cannot rescue a test that would otherwise
    have got lucky."""
    write(repo, ".stackward.toml", '[check]\nmodel_net = "none"\n')
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    plain: ordinary\n")
    git(repo, "add", "-A")
    assert not (repo / ".stackward").exists()

    assert check(repo, monkeypatch) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_none_mode_still_runs_the_heuristic_net(repo, monkeypatch, capsys):
    """Switching off the model net is not switching off the gate."""
    write(repo, ".stackward.toml", '[check]\nmodel_net = "none"\n')
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    dbPassword: value\n")
    git(repo, "add", "-A")
    assert check(repo, monkeypatch) == 1
    assert "config.app:app.dbPassword" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Global Constraint 1: pydantic is test-only.
# ---------------------------------------------------------------------------


def test_stackward_never_imports_pydantic():
    """The shipped binary must neither import nor bundle pydantic. Checked in
    a subprocess because this test process has imported it already — asserting
    against `sys.modules` here would assert nothing.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stackward.cli; "
            "import stackward.commands.sync_declared; "
            "print('pydantic' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "False"


def test_the_gate_path_reaches_no_credential_code():
    """`check-config` and `pre-commit` must never reach the credential store
    or `cryptography`, so a commit cannot be blocked by an expired session or
    a missing native extension. This net added an import to the gate path, so
    the property is worth asserting rather than assuming — checked in a
    subprocess, since this test process has imported half the tool already.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stackward.commands.check_config; "
            "import stackward.commands.pre_commit; "
            "print(sorted(m for m in ('cryptography', 'stackward.store') "
            "if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "[]"


def test_the_generator_is_read_through_the_accessor_the_shipped_code_uses():
    """Reading the file by path instead would assert nothing about a frozen
    build, where that path does not exist."""
    source = generator_source()
    assert "def generate(" in source
    assert "from pydantic import BaseModel" in source
