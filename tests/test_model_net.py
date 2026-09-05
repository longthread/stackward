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

import argparse
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from stackward import cli as cli_module
from stackward.nets import model as model_module
from stackward.bootstrap import generator_source
from stackward.cli import cmd_doctor
from stackward.commands import check_config as check_config_module
from stackward.commands import pre_commit as pre_commit_module
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


def test_a_marked_map_whose_only_key_is_literally_secure_is_an_accepted_false_negative():
    """The adversarial case the plan names, pinned as **accepted**, not fixed.

    A declared field may legitimately hold a mapping whose single key happens
    to be the word `secure` and whose value is an ordinary plaintext string.
    That is byte-for-byte the shape of Pulumi's encryption envelope, so
    `heuristic.is_encrypted` suppresses it and the credential goes unreported.

    This is a deliberate trade, not an oversight. The alternative is to decide
    what a ciphertext looks like -- a version prefix, a length, an alphabet --
    and every such rule reports *genuinely encrypted* values as findings the
    day Pulumi changes its envelope format. A gate that fires on values it has
    already succeeded in protecting is a gate that gets switched off, which
    costs more than this one blind spot does.

    So the test exists to say that out loud, and to fail if someone
    "fixes" it: tightening `is_encrypted` to inspect the wrapped value is
    exactly the change this docstring rules out, and the sibling test above
    (whose value reads like a real ciphertext) would not notice it.
    """
    document = {"config": {"app:app": {"token": {"secure": "not-really-encrypted"}}}}
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


@pytest.mark.parametrize("version", [ARTIFACT_VERSION + 1, ARTIFACT_VERSION - 1])
def test_an_artifact_version_this_matcher_does_not_read_is_refused(version):
    """Both directions refuse, and the older one is the one that matters in
    practice: an artifact written by a previous generator is fresh against
    every source it records and internally consistent, while quietly covering
    less than the repository declares. Only the version can say so."""
    message = parse_should_fail(
        {"version": version, "models": {}, "roots": {}, "sources": {"a": "b"}}
    )
    assert "unsupported version" in message
    assert "sync-declared-secrets" in message


def test_an_artifact_written_by_the_version_1_generator_is_refused():
    """The literal version, pinned rather than expressed relative to the
    constant -- a test written as `ARTIFACT_VERSION - 1` moves with the
    constant and would pass just as well if the bump were reverted.

    Version 1's generator answered "this cannot hold a model" for any class it
    did not recognise, so a pydantic dataclass or `TypedDict` holding a marked
    field left the graph with no refusal. Such an artifact is fresh against
    every source it records and internally consistent; nothing but the version
    can say it covers less than the repository declares.
    """
    message = parse_should_fail(
        {"version": 1, "models": {}, "roots": {}, "sources": {"a": "b"}}
    )
    assert "unsupported version 1" in message
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


def test_a_non_boolean_secret_mark_is_refused_rather_than_silently_unmarking():
    """`"secret": 0` where `false` belongs -- and, just as importantly, where
    `true` belonged.

    The artifact records no blob id for **itself** (see
    `sync_declared.build_artifact`: a self-reference would make every artifact
    stale the moment it was written), so a hand edit or a bad three-way merge
    in the committed JSON is checked by nothing except this parser. That makes
    every guard here load-bearing rather than defensive.

    This one in particular: `0` is falsy, so an accepted `"secret": 0` leaves
    the field unmarked and every credential at or beneath it unreported --
    while reading, in a diff, as a one-character change to a line that still
    says `secret`.
    """
    message = parse_should_fail(
        {
            "version": ARTIFACT_VERSION,
            "models": {"declared:Root": {"token": {"secret": 0}}},
            "roots": {"app:app": "declared:Root"},
            "sources": {"declared.py": "0" * 40},
        }
    )
    assert "must be a boolean" in message
    assert "token" in message


def test_a_null_container_interior_is_refused_rather_than_dropping_the_subtree():
    """`{"kind": "list", "item": null}`.

    The generator never writes this: a container that cannot hold a model is
    written as an ordinary scalar field, so a null interior can only come from
    an edit. Accepted, it produces `ListOf(None)` and the walk has nowhere to
    go -- every model below that field disappears from the net, silently, and
    the same document that used to be refused now reports clean.
    """
    message = parse_should_fail(
        {
            "version": ARTIFACT_VERSION,
            "models": {
                "declared:Root": {"peers": {"child": {"kind": "list", "item": None}}}
            },
            "roots": {},
            "sources": {"declared.py": "0" * 40},
        }
    )
    assert "must not be null" in message


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

# Two independently declarable roots, for the tests about what happens when
# `.stackward.toml` names a namespace the artifact does not cover.
TWO_MODELS = """
    from pydantic import BaseModel, Field


    class Root(BaseModel):
        binding: str = Field(default="", json_schema_extra={"secret": True})


    class Other(Root):
        pass
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


# A class whose fields the walk cannot reach, in each of the three shapes a
# repository actually writes one. None is a `BaseModel` subclass, none has a
# `get_origin`, and none is a bare container -- so each reaches the very last
# branch of `_may_hold_model`, which is where a blanket "no" would let a
# marked field leave the graph with no refusal at all.
STRUCTURED_LEAF = {
    "pydantic_dataclass": """
        from pydantic import Field
        from pydantic.dataclasses import dataclass


        @dataclass
        class Peer:
            handshake: str = Field(default="", json_schema_extra={"secret": True})
    """,
    "typed_dict": """
        from typing import Annotated, TypedDict

        from pydantic import Field


        class Peer(TypedDict):
            handshake: Annotated[str, Field(json_schema_extra={"secret": True})]
    """,
    "named_tuple": """
        from typing import NamedTuple


        class Peer(NamedTuple):
            handshake: str = ""
    """,
}
# A plain annotated class is deliberately absent: pydantic refuses to build a
# model with such a field at all (`PydanticSchemaGenerationError`), so the walk
# never sees it. It still fails closed -- with pydantic's message rather than
# this tool's -- which is why the rule in `_declares_fields_anywhere` is stated
# as a property rather than as this list.


@pytest.mark.parametrize("shape", sorted(STRUCTURED_LEAF))
def test_a_class_whose_fields_cannot_be_walked_is_refused(
    shape, repo, monkeypatch, capsys
):
    """The last branch of the walkability question must not answer "no".

    None of these is a `BaseModel`, a parameterised generic or a bare
    container, so each falls through every earlier branch to the final one.
    Answering "no" there emits an *empty* model and the field vanishes from
    the graph silently -- the whole category, not one annotation -- which is
    R1 inverted: an unanswerable question answered "no".
    """
    write(
        repo,
        "peers.py",
        textwrap.dedent(STRUCTURED_LEAF[shape]),
    )
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel

        from peers import Peer


        class Root(BaseModel):
            peer: Peer
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert not (repo / ARTIFACT_PATH).exists()
    assert "declared:Root.peer" in capsys.readouterr().err


def test_a_scalar_class_is_still_a_leaf(repo, monkeypatch, capsys):
    """The other side of the same rule, and the reason it is about *public*
    annotations rather than any annotations at all.

    `Enum`, `Decimal`, `UUID` and `datetime` declare nothing; pydantic's own
    scalar wrappers declare only underscore-prefixed private attributes, which
    pydantic itself does not collect as fields. Refusing these would make the
    rule unusable, and the remedy for them ("narrow the annotation") does not
    even exist.
    """
    write(
        repo,
        "declared.py",
        """
        import datetime
        import decimal
        import enum
        import uuid

        from pydantic import AnyUrl, BaseModel, SecretStr


        class Mode(enum.Enum):
            FAST = "fast"


        class Root(BaseModel):
            mode: Mode = Mode.FAST
            amount: decimal.Decimal = decimal.Decimal(0)
            ident: uuid.UUID = uuid.UUID(int=0)
            when: datetime.datetime = datetime.datetime(2000, 1, 1)
            endpoint: AnyUrl = AnyUrl("https://example.invalid")
            hidden: SecretStr = SecretStr("")
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"] == {}


def test_an_out_of_repo_base_declaring_no_fields_needs_no_lockfile(
    repo, monkeypatch, capsys
):
    """The shortcut the MRO fix must not take.

    Recording *every* base except `object` and `BaseModel` would make
    `class Config(BaseModel, ABC)` record `abc` as an out-of-repo contributor
    and turn a tracked lockfile into a universal precondition -- the exact
    cost the `BaseModel` exclusion exists to avoid. `ABC` declares no fields,
    so it contributes nothing and is not recorded. No lockfile exists in this
    repository, so generation would refuse if it were.
    """
    write(
        repo,
        "declared.py",
        """
        from abc import ABC

        from pydantic import BaseModel, Field


        class Root(BaseModel, ABC):
            binding: str = Field(default="", json_schema_extra={"secret": True})
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert set(written["sources"]) == {"declared.py", ".stackward.toml"}


# The same two colliding fields in both declaration orders. `plain` is an
# unmarked scalar, which emits no entry of its own; `token` is marked and
# aliased onto that same data key.
KEY_COLLISION = {
    "unmarked_field_first": """
        from pydantic import BaseModel, Field


        class Root(BaseModel):
            plain: str = ""
            token: str = Field(
                default="", alias="plain", json_schema_extra={"secret": True}
            )
    """,
    "aliased_field_first": """
        from pydantic import BaseModel, Field


        class Root(BaseModel):
            token: str = Field(
                default="", alias="plain", json_schema_extra={"secret": True}
            )
            plain: str = ""
    """,
}


@pytest.mark.parametrize("order", sorted(KEY_COLLISION))
def test_a_data_key_claimed_by_two_fields_is_refused_in_either_order(
    order, repo, monkeypatch, capsys
):
    """The refusal must not depend on declaration order.

    An unmarked scalar emits no entry, so a check keyed on *emitted* entries
    never saw it claim its key: with the unmarked field first, the marked one
    took that key silently, while the same two fields written the other way
    round did refuse. Both orders are asserted, because either alone passes
    against the order-dependent version.
    """
    write(repo, "declared.py", textwrap.dedent(KEY_COLLISION[order]))
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert "claimed by more than one field" in capsys.readouterr().err


def test_a_mark_inherited_from_a_plain_mixin_records_the_mixins_file(
    repo, monkeypatch, capsys
):
    """pydantic collects a marked field from a base that is not itself a
    `BaseModel`, so a source filter keyed on `issubclass(base, BaseModel)`
    takes the mark and discards the file it came from. The mixin is tracked
    and already contributing when the artifact is written, so this is not the
    "newly added file" boundary -- the MRO walk sees it and must record it.
    """
    write(
        repo,
        "mixin_mod.py",
        """
        from pydantic import Field


        class Marks:
            binding: str = Field(default="", json_schema_extra={"secret": True})
        """,
    )
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel

        from mixin_mod import Marks


        class Root(Marks, BaseModel):
            plain: str = ""
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert written["models"]["declared:Root"]["binding"] == {"secret": True}
    assert "mixin_mod.py" in written["sources"]


def test_editing_a_plain_mixin_makes_the_artifact_stale(repo, monkeypatch, capsys):
    """The consequence of the test above, and the thing that actually fails
    open without it: a mark changing in the mixin with no refusal."""
    write(
        repo,
        "mixin_mod.py",
        """
        from pydantic import Field


        class Marks:
            binding: str = Field(default="", json_schema_extra={"secret": True})
        """,
    )
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel

        from mixin_mod import Marks


        class Root(Marks, BaseModel):
            plain: str = ""
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)
    write(
        repo,
        "mixin_mod.py",
        """
        from pydantic import Field


        class Marks:
            binding: str = Field(default="", json_schema_extra={"secret": True})
            added_later: str = Field(default="", json_schema_extra={"secret": True})
        """,
    )
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    added_later: leaked\n")
    git(repo, "add", "-A")

    assert check(repo, monkeypatch) == 2
    assert "mixin_mod.py has changed" in capsys.readouterr().err


def test_the_repository_config_is_a_recorded_source(repo, monkeypatch, capsys):
    """`.stackward.toml` supplies `roots` and `declared_paths_fn`, so it
    decides part of the artifact's content as surely as any model file does.
    Leaving it out means a declaration can be added to it and never checked.
    """
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)
    assert ".stackward.toml" in written["sources"]


def test_adding_a_namespace_to_the_config_makes_the_artifact_stale(
    repo, monkeypatch, capsys
):
    """The fail-open the recorded config file closes: a second namespace
    declared in `.stackward.toml` and staged, with the artifact untouched, so
    the namespace is simply skipped and its credential is never looked at."""
    write(repo, "declared.py", TWO_MODELS)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)

    write(
        repo,
        ".stackward.toml",
        f"""
        python = "{sys.executable}"

        [check]
        model_net = "artifact"
        stack_models = {{ "app:app" = "declared:Root", "app:other" = "declared:Other" }}
        """,
    )
    write(repo, "Pulumi.dev.yaml", "config:\n  app:other:\n    binding: leaked\n")
    git(repo, "add", "-A")

    assert check(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert "sync-declared-secrets" in captured.err
    # Nothing reported: before the fix this exited 0, having silently skipped
    # the namespace the policy declared and the artifact did not cover.
    assert captured.out == ""


def test_a_root_the_config_declares_and_the_artifact_lacks_is_refused(
    repo, monkeypatch, capsys
):
    """The second, independent guard on the same class of drift, needing no
    blob id: the namespaces the artifact starts from must be exactly the ones
    the policy declares. Key sets only -- a model id is `module:QualName` and
    diverges from the config's `module:Class` target for a nested class."""
    net, _sources = parse_net(
        {
            "version": ARTIFACT_VERSION,
            "models": {"declared:Root": {"token": {"secret": True}}},
            "roots": {"app:app": "declared:Root"},
            "sources": {"declared.py": "0" * 40},
        }
    )
    monkeypatch.setattr(model_module, "repo_toplevel", lambda: repo)
    monkeypatch.setattr(model_module, "read_artifact_from_index", lambda _root: {})
    monkeypatch.setattr(model_module, "parse_net", lambda _payload: (net, {}))
    monkeypatch.setattr(model_module, "verify_sources", lambda _root, _sources: None)

    with pytest.raises(ModelNetError) as excinfo:
        model_module.load_model_net(
            CheckConfig(
                model_net="artifact",
                stack_models={"app:app": "declared:Root", "app:other": "declared:Other"},
            )
        )
    assert "app:other" in str(excinfo.value)


def test_a_root_model_is_refused_rather_than_covering_nothing(
    repo, monkeypatch, capsys
):
    """A RootModel appears in a document as its inner value, with no `root`
    key, so walking it as an ordinary model would look up a field name that
    can never occur and produce no coverage while saying nothing about it.
    By R1's own principle, unknowable coverage refuses."""
    write(
        repo,
        "declared.py",
        """
        from pydantic import BaseModel, Field, RootModel


        class Wrapped(RootModel[str]):
            root: str = Field(default="", json_schema_extra={"secret": True})


        class Root(BaseModel):
            wrapped: Wrapped = Wrapped(root="")
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert not (repo / ARTIFACT_PATH).exists()
    error = capsys.readouterr().err
    assert "declared:Wrapped" in error
    assert "RootModel" in error


def test_a_root_model_declared_as_a_stack_root_is_refused_too(
    repo, monkeypatch, capsys
):
    """The refusal lives where every model passes through, so a root model
    reached as a declared root and one reached through a field are refused by
    the same line rather than by two that could drift apart."""
    write(
        repo,
        "declared.py",
        """
        from pydantic import RootModel


        class Root(RootModel[dict[str, str]]):
            pass
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    assert "RootModel" in capsys.readouterr().err


def test_a_refusal_names_the_field_the_class_the_annotation_and_the_way_out(
    repo, monkeypatch, capsys
):
    """A refusal a reader cannot act on is a refusal that gets silenced with
    `model_net = "none"`, which loses the whole net -- and first adoption is
    exactly when several of these arrive at once."""
    write(
        repo,
        "declared.py",
        """
        from typing import Any

        from pydantic import BaseModel


        class Root(BaseModel):
            settings: dict[str, Any] = {}
        """,
    )
    declare(repo)
    git(repo, "add", "-A")
    assert sync(repo, monkeypatch) == 2
    error = capsys.readouterr().err
    assert "settings" in error                    # the field
    assert "declared:Root" in error               # the class
    # The annotation *as written*. Asserting only that "Any" appears would
    # pass against a message that reported the inner type alone and sent the
    # reader looking for a line saying `Any`, which their source does not have.
    assert "dict[str, " in error and "Any]" in error
    assert "narrow the annotation" in error       # remedy 1
    assert "json_schema_extra" in error           # remedy 2
    assert "declared_paths_fn" in error           # remedy 3


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
            repo / ".stackward.toml",
        )
    assert "not readable" in str(excinfo.value)
    assert "undefined model" in str(excinfo.value)


def CONFIG(repo: Path) -> Path:
    """`.stackward.toml`, which `build_artifact` records as a source like any
    other -- see `test_the_repository_config_is_a_recorded_source`."""
    return repo / ".stackward.toml"


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
    declare(repo)
    git(repo, "add", "-A")

    sources = build_artifact(repo, external_result(repo), CONFIG(repo))["sources"]
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
    declare(repo)
    git(repo, "add", "-A")

    sources = build_artifact(repo, external_result(repo), CONFIG(repo))["sources"]
    assert "uv.lock" in sources
    assert "requirements.txt" in sources


def test_an_untracked_lockfile_does_not_pin_an_out_of_repo_contributor(repo):
    """A lockfile on disk but not in the index has no blob id to compare
    against later, so it pins nothing and must not be treated as though it
    did."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(repo, "uv.lock", "# a lockfile\n")
    declare(repo)
    git(repo, "add", "declared.py", ".stackward.toml")  # lockfile left out

    with pytest.raises(SyncError) as excinfo:
        build_artifact(repo, external_result(repo), CONFIG(repo))
    assert "outside.package.conventions" in str(excinfo.value)


def test_an_out_of_repo_contributor_with_no_lockfile_is_refused(repo):
    """There would be nothing at all pinning that module's identity, so the
    artifact could never be found stale on account of it."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")

    with pytest.raises(SyncError) as excinfo:
        build_artifact(repo, external_result(repo), CONFIG(repo))
    message = str(excinfo.value)
    assert "outside.package.conventions" in message
    assert "uv.lock" in message  # the message names what it looked for


def test_no_lockfile_is_recorded_when_nothing_came_from_outside_the_repo(repo):
    """The lockfile is evidence about a dependency, and a repository whose
    marks are all its own has no dependency to pin."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    write(repo, "uv.lock", "# a lockfile\n")
    declare(repo)
    git(repo, "add", "-A")

    sources = build_artifact(
        repo,
        {"roots": {}, "models": {}, "files": [str(repo / "declared.py")], "external": []},
        CONFIG(repo),
    )["sources"]
    assert set(sources) == {"declared.py", ".stackward.toml"}


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


def test_a_recorded_source_removed_from_the_index_is_refused_and_named(
    repo, monkeypatch, capsys
):
    """The sibling of "blob changed": a recorded source that is not in the
    index at all.

    Both branches of `verify_sources` end in exit 2 naming the regeneration
    command, so an exit-code assertion cannot tell them apart -- and with the
    `None` branch removed the comparison below it fires instead, reporting a
    file as *changed* when it has actually been deleted or unstaged. So the
    wording is what this test asserts: the reader is told which of the two
    things happened, because the fix differs.
    """
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)

    write(repo, "Pulumi.dev.yaml", "name: myproject\n")
    git(repo, "add", "Pulumi.dev.yaml")
    git(repo, "rm", "-q", "--cached", "declared.py")

    assert check(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert "declared.py" in captured.err
    assert "no longer in the index" in captured.err
    assert "sync-declared-secrets" in captured.err
    assert captured.out == ""


def test_an_empty_stack_models_is_refused_rather_than_covering_nothing(
    repo, monkeypatch, capsys
):
    """`model_net = "artifact"` with nothing declared to walk.

    Without this refusal the whole chain reads as success: `sync` writes an
    artifact whose `roots` is `{}`, `_verify_roots` compares `set() == set()`
    and passes, and `check-config` exits 0 having scanned with a net that
    covers nothing -- a silent fallback to no model net at all, in a
    repository that explicitly asked for one. That is the single outcome this
    tool's fail-closed rule exists to forbid, and it is invisible.

    The artifact's *absence*, not the exit code, is what proves the refusal
    happened before anything was written: a run that generated an empty net
    and then failed for some other reason would exit 2 as well.
    """
    write(repo, "declared.py", HEURISTIC_BLIND_MODEL)
    write(
        repo,
        ".stackward.toml",
        f"""
        python = "{sys.executable}"

        [check]
        model_net = "artifact"
        stack_models = {{}}
        """,
    )
    git(repo, "add", "-A")

    assert sync(repo, monkeypatch) == 2
    assert not (repo / ARTIFACT_PATH).exists()
    error = capsys.readouterr().err
    assert "stack_models is empty" in error
    assert 'model_net = "none"' in error

    # And the consequence, end to end, on a credential only the model net
    # could ever name: with no artifact there is nothing to scan with, and
    # `check-config` says so instead of reporting clean.
    write(repo, "Pulumi.dev.yaml", BLIND_DOCUMENT)
    git(repo, "add", "-A")
    assert check(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert ARTIFACT_PATH in captured.err
    assert captured.out == ""


def test_a_hand_edited_artifact_that_drops_a_root_is_refused_by_the_roots_check(
    repo, monkeypatch, capsys
):
    """The scenario `_verify_roots` is the only remaining defence for.

    Ordinarily it is unreachable: `build_artifact` records `.stackward.toml`
    as a source, so any change to `check.stack_models` trips the blob-id check
    in `verify_sources` first -- which is exactly why `return` as this guard's
    first statement costs nothing but its own unit test.

    The artifact records no blob id for itself, though, so a hand edit or a
    bad merge in the committed JSON meets nothing but the parser. Drop
    `.stackward.toml` from `sources` *and* a namespace from `roots`, and
    `verify_sources` has nothing left to disagree with: every source still
    recorded matches the index exactly. Only the roots comparison can still
    say the artifact covers less than the repository declares -- and the
    credential in the dropped namespace is one the heuristic net cannot name.
    """
    write(repo, "declared.py", TWO_MODELS)
    write(
        repo,
        ".stackward.toml",
        f"""
        python = "{sys.executable}"

        [check]
        model_net = "artifact"
        stack_models = {{ "app:app" = "declared:Root", "app:other" = "declared:Other" }}
        """,
    )
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)

    edited = artifact(repo)
    del edited["sources"][".stackward.toml"]
    del edited["roots"]["app:other"]
    (repo / ARTIFACT_PATH).write_text(json.dumps(edited, sort_keys=True, indent=2) + "\n")
    write(repo, "Pulumi.dev.yaml", "config:\n  app:other:\n    binding: leaked\n")
    git(repo, "add", "-A")

    assert check(repo, monkeypatch) == 2
    captured = capsys.readouterr()
    assert "app:other" in captured.err
    assert "sync-declared-secrets" in captured.err
    # Nothing reported: without this guard the namespace is simply skipped and
    # its credential never looked at, and the command exits 0.
    assert captured.out == ""


def test_a_root_declared_through_a_module_level_alias_still_verifies(
    repo, monkeypatch, capsys
):
    """Why `_verify_roots` compares **key sets** and not `(namespace, model)`
    pairs.

    A root's model id is `module:QualName` -- what the class says about itself
    -- while `check.stack_models` names whatever attribute the operator
    actually imported it as. A module-level alias, or a class re-exported from
    a package's `__init__`, makes the two diverge for a completely correct
    artifact. Comparing pairs would refuse it, and would call it staleness,
    which is not what happened; regenerating would produce the identical file
    and the refusal would repeat forever.

    (The docstring here used to justify the same choice by nested classes.
    That case is unreachable: `bootstrap.regen._import_object` resolves a
    target with a flat `getattr`, so `declared:Outer.Inner` is refused at sync
    time and never reaches an artifact. An alias is the form that does occur.)
    """
    write(repo, "declared.py", TWO_MODELS + "\n    Alias = Root\n")
    declare(repo, models="declared:Alias")
    git(repo, "add", "-A")
    written = sync_and_commit(repo, monkeypatch, capsys)

    # The divergence itself, made explicit: the config asked for `Alias`, the
    # artifact records what the class calls itself.
    assert written["roots"] == {"app:app": "declared:Root"}

    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    binding: \"\"\n")
    assert check(repo, monkeypatch) == 0
    assert capsys.readouterr().out == ""


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
# The hook: the layer that actually prevents publication.
#
# `check-config` invoked by hand reports a leak that may already be committed;
# only `pre-commit` stops one entering history. A model net that ran only on
# manual invocation would be absent at the one moment it counts, so these
# drive a real `git commit` rather than asserting on command output.
# ---------------------------------------------------------------------------


def install_hook(repo: Path, monkeypatch) -> None:
    monkeypatch.chdir(repo)
    assert main(["hooks", "install"]) == 0


def attempt_commit(repo: Path, message: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "commit", "-m", message], cwd=repo, capture_output=True, text=True
    )


def declared_repo(repo: Path, monkeypatch, capsys, model: str) -> None:
    """A repository with `model` declared, generated and committed, and the
    gate installed as a real git hook."""
    write(repo, "declared.py", model)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)
    install_hook(repo, monkeypatch)


def test_a_model_net_only_finding_blocks_a_real_commit(
    repo, monkeypatch, capsys, real_executable
):
    """The headline claim of wiring the model net into the hook.

    `binding` matches none of the heuristic net's built-in key patterns, and
    it sits two levels down a recursive model. Its companion below proves the
    same commit succeeds with the model net switched off, so this test cannot
    be passing because something else blocked the commit.
    """
    declared_repo(repo, monkeypatch, capsys, HEURISTIC_BLIND_MODEL)
    write(repo, "Pulumi.dev.yaml", BLIND_DOCUMENT)
    git(repo, "add", "Pulumi.dev.yaml")

    result = attempt_commit(repo, "add a declared secret")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "config.app:app.children[0].children[0].binding" in output
    assert "deeply-nested" not in output


def test_the_same_commit_succeeds_with_the_model_net_switched_off(
    repo, monkeypatch, capsys, real_executable
):
    """The other half of the pair. Without it, the test above would pass just
    as well if the hook blocked every commit."""
    declared_repo(repo, monkeypatch, capsys, HEURISTIC_BLIND_MODEL)
    write(repo, ".stackward.toml", '[check]\nmodel_net = "none"\n')
    write(repo, "Pulumi.dev.yaml", BLIND_DOCUMENT)
    git(repo, "add", ".stackward.toml", "Pulumi.dev.yaml")

    assert attempt_commit(repo, "add the same document").returncode == 0


def test_a_stale_artifact_blocks_a_real_commit(
    repo, monkeypatch, capsys, real_executable
):
    """The hook reads the artifact from the index and refuses when it no
    longer matches its sources -- it does not fall back to the heuristic net
    alone, which would be a silent downgrade at commit time."""
    declared_repo(repo, monkeypatch, capsys, RECURSIVE_MODEL)
    write(repo, "declared.py", RECURSIVE_MODEL_EDITED)
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    plain: ordinary\n")
    git(repo, "add", "declared.py", "Pulumi.dev.yaml")

    result = attempt_commit(repo, "change a model without regenerating")
    assert result.returncode != 0
    assert "sync-declared-secrets" in result.stdout + result.stderr


def test_the_hook_exits_2_on_a_stale_artifact_not_1(repo, monkeypatch, capsys):
    """Driven in-process, because a `git commit` reports only "the hook said
    no" and cannot distinguish the two codes -- and the distinction is the
    whole reason exit 2 exists."""
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)
    write(repo, "declared.py", RECURSIVE_MODEL_EDITED)
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    token: value\n")
    git(repo, "add", "declared.py", "Pulumi.dev.yaml")

    monkeypatch.chdir(repo)
    assert main(["pre-commit"]) == 2
    captured = capsys.readouterr()
    assert "sync-declared-secrets" in captured.err
    # Nothing reported: the command does not know what the repository declared.
    assert captured.out == ""


def test_the_hook_reports_a_leaf_both_nets_name_once(repo, monkeypatch, capsys):
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)
    write(repo, "Pulumi.dev.yaml", "config:\n  app:app:\n    token: value\n")
    git(repo, "add", "Pulumi.dev.yaml")

    monkeypatch.chdir(repo)
    assert main(["pre-commit"]) == 1
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "plaintext credential at" in line
    ]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# Talking to git.
# ---------------------------------------------------------------------------


def test_every_git_invocation_runs_under_a_fixed_locale(repo, monkeypatch, capsys):
    """`nets.model.run_git` is the third git call site in this tool, and the
    last one without a fixed locale -- `commands.pre_commit` and
    `commands.install_hooks` have theirs.

    Hardening rather than a live fix here: every caller in this module
    branches on `returncode`, never on git's text. But git's diagnostics are
    gettext-marked, and this module quotes git's stderr verbatim into a
    `ModelNetError` a reader is expected to act on.

    **The merge is the half that matters.** git exports `GIT_DIR` and
    `GIT_INDEX_FILE` to a hook process, and this module's `ls-files`/`show`
    calls are what read the index the commit is being gated on. A bare
    `env={"LC_ALL": "C"}` would silently point them at a different index than
    `pre-commit` is checking -- a fail-open that no assertion about output
    could catch, which is why the ambient marker is asserted alongside the
    locale.

    Driven through `check-config` under `model_net = "artifact"` rather than
    by calling `run_git` directly, so it covers every invocation this module
    makes -- `rev-parse`, `ls-files` and `show` -- rather than the one a
    direct call happens to pick. `check-config` shells out to nothing but
    git, so every recorded call came from this module.
    """
    write(repo, "declared.py", RECURSIVE_MODEL)
    declare(repo)
    git(repo, "add", "-A")
    sync_and_commit(repo, monkeypatch, capsys)
    write(repo, "Pulumi.dev.yaml", "name: myproject\n")

    # Recording starts only now: this file's own `git` helper shells out too,
    # and it is not what is under test.
    monkeypatch.setenv("STACKWARD_TEST_MARKER", "inherited")
    seen: list[dict[str, str] | None] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        if args and args[0] == "git":
            seen.append(kwargs.get("env"))
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    assert check(repo, monkeypatch) == 0
    capsys.readouterr()

    assert seen, "no git invocation was recorded"
    for env in seen:
        assert env is not None, "a git invocation inherited the ambient locale"
        assert env.get("LC_ALL") == "C"
        assert env.get("STACKWARD_TEST_MARKER") == "inherited"


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


@pytest.mark.parametrize(
    "entry_point",
    [check_config_module.scan_file, pre_commit_module._scan_staged_config],
)
def test_neither_scan_entry_point_makes_the_model_net_optional(entry_point):
    """`net` must have no default on either scanner.

    A default would make the model net silently opt-in: a future caller that
    simply forgot the argument would scan with the heuristic net alone and
    report clean, which is the failure this whole net exists to prevent. The
    property is a signature, so it is asserted as one -- there is no
    behaviour to observe, because the whole point is what happens to a call
    site that does not yet exist.
    """
    parameter = inspect.signature(entry_point).parameters["net"]
    assert parameter.default is inspect.Parameter.empty


# The gate-path invariant this net's own imports could have broken — that
# `check-config` and `pre-commit` reach no credential-store or `cryptography`
# code — lives in `tests/test_gate_isolation.py`. It used to live here, and
# it was checked by importing the two command modules *directly*: that
# bypassed `cli.py`, which imported `commands.check_passphrase` and
# `commands.session` at module scope and so pulled `cryptography` and
# `stackward.store` onto the real gate path with this test still green. The
# replacement runs `cli.main(["check-config", ...])`, the path a `git commit`
# actually takes, and asserts the union of every forbidden module.


def test_doctor_reports_the_model_walker_as_readable(capsys):
    """`regen.py` is carried into the frozen bundle by an explicit
    `--add-data` entry rather than by import analysis, so dropping that entry
    produces a binary that builds, starts, and then fails only at
    `sync-declared-secrets`. `doctor` is the release smoke test, so it checks
    the real accessor -- and the release workflow greps for this exact line,
    which is why the prefix is asserted verbatim rather than loosely.
    """
    assert cmd_doctor(argparse.Namespace()) == 0
    reported = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("model walker    regen.py readable")
    ]
    assert len(reported) == 1


def test_doctor_fails_when_the_model_walker_cannot_be_read(monkeypatch, capsys):
    """The check has to discriminate, or the release workflow's grep is
    decoration. Verified against a real frozen bundle built without the
    `--add-data` entry as well; this is the version that runs every time."""
    monkeypatch.setattr(
        cli_module,
        "generator_source",
        lambda: (_ for _ in ()).throw(FileNotFoundError("regen.py")),
    )
    assert cmd_doctor(argparse.Namespace()) == 1
    assert "model walker    UNAVAILABLE" in capsys.readouterr().out


def test_doctor_fails_when_the_bundled_file_is_not_the_generator(
    monkeypatch, capsys
):
    """Present but wrong is a distinct failure from absent, and it would
    otherwise read as success."""
    monkeypatch.setattr(cli_module, "generator_source", lambda: "# not it\n")
    assert cmd_doctor(argparse.Namespace()) == 1
    assert "model walker    FAILED" in capsys.readouterr().out


def test_the_generator_is_read_through_the_accessor_the_shipped_code_uses():
    """Reading the file by path instead would assert nothing about a frozen
    build, where that path does not exist."""
    source = generator_source()
    assert "def generate(" in source
    assert "from pydantic import BaseModel" in source
