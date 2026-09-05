"""Generate the declared-secrets graph. Runs under the *repository's*
interpreter, never under `stackward`'s own.

This program is never imported by `stackward`. `commands.sync_declared` reads
it as text (see `bootstrap.generator_source`) and pipes it to the interpreter
named by `[python]` in `.stackward.toml`, which is the only interpreter that
can import the repository's models. That is what keeps pydantic a test-only
dependency of `stackward` itself: the half that needs pydantic runs somewhere
else, and only JSON crosses back.

It reads a request as JSON in `argv[1]` and writes the graph as JSON to the
file that request names (see `main` for why not stdout). It resolves no
credentials, reads no secret store and contacts nothing — it imports declared
classes and inspects their fields.

**It refuses rather than emitting nothing.** Silently skipping an annotation
this program cannot walk is the fail-open the whole model-net design exists
to prevent: the artifact would look complete while covering less than the
repository declared. So an *unmarked* field whose annotation could hold a
model but is not one of the walkable forms is an error naming the field and
the two ways to clear it. A *marked* field's annotation is never examined at
all — coverage is prefix-based, so everything beneath a mark is covered
whatever its type.

The walkable forms are exactly: a `BaseModel` subclass, `list[X]`,
`dict[str, X]`, `Optional[X]` / `X | None`, and any annotation that provably
cannot hold a model (`str`, `int`, an `Enum`, `Literal[...]`, `list[str]`,
`Sequence[str]`, and so on — these become leaves, with nothing beneath them
to declare). Everything else is refused: `Union[A, B]`, `Sequence[A]`,
`Mapping[str, A]`, `set[A]`, `tuple[A, ...]`, a bare `dict` or `list`, `Any`
and `object` — the last four because "could this hold a model?" has no
answer for them, and an unanswerable question must not be answered "no".
"""

from __future__ import annotations

import inspect
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any, NoReturn

# Containers written without parameters. Whether one holds a model is not
# knowable from the annotation, so an unmarked field annotated with one is
# refused rather than guessed at.
_BARE_CONTAINERS = (dict, list, set, frozenset, tuple)

# Union spellings: `typing.Union[...]` and PEP 604's `X | Y`, which
# `get_origin` reports as `types.UnionType` rather than `typing.Union`.
_UNION_ORIGINS = (typing.Union, types.UnionType)


class Unwalkable(Exception):
    """An annotation, alias form or model this program refuses to guess at."""


def _die(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _strip_annotated(annotation: Any) -> Any:
    """`Annotated[X, ...]` down to `X`.

    pydantic already strips the top-level `Annotated` off `FieldInfo.annotation`
    (which is how a mark carried by an alias defined in another module lands
    on the field at all), but a nested one — `list[Annotated[X, ...]]` — is
    still there, so the walk strips as it descends rather than once at entry.
    """
    while hasattr(annotation, "__metadata__") and hasattr(annotation, "__origin__"):
        annotation = annotation.__origin__
    return annotation


def _is_model(annotation: Any) -> bool:
    from pydantic import BaseModel

    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _is_root_model(model: type) -> bool:
    """A model whose whole value *is* its single `root` field.

    Refused, because its coverage is unknowable rather than empty. In the
    document such a model appears as the inner value directly, with no `root`
    key — so walking it as an ordinary model would look up field names that
    can never occur, produce no coverage at all, and say nothing about having
    done so. That is exactly the silent under-coverage this artifact exists to
    eliminate, so it gets the same treatment as an unwalkable annotation.

    Walking one correctly is not the small change it looks. A mark on the
    `root` field means the whole value at the *parent* position is covered,
    and there is no way to say that in this schema when the model is reached
    through a `list[...]` or `dict[str, ...]` — the mark would have to
    propagate through a container node, which carries no `secret` of its own.
    Refusing is the honest answer until the schema can express it.

    Imported defensively: `RootModel` exists in every pydantic 2.x, but a
    generator that crashed on an unexpected import would fail *less* clearly
    than one that reports what it could not decide.
    """
    try:
        from pydantic import RootModel
    except ImportError:  # pragma: no cover - pydantic 2.x always has it
        return False
    return isinstance(model, type) and issubclass(model, RootModel)


def _declares_fields(klass: type) -> bool:
    """Does `klass` declare fields of its own?

    True when it carries at least one annotation of its own whose name does
    not start with an underscore. That is pydantic's own rule, not an
    approximation of it: an underscore-prefixed annotated name is a *private
    attribute*, which pydantic never collects as a field, so it can never
    carry a mark under this tool's convention either.

    The distinction is what separates a structure from a scalar without
    enumerating either. A pydantic dataclass, a `TypedDict`, a `NamedTuple`
    and a plain annotated mixin all declare public annotations; `str`, an
    `Enum`, `Decimal`, `UUID`, `datetime` and `abc.ABC` declare none, and
    pydantic's own scalar wrappers (`AnyUrl`, `SecretStr`) declare only
    private ones. Refusing that second group would make the rule unusable —
    "narrow the annotation" is not even available for `AnyUrl`.

    Read from `__dict__` rather than the attribute: since Python 3.10,
    `klass.__annotations__` lazily *creates* an empty dict on a class that
    has none, and on some builtins raises instead.

    Known conservative case, stated rather than hidden: a class whose only
    annotations are `ClassVar` declares no fields to pydantic but does to
    this rule, so it refuses. That fails closed with an actionable message,
    and detecting `ClassVar` through a possibly-stringised annotation would
    be a guess where this is a fact.
    """
    annotations = klass.__dict__.get("__annotations__")
    if not annotations:
        return False
    return any(not name.startswith("_") for name in annotations)


def _declares_fields_anywhere(klass: type) -> bool:
    """`_declares_fields` over the whole MRO — a class inherits its shape."""
    for base in getattr(klass, "__mro__", (klass,)):
        if base is not object and _declares_fields(base):
            return True
    return False


def _model_id(model: type) -> str:
    """`module:QualName` — stable, readable, and meaningful in a diff of the
    committed artifact, which synthetic indices would not be."""
    return f"{model.__module__}:{model.__qualname__}"


def _may_hold_model(annotation: Any) -> bool:
    """Could a value described by `annotation` contain a declared model?

    True also when the answer is *unknown* — `Any`, `object`, a bare
    container, an unresolved forward reference, an annotation form this
    program does not recognise. That direction is the whole point: an
    unanswerable question is answered "yes, refuse", never "no, emit
    nothing".
    """
    annotation = _strip_annotated(annotation)
    if annotation is None or annotation is type(None) or annotation is Ellipsis:
        return False
    if annotation is Any or annotation is object:
        return True
    if isinstance(annotation, (str, typing.ForwardRef)):
        # An unresolved forward reference: the model was never rebuilt, so
        # what it refers to cannot be inspected here.
        return True
    if _is_model(annotation):
        return True
    if annotation in _BARE_CONTAINERS:
        return True

    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        # Literal's arguments are *values*, not types; none can be a model.
        return False
    args = typing.get_args(annotation)
    if origin is not None:
        # A parameterised generic: recurse. An unparameterised one
        # (`typing.Dict`) has an origin but no arguments, and is as unknowable
        # as a bare `dict`.
        return True if not args else any(_may_hold_model(arg) for arg in args)
    if isinstance(annotation, (list, tuple)):
        # `Callable[[A], B]` carries its parameter types in a list.
        return any(_may_hold_model(item) for item in annotation)
    if isinstance(annotation, type):
        # The last branch, and the one a blanket `False` gets catastrophically
        # wrong. A pydantic dataclass, a `TypedDict` and a `NamedTuple` are
        # none of the things tested above — not a `BaseModel`, no
        # `get_origin`, not a bare container — so they all arrive here, and
        # answering "no" would drop every marked field inside them from the
        # graph with no refusal: the whole category silently uncovered. Only a
        # class that declares no fields of its own anywhere in its MRO is a
        # leaf; see `_declares_fields`.
        return _declares_fields_anywhere(annotation)
    return True


class _GraphBuilder:
    """Walks declared classes into the graph the matcher reads.

    Holds the model worklist so recursion and mutual recursion terminate:
    a model already in `models` is referenced by id and not re-walked, which
    is safe precisely because the artifact is a graph — the matcher follows
    the reference as many times as the *data* requires.
    """

    def __init__(self, is_secret: Any) -> None:
        self.is_secret = is_secret
        self.models: dict[str, dict[str, dict[str, Any]]] = {}
        self.classes: dict[str, type] = {}

    def add(self, model: type) -> str:
        model_id = _model_id(model)
        if model_id in self.models:
            return model_id
        if _is_root_model(model):
            # Checked here rather than at each call site, so a root model
            # reached as a declared root and one reached through a field are
            # refused by the same line. See `_is_root_model`.
            raise Unwalkable(
                f"{model_id}: a RootModel appears in a document as its inner "
                "value, with no field name to match, so any mark it carries "
                "would cover nothing. Declare an ordinary BaseModel with named "
                "fields, or mark the field that holds it "
                "(json_schema_extra={'secret': True}), which covers everything "
                "beneath it whatever its type"
            )
        # Registered before its fields are walked, so a self-reference
        # resolves to this same id instead of recursing forever.
        self.models[model_id] = {}
        self.classes[model_id] = model
        self.models[model_id] = self._fields(model, model_id)
        return model_id

    def _fields(self, model: type, model_id: str) -> dict[str, dict[str, Any]]:
        fields: dict[str, dict[str, Any]] = {}
        # Every key any field could appear under, whether or not that field
        # ends up emitted. Tracked separately from `fields` because an entry
        # is only stored when it says something (see below): keying the
        # collision check on `fields` alone made it order-dependent — an
        # unmarked scalar never claimed its key, so a later marked field
        # aliased to that name took it silently, while the same two fields
        # declared in the other order refused.
        claimed: set[str] = set()
        marked = self.is_secret(model)
        for name, info in model.model_fields.items():
            where = f"{model_id}.{name}"
            entry: dict[str, Any] = {}
            if name in marked:
                # A marked field's annotation is never examined: coverage is
                # prefix-based, so everything beneath it is covered whatever
                # it is, and refusing an unwalkable annotation there would
                # refuse a declaration that is already complete.
                entry["secret"] = True
            else:
                child = self._resolve(
                    info.annotation, where, model_id, info.annotation
                )
                if child is not None:
                    entry["child"] = child
            for key in _data_keys(name, info, where):
                if key in claimed:
                    raise Unwalkable(
                        f"{where}: data key {key!r} is claimed by more than one "
                        "field; give the fields distinct names or aliases"
                    )
                claimed.add(key)
                if entry:
                    # A field with neither a mark nor a model beneath it is
                    # omitted rather than written as `{}`: the matcher treats
                    # "absent" and "declares nothing" identically, and a model
                    # with fifty scalar fields would otherwise bury the marks —
                    # the only thing this artifact exists to carry — under
                    # forty-nine empty objects in every review of it.
                    fields[key] = entry
        return fields

    def _resolve(
        self, annotation: Any, where: str, model_id: str, written: Any
    ) -> dict[str, Any] | None:
        """One annotation, as a graph node — or `None` when nothing beneath it
        is a declared model, which is the common case and is written as the
        absence of a `child` entry."""
        annotation = _strip_annotated(annotation)
        if annotation is None or annotation is type(None):
            return None
        if _is_model(annotation):
            return {"kind": "model", "model": self.add(annotation)}

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        if origin in _UNION_ORIGINS:
            present = [a for a in args if a is not type(None)]
            if len(present) == 1:
                # `Optional[X]` / `X | None`: one real member, so there is
                # exactly one thing to walk. `None` itself is not a mapping
                # and has nothing beneath it.
                return self._resolve(present[0], where, model_id, written)
            if any(_may_hold_model(member) for member in present):
                raise Unwalkable(
                    f"{where}: a union of more than one type cannot be walked, "
                    "because which member a value belongs to is not knowable "
                    "from the annotation"
                )
            return None
        if origin is list and len(args) == 1:
            item = self._resolve(args[0], where, model_id, written)
            return None if item is None else {"kind": "list", "item": item}
        if origin is dict and len(args) == 2 and _strip_annotated(args[0]) is str:
            value = self._resolve(args[1], where, model_id, written)
            return None if value is None else {"kind": "dict", "value": value}

        if _may_hold_model(annotation):
            # Names the field, its class and the annotation itself, then every
            # way forward. A refusal a reader cannot act on is a refusal that
            # gets silenced with model_net = "none", which loses the whole net
            # -- and first adoption of this tool is exactly when several of
            # these arrive at once.
            raise Unwalkable(
                f"{where}: annotation {_describe(written)}"
                f"{_at(written, annotation)} may hold a "
                f"declared model, and {model_id} cannot be walked through it. "
                "Walkable forms are a model, list[X], dict[str, X] and "
                "Optional[X]. Three ways forward: narrow the annotation to one "
                "of those; or mark the field "
                "(json_schema_extra={'secret': True}), which covers everything "
                "beneath it whatever its type; or, if this repository's marking "
                "convention is not this tool's, declare "
                "check.declared_paths_fn"
            )
        return None


def _describe(annotation: Any) -> str:
    """An annotation as a reader would recognise it in their own source.

    `repr` on a bare class is `<class 'str'>`, which reads as noise in a
    message whose whole job is to point at a line of code; typing constructs
    already repr as they were written. Carries type names from the consuming
    repository, which are not credentials -- the same category as the module
    and class names its `.stackward.toml` already states in the clear.
    """
    if isinstance(annotation, type):
        return getattr(annotation, "__qualname__", None) or repr(annotation)
    return repr(annotation)


def _at(written: Any, offending: Any) -> str:
    """` (at X)`, when the part that refused is not the whole annotation.

    The message has to name the annotation as the reader *wrote* it —
    `dict[str, Any]` — or they are sent looking for a line that says `Any`
    and does not exist. Naming the offending part as well keeps the precision
    that made the inner-only message tempting.
    """
    if written is offending:
        return ""
    return f" (at {_describe(offending)})"


def _data_keys(name: str, info: Any, where: str) -> list[str]:
    """Every key this field may appear under in a stack config document.

    The field's own name plus any string alias. Keying only by field name
    would miss an aliased field silently, which is a fail-open; keying by
    every name it could carry over-matches at worst, and an over-match is
    suppressed by the same encrypted/empty rules as any other leaf.

    A non-string `validation_alias` (`AliasChoices`, `AliasPath`) is refused,
    marked field included: without a determinable data key there is no
    position at which to apply coverage at all, and a mark that covers
    nothing is worse than one that does not exist.
    """
    keys = [name]
    for attribute in ("alias", "validation_alias", "serialization_alias"):
        value = getattr(info, attribute, None)
        if value is None:
            continue
        if not isinstance(value, str):
            raise Unwalkable(
                f"{where}: {attribute} is not a plain string, so the key this "
                "field appears under is not knowable; use a string alias"
            )
        if value not in keys:
            keys.append(value)
    return keys


def _standard_marks(model: type) -> set[str]:
    """This tool's published marking convention:
    `json_schema_extra={"secret": True}`.

    A callable `json_schema_extra` is refused rather than ignored — its result
    depends on a schema-generation context that does not exist here, so
    whether the field is marked cannot be determined, and "cannot determine"
    must not become "not marked".
    """
    marked: set[str] = set()
    for name, info in model.model_fields.items():
        extra = getattr(info, "json_schema_extra", None)
        if extra is None:
            continue
        if callable(extra):
            raise Unwalkable(
                f"{_model_id(model)}.{name}: json_schema_extra is a callable, so "
                "whether this field is marked cannot be determined here; use a "
                "plain dict, or declare check.declared_paths_fn"
            )
        if isinstance(extra, dict) and extra.get("secret") is True:
            marked.add(name)
    return marked


def _custom_marks(function: Any) -> Any:
    """`check.declared_paths_fn` as a marking-convention override.

    Called as `fn(model_class)` and must return the *field names* that model
    marks as secret. Field names, not paths: the graph is what makes
    recursion work, and a hook returning flat paths would reintroduce exactly
    the enumeration this design replaced — so a returned string containing
    `.` or `[` is refused rather than interpreted. A name that is not a field
    of the model is refused too, since a typo that marked nothing would be
    indistinguishable from a field deliberately left unmarked.
    """

    def marks(model: type) -> set[str]:
        try:
            returned = function(model)
        except Exception as exc:  # noqa: BLE001 - the hook is repository code
            raise Unwalkable(
                f"{_model_id(model)}: check.declared_paths_fn raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        names: set[str] = set()
        for item in returned:
            if not isinstance(item, str):
                raise Unwalkable(
                    f"{_model_id(model)}: check.declared_paths_fn returned a "
                    f"{type(item).__name__}; it must return field names"
                )
            if "." in item or "[" in item:
                raise Unwalkable(
                    f"{_model_id(model)}: check.declared_paths_fn returned "
                    f"{item!r}; it must return field names, not paths"
                )
            if item not in model.model_fields:
                raise Unwalkable(
                    f"{_model_id(model)}: check.declared_paths_fn returned "
                    f"{item!r}, which is not a field of this model"
                )
            names.add(item)
        return names

    return marks


def _import_object(target: str, what: str) -> Any:
    module_name, _, attribute = target.partition(":")
    try:
        module = __import__(module_name, fromlist=["*"])
    except Exception as exc:  # noqa: BLE001 - repository code, any failure
        raise Unwalkable(
            f"{what} {target!r}: cannot import {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise Unwalkable(
            f"{what} {target!r}: {module_name!r} has no {attribute!r}"
        ) from exc


def _annotated_aliases(model: type) -> list[Any]:
    """Every `Annotated[...]` object appearing in `model`'s own annotations.

    Collected from the *raw* hints rather than from `FieldInfo`, because
    pydantic resolves an alias's mark onto the field and discards the wrapper
    — which is exactly why an alias defined in another module can change a
    mark with the model's own file untouched. The objects collected here are
    matched by identity against module globals in `_collect_sources`, which
    is what finds the module that defines the alias.
    """
    try:
        hints = typing.get_type_hints(model, include_extras=True)
    except Exception as exc:  # noqa: BLE001 - repository code, any failure
        raise Unwalkable(
            f"{_model_id(model)}: annotations cannot be resolved "
            f"({type(exc).__name__}: {exc}), so the modules contributing them "
            "cannot be recorded; a model whose sources cannot be recorded "
            "cannot be checked for staleness"
        ) from exc

    found: list[Any] = []
    seen: set[int] = set()

    def visit(annotation: Any) -> None:
        if id(annotation) in seen:
            return
        seen.add(id(annotation))
        if hasattr(annotation, "__metadata__"):
            found.append(annotation)
        for argument in typing.get_args(annotation):
            visit(argument)

    for hint in hints.values():
        visit(hint)
    return found


def _inside(source: Any, repo_root: Path) -> bool:
    try:
        Path(source).resolve().relative_to(repo_root)
    except ValueError:
        return False
    return True


def _collect_sources(
    classes: list[type], extra_objects: list[Any], repo_root: Path
) -> tuple[list[str], list[str]]:
    """Every file and out-of-repo module that contributed a mark.

    Two mechanisms, because a mark can arrive two ways that neither one alone
    sees:

    - the **MRO** of every reachable model, so a mark inherited from a base
      class in another file is recorded. `getsourcefile(cls)` alone gives
      only the subclass's own file, which is the case this closes. The walk
      is *not* restricted to `BaseModel` subclasses: pydantic collects a
      marked field from a plain `class Mixin:` into
      `Root(Mixin, BaseModel).model_fields`, so a filter keyed on
      `issubclass(base, BaseModel)` takes the mark and discards the file it
      came from.
    - **identity** of every `Annotated` alias used by those models, matched
      against the globals of every imported module. An alias is a module-level
      object, and the annotation resolves to that very object, so the module
      holding it can be found this way and no other — nothing on the resolved
      field records where the alias was written.

    `pydantic.BaseModel` itself is excluded from the MRO walk. Including it
    would make pydantic an out-of-repo contributor for *every* repository,
    and so make a tracked lockfile a universal precondition, for no signal:
    the marks this tool reads are its own convention, not pydantic's. Every
    *other* base is recorded when it is repo-local (free — the blob id is in
    the index already) and, when it is not, only if it declares fields of its
    own. Without that second condition `class Config(BaseModel, ABC)` would
    record `abc` and reimpose the very lockfile precondition the `BaseModel`
    exclusion exists to avoid.

    A module with no `__file__` (a builtin, a namespace package, this program
    itself) is skipped: there is nothing to record a blob id for. That is a
    stated boundary of the blob-id layer, not a claim it does not exist.
    """
    from pydantic import BaseModel

    files: set[str] = set()
    external: set[str] = set()

    def record(source: Any, module_name: str) -> None:
        if not source:
            return
        if _inside(source, repo_root):
            files.add(str(Path(source).resolve()))
        else:
            external.add(module_name)

    for model in classes:
        for base in model.__mro__:
            if base is object or base is BaseModel:
                continue
            try:
                source = inspect.getsourcefile(base)
            except (TypeError, OSError):
                # TypeError for a builtin; OSError ("source code not
                # available") for a class whose module has no file at all.
                source = None
            if source is None:
                continue
            if _inside(source, repo_root):
                # Repo-local: recorded unconditionally. A blob id for a file
                # already in the index costs nothing, and a base that declares
                # no field today is one edit away from declaring a marked one.
                record(source, base.__module__)
            elif _declares_fields(base):
                # Out of the repository: recorded only when it actually
                # declares fields, because recording one makes a tracked
                # lockfile a precondition for generating at all. That is why
                # this cannot simply be "every base except object" —
                # `class Config(BaseModel, ABC)` would pin `abc`.
                external.add(base.__module__)

    wanted = {id(obj) for model in classes for obj in _annotated_aliases(model)}
    wanted.update(id(obj) for obj in extra_objects)
    if wanted:
        for module_name, module in list(sys.modules.items()):
            if module is None or module_name == "__main__":
                continue
            namespace = getattr(module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            if any(id(value) in wanted for value in list(namespace.values())):
                record(getattr(module, "__file__", None), module_name)

    return sorted(files), sorted(external)


def generate(request: dict[str, Any]) -> dict[str, Any]:
    """The whole job: import the declared classes, build the graph, and
    report the files and out-of-repo modules that contributed to it."""
    repo_root = Path(request["repo_root"]).resolve()
    sys.path.insert(0, str(repo_root))

    hook_target = request.get("declared_paths_fn")
    extra_objects: list[Any] = []
    if hook_target:
        hook = _import_object(hook_target, "check.declared_paths_fn")
        is_secret = _custom_marks(hook)
        extra_objects.append(hook)
    else:
        is_secret = _standard_marks

    builder = _GraphBuilder(is_secret)
    roots: dict[str, str] = {}
    for namespace, target in sorted(request["stack_models"].items()):
        model = _import_object(target, "check.stack_models")
        if not _is_model(model):
            raise Unwalkable(
                f"check.stack_models {target!r}: not a pydantic BaseModel subclass"
            )
        roots[namespace] = builder.add(model)

    classes = [builder.classes[key] for key in sorted(builder.classes)]
    # The hook itself decides which fields are marked, so the module defining
    # it is as much a source as any model's file. It is passed in as an object
    # rather than resolved by name: the identity scan in `_collect_sources`
    # finds it in its own module's globals, and so classifies it repo-local or
    # out-of-repo by exactly the same rule as everything else.
    files, external = _collect_sources(classes, extra_objects, repo_root)
    return {
        "roots": roots,
        "models": builder.models,
        "files": files,
        "external": external,
    }


def main(argv: list[str]) -> int:
    """Read the request from `argv[1]`, write the graph to the file it names.

    The result goes to a file rather than to stdout on purpose: importing a
    repository's modules runs that repository's import-time code, which is
    free to print. A banner on stdout would corrupt a JSON document written
    there, and would do it intermittently — only in the repositories that
    happen to print something. stdout and stderr stay available for exactly
    that output, and for this program's own error messages.
    """
    if len(argv) != 2:
        _die("usage: <program> <request-json>")
    try:
        request = json.loads(argv[1])
    except ValueError as exc:
        _die(f"request is not valid JSON: {exc}")
    try:
        result = generate(request)
    except Unwalkable as exc:
        _die(str(exc))
    Path(request["output"]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
