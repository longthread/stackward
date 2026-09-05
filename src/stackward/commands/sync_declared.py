"""`sync-declared-secrets`: write `.stackward/declared-secrets.json`.

Registered in `cli.build_parser` as `stackward sync-declared-secrets`. It
imports the repository's pydantic models, walks their `SECRET` marks into the
graph `nets.model` reads, records the git blob id of everything that
contributed a mark, and writes the artifact atomically.

**It is not on the gate path.** `check-config` and `pre-commit` never call
this — they read the committed artifact. This command runs when a developer
changes a model, and it is the only part of the system that imports
repository code at all. It resolves no credentials, opens no store and
contacts nothing.

**It runs the walk under the repository's interpreter.** A frozen `stackward`
bundles its own interpreter and cannot import the consuming repository's
classes, so the walk runs as a subprocess under `[python]` from
`.stackward.toml` and only JSON crosses back (`bootstrap.regen`). That
separation is what keeps pydantic a *test-only* dependency of `stackward`
itself, which Global Constraint 1 requires.

**Everything that could make the result incomplete is a refusal.** An
annotation that might hold a model but cannot be walked, a contributing file
that git does not track, an out-of-repo contributor with no tracked lockfile
to pin it: each raises rather than producing an artifact that looks complete
and is not. A gate whose declarations quietly cover less than the repository
declared is the one failure mode the whole model-net design exists to
prevent.

**The output is byte-deterministic.** Sorted keys, fixed indentation, one
trailing newline, and every list sorted. The sound closure for staleness is
regenerating in CI and failing on any diff; a generator whose output churned
for unchanged input would make that check noise, and a noisy check gets
switched off.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ..bootstrap import generator_source
from ..config import Config, ConfigError, find_repo_config, load_config
from ..nets.model import (
    ARTIFACT_PATH,
    ARTIFACT_VERSION,
    ModelNetError,
    index_blob_ids,
    parse_net,
    repo_toplevel,
)
from ..store import atomic_write
from .check_config import fail_closed

# Standard Python dependency lockfiles, checked at the repository root. These
# are ecosystem filenames, not knowledge about any particular organisation or
# vendor — `pre_commit` already recognises Pulumi's own `Pulumi.<stack>.yaml`
# on the same basis. *Every* one that is tracked is recorded, not the first
# match: a repository can carry two, and pinning one while ignoring the other
# would leave a real dependency change invisible.
LOCKFILE_NAMES = (
    "uv.lock",
    "poetry.lock",
    "pdm.lock",
    "Pipfile.lock",
    "requirements.lock",
    "requirements.txt",
)

# The artifact is not read from the working tree, so mode is about who may
# look at a file naming a repository's config field names — not a credential,
# and committed anyway.
_ARTIFACT_MODE = 0o644

_GENERATION_TIMEOUT_SECONDS = 300


class SyncError(Exception):
    """Generation could not produce a complete artifact. Never partial: the
    caller writes nothing when this is raised."""


def _load_repo_config() -> tuple[Path, Config]:
    """The repository's `.stackward.toml`, which this command requires.

    Unlike the gate, absence is *not* a normal state here: there is nothing to
    generate without `check.stack_models`, and defaulting would write an empty
    artifact that then passes every freshness check while declaring nothing.
    """
    path = find_repo_config()
    if path is None:
        raise SyncError(
            "no .stackward.toml found; this command needs check.stack_models "
            "and the python interpreter to walk them with"
        )
    try:
        return path, load_config(path)
    except ConfigError as exc:
        raise SyncError(str(exc)) from exc


def _interpreter(config: Config, config_path: Path) -> str:
    """The repository's own interpreter, resolved against the config's
    directory so a relative `python = ".venv/bin/python"` means the same
    thing from any working directory inside the repository."""
    if not config.python:
        raise SyncError(
            f"{config_path}: no 'python' key; this command needs the "
            "repository's own interpreter to import its models"
        )
    candidate = Path(config.python)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    if not candidate.is_file():
        raise SyncError(f"{config_path}: python interpreter not found: {candidate}")
    return str(candidate)


def run_generator(interpreter: str, request: dict[str, object]) -> dict[str, object]:
    """Run the generator under `interpreter` and return its result.

    The program's *text* goes in on stdin (`python -`), the request goes in
    `argv`, and the result comes back through a temporary file named by the
    request. Three channels rather than one because importing a repository's
    modules runs that repository's import-time code: anything it prints would
    corrupt a JSON document sharing stdout with it, and only in the
    repositories that happen to print, which is the worst kind of
    intermittent.

    The request carries module and class names only — never a credential —
    so putting it in `argv` discloses nothing that `.stackward.toml` does not
    already say in the clear.
    """
    with tempfile.TemporaryDirectory(prefix="stackward-regen-") as workspace:
        output = Path(workspace) / "graph.json"
        payload = json.dumps({**request, "output": str(output)})
        try:
            proc = subprocess.run(
                [interpreter, "-", payload],
                input=generator_source(),
                capture_output=True,
                text=True,
                check=False,
                timeout=_GENERATION_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise SyncError(
                f"the model walk did not finish within "
                f"{_GENERATION_TIMEOUT_SECONDS}s"
            ) from exc
        except OSError as exc:
            raise SyncError(f"cannot run {interpreter}: {exc}") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or "").strip() or f"exit status {proc.returncode}"
            # The generator prefixes its own refusals with "error: ", and so
            # does the caller printing this. Strip one so the message does not
            # read "error: error: ...".
            raise SyncError(detail.removeprefix("error: "))
        try:
            text = output.read_text(encoding="utf-8")
        except OSError as exc:
            raise SyncError(f"the model walk produced no result: {exc}") from exc

    try:
        result = json.loads(text)
    except ValueError as exc:
        raise SyncError(f"the model walk produced invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise SyncError("the model walk produced a result that is not an object")
    return result


def _relative_sources(root: Path, absolute: list[str]) -> list[str]:
    """Contributing files as repository-relative POSIX paths.

    A file outside the repository never reaches here — the generator
    classifies those as out-of-repo modules, pinned by lockfile instead — so
    a path that will not relativise is a bug, not a supported case, and is
    refused rather than dropped.
    """
    relative: list[str] = []
    for item in absolute:
        try:
            relative.append(Path(item).resolve().relative_to(root).as_posix())
        except ValueError as exc:
            raise SyncError(f"{item}: not inside the repository") from exc
    return sorted(set(relative))


def _lockfile_sources(root: Path) -> dict[str, str]:
    """Every *tracked* dependency lockfile at the repository root, by blob id.

    Recorded only when something outside the repository contributed a mark —
    a base model or a marking alias imported from a dependency. `git show`
    cannot reach site-packages, so the lockfile that decides which version is
    installed is the closest thing to that module's identity that the index
    holds. A lockfile present on disk but untracked is not returned: it has no
    blob id to compare against later.
    """
    candidates = [name for name in LOCKFILE_NAMES if (root / name).is_file()]
    if not candidates:
        return {}
    return index_blob_ids(root, candidates)


def build_artifact(
    root: Path, result: dict[str, object], config_path: Path
) -> dict[str, object]:
    """Assemble the artifact, refusing anything that would leave it unverifiable.

    `config_path` is `.stackward.toml`, and it is a source like any other:
    `check.stack_models` supplies the artifact's `roots` and
    `check.declared_paths_fn` decides which fields are marked at all, so it
    decides part of the content as surely as a model file does. Leaving it
    out meant a namespace could be added to the policy, staged, and committed
    with the artifact untouched — the artifact then has no root for it, the
    namespace is skipped, and the gate reports clean.

    Two refusals matter more than they look:

    - a contributing file that git does not track. Omitting it would mean its
      blob id can never be compared, so a mark could change in it forever
      without the freshness check noticing — the exact fail-open this artifact
      exists to close.
    - an out-of-repo contributor with no tracked lockfile to stand in for it.
      There would be nothing at all pinning that module's identity.

    The artifact never records **itself**: its own blob id changes on every
    regeneration, so a self-reference would make every artifact stale the
    moment it was written.
    """
    files = _relative_sources(
        root,
        [str(item) for item in result.get("files", [])] + [str(config_path)],
    )
    external = sorted({str(item) for item in result.get("external", [])})

    tracked = index_blob_ids(root, files) if files else {}
    missing = sorted(set(files) - set(tracked))
    if missing:
        raise SyncError(
            f"{missing[0]} contributes a declared secret but is not tracked by "
            "git, so it cannot be checked for staleness; add it to the index "
            "and run this command again"
        )

    sources = dict(tracked)
    if external:
        lockfiles = _lockfile_sources(root)
        if not lockfiles:
            raise SyncError(
                f"{external[0]} contributes a declared secret from outside the "
                "repository, and no tracked dependency lockfile was found to "
                f"pin it (looked for: {', '.join(LOCKFILE_NAMES)})"
            )
        sources.update(lockfiles)

    sources.pop(ARTIFACT_PATH, None)
    if not sources:
        raise SyncError(
            "no source file contributed a declared secret, so nothing could be "
            "recorded to check for staleness"
        )

    artifact = {
        "version": ARTIFACT_VERSION,
        "roots": result.get("roots", {}),
        "models": result.get("models", {}),
        "sources": sources,
    }
    # Read it back with the matcher's own parser before writing it. The
    # generator and the matcher are the two halves of one schema, and they run
    # in different processes under different interpreters — so nothing else
    # would notice them drifting apart until the next commit, in someone
    # else's repository, as a refusal they did not cause. Checking here turns
    # that into a failure at the moment it is introduced.
    try:
        parse_net(artifact)
    except ModelNetError as exc:
        raise SyncError(f"the generated artifact is not readable: {exc}") from exc
    return artifact


def serialise(artifact: dict[str, object]) -> bytes:
    """The artifact's committed bytes: sorted keys, two-space indent, one
    trailing newline.

    Byte-determinism is the property CI leans on — regenerate, diff, fail on
    any change. `sort_keys` is what supplies it, since dictionary insertion
    order here follows the order classes happened to be walked in.
    """
    return (json.dumps(artifact, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sync(root: Path, config_path: Path, config: Config) -> Path:
    """Generate and write the artifact. Returns the path written."""
    if not config.check.stack_models:
        raise SyncError(
            f"{config_path}: check.stack_models is empty; there is nothing "
            'to walk. Declare a model, or set check.model_net = "none"'
        )

    result = run_generator(
        _interpreter(config, config_path),
        {
            "repo_root": str(root),
            "stack_models": dict(config.check.stack_models),
            "declared_paths_fn": config.check.declared_paths_fn,
        },
    )
    artifact = build_artifact(root, result, config_path)

    target = root / ARTIFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(target, serialise(artifact), _ARTIFACT_MODE)
    return target


@fail_closed
def cmd_sync_declared(_args: argparse.Namespace) -> int:
    """Entry point for `stackward sync-declared-secrets`.

    Exit codes match the rest of the tool's fail-closed convention: 0 on
    success, 2 on any refusal. There is no exit 1 here — this command does
    not look for credentials, so it can never mean "a credential was found",
    and `@fail_closed` keeps an unanticipated exception from exiting 1 by
    Python's own default.
    """
    try:
        config_path, config = _load_repo_config()
        root = repo_toplevel()
        written = sync(root, config_path, config)
    except (SyncError, ModelNetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {written.relative_to(root).as_posix()}")
    print("stage it with the change that caused it: git add " + ARTIFACT_PATH)
    return 0
