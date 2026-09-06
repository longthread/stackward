"""Command dispatch.

`v0.1.0` shipped `--version` and `doctor` alone, deliberately: the point of
that release was to prove the distribution path — build, sign, publish,
download, install, run — while it was still cheap to change. That path held,
and the commands below landed on top of a distribution mechanism already
known to work, rather than the other way round.

Every subcommand that can touch a credential is registered through
`_dispatch`, which imports its module only when the subcommand actually
runs. That is not a startup-time optimisation; see `_dispatch` for what it
is.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .bootstrap import GENERATOR_NAME, generator_source
from .commands.check_config import cmd_check_config
from .commands.install_hooks import cmd_install_hooks
from .commands.pre_commit import cmd_pre_commit
from .config import (
    MinVersionError,
    enforce_min_version,
    find_repo_config,
    load_config,
)

# Commands that must keep working even when a repo demands a newer stackward
# than the one installed. Without this exemption the upgrade instruction would
# itself be blocked by the check that prints it.
#
# `self-update` is *reserved*, not registered: `build_parser` below defines no
# such subcommand, and `enforce_min_version`'s message points at `install.sh`
# instead. The name stays here so that a future `self-update` is exempt on the
# day it is added rather than the day someone notices it is not -- an exempt
# name that no parser accepts costs nothing, while a missing one would block
# the very command that fixes the block.
VERSION_CHECK_EXEMPT = frozenset({"doctor", "self-update"})


# Every command module reached through `_dispatch`, as literal `import`
# statements. See `_import_command` for why this is not a dictionary, a
# registry, or `importlib.import_module`.
def _import_command(module_name: str):
    """The `commands` submodule named by `module_name`, imported now.

    Written as one literal `import` statement per module rather than as
    `importlib.import_module(f".commands.{module_name}", __package__)`,
    which is what this was, because of what a *frozen* build can see.
    PyInstaller resolves imports by reading `import` statements out of
    compiled bytecode -- including statements inside a function body, which
    is exactly what keeps these lazy (`crypto_selftest`'s own function-level
    `import cryptography` is why the cryptography backend is in the bundle at
    all). It cannot resolve a module name assembled at runtime.

    So with the f-string form, **none** of the lazily dispatched commands was
    in the shipped binary. Verified against the release workflow's own build
    command: `stackward exec` in the bundle died with `ModuleNotFoundError:
    No module named 'stackward.commands.session'`, a raw traceback, and exit
    code 1 -- the code this tool reserves exclusively for "a credential was
    found". The failure existed only in the artifact everyone installs, which
    is the same shape as the two failures `doctor`'s self-tests were added
    to catch, and for the same reason it is not something a source checkout
    can notice.

    An `if` chain, and not a mapping of names to import thunks, because a
    mapping built at module scope would run every one of those thunks when
    this module is imported -- which is the whole cost being avoided. The
    same shape, for the same reason, as
    `session._build_credential_store`'s provider dispatch.
    """
    if module_name == "session":
        from .commands import session as module
    elif module_name == "credentials":
        from .commands import credentials as module
    elif module_name == "set_secrets":
        from .commands import set_secrets as module
    elif module_name == "sync_declared":
        from .commands import sync_declared as module
    elif module_name == "check_passphrase":
        from .commands import check_passphrase as module
    else:  # pragma: no cover - a name `build_parser` never passes
        raise RuntimeError(f"no command module named {module_name!r}")
    return module


def _dispatch(module_name: str, function_name: str):
    """A subcommand entry point that imports its module only when the
    subcommand is actually run.

    Global Constraint 2 says no code reachable from `check-config` or
    `pre-commit` may prompt, open a network connection, or import a
    credential provider -- so that a commit cannot be blocked by an expired
    session or a flaky connection. Importing the credential commands at this
    module's own scope broke that: `import stackward.cli` pulled in
    `commands.check_passphrase` -> `cryptography`, and `commands.session` ->
    `store` -> `crypto`, on *every* invocation. With `cryptography` made
    unimportable, `stackward pre-commit` died on a raw traceback and exited
    1 -- the code this tool reserves exclusively for "a credential was
    found", reporting a leak in a repository that had none.

    Written as a thunk rather than a module-level `__getattr__`, because
    `build_parser` names every entry point on every run: `__getattr__` would
    import all of them again the moment the parser was built, which is the
    whole cost being avoided. The lookup is `getattr` at *call* time, so
    `monkeypatch.setattr(stackward.commands.session, "cmd_login", ...)`
    still takes effect -- a thunk that captured the function at import time
    would silently bypass such a patch.
    """

    def dispatch(args: argparse.Namespace) -> int:
        return getattr(_import_command(module_name), function_name)(args)

    dispatch.__name__ = function_name
    dispatch.__qualname__ = f"{module_name}.{function_name}"
    return dispatch


def config_home() -> Path:
    """Where profiles and credentials live. XDG, with the usual fallback."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "stackward"


def _tool_version(name: str) -> str:
    """Report an external tool's version, or why it cannot be used."""
    path = shutil.which(name)
    if path is None:
        return "not found on PATH"
    try:
        proc = subprocess.run(
            [name, "--version"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"{path} (unusable: {exc})"
    first = (proc.stdout or proc.stderr).strip().splitlines()
    return f"{first[0] if first else '?'}  [{path}]"


def crypto_selftest() -> str:
    """Prove the cryptography backend works, rather than merely importing.

    Every credential feature depends on this, and the characteristic PyInstaller
    failure is a bundle that builds cleanly and then cannot load a native
    extension on a machine that is not the build machine. An import check would
    miss a broken backend; a real round-trip does not. `doctor` is therefore the
    smoke test a release runs in a clean container.
    """
    try:
        import cryptography
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

        key = Argon2id(
            salt=b"stackward-selftest",
            length=32,
            iterations=1,
            lanes=1,
            memory_cost=8,
        ).derive(b"selftest")

        nonce, plaintext = b"\0" * 12, b"ok"
        sealed = AESGCM(key).encrypt(nonce, plaintext, b"selftest")
        if AESGCM(key).decrypt(nonce, sealed, b"selftest") != plaintext:
            return "FAILED: AES-GCM round trip did not reproduce the plaintext"
        return f"cryptography {cryptography.__version__}  (Argon2id + AES-256-GCM ok)"
    except Exception as exc:  # noqa: BLE001 - report any failure, never crash doctor
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def generator_selftest() -> str:
    """Prove the model walker's source is reachable, rather than assuming it.

    `sync-declared-secrets` ships `bootstrap/regen.py` as a **data** file and
    pipes its text to the consuming repository's interpreter; nothing imports
    it, which is what keeps pydantic out of the bundle. The cost of that is
    that PyInstaller's import analysis cannot see it either, so it is carried
    by an explicit `--add-data` entry in the release workflow -- and if that
    entry is ever dropped, the binary builds cleanly, starts cleanly, and
    then `sync-declared-secrets` fails in the one artifact everyone installs
    and nowhere else.

    That is the same failure shape as `crypto_selftest`'s: something that
    works everywhere except the shipped bundle. So it gets the same treatment
    -- `doctor` is what a release smoke test runs, and it checks the real
    accessor the shipped code uses rather than the file's path on disk, which
    does not exist in a bundle. It can only fail if the packaging is wrong:
    an installed wheel and a source checkout both carry the file already.
    """
    try:
        source = generator_source()
    except Exception as exc:  # noqa: BLE001 - report any failure, never crash doctor
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    if "def generate(" not in source:
        return f"FAILED: {GENERATOR_NAME} is present but is not the generator"
    return f"{GENERATOR_NAME} readable ({len(source)} bytes)"


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Report what stackward can see. Never prints a credential value."""
    config = find_repo_config()
    home = config_home()

    print(f"stackward       {__version__}")
    print(f"python          {platform.python_version()} ({sys.platform})")
    print(f"executable      {sys.executable}")
    print(f"frozen binary   {getattr(sys, 'frozen', False)}")
    print()
    print(f"config home     {home}{'' if home.is_dir() else '  (not created yet)'}")
    print(f"repo config     {config or 'none found — commands needing one will refuse'}")
    print()
    print(f"pulumi          {_tool_version('pulumi')}")
    print(f"git             {_tool_version('git')}")
    print()
    crypto = crypto_selftest()
    print(f"crypto          {crypto}")
    generator = generator_selftest()
    print(f"model walker    {generator}")
    # Exit non-zero when the credential layer or the model walker could not
    # work here. doctor is what a release smoke test runs, so it has to fail
    # rather than narrate -- and both of these can only break in a bundle.
    broken = ("UNAVAILABLE", "FAILED")
    return 1 if crypto.startswith(broken) or generator.startswith(broken) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stackward",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--version", action="version", version=f"stackward {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = sub.add_parser("doctor", help="report resolved configuration and environment")
    doctor.set_defaults(func=cmd_doctor)

    check_config = sub.add_parser(
        "check-config", help="scan stack config file(s) for a plaintext credential"
    )
    check_config.add_argument("files", nargs="+", metavar="FILE")
    check_config.set_defaults(func=cmd_check_config)

    pre_commit = sub.add_parser(
        "pre-commit",
        help="check staged content for a plaintext credential (git hook body)",
    )
    pre_commit.set_defaults(func=cmd_pre_commit)

    hooks = sub.add_parser("hooks", help="manage this repository's git hooks")
    hooks_sub = hooks.add_subparsers(dest="hooks_command", metavar="<command>")
    hooks_install = hooks_sub.add_parser(
        "install", help="install the pre-commit gate into this repository's hooks"
    )
    hooks_install.set_defaults(func=cmd_install_hooks)

    login = sub.add_parser("login", help="point pulumi at a profile's backend")
    login.add_argument("--profile", help="profile to use (overrides selection precedence)")
    login.set_defaults(func=_dispatch("session", "cmd_login"))

    exec_ = sub.add_parser(
        "exec", help="run a command with a profile's credentials injected"
    )
    exec_.add_argument("--profile", help="profile to use (overrides selection precedence)")
    exec_.add_argument(
        "argv", nargs="*", metavar="COMMAND", help="command to run, after --"
    )
    exec_.set_defaults(func=_dispatch("session", "cmd_exec"))

    shell = sub.add_parser(
        "shell", help="open $SHELL with a profile's credentials injected"
    )
    shell.add_argument("--profile", help="profile to use (overrides selection precedence)")
    shell.set_defaults(func=_dispatch("session", "cmd_shell"))

    set_secrets = sub.add_parser(
        "set-secrets", help="publish declared stack secrets into pulumi config"
    )
    set_secrets.add_argument("--stack", help="stack name, forwarded to pulumi")
    set_secrets.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be set without running pulumi",
    )
    set_secrets.set_defaults(func=_dispatch("set_secrets", "cmd_set_secrets"))

    sync_declared = sub.add_parser(
        "sync-declared-secrets",
        help="regenerate the declared-secrets artifact from this repo's models",
    )
    sync_declared.set_defaults(func=_dispatch("sync_declared", "cmd_sync_declared"))

    # `credentials`, like the session commands, is registered through
    # `_dispatch`: its module imports `store` -> `crypto` -> `cryptography`,
    # and `tests/test_gate_isolation.py` asserts that a `check-config` run
    # loads none of those. Nothing in the help text below may name anything
    # from that module either -- an f-string interpolating
    # `session.CREDENTIAL_NAMES` here would import it at *this* module's
    # scope and break the same property in a way no reader would connect to
    # a help string.
    credentials = sub.add_parser(
        "credentials", help="manage the encrypted credential store"
    )
    credentials_sub = credentials.add_subparsers(
        dest="credentials_command", metavar="<command>"
    )

    credentials_init = credentials_sub.add_parser(
        "init", help="create the credential store (asks for a new password twice)"
    )
    credentials_init.set_defaults(func=_dispatch("credentials", "cmd_credentials_init"))

    credentials_set = credentials_sub.add_parser(
        "set",
        help=(
            "seal a profile's credentials, read as KEY=VALUE lines on stdin "
            "(replaces the profile's whole envelope; never takes a value as "
            "an argument)"
        ),
    )
    credentials_set.add_argument(
        "--profile", help="profile to use (overrides selection precedence)"
    )
    credentials_set.set_defaults(func=_dispatch("credentials", "cmd_credentials_set"))

    credentials_list = credentials_sub.add_parser(
        "list", help="list the profiles that have credentials sealed (names only)"
    )
    credentials_list.set_defaults(func=_dispatch("credentials", "cmd_credentials_list"))

    credentials_show = credentials_sub.add_parser(
        "show",
        help="list the credential names a profile carries, never their values",
    )
    credentials_show.add_argument(
        "--profile", help="profile to use (overrides selection precedence)"
    )
    credentials_show.set_defaults(func=_dispatch("credentials", "cmd_credentials_show"))

    credentials_rotate = credentials_sub.add_parser(
        "rotate", help="re-seal every envelope under a new password, or none"
    )
    credentials_rotate.set_defaults(
        func=_dispatch("credentials", "cmd_credentials_rotate")
    )

    check_passphrase = sub.add_parser(
        "check-passphrase",
        help="prove a passphrase decrypts one or more stacks (never as an argument)",
    )
    check_passphrase.add_argument("stacks", nargs="+", metavar="STACK")
    check_passphrase.set_defaults(func=_dispatch("check_passphrase", "cmd_check_passphrase"))

    return parser


def _min_version_refusal(command: str) -> int | None:
    """`2` if this repository demands a newer `stackward` than the installed
    one, otherwise `None`.

    One guard, not two. Everything from finding the config to comparing the
    versions is inside a single `try`, and every failure other than
    `MinVersionError` means "there is no floor I can establish here" and
    lets the command run. `doctor` is the command you run *because*
    `.stackward.toml` is broken, and Task 2's carry-forward ruling requires
    it to keep working when it is; a floor check that refused to run before
    dispatch would take that away. Nothing is lost by staying quiet:
    anything genuinely wrong with the file is reported, in better words, by
    whichever command actually needs it — both gate commands refuse outright
    on an unparseable or missing policy.

    A second, outer guard around this would be worse than none: with two,
    neither is falsifiable, and a test that deletes either one still passes.

    It reads the *working tree*, deliberately, even though `pre-commit`
    reads its `[check]` policy from the index. `min_version` is a statement
    about the binary you are running right now, not about the commit you are
    making, and this runs before dispatch — before there is any reason to
    believe the current directory is even a git repository.
    """
    try:
        path = find_repo_config()
        config = load_config(path) if path is not None else None
        enforce_min_version(config, command)
    except MinVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - a broken config must not break dispatch
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2

    # The `min_version` floor, enforced here rather than in each command, so
    # it applies to every subcommand instead of whichever ones remembered to
    # ask. It runs *before* the gate commands' own missing-policy refusal,
    # which costs nothing: with no config there is no floor to enforce
    # (`enforce_min_version(None, ...)` returns immediately) and the command
    # then refuses with its own, more useful message. The order only matters
    # when a config exists, and there the floor should win -- a repository
    # whose policy needs a newer stackward is telling you that this binary
    # may not understand the policy it is about to read.
    #
    # `main` is outside `@fail_closed`, so an escaping exception here would
    # exit 1: the code reserved for "a credential was found". That is why
    # `_min_version_refusal` returns an exit code rather than raising, and
    # why its own guard is broad.
    refusal = _min_version_refusal(getattr(args, "command", "") or "")
    if refusal is not None:
        return refusal

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
