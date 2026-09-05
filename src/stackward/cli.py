"""Command dispatch.

v0.1.0 deliberately ships only `--version` and `doctor`. The point of this
release is to prove the distribution path — build, sign, publish, download,
install, run — while it is still cheap to change. Features land on top of a
distribution mechanism that is already known to work, not the other way round.
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
from .commands.check_config import cmd_check_config
from .commands.install_hooks import cmd_install_hooks
from .commands.pre_commit import cmd_pre_commit
from .commands.session import cmd_exec, cmd_login, cmd_shell
from .commands.set_secrets import cmd_set_secrets
from .config import find_repo_config

# Commands that must keep working even when a repo demands a newer stackward
# than the one installed. Without this exemption the upgrade instruction would
# itself be blocked by the check that prints it.
VERSION_CHECK_EXEMPT = frozenset({"doctor", "self-update"})


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
    # Exit non-zero when the credential layer could not work here. doctor is
    # what a release smoke test runs, so it has to fail rather than narrate.
    return 1 if crypto.startswith(("UNAVAILABLE", "FAILED")) else 0


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
    login.set_defaults(func=cmd_login)

    exec_ = sub.add_parser(
        "exec", help="run a command with a profile's credentials injected"
    )
    exec_.add_argument("--profile", help="profile to use (overrides selection precedence)")
    exec_.add_argument(
        "argv", nargs="*", metavar="COMMAND", help="command to run, after --"
    )
    exec_.set_defaults(func=cmd_exec)

    shell = sub.add_parser(
        "shell", help="open $SHELL with a profile's credentials injected"
    )
    shell.add_argument("--profile", help="profile to use (overrides selection precedence)")
    shell.set_defaults(func=cmd_shell)

    set_secrets = sub.add_parser(
        "set-secrets", help="publish declared stack secrets into pulumi config"
    )
    set_secrets.add_argument("--stack", help="stack name, forwarded to pulumi")
    set_secrets.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be set without running pulumi",
    )
    set_secrets.set_defaults(func=cmd_set_secrets)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
