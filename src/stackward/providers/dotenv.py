"""`dotenv`: the process environment layered over parsed `KEY=VALUE` files,
adapted to the `SecretSource` role — this is `set-secrets`'s existing,
default value resolution, unchanged, and the default when
`[secrets.source].provider` is absent.

`commands.set_secrets.LocalSecretSource` already has this role's exact
shape (`resolve(name) -> str | None`, environment first, then each parsed
file in declared order, later files overriding earlier ones) — its own
docstring says it was written to this shape specifically so this interface
could adapt it rather than needing `commands.set_secrets._publish_entries`/
`_check_drift` rewritten. This module is that adapter: it does the file
parsing `commands.set_secrets.cmd_set_secrets` used to do inline, and then
delegates to `LocalSecretSource` for resolution itself.

Importing `commands.set_secrets` from here (rather than the other way
round) is deliberate, not a layering inversion: `set_secrets.py` must not
import this package at its own module scope — see `providers/__init__.py`'s
module docstring on why — so the dependency has to run in this direction,
and it costs nothing new, since `commands.set_secrets` is already imported
at `cli.py`'s module scope by every `stackward` invocation today.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..commands.set_secrets import LocalSecretSource, parse_env_file


class DotenvSecretSource:
    """`LocalSecretSource`, built from `environ` and the already-resolved
    `files` list (`commands.set_secrets.resolve_source_files`'s output) —
    each existing file parsed with `parse_env_file`, a missing one silently
    contributing nothing, exactly as `cmd_set_secrets` did before this
    provider existed.
    """

    def __init__(self, environ: Mapping[str, str], files: list[Path]) -> None:
        file_values = [parse_env_file(f) for f in files if f.exists()]
        self._inner = LocalSecretSource(environ, file_values)

    def resolve(self, name: str) -> str | None:
        return self._inner.resolve(name)
