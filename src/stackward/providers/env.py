"""`env`: for CI, where credentials and secret values already live directly
in the process environment — no store to decrypt, no file to parse.

Reading a mapping never fails, so neither class here has a `ProviderError`
path: a name this provider does not have is "not set", the same legitimate,
silent outcome any other provider reports the same way (see
`providers/__init__.py`'s module docstring), not a malfunction.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


class EnvCredentialStore:
    """Resolves a profile's bootstrap set by reading exactly `names` out of
    `environ` — never the whole environment (see `providers/__init__.py`'s
    caution about this: a caller that iterates an unfiltered `dict(os.
    environ)` back as though it were a credential mapping would leak
    everything else the process happens to have set).

    `profile` is accepted, by `CredentialStore`'s shape, and ignored: a CI
    runner's environment is not partitioned by profile, so there is nothing
    here for a profile name to select between — whichever profile asks gets
    the same environment.
    """

    def __init__(self, names: Sequence[str], environ: Mapping[str, str] | None = None) -> None:
        self._names = tuple(names)
        self._environ = os.environ if environ is None else environ

    def resolve(self, profile: str) -> dict[str, str]:
        return {name: self._environ[name] for name in self._names if name in self._environ}


class EnvSecretSource:
    """Resolves a logical name directly from the process environment. Unlike
    `dotenv` (`providers.dotenv.DotenvSecretSource`), no file is ever
    consulted — a manifest declaring `[secrets.source]` `provider = "env"`
    with a `files` list still gets *only* the environment; `files` is
    `dotenv`'s own key, and this provider does not look at it.

    Presence, not truthiness: an environment variable exported as `""` is
    still returned as `""`, not `None` — matching `SecretSource.resolve`'s
    contract that the two are different facts (see
    `commands.set_secrets.LocalSecretSource`, whose environment tier this
    mirrors exactly).
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def resolve(self, name: str) -> str | None:
        return self._environ.get(name)
