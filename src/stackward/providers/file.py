"""`file`: the encrypted store, adapted to the `CredentialStore` role.

`store.resolve_credentials` already has this role's exact shape — a profile
name and a password in, a mapping of every sealed name back out — so this
class adds nothing beyond binding the password and the store directory at
construction time, which `CredentialStore.resolve`'s single-argument
signature has no room for otherwise. See `store.resolve_credentials`'s own
docstring for the resolution semantics themselves; this module does not
duplicate them.

`StoreError` (and its subclasses `ProfileError`, `PasswordError`) propagate
unchanged rather than being rewrapped as `ProviderError`. That is a
deliberate exception to this package's own "`ProviderError`, uniformly"
stance (see `providers/__init__.py`): `commands.session`'s existing,
reviewed tests already assert `pytest.raises(StoreError)` for a wrong
password and for a profile with no envelope, and this task's brief asks for
adaptation, not a rewrite of an error type nothing about this task requires
changing.
"""

from __future__ import annotations

from pathlib import Path

from .. import store


class FileCredentialStore:
    """`store.resolve_credentials`, bound to one password and directory."""

    def __init__(self, password: str, *, directory: Path | None = None) -> None:
        self._password = password
        self._directory = directory

    def resolve(self, profile: str) -> dict[str, str]:
        return store.resolve_credentials(profile, self._password, directory=self._directory)
