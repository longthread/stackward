"""Provider interface: swapping the credential source is a configuration
change, not a migration.

**Two roles, one method each.** `CredentialStore.resolve(profile)` returns a
profile's bootstrap set (the names `commands.session` needs to run `pulumi`
at all: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`PULUMI_CONFIG_PASSPHRASE`), the same shape `store.resolve_credentials`
already has. `SecretSource.resolve(name)` returns one manifest logical name's
value, or `None`, the same shape `commands.set_secrets.LocalSecretSource`
already has. Both are `typing.Protocol`s — structural, not a base class a
provider must inherit from — because the whole point is that a caller written
against the method, not against a particular module, cannot tell which
provider answered it. `file` and `dotenv` in this package are thin adapters
over exactly those two pre-existing, reviewed functions (see `file.py` and
`dotenv.py`); this task adapts them rather than rewriting them.

**Construction is per-provider; only the method is uniform.** A `file`
`CredentialStore` needs a password and a store directory; an `env` one needs
a list of names; a `command` one needs an argv prefix. None of that belongs
in one generic factory signature that would have to grow a parameter for
every provider's own configuration — so `providers/__init__.py` defines no
`build_*` function at all. `commands.session` and `commands.set_secrets` each
own a small `if provider == "...":` dispatch instead, exactly the same shape
`commands.set_secrets._pulumi_config_set` already uses for `_MODE_FLAGS`, and
that is the whole point of the pattern being small enough to read in place
rather than a registry needing its own tests.

**A provider failure is `ProviderError`, reported, never a silent empty
value.** `SecretSource.resolve` returning `None` and `SecretSource.resolve`
raising are different facts: `None` is "not set", which
`commands.set_secrets._publish_entries` already treats as a normal skip; a
raise is "this provider could not answer", which must surface as a `failed`
outcome instead — an empty value and a broken provider are indistinguishable
to everything downstream unless this distinction is kept. `file`'s
`CredentialStore` is the one exception to `ProviderError` specifically: it
lets `store.StoreError` (a wrong password, a missing envelope) propagate
unchanged, because `commands.session`'s existing tests already assert on
`StoreError` for exactly those failures, and rewrapping a pre-existing,
correct error type would be a rewrite this task's brief does not ask for, not
an adaptation.

**No provider module is ever imported from the gate path.** `check-config`
and `pre-commit` resolve no credentials at all — that is what stops a commit
being blocked by an expired session or a flaky network — so neither may ever
cause a `stackward.providers*` module to be imported, even transitively.
`commands.session` and `commands.set_secrets` are both imported at module
scope by `cli.py` (and so, transitively, by every `stackward` invocation,
`check-config` and `pre-commit` included), so the only way to keep that
property is for *both* modules to import from this package **only inside the
function that is actually about to resolve a credential** (`session.
_build_credential_store`, `set_secrets._build_secret_source`, and the
`ProviderError` imports inside `set_secrets._publish_entries`/`_check_drift`)
— never at their own module scope. This mirrors `store.store_dir`'s own
`from .cli import config_home` and `config.enforce_min_version`'s `from .cli
import VERSION_CHECK_EXEMPT`, both local for the same reason: an import cycle
or an eager load neither module can afford at its own top level.
`tests/test_providers.py`'s
`test_the_check_config_entry_point_never_loads_a_provider_module` (and its
stronger, `cli.main`-driven sibling) is what proves this holds, in a
subprocess, rather than merely by this docstring's say-so.

This package's own module scope (this file) imports nothing from `file.py`,
`env.py`, `dotenv.py` or `command.py` — each of those is imported directly by
whichever command needs it, so importing `stackward.providers` bare costs
nothing beyond defining the two protocols and `ProviderError` below.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """A provider could not answer: a non-zero exit, a timeout, a missing
    executable, or output it could not parse. Never carries a credential
    value — every message here names a command, an exit code or a profile,
    never anything the command printed.

    The distinction this type exists to make: an empty resolved value and a
    provider that failed to produce one are different facts (see the module
    docstring). Raise this rather than returning `""` or an incomplete
    mapping on failure; it is the caller's job, not the provider's, to decide
    what "not set" itself means for the operation at hand.
    """


@runtime_checkable
class CredentialStore(Protocol):
    """Resolves a profile's bootstrap credential set."""

    def resolve(self, profile: str) -> dict[str, str]:
        """Every credential name known for `profile`, decrypted/fetched.

        May return a mapping missing a name the caller needs — `file`'s
        envelope legitimately might not hold every name, and `env`'s process
        environment legitimately might not export every name — callers
        (`commands.session._require_credential_names`) already tell a
        missing name apart from a resolution failure. A resolution failure
        itself (the store could not be opened, the remote CLI failed) is
        raised, never folded into an empty or partial mapping — see
        `ProviderError`.
        """
        ...


@runtime_checkable
class SecretSource(Protocol):
    """Resolves one manifest logical name to a value."""

    def resolve(self, name: str) -> str | None:
        """`name`'s value, or `None` when it is simply not set — a
        legitimate, silent outcome the caller treats as a skip (see
        `commands.set_secrets.LocalSecretSource`, whose contract this
        matches exactly: presence, not truthiness, is what `None` means).

        Raises `ProviderError` when the provider itself could not answer at
        all, which must never be conflated with `None` — see the module
        docstring.
        """
        ...
