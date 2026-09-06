"""`credentials`: the only supported way to *write* to the encrypted store.

**Why this module exists.** `store.py` implemented the store as a library and
`commands/session.py` implemented the three commands that *read* from it, and
between the two nothing ever owned the command line that fills it in. The
result shipped as a store that could be opened and never populated: a fresh
user's only working path was the `env` credential provider, which reads
credentials straight out of the process environment and so gives up the thing
the encrypted store exists for. Every function called below
(`store.init_store`, `store.set_credentials`, `store.resolve_credentials`,
`store.rotate_password`) was already written, reviewed and tested. This is
the surface over them, and nothing else.

**Nothing here reimplements a mechanism `session.py` already owns.** The
master password is read by `session._read_store_password` — one place that
decides `STACKWARD_PASSWORD`, the `/dev/tty` check and the refusal when
neither is available. The profile is selected by `session._resolve_profile`,
so `--profile`, `STACKWARD_PROFILE`, the repository's `profile =` and
`default_profile` mean here exactly what they mean for `exec`. The names a
profile must carry come from `session.CREDENTIAL_NAMES` via
`session._require_credential_names`, so `credentials set` refuses the same
envelope `exec` would refuse — at the point it is written, rather than at
the point it is needed. A second copy of any of those would be a second
answer to a question the tool must only have one answer to.

**A value is never an argument.** `credentials set` reads `KEY=VALUE` lines
from **stdin** (the format of the `.env` file this command exists to
replace, parsed by the same `set_secrets.parse_env_text` that reads one), or
prompts for each name without echo when stdin is a terminal. There is no
positional value and no `--value` flag: a command line is visible in `ps`
for the life of the call and lands in shell history forever. Unlike
`session.py`, this module is free to read `sys.stdin` — it spawns no child
that would need those bytes.

**Nothing here prints a credential value, on any path.** `list` prints
profile names; `show` prints credential *names* out of a decrypted envelope
and never the values beside them; `set` prints a count. Prompts are
`getpass`, so nothing typed is echoed either.

**A wrong master password is caught by the store's verifier**, not by
anything here: `set_credentials` and `rotate_password` both open the
known-plaintext envelope before they write, so a mistyped password refuses
with `PasswordError` rather than sealing something the rest of the store
cannot open. That is the failure mode the verifier was built for, and this
module's job is to report it, not to re-detect it.

**Rotation is all-or-nothing, and this module does not soften that.**
`store.rotate_password` rebuilds the whole document in memory and writes it
once; any failure leaves the store exactly as it was. The refusal is
surfaced verbatim — a partial rotation reported as success would be the
worst outcome available, and so would a total failure reported as one.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from .. import store
from ..store import StoreError
from . import session
from .check_config import fail_closed
from .session import ENV_STORE_PASSWORD
from .set_secrets import SetSecretsError, parse_env_text

# Prompts, kept together so that the wording of the two-step "type it twice"
# read is visible in one place. Each is passed to
# `session._read_store_password`, which is what decides whether a prompt
# happens at all -- see this module's docstring.
NEW_PASSWORD_PROMPT = "new credential store password: "
CONFIRM_PASSWORD_PROMPT = "confirm new credential store password: "
OLD_PASSWORD_PROMPT = "current credential store password: "


class CredentialsError(Exception):
    """Something specific to this module stopped a `credentials` subcommand:
    input that is not `KEY=VALUE`, a confirmation that did not match, an
    empty value for a name that must have one.

    Never carries a credential value -- every message here names a profile, a
    credential name, a line number or an environment variable."""


# The errors these entry points recognise and report as one line rather than
# letting `fail_closed` render them as "could not run (<type>)". `StoreError`
# covers `ProfileError` and `PasswordError`, which subclass it;
# `session.SessionError` arrives from the shared profile-selection and
# password-reading helpers.
_KNOWN_ERRORS = (StoreError, session.SessionError, CredentialsError)


def _report(exc: Exception) -> int:
    """Print `exc` as `error: ...` on stderr and return 2 -- the "could not
    run" code every refusal in this module and in `session.py` uses. Never 1:
    that code is reserved, across this whole tool, for "a credential was
    found" by the gate."""
    print(f"error: {exc}", file=sys.stderr)
    return 2


def _read_new_password() -> str:
    """A password that is about to be *set*, read twice and compared.

    Both reads go through `session._read_store_password`, so a store password
    is obtained in exactly one way in this codebase whatever it is wanted
    for; only the wording differs. When `STACKWARD_PASSWORD` is set, that
    function answers both reads from it without prompting and the comparison
    is trivially satisfied -- which is correct for the non-interactive path
    and is why `_refuse_unrotatable` exists to catch the one case where it
    silently means "nothing would change".

    The confirmation is not ceremony: a typo in a password that seals a store
    is not recoverable later by any means, because there is nothing else that
    can open what it sealed.
    """
    first = session._read_store_password(NEW_PASSWORD_PROMPT)
    second = session._read_store_password(CONFIRM_PASSWORD_PROMPT)
    if first != second:
        raise CredentialsError(
            "the two passwords do not match; nothing has been changed"
        )
    return first


def _prompt_for_values() -> dict[str, str]:
    """Ask for each required credential, without echo.

    `getpass`, not `input`: these are the values the whole store exists to
    keep off disk and out of scrollback. Exactly the names `exec`/`shell`
    inject are asked for; an envelope may hold more than that (see
    `session._require_credential_names`), but a name this tool does not
    consume is not one it should be inventing a prompt for -- pipe them in
    instead.

    A value is taken exactly as typed. The stdin path strips surrounding
    whitespace (see `set_secrets.parse_env_text`, whose `.env` semantics this
    command inherits on purpose); nothing is stripped here, because there is
    no file convention to match and no way to quote an intentional space at
    a prompt.
    """
    return {name: getpass.getpass(f"{name}: ") for name in session.CREDENTIAL_NAMES}


def _read_values() -> dict[str, str]:
    """The credentials to seal: `KEY=VALUE` lines from stdin, or prompts when
    stdin is a terminal.

    The branch is on `sys.stdin.isatty()` -- the literal question "was
    something piped into *this* descriptor", which is the right one to ask
    because this command, unlike `session.py`, is free to consume stdin.
    (`session.py` asks a different question, `/dev/tty`, for the different
    reason stated there: its stdin belongs to a child it is about to run.)

    `parse_env_text` raises in `set_secrets`'s vocabulary, which is re-raised
    here in this module's. Forwarding its message is safe, and deliberately
    so: it names an origin and a line number and never the line's own text.
    """
    if sys.stdin.isatty():
        return _prompt_for_values()

    text = sys.stdin.read()
    if not text.strip():
        raise CredentialsError(
            "no credentials on stdin: pipe KEY=VALUE lines in "
            "(for example `stackward credentials set --profile NAME < .env`), "
            "or run this from an interactive terminal to be prompted"
        )
    try:
        return parse_env_text(text, "stdin")
    except SetSecretsError as exc:
        raise CredentialsError(str(exc)) from exc


def _require_values(values: dict[str, str], profile: str) -> None:
    """Refuse an envelope `exec` would refuse, before it is written.

    Two checks, in the order that produces the more useful message.
    `session._require_credential_names` is the authority on *which* names
    must be present -- reused rather than restated, so that the set cannot
    drift between the command that writes an envelope and the command that
    reads it.

    A name present with an **empty** value is refused too, which that
    function does not do: it asks about presence, since an envelope is
    resolved by a provider that may legitimately have nothing for a name. At
    the moment of sealing there is no such ambiguity -- `AWS_ACCESS_KEY_ID=`
    in a piped `.env` is a placeholder line, not a credential, and sealing it
    produces a store that opens fine and then fails to authenticate with
    nothing pointing back here. Only the required names are checked: an extra
    name carried along in the same file is not this command's business.
    """
    session._require_credential_names(values, profile)
    empty = sorted(
        name for name in session.CREDENTIAL_NAMES if not values[name]
    )
    if empty:
        raise CredentialsError(
            f"profile {profile!r}: no value given for {', '.join(empty)}; "
            "an empty credential would seal cleanly and then fail to "
            "authenticate"
        )


def _refuse_unrotatable(old: str, new: str) -> None:
    """Refuse a rotation that would change nothing, naming the actual cause.

    The cause is almost always `STACKWARD_PASSWORD`: `_read_store_password`
    answers from it without prompting, so with it set both the old and the
    new password are the same string and the user was never asked anything.
    Reporting that as a bare "the passwords are the same" would describe a
    sameness they did not type and cannot explain, so the variable is named
    and the fix with it.
    """
    if old != new:
        return
    if ENV_STORE_PASSWORD in os.environ:
        raise CredentialsError(
            f"{ENV_STORE_PASSWORD} is set, so the current and the new password "
            f"were both read from it and rotation would change nothing: unset "
            f"{ENV_STORE_PASSWORD} and run this from an interactive terminal"
        )
    raise CredentialsError(
        "the new password is the same as the current one; nothing to rotate"
    )


@fail_closed
def cmd_credentials_init(_args: argparse.Namespace) -> int:
    """`stackward credentials init`: create the store.

    Refuses to overwrite an existing one -- `store.init_store` enforces that,
    because overwriting would discard every sealed envelope irrecoverably in
    response to a command a person could plausibly run twice.

    Says which password was used when it came from the environment. That is
    the supported non-interactive path, but a store created from an ambient
    variable the user has forgotten about is a store they cannot open
    tomorrow, and silence is what makes that possible.
    """
    from_env = ENV_STORE_PASSWORD in os.environ
    try:
        password = _read_new_password()
        store.init_store(password)
    except _KNOWN_ERRORS as exc:
        return _report(exc)

    print(f"credential store created at {store.credentials_path()}")
    if from_env:
        print(f"(sealed with the password in {ENV_STORE_PASSWORD})")
    print("next: stackward credentials set --profile <name>")
    return 0


@fail_closed
def cmd_credentials_set(args: argparse.Namespace) -> int:
    """`stackward credentials set [--profile NAME]`: seal a profile's
    credentials.

    **Replaces the profile's whole envelope**, and does not merge into it:
    `store.set_credentials` writes what it is given. That is why every
    required name must be supplied on every run -- a "set one name" that
    silently dropped the other two would be far worse than a refusal.

    The password is asked for last, so that malformed input, a missing name
    or an empty value is reported before anyone types it.
    """
    try:
        profile = session._resolve_profile(args.profile)
        values = _read_values()
        _require_values(values, profile.name)
        password = session._read_store_password()
        store.set_credentials(profile.name, values, password)
    except _KNOWN_ERRORS as exc:
        return _report(exc)

    print(f"sealed {len(values)} credential(s) for profile {profile.name!r}")
    print(f"check with: stackward credentials show --profile {profile.name}")
    return 0


@fail_closed
def cmd_credentials_list(_args: argparse.Namespace) -> int:
    """`stackward credentials list`: the profiles that have an envelope.

    Opens nothing and asks for no password -- `store.store_profiles` reads
    the document's keys. A profile that appears here has credentials sealed
    for it; one that does not appear may still exist in `config` and be
    perfectly usable with `login`, which never reads this file.

    The reverse case is real too, and is why this listing and `show` can
    disagree: an envelope whose `[profile.<name>]` table has been deleted
    from `config` is still in this file (deliberately -- see
    `store.rotate_password`, which re-seals it rather than destroying it)
    and is listed here, while `show` refuses it, because selecting a profile
    means finding its backend and there is no longer one to find. That is
    the honest answer in both places: the envelope exists, and nothing can
    currently use it.
    """
    try:
        profiles = store.store_profiles()
    except _KNOWN_ERRORS as exc:
        return _report(exc)

    if not profiles:
        print(
            "no profile has credentials in the store yet: "
            "stackward credentials set --profile <name>",
            file=sys.stderr,
        )
        return 0
    for name in profiles:
        print(name)
    return 0


@fail_closed
def cmd_credentials_show(args: argparse.Namespace) -> int:
    """`stackward credentials show [--profile NAME]`: which credential names
    a profile carries.

    **Names only, never values.** This is the command that answers "did that
    `set` land, and is anything missing" without anyone having to run `exec`
    against a real backend to find out. It decrypts, because the names live
    inside the envelope and there is nowhere else to read them from.

    Exit code 0 even when a required name is absent, with the shortfall
    noted on stderr: the question this command was asked -- what does this
    profile carry -- was answered correctly, and the store is not in a state
    anything needs to refuse. `exec` is what refuses an incomplete envelope,
    and it already does.
    """
    try:
        profile = session._resolve_profile(args.profile)
        password = session._read_store_password()
        credentials = store.resolve_credentials(profile.name, password)
    except _KNOWN_ERRORS as exc:
        return _report(exc)

    names = sorted(credentials)
    print(f"profile {profile.name!r} carries {len(names)} credential(s):")
    for name in names:
        print(f"  {name}")

    missing = [name for name in session.CREDENTIAL_NAMES if name not in credentials]
    if missing:
        print(
            f"missing, and required by exec/shell: {', '.join(missing)}",
            file=sys.stderr,
        )
    return 0


@fail_closed
def cmd_credentials_rotate(_args: argparse.Namespace) -> int:
    """`stackward credentials rotate`: re-seal every envelope under a new
    password.

    Every envelope in the credentials file, not every profile in `config`:
    `store.rotate_password` re-seals what is actually there, so an envelope
    whose `[profile.<name>]` table was removed is carried across rather than
    quietly destroyed.

    All-or-nothing. If any envelope fails to open, `store.rotate_password`
    raises and writes nothing, and this reports that refusal as-is -- the
    store still opens under the current password, which is what the message
    must not obscure.
    """
    try:
        old = session._read_store_password(OLD_PASSWORD_PROMPT)
        new = _read_new_password()
        _refuse_unrotatable(old, new)
        store.rotate_password(old, new)
    except _KNOWN_ERRORS as exc:
        return _report(exc)

    # No "remember to update STACKWARD_PASSWORD" note here: reaching this
    # line means the variable was unset, because `_refuse_unrotatable` above
    # cannot be passed while it is set.
    print("every envelope in the credential store was re-sealed")
    return 0
