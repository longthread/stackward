# stackward

Operations toolkit for Pulumi stacks that use a [DIY (self-managed) backend][diy] —
object storage or a local directory, rather than Pulumi Cloud.

[diy]: https://www.pulumi.com/docs/iac/concepts/state-and-backends/

Committing `Pulumi.<stack>.yaml` is the intended workflow: the `encryptionsalt`
at the top of the file is what makes it safe. What makes it *un*safe is a value
hand-edited in as plaintext, which looks exactly like every other line in the
diff. `stackward` exists to make that mistake hard.

> **Status: early.** `v0.1.0` ships `--version` and `doctor` only. It exists to
> prove the release and install path before features are built on top of it.

## What it will do

| Command | Purpose |
|---|---|
| `check-config` | Refuse plaintext credentials in a stack config file |
| `pre-commit` | The above, over staged content, as a git hook |
| `hooks install` | Install that hook, honouring `core.hooksPath` |
| `credentials` | Keep backend credentials in one encrypted, profile-scoped store |
| `exec` / `shell` | Run `pulumi` with those credentials injected into the child process |
| `login` | Point Pulumi at a profile's backend |
| `set-secrets` | Publish stack secrets from a declared manifest, values on stdin |
| `check-passphrase` | Verify a passphrase actually decrypts a stack, rather than merely exiting 0 |
| `sync-declared-secrets` | Regenerate the committed graph of secrets a repo's own models declare |

## Design commitments

These are constraints, not aspirations — each has a test.

- **The gate resolves no credentials.** `check-config` and `pre-commit` read
  files. They never prompt, never touch the network, and never reach credential
  provider code, so a commit cannot be blocked by an expired session or a flaky
  connection.
- **Policy lives in the consuming repository.** `stackward` supplies the
  mechanism; which keys are sensitive, which paths are managed and which model
  declares them are all read from the repo's `.stackward.toml`. The tool ships
  no defaults about anyone's infrastructure.
- **No backend assumptions.** A profile carries a Pulumi backend URL and it is
  passed through, so `s3://`, `gs://`, `azblob://` and `file://` all work. No
  storage vendor is named anywhere in this codebase.
- **Values reach the Pulumi CLI on stdin**, never in a command line, so they do
  not appear in `ps` or in shell history.
- **Failures are closed.** If the tool cannot determine whether a file is safe,
  it exits non-zero and says what to fix. It never passes a file because a
  check could not run.

## Install

Prebuilt binaries are published per release. Nothing else is needed to run
them — no Python, no package manager.

```sh
curl -fsSL https://raw.githubusercontent.com/longthread/stackward/main/install.sh | sh
```

The installer verifies the published checksum, and build provenance too when
`gh` is available. It installs each version into its own directory and swaps a
symlink, so upgrading never disturbs a running process — a git hook, for
instance.

In CI, pin both halves. `raw` serves any ref, so a tag pins the installer, and
`STACKWARD_VERSION` pins the binary it fetches:

```sh
curl -fsSL https://raw.githubusercontent.com/longthread/stackward/v0.1.1/install.sh \
  | STACKWARD_VERSION=v0.1.1 sh
```

Both endpoints are cached — `raw` for five minutes, and the
`releases/latest/download` redirect for around a minute after a release — so an
unpinned install can briefly return the previous version. That is the practical
argument for naming the version rather than asking for whatever is newest.

The installer comes from the git tree rather than a release asset so that a fix
to it ships without cutting a release. The binaries cannot work that way: they
are build artifacts rather than repository files, so they are fetched from
release assets, which is also what the provenance attestation covers.

| Platform | Requirement |
|---|---|
| Linux x86-64 | glibc 2.35 or newer — Ubuntu 22.04, Debian 12, RHEL 9 and later |
| macOS arm64 | Apple silicon |

Older glibc, other architectures, and Windows are not built. The installer says
so plainly rather than installing something that cannot start.

## Development

```sh
uv sync
uv run pytest
uv run stackward doctor
```

### Install this repository's own hooks

Run once per clone — git does not version hooks:

```sh
./scripts/install-hooks.sh /path/to/denylist.txt
```

This repository is public and was extracted from a private environment, so it
must not name anything belonging to it. Two layers enforce that, and they are
not interchangeable:

- **`scripts/pre-push`** refuses to push a commit containing a denied term.
  This is the layer that *prevents* a leak, because it runs before anything
  leaves the machine.
- **The `neutrality` CI job** checks the same thing on pull requests, as a
  backstop for a clone that never installed the hook.

CI alone would not be enough. It runs after a commit exists, so on a direct
push it can only report a leak that is already published — and a later commit
does not unpublish it. `main` therefore requires a pull request with that check
green, enforced for administrators too.

The denied terms are not stored here; the list would name everything it blocks.
Keep it somewhere private and point `STACKWARD_DENYLIST` at it. Both the hook
and the CI job refuse to run without one rather than passing by default.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
