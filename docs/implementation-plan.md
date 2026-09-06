# stackward — implementation plan

Task breakdown for the credential gate and the credential store. Each task
names the exact behaviour required; where a value is stated here, use it
verbatim rather than inventing an equivalent.

## Global Constraints

These bind every task. A change that violates one is wrong even if its own
tests pass.

1. **Runtime dependencies are `pyyaml` and `cryptography`, and nothing else.**
   Never import the Pulumi SDK — shell out to the `pulumi` binary, as the code
   already does for `git`. CI fails on a new direct dependency.
2. **The gate resolves no credentials.** `check-config` and `pre-commit` read
   files only. No code reachable from them may prompt, open a network
   connection, or import a credential provider. This is what keeps a commit
   from being blocked by an expired session, and it is asserted by a test.
3. **Fail closed.** If the tool cannot determine whether something is safe, it
   exits non-zero and names the fix. It never passes because a check could not
   run. A missing policy file is a refusal, not a skip.
4. **Never print a credential, and never print a denied term.** Report
   locations — file, line, config path — not the matching text. Errors and
   test failures included.
5. **No environment-specific knowledge.** No organisation, vendor, domain,
   bucket, region, namespace or model-class name appears in this repository.
   Everything specific to a user's environment arrives through
   `.stackward.toml` or a file that config names.
6. **Values reach external commands on stdin**, never in `argv` — command
   lines are visible in `ps` and shell history.
7. **Python 3.11+.** Use `tomllib` from the standard library; do not add a TOML
   dependency.

---

## Task 1: Path grammar

**Goal.** One canonical representation for a config path, used by findings,
manifests and matching alike. Two grammars in one tool is why the same helper
gets written twice.

**File.** `src/stackward/paths.py`, tests in `tests/test_paths.py`.

**Requirements.**
- Follow Pulumi's `--path` syntax: dotted segments, `[n]` for list indices, and
  `["..."]` for any key containing `.`, `[`, `]` or `"`.
- `parse(text) -> list[str | int]` — segments, list indices as `int`.
- `render(segments) -> str` — the inverse; bracket-quote only when required.
- `render(parse(x)) == x` for every canonical input, and
  `parse(render(y)) == y` for every segment list.
- A key containing a quote is escaped as `\"` inside the bracket form.
- Malformed input raises `ValueError` naming the offending position.

**Tests must cover.** A plain dotted path; a list index; a key containing a
dot; a key containing a bracket; a key containing a quote; an empty-string key;
a segment that looks like an integer but is a dict key (`["1"]` is the string
`"1"`, `[1]` is the index `1` — they must not collapse); round-trip both
directions; three distinct malformed inputs.

---

## Task 2: Repository configuration

**Goal.** Load and validate `.stackward.toml`, and enforce `min_version`.

**File.** `src/stackward/config.py`, tests in `tests/test_config.py`. Move
`find_repo_config` out of `cli.py` into here and update its callers.

**Requirements.**
- Parse with `tomllib`. Unknown top-level keys are an error naming the key —
  a typo must not be silently ignored.
- Recognised: `profile`, `python`, `min_version`, and tables `[check]`,
  `[secrets.*]`, `[secrets.source]`.
- `[check]` keys: `model_net` (`"artifact"` or `"none"`, default `"none"`),
  `stack_models` (table of `"<ns>" = "module:Class"`), `sensitive_keys` (list,
  **extends** the built-ins), `sensitive_parents` (list, **replaces** the
  built-ins), `allowed_references` (list). The extend/replace asymmetry is
  deliberate and must be documented in a comment where it is implemented.
- Built-in `sensitive_keys` default: `password`, `passphrase`, `token`,
  `secret`, `apikey`, `api_key`, `jwt`, `credential`, `private_key`,
  `access_key_id`, `secret_access_key`.
- `min_version` compares against `stackward.__version__` using a numeric
  tuple comparison, not string ordering. If the installed version is lower,
  exit non-zero naming the required version and the upgrade command — except
  for the commands in `cli.VERSION_CHECK_EXEMPT`, which must still run.
- Config absent → return `None`; callers decide. Config present but invalid →
  raise, never fall back to defaults.

**Tests must cover.** Each recognised key; an unknown key erroring; the
extend-vs-replace asymmetry proven for both lists; `min_version` accepting an
equal and a higher installed version and rejecting a lower one; that
`"0.10.0" > "0.9.0"` compares correctly (string ordering gets this wrong);
exemption of `doctor` and `self-update`.

---

## Task 3: The heuristic net and `check-config`

**Goal.** Detect a plaintext credential in a stack config file.

**Files.** `src/stackward/nets/heuristic.py`, `src/stackward/commands/check_config.py`,
tests in `tests/test_heuristic.py`.

**Requirements.**
- Walk the parsed YAML structure. Do **not** re-parse rendered path strings —
  a dict key may legally contain `.` or `[`, and string matching loses it.
  Emit paths using Task 1's `render`.
- A leaf is a **finding** when its key matches `sensitive_keys`
  (case-insensitive substring) **or** any ancestor key is in
  `sensitive_parents`, unless it is encrypted or empty.
- **Encrypted** means the leaf's *parent dict* is exactly `{"secure": ...}` —
  one key, named `secure`. Test the parent mapping, never
  `path.endswith(".secure")`: a malformed `{"secure": ..., "other": ...}` is
  not encrypted, and a suffix test would pass it.
- **Empty** means `value is None` or `value == ""`. Use identity for booleans:
  `value is True` / `value is False` are not empty. In Python `0 == False` and
  `1 == True`, so a containment test wrongly skips integer `0` and `1` — and
  integer leaves occur in real config. Integers `0` and `1` **are** findings.
- `allowed_references` lists paths that name another secret rather than
  holding one; they are never findings.
- `check-config FILE...` exits 0 when clean, 1 with findings, 2 on a usage or
  parse error. Findings print as `<file>: plaintext credential at '<path>'`,
  sorted, one per line. Never print the value.
- Unparseable YAML, or a YAML document that is not a mapping, is an error —
  not "no findings".

**Tests must cover.** Every rule above, each as its own test: a clean config;
a plaintext match by key; a match by sensitive parent; an encrypted leaf
passing; the malformed `{secure, other}` leaf being flagged; `None` and `""`
passing; `True`/`False` passing; integers `0`, `1` and a larger integer being
flagged; a dict key containing a `.` being reported with correct bracket
quoting; an `allowed_references` path passing; a non-mapping document erroring.

---

## Task 4: `pre-commit`

**Goal.** Run the gate over staged content, as a git hook body.

**Files.** `src/stackward/commands/pre_commit.py`, tests in `tests/test_pre_commit.py`.

**Requirements.**
- Select staged files with `git diff --cached --name-only --diff-filter=ACMR`.
  `R` matters: a rename plus an edit stages as `R`, and omitting it lets that
  change bypass the gate entirely.
- Match stack config files by basename: `Pulumi.<stack>.yaml`, where `<stack>`
  may contain dots (`Pulumi.prod.eu.yaml` is legal). Match `Pulumi.` prefix and
  `.yaml` suffix, not a restrictive character class.
- Refuse outright — before any other check — if a Pulumi **state export** is
  staged: a file named `state.json` or matching `*.stack-export.json`. These
  carry every resource input including credentials.
- Check **staged** content, not the working tree: read each file with
  `git show ":<path>"`. An unstaged fix must not let a staged credential
  through.
- Handle a path containing spaces or non-ASCII: use `-z` and split on NUL.
- An unmerged index entry, or a staged symlink whose blob is not YAML, is an
  error, not a pass.
- Exit 0 clean, 1 blocked. On block, print the findings and the exact
  `pulumi config set --secret --path '<path>'` remediation, and state that
  `--no-verify` bypasses.

**Tests must cover.** Each rule, using a real temporary git repository (create
it in the test with `git init`; do not mock git). Include: a rename+edit
staged as `R` being caught; a staged state export being refused; the staged
version being checked rather than the working tree; a filename containing a
space.

---

## Task 5: `hooks install`

**Goal.** Install the `pre-commit` hook into a repository, safely.

**Files.** `src/stackward/commands/install_hooks.py`, tests in
`tests/test_install_hooks.py`.

**Requirements.**
- Resolve the hooks directory with `git rev-parse --git-path hooks`. Do not
  hardcode `.git/hooks`: `core.hooksPath` is redirected by several common
  tools, and installing where git does not look is a silent no-op.
- **Refuse if the resolved hooks directory is tracked by git** (test with
  `git ls-files --error-unmatch <dir>` or equivalent). Writing there commits a
  hook that runs a binary other clones may not have installed. Print the
  reason and suggest chaining from the tracked hook instead.
- The hook body contains **no logic**: it locates the repository and execs
  `stackward pre-commit`. Embed the absolute path to the running executable,
  with a `PATH` lookup as fallback. A bare `exec stackward` from a GUI client
  with a minimal `PATH` exits 127 and blocks every commit.
- Mark the generated hook with a fixed marker line so it can be recognised
  later. If an existing hook is present and lacks the marker, back it up to
  `<hook>.backup.<UTC timestamp>` rather than refusing or overwriting.
- `chmod 0o755` explicitly.
- Idempotent: installing twice leaves one hook and creates no second backup.

**Tests must cover.** Each rule, against a real temporary git repository:
default location; a redirected `core.hooksPath`; refusal on a tracked hooks
directory; an existing foreign hook being backed up; idempotence; the mode
being `0o755`; the body containing an absolute path.

---

## Task 6: Credential store — profiles and encryption

**Goal.** One profile-scoped store for backend credentials, encrypted at rest.

**Files.** `src/stackward/store.py`, `src/stackward/crypto.py`, tests in
`tests/test_store.py` and `tests/test_crypto.py`.

**Requirements.**
- Location: `${XDG_CONFIG_HOME:-~/.config}/stackward/`, holding `config`
  (TOML, mode 0o600, backend identity) and `credentials`
  (mode 0o600, directory 0o700).
  - **Revised during implementation**, from "mode 0o644, backend identity, no
    credentials". Both halves were superseded by the requirement below that
    `backend_url` be passed through **verbatim**: `postgres://user:password@host/db`
    is a documented, supported backend form, so `config` can hold a credential
    after all. Enforcing "no credentials" would mean parsing a URL this project
    deliberately does not parse, and would reject a backend Pulumi accepts.
    The consequences are carried instead: mode `0o600` (`store.CONFIG_MODE`),
    and a parse failure reports a coordinate rather than tomllib's own message,
    which quotes document text (`config.toml_position`). Recorded here rather
    than corrected silently, because "no credentials" reads like a rule the
    code broke when it is a premise the code disproved.
- `config` holds `default_profile` and `[profile.<name>]` tables. A profile
  carries **either** `backend_url` (passed through verbatim — `s3://`, `gs://`,
  `azblob://`, `file://` must all work) **or** the component form `bucket`,
  `prefix`, `endpoint`, `region` for an S3-compatible store. There is no
  default backend: a profile is required and an absent one is an error.
- Profile selection precedence, first match wins: `--profile`,
  `STACKWARD_PROFILE`, the repo config's `profile`, `default_profile`, then
  error. Never guess.
- **Encryption**: one envelope per profile section. Argon2id for key
  derivation (`cryptography.hazmat.primitives.kdf.argon2.Argon2id`) and
  AES-256-GCM for sealing. Envelope is versioned JSON with fields `v`, `kdf`,
  `salt`, `nonce`, `ct`, each base64. A fresh random salt and nonce per seal.
- **The profile name is the AAD**, so an envelope cannot be moved between
  profiles undetected.
- A **store-level verifier**: a known-plaintext envelope written at init, so a
  mistyped password fails immediately rather than weeks later at first use.
- Password rotation re-seals **every** envelope or none.
- All writes are atomic: write to a temporary file in the same directory,
  `fsync`, set the mode, then `rename`. A crash must not leave a truncated
  credentials file.
- Warn when a file's mode is more permissive than required.

**Tests must cover.** Seal/open round trip; a wrong password failing cleanly
via the GCM tag rather than returning garbage; an envelope resealed under
profile A failing to open as profile B (AAD); the ciphertext containing none
of the plaintext (assert a known value's bytes are absent); rotation being
all-or-nothing under an induced failure; atomic write leaving the original
intact when the write fails; every precedence branch of profile selection;
the absent-profile error.

---

## Task 7: `login`, `exec`, `shell`

**Goal.** Point Pulumi at a profile's backend, and run commands with that
profile's credentials in the environment.

**Files.** `src/stackward/commands/session.py`, tests in `tests/test_session.py`.

**Requirements.**
- `pulumi login` persists only the backend URL. `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` and `PULUMI_CONFIG_PASSPHRASE` are read from the
  environment of **every subsequent** `pulumi` process. A store only
  `stackward` can read therefore breaks a plain `pulumi up`, and `exec` is what
  makes the store usable at all.
- `stackward exec -- <cmd...>` decrypts the profile, injects those three
  variables **and** `PULUMI_BACKEND_URL`, and runs the command as a child.
  Secrets exist only in the child's environment; never write them to disk,
  and never log them.
- `stackward shell` is the same with the user's `$SHELL` as the command.
- `stackward login` resolves the profile's backend URL and runs
  `pulumi login <url>`.
- **Backend guard:** before any mutating command, compare the resolved
  profile's backend URL with Pulumi's current backend; refuse on mismatch,
  naming both. This is what makes profiles safe rather than merely convenient.
- Propagate the child's exit status exactly.
- `--` terminates option parsing; `exec` with no command is a usage error.

**Tests must cover.** The injected environment containing exactly the expected
keys (assert the values are passed but never logged); exit-status propagation
including a non-zero and a signal; the backend guard refusing a mismatch and
allowing a match; `exec` with no command erroring; that no secret appears in
captured stdout or stderr on any path.

---

## Task 8: `set-secrets`

**Goal.** Publish stack secrets from a declared manifest, values on stdin.

**Files.** `src/stackward/commands/set_secrets.py`, tests in
`tests/test_set_secrets.py`.

**Requirements.**
- Manifest lives in `.stackward.toml` under `[secrets."<project dir>"]` with
  `secret`, `plaintext` and `unmanaged` tables mapping config path → logical
  name. A single-project repository uses one table; project directories are
  always repository-root-relative.
- Value sources, precedence highest first: the process environment, then each
  file in `[secrets.source].files` in order, later overriding earlier.
  `{stack}` in a filename is substituted.
- Parse `KEY=VALUE` files: ignore blanks and `#` comments, strip a leading
  `export `, strip one matched pair of surrounding quotes. Warn when such a
  file's mode is more permissive than 0o600.
- Values are passed to `pulumi config set --secret --path <path>` **on stdin**.
- A name whose value is unset is **skipped, never cleared** — the command only
  ever touches what it was given a value for.
- `--dry-run` reports what would be set without running anything.
- `--stack S` is forwarded to `pulumi`; omitted when not given.
- Paths listed under `[required]` cause a non-zero exit when unresolved.
- **Drift detection**: for declared `[[drift_pairs]]` of a bootstrap and a
  managed variable, compare the local bootstrap value against the published
  value (read with `pulumi config get`, which decrypts — `pulumi config` list
  does not, and exits 0 under any passphrase). Warn only when they differ and
  the published value was readable. Never print either value.
- A timeout or a missing `pulumi` binary is reported and returns failure for
  that entry without aborting the run mid-rotation; the summary names what was
  not set.

**Tests must cover.** Precedence across environment and two files; skip-if-unset;
`--dry-run` writing nothing; the value reaching the subprocess via stdin and
**not** via `argv`; a required unresolved path failing; drift warning firing on
a difference and staying silent when equal or unreadable; a subprocess timeout
being reported without aborting the remaining entries; that no test prints a
value.

---

## Task 9: `check-passphrase`

**Goal.** Prove a passphrase actually decrypts a stack, which no ordinary
`pulumi` command does.

**Files.** `src/stackward/commands/check_passphrase.py`, tests in
`tests/test_check_passphrase.py`.

**Requirements.**
- Read the passphrase from stdin, falling back to `getpass` when stdin is a
  TTY. Never accept it as an argument.
- Obtain a stack's state with `pulumi stack export --stack <name>` and read
  `deployment.secrets_providers.state.salt`, whose format is
  `v1:<b64 salt>:v1:<b64 nonce>:<b64 ciphertext>`.
- Derive with **PBKDF2-HMAC-SHA256, 1,000,000 iterations, 32-byte key** — this
  must match Pulumi's own scheme, and is deliberately different from the
  Argon2id used for our own store, which is not interoperating with anything.
- Open with AES-GCM; the plaintext is the literal `b"pulumi"`. Report per
  stack: accepted or rejected.
- Exit 0 only if every named stack accepted.
- Report a fingerprint (first 16 hex of SHA-256) rather than any part of the
  passphrase, so two candidates can be told apart in a log safely.

**Tests must cover.** A synthesised salt that the correct passphrase opens; the
same salt rejecting a wrong passphrase; a malformed salt string erroring rather
than crashing; the passphrase never appearing in output; the fingerprint being
stable and 16 characters.

---

## Task 10: The model net

**Goal.** Detect a credential the heuristic cannot name, using declarations
carried in a committed artifact.

**Files.** `src/stackward/nets/model.py`, `src/stackward/bootstrap/regen.py`,
`src/stackward/commands/sync_declared.py`, tests in `tests/test_model_net.py`.

**Background that determines the design.** A shipped binary bundles its own
interpreter and cannot import the consuming repository's classes. The
declarations must therefore cross that boundary as data. The obvious form — a
flat list of path patterns — is wrong: a self-referential model yields a
finite pattern list while the data it describes is unbounded, so the flat form
fails **open** exactly where it must fail closed.

**Requirements.**
- The artifact is a **graph**, not a pattern list: models → fields → marks →
  child-model references, written to `.stackward/declared-secrets.json`. The
  matcher walks the graph and the data together, so recursion and mutual
  recursion work by construction.
- Marking convention, published as this tool's contract:
  `json_schema_extra={"secret": True}` on a pydantic field. `[check].declared_paths_fn`
  overrides it for a non-standard convention.
- A marked field covers **everything beneath it**, whatever its annotation —
  matching is prefix-based, not exact.
- `sync-declared-secrets` shells out to the interpreter named by `python` in
  the repo config, imports the declared classes, and writes the artifact. It
  **refuses** any annotation containing a model that it cannot walk —
  `Union[A, B]`, `Sequence[A]`, `Mapping[str, A]`, bare `dict` — rather than
  silently emitting nothing for it.
- Record a `sources` map: every file contributing a marked field, by git blob
  id, walking the **MRO** so a mark inherited from a base class in another
  file is captured. For modules outside the repository, record the dependency
  lockfile's hash instead.
- Freshness: `check-config` reads the artifact **from the git index**
  (`git show ":.stackward/declared-secrets.json"`), not the working tree — an
  unstaged regeneration must not let a stale committed artifact pass. A blob-id
  mismatch is a hard failure naming the regeneration command.
- `model_net = "none"` is a declared mode. It is never a fallback: a missing or
  stale artifact under `"artifact"` mode fails, it does not degrade.

**Tests must cover.** Synthetic pydantic models only (pydantic is a test-only
dependency). Include: a self-referential model (`Node` with `children:
list[Node]`) matching at depth; mutual recursion; a mark inherited from a base
class in another module; a mark carried by an `Annotated` alias defined
elsewhere; prefix coverage of a marked container's descendants; a dict key
containing `.`; refusal of each unwalkable annotation; a stale artifact
failing; `"none"` mode skipping cleanly.

---

## Task 11: Provider interface

**Goal.** Make the credential source swappable, so adopting a managed secret
store is a configuration change rather than a migration.

**Files.** `src/stackward/providers/__init__.py` and one module per provider,
tests in `tests/test_providers.py`.

**Requirements.**
- Two roles. `CredentialStore` resolves a profile's bootstrap set;
  implementations: `file` (Task 6's encrypted store) and `env`.
  `SecretSource` resolves manifest values; implementations: `dotenv` and `env`.
- Selection is by `provider = "..."` in the profile's `credentials` table and
  in `[secrets.source]` respectively.
- A remote provider shells out to its CLI. Adding one must not add a Python
  dependency.
- **No provider module may be imported from the gate path.** Enforce with a
  test that imports the `check-config` entry point in a subprocess and asserts
  no provider module appears in `sys.modules`.
- Fetched values are held in memory for the invocation only; never written to
  disk.

**Tests must cover.** The same operation producing identical results and
identical subprocess invocations against two different providers seeded with
the same values — this is the "swapping the provider changes nothing for the
caller" guarantee, and asserting it is what makes it true; the gate-path import
test; that a provider failure is reported rather than silently yielding an
empty value.
