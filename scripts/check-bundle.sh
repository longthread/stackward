#!/usr/bin/env bash
# Prove that a built bundle actually works. Takes the binary to check as $1.
#
# **One definition, run by both release jobs.** `build` runs it against what
# it has just produced; `publish` runs it against the artifact it is about to
# release, unpacked exactly as a user would. A second, separately-maintained
# copy is precisely how the publish job came to smoke-test only `--version`
# and `doctor` -- the two commands registered directly rather than through
# `cli._dispatch`, and so the only two that kept passing while every lazily
# dispatched command was missing from the binary entirely. The bundle that
# regression would have shipped passes a `--version`/`doctor` check by
# construction.
#
# bash, not sh: `set -o pipefail` is not POSIX and is silently absent under
# dash, which would let the `doctor | tee` pipeline below report the exit
# status of `tee`.
set -euo pipefail

bundle="${1:?usage: check-bundle.sh <path to the stackward binary>}"
# Resolved to an absolute path once, here: the checks below run from
# elsewhere on purpose, and each caller invokes this from a different
# directory.
case "$bundle" in
  /*) ;;
  *) bundle="$PWD/$bundle" ;;
esac

echo "Checking bundle: $bundle"

# A bundle that builds is not a bundle that works. doctor performs a real
# Argon2id + AES-GCM round trip, so this catches a native extension that
# failed to freeze -- the classic failure of this route.
#
# doctor also reads `regen.py` back out of the bundle through the same
# accessor `sync-declared-secrets` uses. That file is carried by the
# `--add-data` entry in the build job rather than by import analysis, so
# dropping it would produce a binary that builds, starts, and then fails at
# the one command that needs it -- in the shipped artifact and nowhere else.
# doctor exits non-zero on that, and the grep states the expectation so a
# future edit to doctor's output cannot silently retire this check.
"$bundle" --version

# A temporary file, not `doctor.txt` in the working directory: one caller
# runs this from the checkout and the other from the directory it unpacked
# the release artifacts into.
#
# Every `mktemp` here passes an explicit `X`-suffixed template, which GNU and
# BSD both accept. The build matrix includes a macOS runner and this workflow
# only ever fires on a tag, so a bare `mktemp` that BSD rejects would first be
# discovered during a release.
doctor_out="$(mktemp "${TMPDIR:-/tmp}/stackward-doctor.XXXXXX")"
trap 'rm -f "$doctor_out"' EXIT
"$bundle" doctor | tee "$doctor_out"
grep -q "^model walker    regen.py readable" "$doctor_out"

# A lazily dispatched command must actually load in the bundle.
# `cli._dispatch` imports each command module only when its subcommand runs,
# and PyInstaller can only see an import it can read as a literal statement:
# while that dispatch named its module dynamically, every one of these
# commands was missing from the binary and died on `ModuleNotFoundError` with
# a traceback and exit code 1 -- the code reserved for "a credential was
# found" -- while `--version` and `doctor` kept passing this very check.
#
# Two commands, from two different modules, and neither needs a credential, a
# store or a network: each refuses with exit code 2 after its module has
# loaded, which is exactly the evidence wanted. `set -e` is off for these
# because a non-zero exit is the expected outcome.
#
# Run from an empty directory, not the caller's: `cli.main` enforces a
# repository's `min_version` floor *before* dispatch, so a `.stackward.toml`
# above the working directory could make both commands exit 2 without either
# module ever loading -- the check would then pass on exactly the evidence it
# exists to reject. That floor is now read on its own, past any key the
# binary does not recognise (`config.read_min_version`), so an *invalid*
# config above the working directory can produce the refusal too, which makes
# the empty directory load-bearing rather than merely tidy.
for command in "credentials list" "exec"; do
  # Made before `set -e` is lifted: a `mktemp` that fails is a broken runner,
  # not evidence about the bundle, and must not be read as one.
  probe="$(mktemp -d "${TMPDIR:-/tmp}/stackward-probe.XXXXXX")"
  set +e
  # shellcheck disable=SC2086
  out="$(cd "$probe" && XDG_CONFIG_HOME="$PWD" "$bundle" $command 2>&1)"
  code=$?
  set -e
  if [ "$code" -ne 2 ]; then
    echo "::error::\`stackward $command\` exited $code, expected 2" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  rm -rf "$probe"
  case "$out" in
    *ModuleNotFoundError*|*Traceback*|*"invalid choice"*)
      # "invalid choice" is the other way this passes for the wrong reason:
      # an unregistered subcommand also exits 2, from argparse, having loaded
      # nothing at all.
      echo "::error::\`stackward $command\` did not load in the bundle" >&2
      printf '%s\n' "$out" >&2
      exit 1
      ;;
  esac
done

echo "Bundle check passed: $bundle"
