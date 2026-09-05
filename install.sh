#!/bin/sh
# Install stackward.
#
#   curl -fsSL https://github.com/longthread/stackward/releases/latest/download/install.sh | sh
#
# Environment:
#   STACKWARD_VERSION  tag to install (default: latest). Pin this in CI — a
#                      build must not move underneath you.
#   STACKWARD_PREFIX   install root (default: ~/.local)
set -eu

REPO="longthread/stackward"
VERSION="${STACKWARD_VERSION:-latest}"
PREFIX="${STACKWARD_PREFIX:-$HOME/.local}"
LIBDIR="$PREFIX/share/stackward"
BINDIR="$PREFIX/bin"

die() { printf 'install: %s\n' "$*" >&2; exit 1; }

# Asset names are built from uname, and the release matrix uses those values
# verbatim. Anything else is a 404 with a confusing message.
OS="$(uname -s)"
ARCH="$(uname -m)"
ASSET="stackward-${OS}-${ARCH}.tar.gz"

case "$OS-$ARCH" in
  Linux-x86_64|Darwin-arm64) ;;
  *) die "no prebuilt binary for ${OS}-${ARCH}. Build from source: https://github.com/${REPO}" ;;
esac

if [ "$VERSION" = "latest" ]; then
  BASE="https://github.com/${REPO}/releases/latest/download"
else
  BASE="https://github.com/${REPO}/releases/download/${VERSION}"
fi

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar  >/dev/null 2>&1 || die "tar is required"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

printf 'Downloading %s (%s)\n' "$ASSET" "$VERSION"
curl -fsSL "${BASE}/${ASSET}"        -o "$TMP/$ASSET"    || die "download failed: ${BASE}/${ASSET}"
curl -fsSL "${BASE}/${ASSET}.sha256" -o "$TMP/$ASSET.sha256" || die "checksum download failed"

# Transfer integrity. This is not provenance: whoever could replace the tarball
# could replace the checksum beside it. The attestation check below is the one
# that establishes where the artifact came from.
printf 'Verifying checksum\n'
( cd "$TMP" && if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --check --status "$ASSET.sha256"
  else
    shasum -a 256 --check --status "$ASSET.sha256"
  fi ) || die "checksum mismatch — do not use this download"

if command -v gh >/dev/null 2>&1; then
  printf 'Verifying build provenance\n'
  gh attestation verify "$TMP/$ASSET" --repo "$REPO" >/dev/null 2>&1 \
    && printf '  provenance ok: built by %s CI\n' "$REPO" \
    || printf '  WARNING: provenance could not be verified (checksum only)\n'
else
  printf 'Note: install gh to verify build provenance; checksum verified only\n'
fi

# Determine the version so the payload can live in a versioned directory.
# Extracting over a running onedir truncates mmapped shared objects and
# SIGBUSes anything mid-invocation — a git hook, for instance.
mkdir -p "$TMP/x" && tar -xzf "$TMP/$ASSET" -C "$TMP/x"
[ -x "$TMP/x/stackward/stackward" ] || die "archive does not contain the expected layout"
RESOLVED="$("$TMP/x/stackward/stackward" --version | awk '{print $2}')"
[ -n "$RESOLVED" ] || die "extracted binary did not report a version"

TARGET="$LIBDIR/$RESOLVED"
mkdir -p "$LIBDIR" "$BINDIR"
rm -rf "$TARGET.incoming"
mv "$TMP/x/stackward" "$TARGET.incoming"
rm -rf "$TARGET"
mv "$TARGET.incoming" "$TARGET"

# Swap the symlink atomically: rename(2) over an existing path replaces it in
# one step, so no invocation ever sees a missing or half-written binary.
ln -sfn "$TARGET/stackward" "$BINDIR/.stackward.new"
mv -f "$BINDIR/.stackward.new" "$BINDIR/stackward"

printf '\nInstalled stackward %s -> %s\n' "$RESOLVED" "$BINDIR/stackward"
case ":$PATH:" in
  *":$BINDIR:"*) ;;
  *) printf 'Add it to your PATH:\n  export PATH="%s:$PATH"\n' "$BINDIR" ;;
esac
printf 'Verify with: stackward doctor\n'
