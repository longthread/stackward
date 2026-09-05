#!/bin/sh
# Install this repository's own git hooks.
#
# stackward installs commit gates on other repositories; it should hold itself
# to the same standard. Run once per clone — git does not version hooks.
#
#     ./scripts/install-hooks.sh [path-to-denylist]
#
# The denylist is not stored in this repository, because it names the terms it
# blocks. Pass its path, or set STACKWARD_DENYLIST in your environment. The
# copy is written to .git/, which git never tracks.
set -eu

REPO_ROOT="$(git rev-parse --show-toplevel)"
GIT_DIR="$(git rev-parse --git-dir)"
# Honour core.hooksPath: husky, pre-commit and corporate global config all
# redirect it, and hardcoding .git/hooks installs where git never looks.
HOOKS_DIR="$(git rev-parse --git-path hooks)"
mkdir -p "$HOOKS_DIR"

SOURCE="${1:-${STACKWARD_DENYLIST:-}}"
TARGET="$GIT_DIR/neutrality-denylist.txt"

if [ -n "$SOURCE" ]; then
  [ -f "$SOURCE" ] || { echo "install-hooks: no such denylist: $SOURCE" >&2; exit 1; }
  cp "$SOURCE" "$TARGET"
  chmod 600 "$TARGET"
  echo "Denylist copied to $TARGET"
elif [ -f "$TARGET" ]; then
  echo "Using existing denylist at $TARGET"
else
  cat >&2 <<EOF
install-hooks: no denylist supplied.

  ./scripts/install-hooks.sh /path/to/denylist.txt

The hook refuses to push without one rather than passing by default.
EOF
  exit 1
fi

# A one-line shim rather than a copy: the logic then updates with git pull,
# instead of going stale in every clone that installed it once.
cat > "$HOOKS_DIR/pre-push" <<EOF
#!/bin/sh
# Installed by scripts/install-hooks.sh - edit that, not this.
exec "$REPO_ROOT/scripts/pre-push" "\$@"
EOF
chmod 755 "$HOOKS_DIR/pre-push"

echo "Installed pre-push hook -> $HOOKS_DIR/pre-push"
echo "Verify with: git push --dry-run"
