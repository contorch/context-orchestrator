#!/usr/bin/env bash
# install-claude-context.sh — connect Claude Code to this checkout's
# context-orchestrator: the MCP server, the auto-context hook (contorch-hook),
# the marked CLAUDE.md block and the transcripts skill.
#
# All of it is done by `contorch-memory claude install` (the one owner of
# Claude Code integration); this script only finds the checkout's venv.
# Idempotent. Run setup.sh first.
#
# Usage:
#   ./install-claude-context.sh                # install / update
#   ./install-claude-context.sh --uninstall    # remove what Contorch installed
#   (--copy / --symlink are accepted for old callers and ignored)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORY="$REPO_ROOT/.venv/bin/contorch-memory"
ACTION="install"

for arg in "$@"; do
    case "$arg" in
        --uninstall) ACTION="uninstall" ;;
        --copy|--symlink) ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "unknown arg: $arg (try --help)" >&2
            exit 2 ;;
    esac
done

# Channel guard: another install (Contorch.app, Homebrew) may own this Mac.
if [ -f "$REPO_ROOT/scripts/contorch_channel_guard.sh" ]; then
    . "$REPO_ROOT/scripts/contorch_channel_guard.sh"
    contorch_channel_guard || exit $?
fi

if [ ! -x "$MEMORY" ]; then
    echo "✗ $MEMORY not found — run setup.sh first" >&2
    exit 1
fi

"$MEMORY" claude "$ACTION" --channel "${CONTORCH_CHANNEL:-dev}"
echo ""
echo "Restart Claude Code to pick up the changes (settings are read at startup)."
