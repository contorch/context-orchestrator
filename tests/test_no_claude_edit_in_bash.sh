#!/bin/bash
# Claude Code integration has one owner: `contorch-memory claude …`
# (src/context_orchestrator/claude_install.py). No shell script in this repo
# may register the MCP server or edit Claude Code's settings.json, CLAUDE.md,
# hooks or skills itself. Runs under macOS's /bin/bash 3.2.
set -u
cd "$(dirname "$0")/.." || exit 2
status=0
for f in $(git ls-files '*.sh'); do
    case "$f" in tests/test_no_claude_edit_in_bash.sh) continue ;; esac
    # code lines only (comments may describe what the owner does)
    if grep -nE 'claude[[:space:]]+mcp[[:space:]]+(add|add-json|remove)|CLAUDE\.md|UserPromptSubmit|settings\.json|\.claude/(hooks|skills)' "$f" \
        | grep -vE '^[0-9]+:[[:space:]]*#'; then
        echo "✗ $f edits Claude Code directly — call \`contorch-memory claude install|uninstall\` instead" >&2
        status=1
    fi
done
[ "$status" -eq 0 ] && echo "✓ no shell script edits Claude Code directly"
exit "$status"
