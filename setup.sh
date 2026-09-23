#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# bootstrap.sh exports PYTHON when Apple's python3 (3.9) is too old; honor it.
PYTHON="${PYTHON:-python3}"
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
TEMPLATE="$SCRIPT_DIR/claude-md-template.md"
MARKER="CONTEXT-ORCHESTRATOR"
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
SRC_PATH="$SCRIPT_DIR/src"

INSTALL_LAUNCHD=1
for arg in "$@"; do
    case "$arg" in
        --no-launchd) INSTALL_LAUNCHD=0 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--no-launchd]

  --no-launchd    Install MCP server but skip transcript-watcher launchd
                  auto-start. Use if you want to run the watcher manually.
EOF
            exit 0
            ;;
    esac
done

echo "Setting up context-orchestrator..."

# ---------------------------------------------------------------------------
# Prereq check
# ---------------------------------------------------------------------------
PREREQS_OK=1
fail() { echo "  ✗ $1"; PREREQS_OK=0; }
ok()   { echo "  ✓ $1"; }

echo ""
echo "Checking prerequisites..."

if command -v "$PYTHON" &>/dev/null; then
    PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        ok "Python $PY_VERSION ($PYTHON)"
    else
        fail "Python $PY_VERSION too old. Need 3.10+ (set PYTHON=/path/to/python3.12 or brew install python@3.12)."
    fi
else
    fail "$PYTHON not on PATH. Install via Homebrew (brew install python@3.12) or python.org."
fi

if command -v git &>/dev/null; then
    ok "git $(git --version | awk '{print $3}')"
else
    fail "git not on PATH. Comes with Xcode CLI tools."
fi

if command -v claude &>/dev/null; then
    ok "claude CLI on PATH"
else
    echo "  ⚠ claude CLI not on PATH. MCP registration will print manual instructions instead."
fi

if [ "$PREREQS_OK" -eq 0 ]; then
    echo ""
    echo "Prerequisites missing. Fix and re-run."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# 1. venv + install
# ---------------------------------------------------------------------------
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$SCRIPT_DIR/.venv"
fi

echo "Installing dependencies..."
"$SCRIPT_DIR/.venv/bin/pip" install -q --upgrade pip
"$SCRIPT_DIR/.venv/bin/pip" install -q -e "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 2. Update ~/.claude/CLAUDE.md (append, idempotent)
# ---------------------------------------------------------------------------
mkdir -p "$HOME/.claude"

if [ -f "$CLAUDE_MD" ] && grep -q "$MARKER" "$CLAUDE_MD"; then
    echo "CLAUDE.md already has context-orchestrator instructions — skipping."
else
    echo "Appending context-orchestrator instructions to $CLAUDE_MD..."
    cat "$TEMPLATE" >> "$CLAUDE_MD"
fi

# ---------------------------------------------------------------------------
# 3. Register MCP server with Claude Code
# ---------------------------------------------------------------------------
if command -v claude &>/dev/null; then
    # Remove existing entry if present (idempotent)
    claude mcp remove --scope user context-orchestrator 2>/dev/null || true

    echo "Registering MCP server with Claude Code..."
    claude mcp add -t stdio -s user \
        -e "PYTHONPATH=$SRC_PATH" \
        -- context-orchestrator "$PYTHON_PATH" -m context_orchestrator.server
else
    cat <<EOF

Claude Code CLI not found. Add manually to ~/.claude.json under "mcpServers":

  "context-orchestrator": {
    "type": "stdio",
    "command": "$PYTHON_PATH",
    "args": ["-m", "context_orchestrator.server"],
    "env": {
      "PYTHONPATH": "$SRC_PATH"
    }
  }
EOF
fi

# ---------------------------------------------------------------------------
# 4. Cut over to chroma HTTP server (single source of truth, no SQLite contention)
# ---------------------------------------------------------------------------
CHROMA_DIR="$HOME/.context-orchestrator/chroma"
if [ "$INSTALL_LAUNCHD" -eq 1 ]; then
    echo ""
    echo "Setting up chroma HTTP server..."

    # Stop the watcher if it's running, so it releases any open handle while we cut over.
    if [ -f "$HOME/Library/LaunchAgents/com.contorch.transcript-watcher.plist" ]; then
        launchctl unload "$HOME/Library/LaunchAgents/com.contorch.transcript-watcher.plist" 2>/dev/null || true
    fi

    # One-time backup of any existing chroma data before flipping to the daemon.
    if [ -d "$CHROMA_DIR" ] && [ -f "$CHROMA_DIR/chroma.sqlite3" ]; then
        STAMP=$(date +%Y%m%d-%H%M%S)
        BACKUP="$HOME/.context-orchestrator/chroma.backup-$STAMP"
        if [ ! -e "$BACKUP" ]; then
            echo "  Backing up chroma data to $BACKUP (one-time, ~$(du -sh "$CHROMA_DIR" | cut -f1))..."
            cp -R "$CHROMA_DIR" "$BACKUP"
        fi
    fi

    "$SCRIPT_DIR/.venv/bin/context-orchestrator-chroma" install

    # Wait for the daemon to come up (a cold chroma start can take 20-30s on
    # a slow disk / first import), then verify — failing loudly, with the log.
    CHROMA_UP=0
    for _ in $(seq 1 60); do
        if "$SCRIPT_DIR/.venv/bin/context-orchestrator-chroma" status >/dev/null 2>&1; then
            CHROMA_UP=1; break
        fi
        sleep 1
    done
    if [ "$CHROMA_UP" -ne 1 ]; then
        echo "chroma daemon did not come up within 60s. Last log lines:"
        tail -20 "$HOME/.context-orchestrator/chroma-daemon.log" 2>/dev/null || true
        exit 1
    fi
    "$SCRIPT_DIR/.venv/bin/context-orchestrator-chroma" status

    # No watcher daemon: the MCP server indexes new transcripts on demand.
    # Retire an agent left by an older install.
    if [ -f "$HOME/Library/LaunchAgents/com.contorch.transcript-watcher.plist" ]; then
        echo ""
        echo "Removing the old transcript-watcher daemon (indexing is on demand now)..."
        "$SCRIPT_DIR/.venv/bin/transcript-watcher" uninstall || true
    fi
else
    echo ""
    echo "Skipping launchd install (--no-launchd flag)."
    echo "  Note: search.py defaults to HttpClient — without the chroma daemon"
    echo "  running, queries will fail. Start it manually:"
    echo "    $SCRIPT_DIR/.venv/bin/chroma run --path $CHROMA_DIR --host 127.0.0.1 --port 8765"
fi

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
cat <<EOF

Done. Diagnostic:
  $SCRIPT_DIR/.venv/bin/transcript-watcher doctor   # health check
  $SCRIPT_DIR/.venv/bin/transcript-watcher status

Restart Claude Code so the new MCP server is loaded.
EOF
