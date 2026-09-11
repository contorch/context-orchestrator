#!/usr/bin/env bash
# bootstrap.sh — full-stack installer for the meeting-capture +
# context-orchestrator + pipeline-monitor pipeline.
#
# Designed for a fresh Mac with Claude Code already installed. Walks
# through every step interactively, prompts for the Gemini key when it
# needs it, and is idempotent — safe to re-run if anything failed
# partway through.
#
# Run on a fresh laptop:
#   curl -fsSL https://raw.githubusercontent.com/contorch/context-orchestrator/main/bootstrap.sh | bash
#
# Or from a local checkout:
#   bash bootstrap.sh

set -uo pipefail

# ============================================================ config

BASE_DIR="${BASE_DIR:-$HOME/tasks}"
GEMINI_KEY_FILE="$HOME/.config/google/key"

CONTEXT_ORCH_REPO="https://github.com/contorch/context-orchestrator.git"
MEETING_CAPTURE_REPO="https://github.com/contorch/meeting-capture.git"
PIPELINE_MONITOR_REPO="https://github.com/contorch/pipeline-monitor.git"

CONTEXT_ORCH_DIR="$BASE_DIR/context-orchestrator"
MEETING_CAPTURE_DIR="$BASE_DIR/meeting-capture"
PIPELINE_MONITOR_DIR="$BASE_DIR/pipeline-monitor"

# ============================================================ output

if [ -t 1 ]; then
    BOLD='\033[1m'; RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'
    BLUE='\033[34m'; CYAN='\033[36m'; DIM='\033[2m'; RESET='\033[0m'
else
    BOLD=''; RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; DIM=''; RESET=''
fi

step()  { printf "\n${BOLD}${BLUE}▶ %s${RESET}\n" "$*"; }
ok()    { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
skip()  { printf "  ${DIM}○${RESET} %s ${DIM}(already done)${RESET}\n" "$*"; }
warn()  { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }
fail()  { printf "  ${RED}✗${RESET} %s\n" "$*" >&2; exit 1; }
info()  { printf "  ${CYAN}·${RESET} %s\n" "$*"; }
ask()   { printf "${BOLD}${YELLOW}?${RESET} %s " "$*"; }

banner() {
    cat <<EOF

${BOLD}${CYAN}╭──────────────────────────────────────────────────────╮
│   ${YELLOW}△${CYAN} ${BOLD}Contorch${RESET}${CYAN} — persistent context layer for Claude   │
│        bootstrap installer                            │
╰──────────────────────────────────────────────────────╯${RESET}

This will install:
  ${CYAN}1.${RESET} ${BOLD}context-orchestrator${RESET} — task + context store, MCP server, semantic search
  ${CYAN}2.${RESET} ${BOLD}meeting-capture${RESET}     — auto-recording when your mic activates
  ${CYAN}3.${RESET} ${BOLD}pipeline-monitor${RESET}    — menu bar dashboard for the whole stack
  ${CYAN}4.${RESET} ${BOLD}Gemini integration${RESET}  — better embeddings + transcription (optional)
  ${CYAN}5.${RESET} ${BOLD}auto-context hook${RESET}   — Claude Code pre-loads context on every prompt

Idempotent — re-run anytime to fix a partial install.
Install root: ${BOLD}$BASE_DIR${RESET}

EOF
}

# ============================================================ prereqs

check_macos() {
    [ "$(uname)" = "Darwin" ] || fail "macOS required (you're on $(uname))"
    ok "macOS $(sw_vers -productVersion)"
}

check_xcode_clt() {
    if xcode-select -p >/dev/null 2>&1; then
        ok "Xcode Command Line Tools installed"
    else
        warn "Xcode Command Line Tools missing"
        info "Triggering installer dialog — accept it, then re-run this script"
        xcode-select --install 2>/dev/null || true
        fail "Re-run after Xcode CLT finishes installing"
    fi
}

check_homebrew() {
    if command -v brew >/dev/null 2>&1; then
        ok "Homebrew installed"
    else
        warn "Homebrew missing"
        info "Install with:"
        info "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fail "Install Homebrew, then re-run this script"
    fi
}

ensure_brew_pkg() {
    local pkg="$1"
    local cmd="${2:-$1}"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$pkg present"
    else
        info "installing $pkg via brew…"
        brew install "$pkg" >/dev/null 2>&1 || fail "brew install $pkg failed"
        ok "$pkg installed"
    fi
}

check_python() {
    local py="${PYTHON:-python3}"
    if ! command -v "$py" >/dev/null 2>&1; then
        warn "python3 missing — installing python@3.10 via brew"
        brew install python@3.10 >/dev/null 2>&1
    fi
    local v
    v=$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    case "$v" in
        3.1[0-9]|3.[2-9][0-9])
            ok "python $v"
            ;;
        *)
            warn "python $v < 3.10 — installing python@3.10 via brew"
            brew install python@3.10 >/dev/null 2>&1
            ok "python@3.10 installed (use it via /opt/homebrew/opt/python@3.10/bin/python3.10)"
            ;;
    esac
}

check_claude_code() {
    if command -v claude >/dev/null 2>&1; then
        ok "Claude Code CLI installed ($(claude --version 2>&1 | head -1))"
    else
        warn "Claude Code not detected on PATH"
        info "Get it from claude.ai/download — re-run this script after install"
        info "(continuing anyway; some steps may need a restart afterward)"
    fi
}

# ============================================================ repo clone/update

clone_or_update() {
    local repo="$1" dir="$2" name="$3"
    if [ -d "$dir/.git" ]; then
        info "git pull $name…"
        git -C "$dir" pull --quiet --ff-only 2>/dev/null && ok "$name up to date" || warn "$name has local changes — skipping pull"
    else
        mkdir -p "$(dirname "$dir")"
        info "git clone $name…"
        git clone --quiet "$repo" "$dir" || fail "clone failed: $repo"
        ok "$name cloned to $dir"
    fi
}

# ============================================================ Gemini key

prompt_gemini_key() {
    if [ -f "$GEMINI_KEY_FILE" ] && [ -s "$GEMINI_KEY_FILE" ]; then
        local existing
        existing=$(cat "$GEMINI_KEY_FILE" | tr -d '\n')
        if [[ "$existing" =~ ^AIza[0-9A-Za-z_-]{35}$ ]]; then
            ok "Gemini key already at $GEMINI_KEY_FILE"
            return 0
        else
            warn "$GEMINI_KEY_FILE exists but doesn't look like a valid key (expected AIza... 39 chars)"
        fi
    fi

    cat <<EOF

  ${CYAN}Gemini powers the better embeddings (3072d, much higher quality than the
  default 384d local model) AND transcription in meeting-capture. It's optional
  — skip with empty input and the pipeline runs entirely local.${RESET}

  Get a free key at: ${BOLD}https://aistudio.google.com/app/apikey${RESET}
  Free tier covers ~all personal use.

EOF
    ask "Paste your Gemini API key (or Enter to skip):"
    local key
    if [ -t 0 ]; then
        read -r key
    else
        # piped install — read from /dev/tty
        read -r key < /dev/tty
    fi
    if [ -z "$key" ]; then
        warn "skipped — Gemini features disabled (you can re-run later)"
        return 1
    fi
    if ! [[ "$key" =~ ^AIza[0-9A-Za-z_-]{35}$ ]]; then
        warn "key format looks wrong (expected AIza... 39 chars total). Saving anyway, but it may not work."
    fi
    mkdir -p "$(dirname "$GEMINI_KEY_FILE")"
    printf '%s' "$key" > "$GEMINI_KEY_FILE"
    chmod 600 "$GEMINI_KEY_FILE"
    ok "Gemini key saved to $GEMINI_KEY_FILE (mode 600)"
    return 0
}

# ============================================================ pipeline steps

setup_context_orch() {
    step "[1/5] context-orchestrator"
    clone_or_update "$CONTEXT_ORCH_REPO" "$CONTEXT_ORCH_DIR" "context-orchestrator"
    info "running setup.sh…"
    if (cd "$CONTEXT_ORCH_DIR" && bash setup.sh 2>&1 | sed 's/^/    /'); then
        ok "context-orchestrator setup complete"
    else
        fail "context-orchestrator setup failed — see output above"
    fi
}

setup_meeting_capture() {
    step "[2/5] meeting-capture"
    clone_or_update "$MEETING_CAPTURE_REPO" "$MEETING_CAPTURE_DIR" "meeting-capture"
    if [ -f "$MEETING_CAPTURE_DIR/setup.sh" ]; then
        info "running setup.sh…"
        if (cd "$MEETING_CAPTURE_DIR" && bash setup.sh 2>&1 | sed 's/^/    /'); then
            ok "meeting-capture setup complete"
        else
            warn "meeting-capture setup had errors — review output above"
        fi
    else
        warn "no setup.sh in meeting-capture — skipping (repo may need manual setup)"
    fi
}

setup_gemini() {
    step "[3/5] Gemini activation (optional)"
    if prompt_gemini_key; then
        info "running enable-gemini-pipeline.sh…"
        export CONTEXT_ORCH="$CONTEXT_ORCH_DIR"
        export MEETING_CAPTURE="$MEETING_CAPTURE_DIR"
        if (cd "$CONTEXT_ORCH_DIR" && bash enable-gemini-pipeline.sh 2>&1 | sed 's/^/    /'); then
            ok "Gemini pipeline enabled"
        else
            warn "Gemini activation had errors — re-run manually if needed:"
            warn "  cd $CONTEXT_ORCH_DIR && bash enable-gemini-pipeline.sh"
        fi
    else
        skip "Gemini activation (no key)"
    fi
}

setup_auto_context_hook() {
    step "[4/5] auto-context hook for Claude Code"
    if [ -f "$CONTEXT_ORCH_DIR/install-claude-context.sh" ]; then
        if (cd "$CONTEXT_ORCH_DIR" && bash install-claude-context.sh 2>&1 | sed 's/^/    /'); then
            ok "auto-context hook installed"
        else
            warn "hook install had errors — see above"
        fi
    else
        warn "install-claude-context.sh missing — skip (older context-orchestrator?)"
    fi
}

setup_pipeline_monitor() {
    step "[5/5] pipeline-monitor (menu bar dashboard)"
    clone_or_update "$PIPELINE_MONITOR_REPO" "$PIPELINE_MONITOR_DIR" "pipeline-monitor"
    info "running install.sh --autostart…"
    if (cd "$PIPELINE_MONITOR_DIR" && bash install.sh --autostart 2>&1 | sed 's/^/    /'); then
        ok "pipeline-monitor installed and running (look for ○ in menu bar)"
    else
        warn "pipeline-monitor install had errors — see above"
    fi
}

# ============================================================ TCC permissions

open_tcc_panes() {
    step "Open System Settings → grant Microphone + Screen Recording"
    info "The daemons just started via launchd; they've already tried to access"
    info "the mic and the system-audio capture API. Their entries are now in the"
    info "Privacy panes — toggled OFF. Just flip them ON."
    info ""
    info "Opening Microphone pane…"
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone" 2>/dev/null || true
    sleep 1
    info "Opening Screen & System Audio Recording pane…"
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture" 2>/dev/null || true
    info ""
    info "Look for entries named ${BOLD}sysaudio${RESET} (Screen Recording) and"
    info "${BOLD}meeting-capture${RESET} or ${BOLD}python3${RESET} (Microphone). Toggle each ON."
    info ""
    if [ -t 0 ] || [ -e /dev/tty ]; then
        ask "Press Enter once you've toggled both ON (or Ctrl-C to skip):"
        if [ -t 0 ]; then read -r _; else read -r _ < /dev/tty; fi
    else
        warn "non-interactive shell — skipping the wait. Toggle them when you can."
    fi
    # After granting, bounce the daemons so they pick up the new permissions.
    info "Restarting daemons so they pick up the new grants…"
    for label in com.contorch.meeting-capture com.contorch.transcript-watcher; do
        launchctl kickstart -k "gui/$(id -u)/$label" 2>/dev/null && ok "kicked $label" || true
    done
}

# ============================================================ final report

final_report() {
    cat <<EOF

${BOLD}${GREEN}╭──────────────────────────────────────────────────────╮
│  ✓ Bootstrap complete                                │
╰──────────────────────────────────────────────────────╯${RESET}

${BOLD}Installed at:${RESET}
  $CONTEXT_ORCH_DIR
  $MEETING_CAPTURE_DIR
  $PIPELINE_MONITOR_DIR

${BOLD}${YELLOW}One thing left for you:${RESET}

  ${CYAN}▶${RESET} ${BOLD}Restart Claude Code${RESET}
       Quit and relaunch the app so it picks up the new MCP server +
       UserPromptSubmit hook from ~/.claude/settings.json.

${BOLD}Verify everything works:${RESET}
  Click the ○ icon in your menu bar → ${BOLD}"Run end-to-end smoke test"${RESET}.
  Should show ✓ in ~1.5s.

${BOLD}Useful one-liners:${RESET}
  launchctl list | grep com.contorch            ${DIM}# all daemons${RESET}
  curl http://127.0.0.1:8765/api/v2/heartbeat   ${DIM}# chroma daemon${RESET}
  ~/.claude/hooks/auto-context.py < /dev/null   ${DIM}# probe the hook${RESET}

EOF
}

# ============================================================ main

main() {
    banner

    step "Prerequisites"
    check_macos
    check_xcode_clt
    check_homebrew
    ensure_brew_pkg git
    ensure_brew_pkg node npm
    ensure_brew_pkg gh
    check_python
    check_claude_code

    setup_context_orch
    setup_meeting_capture
    setup_gemini
    setup_auto_context_hook
    setup_pipeline_monitor

    open_tcc_panes

    final_report
}

if [ "${BASH_SOURCE[0]:-}" = "${0}" ] || [ -z "${BASH_SOURCE[0]:-}" ]; then
    main "$@"
fi
