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

# Non-interactive mode (CI, scripted installs): skip the Gemini prompt and the
# permissions wait instead of trying to read a terminal that is not there.
NONINTERACTIVE="${CONTORCH_NONINTERACTIVE:-0}"
# Comma-separated components to skip, e.g. CONTORCH_SKIP=pipeline-monitor
SKIP=",${CONTORCH_SKIP:-},"

# ============================================================ config

BASE_DIR="${BASE_DIR:-$HOME/tasks}"
GEMINI_KEY_FILE="$HOME/.config/google/key"

# Overridable so CI can install from a local checkout / branch.
CONTEXT_ORCH_REPO="${CONTEXT_ORCH_REPO:-https://github.com/contorch/context-orchestrator.git}"
MEETING_CAPTURE_REPO="${MEETING_CAPTURE_REPO:-https://github.com/contorch/meeting-capture.git}"
PIPELINE_MONITOR_REPO="${PIPELINE_MONITOR_REPO:-https://github.com/contorch/pipeline-monitor.git}"
PY_FORMULA="python@3.12"

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

INCOMPLETE=""
incomplete() { INCOMPLETE="${INCOMPLETE}  ✗ $1\n"; }
skipped() { case "$SKIP" in *",$1,"*) return 0 ;; esac; return 1; }

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
    if ! command -v brew >/dev/null 2>&1; then
        for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
            [ -x "$b" ] && eval "$("$b" shellenv)" && break
        done
    fi
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

py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
py_ver() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "none"; }

check_python() {
    local py="${PYTHON:-python3}"
    if ! command -v "$py" >/dev/null 2>&1 || ! py_ok "$py"; then
        warn "python3 $(py_ver "$py") is missing or older than 3.10 (Apple's CLT python is 3.9)"
        info "installing $PY_FORMULA via brew…"
        brew install "$PY_FORMULA" >/dev/null 2>&1 || fail "brew install $PY_FORMULA failed — run it by hand and re-run this script"
        # Homebrew does not link a bare `python3` for versioned formulas, so the
        # setup.sh scripts below would still find Apple's 3.9. Put the formula's
        # unversioned symlinks first on PATH for the rest of this run.
        local prefix; prefix="$(brew --prefix "$PY_FORMULA")"
        export PATH="$prefix/libexec/bin:$PATH"
        py="$prefix/libexec/bin/python3"
        py_ok "$py" || fail "$py is not a working Python >= 3.10"
    fi
    export PYTHON="$py"
    ok "python $(py_ver "$py") ($(command -v "$py"))"
}

check_claude_code() {
    if command -v claude >/dev/null 2>&1; then
        ok "Claude Code CLI installed ($(claude --version 2>&1 | head -1))"
    else
        warn "Claude Code not detected on PATH"
        info "Get it from claude.ai/download — re-run this script after install"
        info "(continuing anyway; the MCP server will not be registered)"
        incomplete "Claude Code not installed — install it, then re-run this script to register the MCP server + hook"
    fi
}

# ============================================================ repo clone/update

clone_or_update() {
    local repo="$1" dir="$2" name="$3"
    if [ -d "$dir/.git" ]; then
        info "git pull ${name}…"
        git -C "$dir" pull --quiet --ff-only 2>/dev/null && ok "$name up to date" || warn "$name has local changes — skipping pull"
    else
        mkdir -p "$(dirname "$dir")"
        info "git clone ${name}…"
        git clone --quiet "$repo" "$dir" || fail "clone failed: $repo"
        ok "$name cloned to $dir"
    fi
}

# ============================================================ Gemini key

# True only if we can actually read from the controlling terminal (a /dev/tty
# node can exist and still be unopenable, e.g. under CI or a detached script).
have_tty() { [ -t 0 ] || { [ -e /dev/tty ] && ( : < /dev/tty ) 2>/dev/null; }; }

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

  ${CYAN}Meeting transcription runs on Google's Gemini and needs an API key.
  Skip it (empty input) and meeting-capture will record nothing until you add
  one — notes, sources, and search still work, fully local. Gemini also
  upgrades embeddings (3072d vs the default local 384d).${RESET}

  Get a key at: ${BOLD}https://aistudio.google.com/apikey${RESET}
  Add one later: write it to ~/.config/google/key (chmod 600) and re-run.

EOF
    local key=""
    if [ "$NONINTERACTIVE" = 1 ] || ! have_tty; then
        warn "no terminal to prompt on — skipping Gemini (write the key to $GEMINI_KEY_FILE later)"
        return 1
    fi
    ask "Paste your Gemini API key (or Enter to skip):"
    if [ -t 0 ]; then
        read -r key
    else
        # piped install — read from /dev/tty
        read -r key < /dev/tty || key=""
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
            incomplete "meeting-capture — re-run: bash $MEETING_CAPTURE_DIR/setup.sh"
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
            incomplete "Gemini activation — re-run: cd $CONTEXT_ORCH_DIR && bash enable-gemini-pipeline.sh"
        fi
    else
        skip "Gemini activation (no key)"
        incomplete "meeting transcription is OFF until a Gemini key is at $GEMINI_KEY_FILE"
    fi
}

setup_auto_context_hook() {
    step "[4/5] auto-context hook for Claude Code"
    if [ -f "$CONTEXT_ORCH_DIR/install-claude-context.sh" ]; then
        if (cd "$CONTEXT_ORCH_DIR" && bash install-claude-context.sh 2>&1 | sed 's/^/    /'); then
            ok "auto-context hook installed"
        else
            warn "hook install had errors — see above"
            incomplete "auto-context hook — re-run: cd $CONTEXT_ORCH_DIR && bash install-claude-context.sh"
        fi
    else
        warn "install-claude-context.sh missing — skip (older context-orchestrator?)"
        incomplete "auto-context hook (install-claude-context.sh missing)"
    fi
}

setup_pipeline_monitor() {
    step "[5/5] pipeline-monitor (menu bar dashboard)"
    if skipped pipeline-monitor; then skip "pipeline-monitor (CONTORCH_SKIP)"; return 0; fi
    clone_or_update "$PIPELINE_MONITOR_REPO" "$PIPELINE_MONITOR_DIR" "pipeline-monitor"
    info "running install.sh --autostart…"
    if (cd "$PIPELINE_MONITOR_DIR" && bash install.sh --autostart 2>&1 | sed 's/^/    /'); then
        ok "pipeline-monitor installed and running (look for ○ in menu bar)"
    else
        warn "pipeline-monitor install had errors — see above"
        incomplete "pipeline-monitor — re-run: cd $PIPELINE_MONITOR_DIR && bash install.sh --autostart"
    fi
}

# ============================================================ TCC permissions

open_tcc_panes() {
    step "Grant macOS permissions to sysaudio"
    local bin="$MEETING_CAPTURE_DIR/bin/sysaudio"
    info "Capture runs through one binary, ${BOLD}sysaudio${RESET}, and macOS attaches the"
    info "permissions to it — grant them to sysaudio, not to your terminal."
    info ""
    info "1. System Settings → Privacy & Security → ${BOLD}Screen & System Audio Recording${RESET}"
    info "   Click ＋, press ⌘⇧G, paste this path, add it, and enable it (also under"
    info "   'System Audio Recording Only' if that list is shown):"
    info "     $bin"
    info "2. ${BOLD}Microphone${RESET} (macOS 15+): nothing to do now — the first real recording"
    info "   pops a prompt for sysaudio; click Allow to get your own voice as 'Me'."
    info ""
    if [ "$NONINTERACTIVE" = 1 ] || ! have_tty; then
        warn "non-interactive — not opening System Settings. Grant the permission above before your first call."
        incomplete "Screen & System Audio Recording grant for $bin (System Settings → Privacy & Security)"
        return 0
    fi
    info "Opening the Screen & System Audio Recording pane…"
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture" 2>/dev/null || true
    ask "Press Enter once sysaudio is added and enabled (or Ctrl-C to do it later):"
    if [ -t 0 ]; then read -r _; else read -r _ < /dev/tty || true; fi
    # Bounce the capture daemon so it picks up the grant.
    launchctl kickstart -k "gui/$(id -u)/com.contorch.meeting-capture" 2>/dev/null && ok "restarted meeting-capture" || true
}

# ============================================================ final report

final_report() {
    cat <<EOF

$( if [ -z "$INCOMPLETE" ]; then printf "%b" "${BOLD}${GREEN}╭──────────────────────────────────────────────────────╮\n│  ✓ Bootstrap complete                                │\n╰──────────────────────────────────────────────────────╯${RESET}"; else printf "%b" "${BOLD}${YELLOW}╭──────────────────────────────────────────────────────╮\n│  ! Bootstrap finished with things left to do         │\n╰──────────────────────────────────────────────────────╯${RESET}\n\n${BOLD}Not done yet:${RESET}\n${INCOMPLETE}"; fi )

${BOLD}Installed at:${RESET}
  $CONTEXT_ORCH_DIR
  $MEETING_CAPTURE_DIR
  $PIPELINE_MONITOR_DIR

${BOLD}${YELLOW}Then:${RESET}

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
    check_python
    check_claude_code

    setup_context_orch
    setup_meeting_capture
    setup_gemini
    setup_auto_context_hook
    setup_pipeline_monitor

    open_tcc_panes

    final_report
    [ -z "$INCOMPLETE" ]
}

if [ "${BASH_SOURCE[0]:-}" = "${0}" ] || [ -z "${BASH_SOURCE[0]:-}" ]; then
    main "$@"
fi
