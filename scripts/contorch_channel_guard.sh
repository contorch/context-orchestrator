# contorch_channel_guard.sh — the channel guard READER for context-orchestrator's
# bash installers (bootstrap.sh, setup.sh, install-claude-context.sh). Source it,
# then call `contorch_channel_guard || exit $?` before touching Claude Code,
# launchd or the venv.
#
# The rule lives in pipeline-monitor (pipeline_monitor.channel), the only writer
# of ~/.contorch/channel.json (contorch.channel/1); it precomputes who may write
# (`writers`) and an operation token (`op`). Readers only test membership:
#
#   no marker                                   -> allowed
#   $CONTORCH_CHANNEL (unset/unknown = dev) in writers
#     and (op absent or $CONTORCH_OP == op.id)  -> allowed
#   anything else (incl. an unreadable marker)  -> print blocked_message, return 3
#
# Contract fixtures: pipeline-monitor contract/channel_guard/*.json, run by
# tests/bootstrap_guard_test.sh. Bash 3.2, no jq, no python: the marker is read
# with macOS's plutil, which parses JSON. Safe under `set -euo pipefail`.

contorch_channel_guard() {
    local marker me msg i w ok op
    marker="${CONTORCH_CHANNEL_MARKER:-$HOME/.contorch/channel.json}"
    [ -e "$marker" ] || return 0
    me="${CONTORCH_CHANNEL:-dev}"
    case "$me" in app|brew|dev) ;; *) me=dev ;; esac
    msg=$(plutil -extract blocked_message raw -o - "$marker" 2>/dev/null) || msg=""
    if [ -z "$msg" ]; then
        msg="Contorch on this Mac is managed by another install (see $marker); this installer will not change it."
    fi
    ok=0
    i=0
    while w=$(plutil -extract "writers.$i" raw -o - "$marker" 2>/dev/null); do
        if [ "$w" = "$me" ]; then ok=1; fi
        i=$((i + 1))
    done
    if op=$(plutil -extract op.id raw -o - "$marker" 2>/dev/null); then
        if [ "${CONTORCH_OP:-}" != "$op" ]; then ok=0; fi
    elif plutil -extract op json -o - "$marker" >/dev/null 2>&1; then
        ok=0    # an op without an id: can't be ours
    fi
    if [ "$ok" -eq 1 ]; then
        return 0
    fi
    printf '%s\n' "$msg" >&2
    return 3
}
