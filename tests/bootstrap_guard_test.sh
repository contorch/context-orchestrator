#!/bin/bash
# Runs scripts/contorch_channel_guard.sh over every channel-guard fixture
# (pipeline-monitor contract/channel_guard, vendored in tests/fixtures with a
# pinned commit + sha256 in PIN) the way installers hit it: macOS /bin/bash
# 3.2, a clean environment, UTF-8, no tty, (1) sourced by a script piped into
# bash (curl | bash) and (2) under `bash -c` with set -euo pipefail.
# Usage: /bin/bash tests/bootstrap_guard_test.sh [fixtures-dir]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GUARD="$ROOT/scripts/contorch_channel_guard.sh"
FIX="${1:-$ROOT/tests/fixtures/channel_guard}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/guard-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
pass=0
failn=0

field() {  # field FILE KEYPATH -> raw value; nothing and status 1 if absent
    # (some macOS versions print plutil's error on stdout, so keep the output
    # only when plutil succeeded)
    local v
    v=$(plutil -extract "$2" raw -o - "$1" 2>/dev/null) || return 1
    printf '%s' "$v"
}

for f in "$FIX"/*.json; do
    name="$(basename "$f" .json)"
    home="$WORK/$name"
    mkdir -p "$home/.contorch"
    if raw=$(field "$f" marker_raw); then
        printf '%s' "$raw" > "$home/.contorch/channel.json"
    elif plutil -extract marker json -o "$WORK/marker.tmp" "$f" >/dev/null 2>&1; then
        mv "$WORK/marker.tmp" "$home/.contorch/channel.json"
    fi
    rm -f "$WORK/marker.tmp"   # `"marker": null`: plutil fails, no file
    # `"marker": null` → no file (plutil can't extract null)
    want="$(field "$f" expect.exit)"
    want_err="$(field "$f" expect.stderr || true)"
    want_sub="$(field "$f" expect.stderr_contains || true)"
    envs="HOME=$home PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8"
    ch="$(field "$f" env.CONTORCH_CHANNEL || true)"
    op="$(field "$f" env.CONTORCH_OP || true)"
    [ -n "$ch" ] && envs="$envs CONTORCH_CHANNEL=$ch"
    [ -n "$op" ] && envs="$envs CONTORCH_OP=$op"

    for mode in pipe bash-c; do
        if [ "$mode" = pipe ]; then
            printf '. "%s"\ncontorch_channel_guard\nexit $?\n' "$GUARD" \
                | env -i $envs /bin/bash >"$WORK/out" 2>"$WORK/err"
            got=$?
        else
            env -i $envs /bin/bash -c "set -euo pipefail; . \"$GUARD\"; contorch_channel_guard || exit \$?; echo allowed" \
                </dev/null >"$WORK/out" 2>"$WORK/err"
            got=$?
        fi
        err="$(cat "$WORK/err")"
        ok=1
        [ "$got" = "$want" ] || ok=0
        if [ "$want" = 3 ]; then
            if [ -n "$want_err" ] && [ "$err" != "$want_err" ]; then ok=0; fi
            if [ -n "$want_sub" ]; then case "$err" in *"$want_sub"*) ;; *) ok=0 ;; esac; fi
        else
            [ -z "$err" ] || ok=0
        fi
        if [ "$mode" = bash-c ] && [ "$want" = 0 ] && [ "$(cat "$WORK/out")" != allowed ]; then ok=0; fi
        if [ "$ok" = 1 ]; then
            pass=$((pass + 1))
        else
            failn=$((failn + 1))
            echo "✗ $name ($mode): exit $got (want $want); stderr: $err" >&2
        fi
    done
done
echo "channel guard: $pass passed, $failn failed ($(ls "$FIX"/*.json | wc -l | tr -d ' ') fixtures × 2 modes)"
[ "$failn" -eq 0 ]
