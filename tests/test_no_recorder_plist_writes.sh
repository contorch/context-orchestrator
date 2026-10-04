#!/bin/bash
# Recorder settings (MEETING_CAPTURE_*) and the Gemini key belong to
# meeting-capture (`meeting-capture stt|mode|source …`, ~/.meeting-capture/env)
# and ~/.config/google/key. Nothing in context-orchestrator may write them into
# a launchd plist (the retired enable-gemini-pipeline.sh did). Bash 3.2.
set -u
cd "$(dirname "$0")/.." || exit 2
status=0
files=$(git ls-files '*.sh' '*.py' | grep -v '^tests/')
for f in $files; do
    # shell: plist editors / EnvironmentVariables lines that name those keys
    # (incl. wrappers like `set_plist_env_var "$PLIST" KEY value`; bootstrap's
    # own MEETING_CAPTURE_DIR / _REPO shell variables are not settings)
    if grep -niE '(plist|PlistBuddy|defaults[[:space:]]+write|EnvironmentVariables)' "$f" \
        | sed -E 's/MEETING_CAPTURE_(DIR|REPO)//g' \
        | grep -E 'MEETING_CAPTURE_|GOOGLE_API_KEY|GEMINI_API_KEY' | grep -vE '^[0-9]+:[[:space:]]*#'; then
        echo "✗ $f writes a recorder setting or the API key into a plist" >&2; status=1
    fi
    # python: those keys as dict keys (a plist payload)
    if grep -nE "[\"'](MEETING_CAPTURE_[A-Z_]+|GOOGLE_API_KEY|GEMINI_API_KEY)[\"'][[:space:]]*:" "$f"; then
        echo "✗ $f builds a payload with a recorder setting or the API key" >&2; status=1
    fi
done
if [ -n "$(git ls-files enable-gemini-pipeline.sh)" ]; then
    echo "✗ enable-gemini-pipeline.sh is back (retired in M1.7)" >&2; status=1
fi
[ "$status" -eq 0 ] && echo "✓ no recorder-setting / API-key plist writes"
exit "$status"
