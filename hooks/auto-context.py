#!/usr/bin/env python3
"""UserPromptSubmit hook — proactively load context for every prompt.

Reads the user's prompt from stdin (JSON: {"prompt": "...", "cwd": "...", ...}).
Returns JSON with `additionalContext` injected into Claude's context.

Sources, in priority order:
  1. context-orchestrator semantic search (top 5 hits from any repo/task)
  2. Git context: current branch, last 8 commits, modified files
  3. Recent commits touching files mentioned in the prompt (best-effort)

Designed to be FAST (sub-2s) and FAIL SILENTLY — a slow or broken hook
must never block the user's prompt. Total timeout: 10s (per Claude Code).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Hard cap: never spend more than this in any single sub-step
SUBSTEP_TIMEOUT = 3

# Locate context-orchestrator. Order: env override → common locations under $HOME.
# Machine-portable: avoids hard-coding any one user's path.
def _find_orch_root() -> str | None:
    env = os.environ.get("CO_REPO")
    if env and Path(env, "src/context_orchestrator").is_dir():
        return env
    home = Path.home()
    for cand in (
        home / "tasks" / "vector_databases_experiments",   # the experiments checkout
        home / "tasks" / "context-orchestrator",
        home / "src" / "context-orchestrator",
        home / "code" / "context-orchestrator",
    ):
        if (cand / "src" / "context_orchestrator").is_dir():
            return str(cand)
    return None


_ORCH_ROOT = _find_orch_root()
CTX_ORCH_DB_PYPATH = f"{_ORCH_ROOT}/src" if _ORCH_ROOT else None
CTX_ORCH_VENV = f"{_ORCH_ROOT}/.venv/bin/python" if _ORCH_ROOT else None


def _exit_silent():
    """Emit empty additionalContext so the prompt proceeds normally."""
    print(json.dumps({"additionalContext": ""}))
    sys.exit(0)


def _git(args, cwd, timeout=SUBSTEP_TIMEOUT):
    try:
        r = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def gather_git_context(cwd: str) -> str:
    """Branch + last 8 commits + uncommitted file list. Empty string if not a git repo."""
    if not _git(["rev-parse", "--git-dir"], cwd):
        return ""
    branch = _git(["branch", "--show-current"], cwd) or "(detached)"
    log = _git(["log", "--oneline", "-8"], cwd)
    status = _git(["status", "--short"], cwd)
    parts = [f"**Git** (branch: `{branch}`)"]
    if log:
        parts.append("Recent commits:\n```\n" + log + "\n```")
    if status:
        parts.append("Uncommitted changes:\n```\n" + status[:500] + "\n```")
    return "\n".join(parts)


def gather_orch_context(prompt: str, project_url: str) -> str:
    """Semantic search context-orch. Returns formatted top hits or ''.

    Runs context_orchestrator.hook.search_lines in the context-orchestrator venv.
    """
    if not CTX_ORCH_VENV or not Path(CTX_ORCH_VENV).exists():
        return ""
    # Run in a subprocess so import errors / Chroma daemon down don't kill the hook.
    # context_orchestrator.hook waits at most 2 s for the in-process Chroma
    # session lock, then answers from SQLite full-text search alone.
    code = f'''
import sys
sys.path.insert(0, {CTX_ORCH_DB_PYPATH!r})
try:
    from context_orchestrator.hook import search_lines
    lines, _mode = search_lines({prompt!r}, {project_url!r})
    print("\\n".join(lines))
except Exception:
    pass
'''
    try:
        r = subprocess.run(
            [CTX_ORCH_VENV, "-c", code],
            capture_output=True, text=True,
            timeout=SUBSTEP_TIMEOUT * 2,  # search can be slower (Gemini API call)
            check=False,
        )
        out = r.stdout.strip()
        return f"**Context-orchestrator search hits:**\n{out}" if out else ""
    except Exception:
        return ""


HEARTBEAT_FILE = Path.home() / ".context-orchestrator" / "auto-context-heartbeat.json"


def _write_heartbeat(injected_chars: int, latency_ms: int, prompt_len: int):
    """Drop a single-line JSON file the dashboard reads to show last fire."""
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_FILE.write_text(json.dumps({
            "ts": __import__("time").time(),
            "iso": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "injected_chars": injected_chars,
            "latency_ms": latency_ms,
            "prompt_len": prompt_len,
        }))
    except Exception:
        pass  # heartbeat is best-effort; don't break the hook


def main():
    import time as _t
    _start = _t.time()
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        _exit_silent()

    prompt = (payload.get("prompt") or "").strip()
    cwd = payload.get("cwd") or os.getcwd()

    # Skip if prompt is trivial — search is wasteful for "ok", "thanks", etc.
    if len(prompt) < 12:
        _exit_silent()

    # Skip if prompt is a slash command — it has its own handling
    if prompt.startswith("/"):
        _exit_silent()

    git_ctx = gather_git_context(cwd)

    # Best-effort project URL for scoping the search
    project_url = ""
    if git_ctx:
        project_url = _git(["remote", "get-url", "origin"], cwd) or ""

    orch_ctx = gather_orch_context(prompt, project_url)

    sections = [s for s in (orch_ctx, git_ctx) if s]
    if not sections:
        _exit_silent()

    body = (
        "**[auto-context]** Pre-loaded for this prompt — use it if relevant, "
        "ignore if not.\n\n" + "\n\n".join(sections)
    )
    print(json.dumps({"additionalContext": body}))
    _write_heartbeat(
        injected_chars=len(body),
        latency_ms=int((_t.time() - _start) * 1000),
        prompt_len=len(prompt),
    )


if __name__ == "__main__":
    main()
