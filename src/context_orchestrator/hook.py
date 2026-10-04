"""`contorch-hook` — Claude Code's UserPromptSubmit hook (auto-context).

Reads the prompt (stdin JSON: {"prompt": "...", "cwd": "...", ...}) and prints
{"additionalContext": "..."}: the top memory search hits plus the git state
of the working directory.

It runs once per prompt, so it must never hold a prompt up and never fail
one: it waits at most chroma_lock.HOOK_LOCK_WAIT_S (2 s) for the in-process
Chroma session lock and otherwise answers from SQLite full-text search; the
whole search has SEARCH_BUDGET_S; any error means "no context", exit 0.
Claude Code's own limit is claude_install.HOOK_TIMEOUT_S (the settings entry).

Installed by `contorch-memory claude install` as a guarded command:
    [ -x "<path>/contorch-hook" ] && "<path>/contorch-hook"; exit 0 # contorch-hook:<channel>
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .chroma_lock import HOOK_LOCK_WAIT_S, LockTimeout

log = logging.getLogger("context-orchestrator")

MAX_HITS = 5
SNIPPET_CHARS = 280
MIN_PROMPT_CHARS = 12          # "ok", "thanks" — nothing to search for
GIT_TIMEOUT_S = 3
SEARCH_BUDGET_S = 6.0          # well inside Claude Code's HOOK_TIMEOUT_S
HEARTBEAT_FILE = Path.home() / ".context-orchestrator" / "auto-context-heartbeat.json"


def _vector_lines(prompt: str, project_url: str, lock_timeout: float) -> list[str]:
    from .search import VectorSearch
    p = os.environ.get("CO_CHROMA_PATH")
    vs = VectorSearch(chroma_path=Path(p) if p else None, lock_timeout=lock_timeout, verify=False)
    if not vs.enabled:
        return []
    hits: list[dict] = []
    if project_url:   # project-scoped first; fall back to global if no hits
        hits = vs.search(query=prompt, where={"project": project_url}, n_results=MAX_HITS,
                         hybrid=True, mmr=True)
    if not hits:
        hits = vs.search(query=prompt, n_results=MAX_HITS, hybrid=True, mmr=True)
    out = []
    for h in hits[:MAX_HITS]:
        meta = h.get("metadata") or {}
        label = meta.get("repo_url") or meta.get("task_name") or meta.get("type", "?")
        text = (h.get("text") or "")[:SNIPPET_CHARS].replace("\n", " ")
        out.append(f"- [{label}] {text}")
    return out


def _keyword_lines(prompt: str, project_url: str, db=None) -> list[str]:
    if db is None:
        from .db import Database
        p = os.environ.get("CO_DB_PATH")
        db = Database(db_path=Path(p) if p else None)
    hits = db.search_text(prompt, limit=MAX_HITS, project=project_url) if project_url else []
    if not hits:
        hits = db.search_text(prompt, limit=MAX_HITS)
    out = []
    for h in hits[:MAX_HITS]:
        label = h.get("repo_url") or h.get("task_name") or h.get("meeting_id") or h.get("kind", "?")
        text = " ".join((h.get("text") or "").split())[:SNIPPET_CHARS]
        out.append(f"- [{label}] {text}")
    return out


def search_lines(prompt: str, project_url: str = "", lock_timeout: float = HOOK_LOCK_WAIT_S,
                 db=None) -> tuple[list[str], str]:
    """(markdown lines, mode). mode is "vector", or "keyword" when the vector
    index was busy past `lock_timeout` or failed."""
    try:
        lines = _vector_lines(prompt, project_url, lock_timeout)
        if lines:
            return lines, "vector"
    except LockTimeout:
        log.info("Chroma busy for more than %.1fs — keyword search only", lock_timeout)
    except Exception as exc:  # the hook must never fail a prompt
        log.info("vector search unavailable (%s) — keyword search only", str(exc)[:160])
    try:
        return _keyword_lines(prompt, project_url, db), "keyword"
    except Exception as exc:
        log.info("keyword search failed: %s", str(exc)[:160])
        return [], "keyword"


def _git(args: list[str], cwd: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                           timeout=GIT_TIMEOUT_S, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def git_context(cwd: str) -> str:
    """Branch + last 8 commits + uncommitted file list; "" outside a repo."""
    if not _git(["rev-parse", "--git-dir"], cwd):
        return ""
    branch = _git(["branch", "--show-current"], cwd) or "(detached)"
    log_ = _git(["log", "--oneline", "-8"], cwd)
    status = _git(["status", "--short"], cwd)
    parts = [f"**Git** (branch: `{branch}`)"]
    if log_:
        parts.append("Recent commits:\n```\n" + log_ + "\n```")
    if status:
        parts.append("Uncommitted changes:\n```\n" + status[:500] + "\n```")
    return "\n".join(parts)


def memory_context(prompt: str, project_url: str, budget_s: Optional[float] = None) -> str:
    """Search hits as a markdown section, or "" (nothing found, or the search
    didn't finish within `budget_s`, default SEARCH_BUDGET_S)."""
    budget_s = SEARCH_BUDGET_S if budget_s is None else budget_s
    box: dict = {}

    def run():
        try:
            box["lines"], box["mode"] = search_lines(prompt, project_url)
        except Exception as exc:  # never fail the prompt
            box["error"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(budget_s)
    lines = box.get("lines") or []
    if not lines:
        return ""
    head = "**Context-orchestrator search hits:**" if box.get("mode") == "vector" else \
        "**Context-orchestrator search hits** (keyword):"
    return head + "\n" + "\n".join(lines)


def build(payload: dict) -> Optional[str]:
    prompt = (payload.get("prompt") or "").strip()
    cwd = payload.get("cwd") or os.getcwd()
    if len(prompt) < MIN_PROMPT_CHARS or prompt.startswith("/"):
        return None
    git = git_context(cwd)
    project_url = _git(["remote", "get-url", "origin"], cwd) if git else ""
    sections = [s for s in (memory_context(prompt, project_url), git) if s]
    if not sections:
        return None
    return ("**[auto-context]** Pre-loaded for this prompt — use it if relevant, ignore if not.\n\n"
            + "\n\n".join(sections))


def _heartbeat(chars: int, latency_ms: int, prompt_len: int) -> None:
    """One line the menu bar reads to show when the hook last fired."""
    try:
        import datetime as _dt
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = HEARTBEAT_FILE.with_name(f".{HEARTBEAT_FILE.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"ts": time.time(),
                                   "iso": _dt.datetime.now().isoformat(timespec="seconds"),
                                   "injected_chars": chars, "latency_ms": latency_ms,
                                   "prompt_len": prompt_len}))
        os.replace(tmp, HEARTBEAT_FILE)
    except Exception:
        pass


def main() -> int:
    """Entry point of `contorch-hook`. Always prints valid JSON and exits 0."""
    start = time.time()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    body = None
    prompt_len = 0
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        prompt_len = len((payload.get("prompt") or "").strip())
        body = build(payload)
    except Exception:
        body = None
    sys.stdout.write(json.dumps({"additionalContext": body or ""}) + "\n")
    sys.stdout.flush()
    if body:
        _heartbeat(len(body), int((time.time() - start) * 1000), prompt_len)
    # Leave now: a search thread past its budget must not keep the prompt
    # waiting, and native runtimes (onnxruntime) can abort in their exit
    # handlers — the answer is already written.
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
