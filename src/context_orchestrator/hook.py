"""The auto-context hook's search: what a prompt pre-loads from memory.

The hook runs once per prompt, so it must never hold a prompt up. It waits at
most chroma_lock.HOOK_TIMEOUT_S (2 s) for the in-process Chroma session lock;
when Chroma stays busy (another process is indexing) — or the vector index is
unavailable for any reason — it answers from SQLite full-text search alone.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .chroma_lock import HOOK_TIMEOUT_S, LockTimeout

log = logging.getLogger("context-orchestrator")

MAX_HITS = 5
SNIPPET_CHARS = 280


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


def search_lines(prompt: str, project_url: str = "", lock_timeout: float = HOOK_TIMEOUT_S,
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


def main_search(prompt: str, project_url: str = "", lock_timeout: Optional[float] = None) -> int:
    """`python -m context_orchestrator.hook PROMPT [PROJECT]` — print the lines
    (used by hooks/auto-context.py, which runs it in a subprocess)."""
    lines, _mode = search_lines(prompt, project_url,
                                HOOK_TIMEOUT_S if lock_timeout is None else lock_timeout)
    if lines:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    raise SystemExit(main_search(sys.argv[1] if len(sys.argv) > 1 else "",
                                 sys.argv[2] if len(sys.argv) > 2 else ""))
