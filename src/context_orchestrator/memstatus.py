"""`contorch-memory status --json` and `where --json`.

status must stay cheap and must NOT import chromadb (unless --deep): it is
what the menu bar and the app's adopt/rollback logic poll, and it has to
answer from any interpreter state. Vector counts are read straight from the
index's own SQLite file (in-process) or over HTTP (server); the chromadb
version comes from this interpreter's dist-info, and who last wrote the index
from <chroma>/contorch-index.json (stamped by every in-process write).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

STATUS_SCHEMA = "contorch-memory.status/1"
WHERE_SCHEMA = "contorch-memory.where/1"


def version_key(v: Optional[str]) -> tuple:
    parts = re.split(r"[.+-]", v or "0")[:3]
    return tuple(int(p) if p.isdigit() else 0 for p in parts) + (0,) * (3 - len(parts))


def index_compatible(written_by: Optional[str], installed: Optional[str]) -> bool:
    """Can this chromadb open an index last written by `written_by`?
    chromadb migrates older on-disk formats forward but cannot read a newer
    one, so: unknown writer (pre-0.5 contorch, or a server) → yes; a writer
    newer than us → no; no chromadb here → no."""
    if not installed:
        return False
    if not written_by:
        return True
    return version_key(written_by) <= version_key(installed)


def effective_identity(choice: str) -> str:
    """The embedding identity search would use, without building the model."""
    from .search import _resolve_gemini_api_key
    if choice == "none":
        return "none"
    if choice == "local":
        return "default"
    if choice.startswith("gemini-"):
        return f"gemini-{choice}"
    if not choice:
        if find_spec("google") and find_spec("google.genai") and _resolve_gemini_api_key():
            return "gemini-gemini-embedding-001"
        return "default"
    return "sentence_transformer"


def _collection_name(chroma_path: Path, identity: str) -> Optional[str]:
    try:
        mapping = json.loads((chroma_path / "contorch-collections.json").read_text())
    except (OSError, ValueError):
        mapping = {}
    if identity in mapping:
        return mapping[identity]
    return None if "context" in mapping.values() else "context"


def docs_in_process(chroma_path: Path, collection: Optional[str]) -> Optional[int]:
    """Documents in one collection, from chroma.sqlite3 (read-only)."""
    db = chroma_path / "chroma.sqlite3"
    if collection is None or not db.exists():
        return 0 if collection is None or not chroma_path.exists() else None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
        try:
            row = con.execute(
                "SELECT COUNT(e.id) FROM collections c JOIN segments s ON s.collection = c.id "
                "AND s.scope = 'METADATA' LEFT JOIN embeddings e ON e.segment_id = s.id "
                "WHERE c.name = ?", (collection,)).fetchone()
        finally:
            con.close()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return None


def docs_server(host: str, port: int, collection: Optional[str]) -> Optional[int]:
    """Documents in one collection, over the chroma server's v2 HTTP API."""
    base = f"http://{host}:{port}/api/v2/tenants/default_tenant/databases/default_database/collections"
    try:
        with urllib.request.urlopen(base, timeout=2) as r:
            cols = json.loads(r.read())
        cid = next((c["id"] for c in cols if c.get("name") == collection), None)
        if cid is None:
            return 0
        with urllib.request.urlopen(f"{base}/{cid}/count", timeout=2) as r:
            return int(json.loads(r.read()))
    except Exception:
        return None


def status(deep: bool = False) -> dict:
    from . import search
    from .db import Database
    from .chroma_daemon import CHROMA_PATH as SERVER_CHROMA_PATH
    choice = search.embedding_choice()
    identity = effective_identity(choice)
    chroma_path = search._embedded_path_if_no_server()
    if choice == "none":
        vector_index = "none"
    elif chroma_path is None:
        vector_index = "server"
    else:
        vector_index = "in_process"
    installed = search.chromadb_version()
    index_dir = chroma_path or SERVER_CHROMA_PATH
    written_by = search.read_index_stamp(index_dir).get("chromadb_version")
    compatible = index_compatible(written_by, installed)
    doc: dict = {"schema": STATUS_SCHEMA, "ok": True, "embeddings": choice or "auto",
                 "embedding_identity": identity, "vector_index": vector_index,
                 "index_path": str(index_dir), "docs": None, "transcripts": None,
                 "pending": None, "chromadb_version": installed,
                 "index_written_by": written_by, "index_compatible": compatible,
                 "deep": deep}
    errors = []
    try:
        p = os.environ.get("CO_DB_PATH")
        db = Database(db_path=Path(p) if p else None)
        total, _ = db.count_transcripts()
        doc["transcripts"] = total
        doc["pending"] = len(db.transcripts_to_index(float("inf"), identity))
        doc["db_path"] = str(db.db_path)
    except Exception as exc:
        errors.append({"code": "db_unreadable", "message": f"{type(exc).__name__}: {exc}"[:300]})
    if vector_index != "none":
        name = _collection_name(index_dir, identity)
        doc["collection"] = name
        if deep:
            try:
                vs = search.VectorSearch(verify=False)
                doc["docs"] = vs.count()
                doc["collection"] = vs.collection_name
            except Exception as exc:
                errors.append({"code": "index_unreadable", "message": f"{type(exc).__name__}: {exc}"[:300]})
        elif vector_index == "in_process":
            doc["docs"] = docs_in_process(index_dir, name)
        else:
            host = os.environ.get("CO_CHROMA_HOST", search.DEFAULT_CHROMA_HOST)
            port = int(os.environ.get("CO_CHROMA_PORT", search.DEFAULT_CHROMA_PORT))
            doc["docs"] = docs_server(host, port, name)
            if doc["docs"] is None:
                errors.append({"code": "server_unreachable",
                               "message": f"chroma server at {host}:{port} did not answer"})
    if vector_index == "in_process" and not compatible:
        errors.append({"code": "chroma_downgrade" if installed else "chromadb_missing",
                       "message": f"index written by chromadb {written_by}, this is {installed}"})
    if errors:
        doc["ok"] = False
        doc["error"] = errors[0]
        if len(errors) > 1:
            doc["errors"] = errors
    return doc


def where(chan: Optional[str] = None) -> dict:
    """Absolute paths of this install's commands and files (pm reads these
    instead of guessing install layouts)."""
    from . import claude_install as ci
    from . import search
    from .db import DEFAULT_DB_PATH
    chan = chan or ci.channel()
    eps = ci.entry_points(chan)
    paths = ci.ClaudePaths.detect()
    return {"schema": WHERE_SCHEMA, "ok": True, "channel": chan, "python": sys.executable,
            "mcp": eps["mcp"], "hook": eps["hook"], "transcripts": eps["transcripts"],
            "memory": eps["memory"], "chroma_cli": eps["chroma_cli"],
            "skills": {"transcripts": str(paths.skill_dir)},
            "db": os.environ.get("CO_DB_PATH") or str(DEFAULT_DB_PATH),
            "chroma": str(search._embedded_path_if_no_server() or search.DEFAULT_CHROMA_PATH),
            "env_file": str(Path.home() / ".context-orchestrator" / "env")}


def describe(doc: dict) -> str:
    lines = [f"vector index: {doc['vector_index']} ({doc.get('docs')} docs, "
             f"collection {doc.get('collection')})",
             f"transcripts:  {doc.get('transcripts')} stored, {doc.get('pending')} pending",
             f"chromadb:     {doc.get('chromadb_version')} (index written by "
             f"{doc.get('index_written_by') or 'unknown'}; "
             f"{'compatible' if doc.get('index_compatible') else 'NOT compatible'})"]
    if doc.get("error"):
        lines.append(f"error:        {doc['error']['code']}: {doc['error']['message']}")
    return "\n".join(lines)
