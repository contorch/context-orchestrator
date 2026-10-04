"""`contorch-memory selftest --json`: the memory end to end, by its owner.

database → (index → embed → write → embed the query → search → clean up),
with a throwaway marker document. Reports the stage reached and, on
failure, a machine code the menu bar can turn into a to-do:

  offline   no network (DNS, refused, unreachable, timed out)
  proxy     a proxy refused or needs credentials
  tls       a certificate could not be verified (often an intercepting proxy)
  key       the embedding API rejected the key
  quota     the embedding API is rate- or quota-limited
  busy      the index stayed locked by another process
  db        the database can't be opened / written
  internal  anything else

Replaces pipeline-monitor's smoketest.py (which needed a checkout on disk).
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

SCHEMA = "contorch-memory.selftest/1"


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def classify(exc: BaseException) -> str:
    from .chroma_lock import LockTimeout
    names = " ".join(type(e).__name__.lower() for e in _chain(exc))
    text = " ".join(str(e).lower() for e in _chain(exc))
    if any(isinstance(e, LockTimeout) for e in _chain(exc)):
        return "busy"
    if "proxyerror" in names or "407" in text or "proxy" in text or "tunnel connection failed" in text:
        return "proxy"
    if "certificate_verify_failed" in text or "certificate verify failed" in text \
            or "sslcertverification" in names:
        return "tls"
    if "api key not valid" in text or "api_key_invalid" in text or "permission_denied" in text \
            or "unauthenticated" in text or " 401" in text or " 403" in text or "needs a gemini api key" in text:
        return "key"
    if "resource_exhausted" in text or " 429" in text or "quota" in text or "rate limit" in text:
        return "quota"
    if any(k in names for k in ("connecterror", "connectionerror", "connecttimeout", "timeout",
                                "gaierror", "nameresolution", "networkerror")) \
            or any(k in text for k in ("nodename nor servname", "name or service not known",
                                       "network is unreachable", "connection refused",
                                       "temporary failure in name resolution", "timed out",
                                       "failed to establish a new connection")):
        return "offline"
    return "internal"


def run() -> dict:
    from . import search
    from .db import Database
    t0 = time.monotonic()
    choice = search.embedding_choice()
    doc = {"schema": SCHEMA, "ok": False, "stage": "db", "ms": 0, "embeddings": choice or "auto"}

    def finish(**kw):
        doc.update(kw)
        doc["ms"] = int((time.monotonic() - t0) * 1000)
        return doc

    try:
        p = os.environ.get("CO_DB_PATH")
        db = Database(db_path=Path(p) if p else None)
        db.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _contorch_selftest (x)")
        db.conn.execute("INSERT INTO _contorch_selftest VALUES (1)")
        db.count_transcripts()
    except Exception as exc:
        return finish(error={"code": "db", "message": f"{type(exc).__name__}: {exc}"[:300]})
    if choice == "none":
        return finish(ok=True, stage="done", vector_index="none",
                      note="embeddings off: keyword search only, nothing to embed")

    marker = "contorchselftest" + uuid.uuid4().hex[:10]
    text = f"{marker} self-test document for the contorch memory; deleted right away"
    vs = None
    try:
        doc["stage"] = "index"
        vs = search.VectorSearch(lock_timeout=15, verify=False)
        doc["vector_index"] = "in_process" if vs.in_process else "server"
        doc["stage"] = "embed"
        vec = vs.embed_documents([text])
        doc["stage"] = "write"
        vs.upsert([marker], [text], [{"type": "selftest"}], embeddings=vec)
        doc["stage"] = "embed_query"
        vs.embed_query(marker)
        doc["stage"] = "search"
        hits = vs.search(marker, n_results=5, hybrid=True)
        rank = next((i + 1 for i, h in enumerate(hits) if h.get("id") == marker), None)
        doc["stage"] = "cleanup"
        vs.remove(marker)
        if rank is None:
            return finish(stage="search", error={"code": "internal",
                                                 "message": f"marker not in the top {len(hits)} hits"})
        return finish(ok=True, stage="done", rank=rank)
    except Exception as exc:
        if vs is not None:
            try:
                vs.remove(marker)
            except Exception:
                pass
        return finish(error={"code": classify(exc), "message": f"{type(exc).__name__}: {exc}"[:300]})
