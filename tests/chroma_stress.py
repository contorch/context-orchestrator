"""Roles for the multi-process Chroma stress test (tests/test_chroma_concurrency.py).

Each role runs as its own interpreter (never fork), against ONE in-process
Chroma folder — the shape of a lightweight install: several MCP servers (one
per Claude Code session, long-lived), CLI writers, and the per-prompt hook.
Ported from the contorch-macos lab (lab-phase1/modules-both-channels/
chroma-concurrency: lab.py, run.py, emb_check.py).

    python tests/chroma_stress.py writer TAG N DELETE_EVERY OUT [--unlocked]
    python tests/chroma_stress.py server TAG DONE_FILE OUT [--unlocked]
    python tests/chroma_stress.py hook N OUT
    python tests/chroma_stress.py verify SPEC_JSON

Locked roles use the real VectorSearch (the code under test). `--unlocked`
roles reproduce the pre-0.5 pattern with raw chromadb — one PersistentClient
per process held for its whole life, reload() = a new client on the cached
System — as the control that shows the loss the lock prevents.

Embeddings are a deterministic 64-d hash ("lab-hash-64"): the test measures
storage concurrency, not ONNX CPU (8 processes x MiniLM took minutes per 150
docs in the lab).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTION = "context"


def _hash_ef():
    import numpy as np
    from chromadb.api.types import EmbeddingFunction

    class LabHashEF(EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input):
            out = []
            for t in input:
                v = np.zeros(64, dtype=np.float32)
                for w in t.lower().split():
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
                n = float(np.linalg.norm(v)) or 1.0
                out.append(v / n)
            return out

        @staticmethod
        def name():
            return "lab-hash-64"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return LabHashEF()

    return LabHashEF()


def install_hash_ef() -> None:
    import context_orchestrator.search as s
    s._build_embedding_function = _hash_ef


def chroma_path() -> Path:
    return Path(os.environ["CO_CHROMA_PATH"])


def text_for(doc_id: str) -> str:
    rnd = random.Random(doc_id)
    words = ["budget", "roadmap", "hiring", "latency", "chroma", "kubernetes", "invoice",
             "migration", "pricing", "onboarding", "sqlite", "launch", "retro", "design",
             "customer", "outage", "refactor", "meeting", "deadline", "contract"]
    return f"DOC {doc_id} " + " ".join(rnd.choice(words) for _ in range(40))


def _vs():
    install_hash_ef()
    from context_orchestrator.search import VectorSearch
    return VectorSearch(chroma_path=chroma_path())


def _raw_collection():
    """The pre-0.5 shape: a PersistentClient opened once and kept."""
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path()))
    return client.get_or_create_collection(name=COLLECTION, embedding_function=_hash_ef(),
                                           metadata={"hnsw:space": "cosine"})


class _ErrorCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage()[:300])


def writer(tag: str, n: int, delete_every: int, out: str, unlocked: bool) -> None:
    res = {"tag": tag, "ok": 0, "errors": [], "deleted": [], "op_ms": []}
    errs = _ErrorCounter()
    logging.getLogger("context-orchestrator").addHandler(errs)
    try:
        if unlocked:
            col = _raw_collection()
        else:
            vs = _vs()
        for k in range(n):
            did = f"{tag}-{k}"
            t = time.monotonic()
            try:
                meta = {"type": "stress", "writer": tag, "k": k}
                if unlocked:
                    col.upsert(ids=[did], documents=[text_for(did)], metadatas=[meta])
                else:
                    vs.add(did, text_for(did), meta)
                res["ok"] += 1
            except Exception as e:  # noqa: BLE001
                res["errors"].append(f"{did}: {type(e).__name__}: {e}"[:300])
            if delete_every and k and k % delete_every == 0:
                victim = f"{tag}-{k - 1}"
                try:
                    if unlocked:
                        col.delete(ids=[victim])
                    else:
                        vs.remove(victim)      # logs (doesn't raise) on failure
                    res["deleted"].append(victim)
                except Exception as e:  # noqa: BLE001
                    res["errors"].append(f"delete {victim}: {type(e).__name__}: {e}"[:300])
            res["op_ms"].append(int((time.monotonic() - t) * 1000))
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"FATAL {type(e).__name__}: {e}"[:800])
    res["errors"] += [f"logged: {m}" for m in errs.messages]
    Path(out).write_text(json.dumps(res))


def server(tag: str, done_file: str, out: str, unlocked: bool) -> None:
    """A long-lived MCP server: one VectorSearch for the process's life,
    reload() + count + hybrid/MMR search every 0.3 s (server.py search())."""
    res = {"tag": tag, "samples": [], "errors": []}
    try:
        if unlocked:
            import chromadb
            col = _raw_collection()
        else:
            vs = _vs()
        t0 = time.monotonic()
        while not Path(done_file).exists() and time.monotonic() - t0 < 1800:
            try:
                if unlocked:
                    client = chromadb.PersistentClient(path=str(chroma_path()))   # old reload()
                    col = client.get_or_create_collection(name=COLLECTION, embedding_function=_hash_ef(),
                                                          metadata={"hnsw:space": "cosine"})
                    cnt = col.count()
                    hits = col.query(query_texts=["budget roadmap latency"], n_results=5)["ids"][0]
                else:
                    vs.reload()
                    cnt = vs.count()
                    hits = vs.search("budget roadmap latency", n_results=5, hybrid=True, mmr=True)
                res["samples"].append({"t": round(time.monotonic() - t0, 2), "count": cnt,
                                       "hits": len(hits)})
            except Exception as e:  # noqa: BLE001
                res["errors"].append(f"{type(e).__name__}: {e}"[:300])
            time.sleep(0.3)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"FATAL {type(e).__name__}: {e}"[:800])
    Path(out).write_text(json.dumps(res))


HOOK_SNIPPET = r'''
import json, sys, time
t0 = time.monotonic()
sys.path.insert(0, {here!r})
import chroma_stress
chroma_stress.install_hash_ef()
from context_orchestrator.hook import search_lines
lines, mode = search_lines("pricing contract deadline", "")
print(json.dumps({{"ms": int((time.monotonic() - t0) * 1000), "hits": len(lines), "mode": mode}}))
'''


def hook(n: int, out: str) -> None:
    """The auto-context hook: a fresh interpreter per prompt, through the
    real hook search (2 s lock wait, then keyword-only)."""
    res = {"runs": [], "errors": []}
    code = HOOK_SNIPPET.format(here=str(HERE))
    for _ in range(n):
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            res["errors"].append("timeout 120s")
            continue
        if p.returncode == 0:
            try:
                res["runs"].append(json.loads(p.stdout.strip().splitlines()[-1]))
            except Exception:  # noqa: BLE001
                res["errors"].append("bad output: " + p.stdout[-200:] + p.stderr[-300:])
        else:
            res["errors"].append(f"rc {p.returncode}: " + p.stderr.strip()[-400:])
    Path(out).write_text(json.dumps(res))


def verify(spec_file: str) -> dict:
    """Fresh-process view, through the session lock: every expected doc present,
    every deleted doc absent, every stored id has a vector, and each doc's own
    text finds it as a nearest neighbour (HNSW consistent with the rows)."""
    spec = json.loads(Path(spec_file).read_text())
    expected, gone = spec["expected"], spec["gone"]
    vs = _vs()
    with vs.session() as col:
        got = col.get(include=[], limit=1_000_000)["ids"]
        got_set = set(got)
        vectorless = []
        for i in got:
            try:
                e = col.get(ids=[i], include=["embeddings"])["embeddings"]
                if e is None or len(e) == 0:
                    vectorless.append(i)
            except Exception as exc:  # "Error finding id" / "Error getting embedding"
                vectorless.append(f"{i}: {str(exc)[:60]}")
        knn_miss = []
        present = sorted(got_set & set(expected))
        ef = _hash_ef()
        for did in present:
            try:
                r = col.query(query_embeddings=[list(map(float, ef([text_for(did)])[0]))], n_results=5)
                if did not in r["ids"][0]:
                    knn_miss.append(did)
            except Exception as exc:  # noqa: BLE001
                knn_miss.append(f"{did}: {str(exc)[:60]}")
        count = col.count()
    missing = sorted(set(expected) - got_set)
    resurrected = sorted(set(gone) & got_set)
    return {"count": count, "ids": len(got), "expected": len(expected),
            "missing": len(missing), "missing_examples": missing[:10],
            "resurrected": len(resurrected), "resurrected_examples": resurrected[:10],
            "dup_ids": len(got) - len(got_set),
            "vectorless": len(vectorless), "vectorless_examples": vectorless[:10],
            "knn_checked": len(present), "knn_miss": len(knn_miss), "knn_miss_examples": knn_miss[:10]}


if __name__ == "__main__":
    role, args = sys.argv[1], sys.argv[2:]
    unlocked = "--unlocked" in args
    args = [a for a in args if a != "--unlocked"]
    if role == "writer":
        writer(args[0], int(args[1]), int(args[2]), args[3], unlocked)
    elif role == "server":
        server(args[0], args[1], args[2], unlocked)
    elif role == "hook":
        hook(int(args[0]), args[1])
    elif role == "verify":
        print(json.dumps(verify(args[0])))
    else:
        raise SystemExit(f"unknown role {role}")
