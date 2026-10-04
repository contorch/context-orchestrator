"""`contorch-memory status --json`, `selftest --json`, `where --json`."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from context_orchestrator import memstatus, search, selftest, settings
from context_orchestrator.chroma_lock import LockTimeout

SRC = Path(__file__).resolve().parent.parent / "src"


class _FakeEF:
    def name(self):
        return "fake-ef"

    def __call__(self, input):
        return [[float((sum(map(ord, t)) >> i) & 1) + 0.01 for i in range(8)] for t in input]

    def embed_query(self, input):
        return self(input if isinstance(input, list) else [input])


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "context.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _FakeEF())
    return tmp_path


def _run_cli(args, env_extra, tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]),
               **env_extra)
    return subprocess.run([sys.executable, "-c",
                           "import sys; from context_orchestrator.settings import main; "
                           f"rc = main({args!r}); "
                           "print('CHROMADB_IMPORTED' if 'chromadb' in sys.modules else 'NO_CHROMADB', file=sys.stderr); "
                           "sys.exit(rc)"], env=env, capture_output=True, text=True, timeout=120)


def test_status_json_schema_and_no_chromadb_import(mem, tmp_path):
    from context_orchestrator import transcripts as T
    from context_orchestrator.db import Database
    db = Database(db_path=tmp_path / "context.db")
    vs = search.VectorSearch(chroma_path=tmp_path / "chroma")
    mid, _ = T.add_text(db, "[10:00:01] Jane: the zebra migration ships Thursday", started_at="2026-09-28T10:00")
    T.index_row(vs, db, db.get_transcript(mid))
    T.add_text(db, "[11:00:01] Bob: a second meeting about pricing", started_at="2026-09-28T11:00")
    env = {"CO_DB_PATH": str(tmp_path / "context.db"), "CO_CHROMA_PATH": str(tmp_path / "chroma"),
           "CO_EMBEDDING_MODEL": "local"}
    r = _run_cli(["status", "--json"], env, tmp_path)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert "NO_CHROMADB" in r.stderr, "status --json must not import chromadb"
    assert doc["schema"] == "contorch-memory.status/1" and doc["ok"]
    assert doc["vector_index"] == "in_process" and doc["index_compatible"] is True
    assert {"embeddings", "vector_index", "docs", "transcripts", "chromadb_version",
            "index_written_by", "index_compatible", "ok"} <= set(doc)
    assert doc["transcripts"] == 2 and doc["pending"] == 2   # identity "default" ≠ the test's fake-ef
    assert doc["chromadb_version"] == "1.5.9" and doc["index_written_by"] == "1.5.9"
    r = _run_cli(["status", "--json", "--deep"], env, tmp_path)
    deep = json.loads(r.stdout)
    assert deep["deep"] and "CHROMADB_IMPORTED" in r.stderr


def test_status_counts_docs_from_the_index_file(mem, tmp_path, monkeypatch):
    vs = search.VectorSearch(chroma_path=tmp_path / "chroma")
    vs.upsert(["a", "b", "c"], ["x", "y", "z"], [{"t": 1}] * 3)
    monkeypatch.setattr(memstatus, "effective_identity", lambda choice: "fake-ef")
    doc = memstatus.status()
    assert doc["docs"] == 3 and doc["collection"] == "context"


def test_status_text_mode_unchanged(mem, capsys):
    assert settings.main(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [l.split(":")[0] for l in lines] == ["embeddings", "transcripts", "vector index", "full-text"]


def test_compatibility_table():
    c = memstatus.index_compatible
    assert c(None, "1.5.9")                 # unknown writer (pre-0.5 contorch / server)
    assert c("1.5.9", "1.5.9")
    assert c("0.6.3", "1.5.9")              # chromadb migrates older formats forward
    assert c("1.5.2", "1.5.9")
    assert not c("1.6.0", "1.5.9")          # written by a newer chromadb: can't downgrade
    assert not c("2.0.0", "1.5.9")
    assert not c("1.5.9", None)             # no chromadb here at all


def test_status_reports_a_downgrade(mem, tmp_path):
    (tmp_path / "chroma").mkdir()
    (tmp_path / "chroma" / "contorch-index.json").write_text(json.dumps({"chromadb_version": "9.0.0"}))
    doc = memstatus.status()
    assert not doc["ok"] and not doc["index_compatible"]
    assert doc["error"]["code"] == "chroma_downgrade"


def test_status_embeddings_none(monkeypatch, tmp_path):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "none")
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "c.db"))
    doc = memstatus.status()
    assert doc["vector_index"] == "none" and doc["ok"] and doc["docs"] is None


def test_selftest_end_to_end_and_it_cleans_up(mem, tmp_path, capfd):
    assert settings.main(["selftest", "--json"]) == 0
    doc = json.loads(capfd.readouterr().out)
    assert doc["schema"] == "contorch-memory.selftest/1"
    assert doc["ok"] and doc["stage"] == "done" and doc["rank"] == 1 and doc["ms"] >= 0
    vs = search.VectorSearch(chroma_path=tmp_path / "chroma")
    assert vs.all_ids() == [], "marker removed"


@pytest.mark.parametrize("exc,code", [
    (ConnectionError("[Errno 8] nodename nor servname provided, or not known"), "offline"),
    (OSError("Network is unreachable"), "offline"),
    (RuntimeError("ProxyError: 407 Proxy Authentication Required"), "proxy"),
    (RuntimeError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"), "tls"),
    (RuntimeError("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key."), "key"),
    (RuntimeError("429 RESOURCE_EXHAUSTED"), "quota"),
    (LockTimeout("busy"), "busy"),
    (ValueError("something else"), "internal"),
])
def test_selftest_classifies_failures(mem, monkeypatch, capfd, exc, code):
    def boom(self, texts):
        raise exc
    monkeypatch.setattr(search.VectorSearch, "embed_documents", boom)
    assert settings.main(["selftest", "--json"]) == 1
    doc = json.loads(capfd.readouterr().out)
    assert not doc["ok"] and doc["stage"] == "embed" and doc["error"]["code"] == code


def test_selftest_with_embeddings_off(monkeypatch, tmp_path):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "none")
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "c.db"))
    doc = selftest.run()
    assert doc["ok"] and doc["vector_index"] == "none"


def test_selftest_chained_offline_error(mem, monkeypatch):
    def boom(self, texts):
        try:
            raise OSError("[Errno 61] Connection refused")
        except OSError as e:
            raise RuntimeError("embedding failed") from e
    monkeypatch.setattr(search.VectorSearch, "embed_documents", boom)
    assert selftest.run()["error"]["code"] == "offline"


def test_where_json(mem, capfd, monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    assert settings.main(["where", "--json", "--channel", "dev"]) == 0
    doc = json.loads(capfd.readouterr().out)
    assert doc["schema"] == "contorch-memory.where/1" and doc["channel"] == "dev"
    for k in ("mcp", "hook", "transcripts", "chroma_cli", "memory"):
        assert Path(doc[k]).is_absolute() and Path(doc[k]).parent == Path(sys.executable).parent
    assert doc["skills"]["transcripts"] == str(tmp_path / "cc" / "skills" / "transcripts")
    assert doc["chroma"] == str(tmp_path / "chroma")
