"""The in-process Chroma session lock (chroma_lock + VectorSearch.session)."""
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from context_orchestrator import chroma_lock, search
from context_orchestrator.chroma_lock import LockTimeout, is_held, lock_path_for
from context_orchestrator.search import VectorSearch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"


class _FakeEF:
    """Deterministic 8-d vectors; no model download, no API."""
    def name(self):
        return "fake-ef"

    def __call__(self, input):
        return [[float((sum(map(ord, t)) >> i) & 1) + 0.01 for i in range(8)] for t in input]

    def embed_query(self, input):
        return self(input if isinstance(input, list) else [input])


@pytest.fixture
def vs(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _FakeEF())
    return VectorSearch(chroma_path=tmp_path / "chroma")


def test_lock_file_sits_next_to_the_folder():
    assert lock_path_for(Path("/x/.context-orchestrator/chroma")) == Path("/x/.context-orchestrator/chroma.lock")


def test_every_operation_opens_fresh_inside_the_lock_and_holds_nothing_after(vs, monkeypatch):
    opened, cleared = [], []
    real_open = VectorSearch._open_local

    def spy_open(self):
        opened.append(is_held(self.lock_path))
        real_open(self)
    monkeypatch.setattr(VectorSearch, "_open_local", spy_open)
    real_clear = search._clear_chroma_system_cache

    def spy_clear():
        cleared.append(is_held(vs.lock_path))
        real_clear()
    monkeypatch.setattr(search, "_clear_chroma_system_cache", spy_clear)

    vs.add("a", "alpha budget", {"type": "t"})
    vs.upsert(["b", "c"], ["beta roadmap", "gamma pricing"], [{"type": "t"}, {"type": "t"}])
    assert vs.count() == 3
    assert vs.search("budget", n_results=2)
    assert vs.search("budget", n_results=2, hybrid=True, mmr=True)
    assert set(vs.all_ids()) == {"a", "b", "c"}
    vs.remove("a")
    assert vs.count() == 2
    assert len(opened) == 8 and all(opened), "one fresh open per operation, always under the lock"
    assert len(cleared) == 16 and all(cleared), "System cache cleared on entry and exit, under the lock"
    assert vs._col is None and vs._client is None, "no client held between sessions"
    assert not is_held(vs.lock_path)
    from chromadb.api.shared_system_client import SharedSystemClient
    assert SharedSystemClient._identifier_to_system == {}


def test_collection_outside_a_session_is_refused(vs):
    with pytest.raises(RuntimeError, match="outside"):
        vs.collection.count()
    with vs.session() as col:
        assert vs.collection is col


def test_nested_sessions_reuse_the_open_collection(vs, tmp_path, monkeypatch):
    other = VectorSearch(chroma_path=tmp_path / "chroma")
    with vs.session(write=True) as col:
        vs.add("n1", "nested write", {"type": "t"})         # same instance: reuses col
        assert vs.collection is col
        other.add("n2", "another instance, same process", {"type": "t"})   # re-entrant lock
        assert is_held(vs.lock_path)
    assert vs.count() == 2 and not is_held(vs.lock_path)


def test_embeddings_are_computed_outside_the_lock(vs, monkeypatch):
    seen = []
    real = vs.embedding_function

    class Spy:
        def name(self):
            return "fake-ef"

        def __call__(self, input):
            seen.append(("doc", is_held(vs.lock_path)))
            return real(input)

        def embed_query(self, input):
            seen.append(("query", is_held(vs.lock_path)))
            return real.embed_query(input)
    vs._ef = Spy()
    vs.add("x", "some text", {"type": "t"})
    vs.search("some", n_results=1)
    assert seen and not any(held for _k, held in seen), seen


def test_rerank_runs_outside_the_lock(vs, monkeypatch):
    vs.add("x", "some text about pricing", {"type": "t"})
    held = []

    def fake_rerank(query, cands, n, model):
        held.append(is_held(vs.lock_path))
        return cands[:n]
    monkeypatch.setattr(search, "_llm_rerank", fake_rerank)
    assert vs.search("pricing", n_results=1, rerank=True, rerank_model="gemini-x")
    assert held == [False]


def test_writes_stamp_the_chromadb_version_reads_do_not(vs):
    stamp = vs.chroma_path / "contorch-index.json"
    vs.count()
    assert not stamp.exists()
    vs.add("x", "text", {"type": "t"})
    data = json.loads(stamp.read_text())
    assert data == {"schema": "contorch-index/1", "chromadb_version": search.chromadb_version()}
    assert data["chromadb_version"] == "1.5.9"


def test_remove_logs_failures_instead_of_swallowing(vs, monkeypatch, caplog):
    def broken(self, *a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(VectorSearch, "_open_local", broken)
    with caplog.at_level(logging.ERROR, logger="context-orchestrator"):
        vs.remove("whatever")
    assert "could not remove" in caplog.text and "disk on fire" in caplog.text


_HOLDER = textwrap.dedent("""
    import fcntl, sys, time
    fh = open(sys.argv[1], "a")
    fcntl.flock(fh, fcntl.LOCK_EX)
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
""")


@pytest.fixture
def held_by_other_process():
    procs = []

    def hold(lock_file: Path, seconds: float = 30):
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen([sys.executable, "-c", _HOLDER, str(lock_file), str(seconds)],
                             stdout=subprocess.PIPE, text=True)
        assert p.stdout.readline().strip() == "held"
        procs.append(p)
        return p
    yield hold
    for p in procs:
        p.kill()
        p.wait()


def test_another_process_holding_the_lock_times_out(vs, held_by_other_process):
    held_by_other_process(vs.lock_path)
    vs.lock_timeout = 0.3
    t = time.monotonic()
    with pytest.raises(LockTimeout):
        vs.count()
    assert 0.25 <= time.monotonic() - t < 2


def test_hook_waits_at_most_two_seconds_then_answers_keyword_only(tmp_path, monkeypatch,
                                                                  held_by_other_process):
    from context_orchestrator import hook
    from context_orchestrator.db import Database
    assert chroma_lock.HOOK_TIMEOUT_S == 2.0
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _FakeEF())
    chroma = tmp_path / "chroma"
    monkeypatch.setenv("CO_CHROMA_PATH", str(chroma))
    db = Database(db_path=tmp_path / "c.db")
    t = db.create_task("launch", project="p")
    db.add_source(t["id"], "text", "The zebra migration ships on Thursday", notes="")
    VectorSearch(chroma_path=chroma).add("v1", "vector-only zebra text", {"type": "source"})

    lines, mode = hook.search_lines("zebra migration", "", db=db)
    assert mode == "vector" and lines

    held_by_other_process(lock_path_for(chroma))
    start = time.monotonic()
    lines, mode = hook.search_lines("zebra migration", "", db=db)
    waited = time.monotonic() - start
    assert mode == "keyword"
    assert lines and "zebra migration ships" in lines[0]
    assert 1.9 <= waited < 3.5, waited


_READER = textwrap.dedent("""
    import json, sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[3])
    import test_chroma_lock as t
    from context_orchestrator import search
    search._build_embedding_function = lambda: t._FakeEF()
    from context_orchestrator.search import VectorSearch
    vs = VectorSearch(chroma_path=Path(sys.argv[1]))      # opened once, like an MCP server
    vs.add("warm", "warm up the reader", {"type": "t"})
    vs.search("warm", n_results=1)
    sig = Path(sys.argv[2])
    (sig / "ready").write_text("1")
    while not (sig / "written").exists():
        time.sleep(0.05)
    vs.reload()
    found = 0
    for i in range(40):
        hits = vs.search(f"late doc number {i} " + "x" * i, n_results=50)
        found += any(h["id"] == f"late-{i}" for h in hits)
    print(json.dumps({"found": found, "count": vs.count()}))
""")


def test_long_lived_reader_sees_another_process_writes(tmp_path, monkeypatch):
    """The lab's stale-reader case: before the lock, a reader that opened
    first never saw another process's vectors (kNN 0/40), even after reload()."""
    chroma, sig = tmp_path / "chroma", tmp_path / "sig"
    sig.mkdir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), str(HERE)]))
    reader = subprocess.Popen([sys.executable, "-c", _READER, str(chroma), str(sig), str(HERE)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 60
    while not (sig / "ready").exists():
        assert reader.poll() is None, reader.stderr.read()[-2000:]
        assert time.monotonic() < deadline
        time.sleep(0.05)
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _FakeEF())
    writer = VectorSearch(chroma_path=chroma)
    writer.upsert([f"late-{i}" for i in range(40)],
                  [f"late doc number {i} " + "x" * i for i in range(40)],
                  [{"type": "t"}] * 40)
    (sig / "written").write_text("1")
    out, err = reader.communicate(timeout=120)
    assert reader.returncode == 0, err[-2000:]
    res = json.loads(out.strip().splitlines()[-1])
    assert res["count"] == 41
    assert res["found"] == 40, res


def test_http_mode_keeps_one_client_and_takes_no_lock(monkeypatch):
    """The chroma server serialises access itself; nothing changes there."""
    calls = []

    class FakeCol:
        name = "context"

        def count(self):
            return 0

    class FakeClient:
        def get_or_create_collection(self, **kw):
            calls.append(kw["name"])
            return FakeCol()
    monkeypatch.setattr(search, "_build_embedding_function", lambda: None)
    monkeypatch.setattr(VectorSearch, "_http_client_with_retry", lambda self: FakeClient())
    vs = VectorSearch(host="127.0.0.1", port=1)
    assert vs.lock_path is None and not vs.in_process
    with vs.session() as col:
        assert col is vs.collection
    assert vs.count() == 0 and calls == ["context"]


def test_server_holds_no_long_lived_client():
    from context_orchestrator import server
    if not server.vs.in_process:
        pytest.skip("server module configured for HTTP")
    assert server.vs._client is None and server.vs._col is None


def test_server_search_answers_from_full_text_when_chroma_stays_busy(tmp_path, monkeypatch,
                                                                     held_by_other_process):
    from context_orchestrator import server
    from context_orchestrator.db import Database
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _FakeEF())
    db = Database(db_path=tmp_path / "ctx.db")
    monkeypatch.setattr(server, "db", db)
    vs = VectorSearch(chroma_path=tmp_path / "chroma", lock_timeout=0.3)
    monkeypatch.setattr(server, "vs", vs)
    t = db.create_task("launch", project="p")
    db.add_source(t["id"], "text", "The zebra migration ships on Thursday", notes="")
    held_by_other_process(vs.lock_path)
    out = server.search("zebra migration", project="p")
    assert "zebra migration ships" in out
