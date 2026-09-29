"""Transcripts stored in SQLite, indexed into Chroma, no files involved."""
import datetime as dt
import io
import json
import tempfile
import time
import zipfile
from pathlib import Path

import pytest

from context_orchestrator import transcripts as T
from context_orchestrator.db import Database
from context_orchestrator.search import VectorSearch


@pytest.fixture
def db(tmp_path):
    return Database(db_path=tmp_path / "context.db")


@pytest.fixture
def vs():
    return VectorSearch(chroma_path=Path(tempfile.mkdtemp()))


VTT = """WEBVTT

00:00:05.000 --> 00:00:09.000
<v Jane Doe>We doubled revenue after moving to annual pricing.

00:01:10.500 --> 00:01:12.000
<v Host>How long did that take?
"""


def test_add_text_stores_and_dedupes(db):
    mid, created = T.add_text(db, "[14:00:01] Host: welcome to the show everyone",
                              title="Podcast with Jane", started_at="2026-09-28T14:00")
    assert created and mid == "2026-09-28-1400-podcast-with-jane"
    assert db.get_transcript(mid)["source"] == ""
    again, created2 = T.add_text(db, "[14:00:01] Host: welcome to the show everyone", title="other")
    assert again == mid and not created2


def test_same_title_same_minute_gets_a_new_id(db):
    a, _ = T.add_text(db, "first episode text here", title="ep", started_at="2026-09-28T14:00")
    b, _ = T.add_text(db, "second episode text here", title="ep", started_at="2026-09-28T14:00")
    assert a != b and b.endswith("-2")


def test_captions_become_wall_clock_lines(db):
    mid, _ = T.add_text(db, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    body = db.get_transcript(mid)["body"]
    assert "[14:00:05] Jane Doe: We doubled revenue" in body
    assert "[14:01:10] Host: How long" in body


def test_index_row_gives_timestamped_searchable_chunks(db, vs):
    mid, _ = T.add_text(db, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    n = T.index_row(vs, db, db.get_transcript(mid))
    assert n == 1
    meta = vs.collection.get(where={"meeting_id": mid}, include=["metadatas"])["metadatas"][0]
    assert meta["chunk_type"] == "speech" and meta["title"] == "Jane"
    assert meta["start_ts_iso"].startswith("2026-09-28T14:00:05")
    assert db.count_transcripts() == (1, 0)


def test_failed_embedding_keeps_text_and_old_chunks(db, vs, monkeypatch):
    mid, _ = T.add_text(db, "[14:00:01] Host: version one of the notes on pricing", started_at="2026-09-28T14:00")
    T.index_row(vs, db, db.get_transcript(mid))
    db.put_transcript(mid, "[14:00:01] Host: version two of the notes on pricing", now=time.time() + 1)

    def boom(**_kw):
        raise RuntimeError("API key not valid")
    monkeypatch.setattr(vs.collection, "upsert", boom)
    assert T.index_pending(vs, db, settle_seconds=0, now=time.time() + 5) == []
    assert "version two" in db.get_transcript(mid)["body"]
    assert db.count_transcripts() == (1, 1), "still pending, retried later"
    docs = vs.collection.get(where={"meeting_id": mid})["documents"]
    assert docs and "version one" in docs[0], "previous index entries stay searchable"


def test_import_zip_reads_members_in_memory(db, tmp_path):
    z = tmp_path / "old.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("transcripts/meeting-2026-05-01T10-00-00.md", "# Meeting\n\n[10:00:01] **Me:** hello there team\n")
        zf.writestr("__MACOSX/._junk.md", "junk")
        zf.writestr("notes/interview.vtt", VTT)
        zf.writestr("image.png", b"\x89PNG")
    stats = T.import_path(db, z)
    assert stats["stored"] == 2
    ids = {r["meeting_id"] for r in db.list_transcripts()}
    assert "meeting-2026-05-01T10-00-00" in ids
    assert any(i.endswith("-interview") for i in ids)
    assert T.import_path(db, z)["unchanged"] == 2


def test_import_dir_delete_removes_only_stored_settled_files(db, tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    old = d / "meeting-2026-05-01T10-00-00.md"
    old.write_text("[10:00:01] **Them:** the old meeting text")
    past = time.time() - 600
    import os
    os.utime(old, (past, past))
    live = d / "meeting-2026-09-28T10-00-00.md"
    live.write_text("[10:00:01] **Them:** still being written")
    stats = T.import_path(db, d, delete=True, backup_dir=tmp_path / "backups")
    backup = stats.pop("backup")
    assert stats == {"stored": 1, "unchanged": 0, "deleted": 1, "skipped_live": 1}
    assert not old.exists() and live.exists()
    with zipfile.ZipFile(backup) as z:                  # the deleted file, intact
        assert z.namelist() == ["meeting-2026-05-01T10-00-00.md"]
        assert z.read("meeting-2026-05-01T10-00-00.md") == b"[10:00:01] **Them:** the old meeting text"
    assert "old meeting text" in db.get_transcript("meeting-2026-05-01T10-00-00")["body"]


class _FakeEF:
    """Deterministic 8-d vectors so a bundle round-trips without any API."""
    def name(self):
        return "fake-ef"

    def __call__(self, input):
        return self.embed_documents(input)

    def embed_documents(self, input):
        return [[float((hash(t) >> i) & 1) for i in range(8)] for t in input]

    def embed_query(self, input):
        return self.embed_documents(input if isinstance(input, list) else [input])


def test_bundle_round_trip_loads_vectors_without_embedding(db, tmp_path, monkeypatch):
    src_db = Database(db_path=tmp_path / "other-machine.db")
    mid, _ = T.add_text(src_db, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    buf = io.StringIO()
    assert T.write_bundle([src_db.get_transcript(mid)], buf, _FakeEF()) == 1
    bundle = tmp_path / "b.jsonl"
    bundle.write_text(buf.getvalue())
    assert T.is_bundle(bundle)

    vs = VectorSearch(chroma_path=tmp_path / "chroma")
    monkeypatch.setattr(T, "_ef_identity", lambda ef: "fake-ef")
    def no_embedding(*a, **k):
        raise AssertionError("import must not call the embedding API")
    monkeypatch.setattr(vs.collection, "_embedding_function", no_embedding, raising=False)

    stats = T.import_bundle(vs, db, bundle)
    assert stats["vectors_loaded"] == 1 and stats["compatible"]
    got = vs.collection.get(where={"meeting_id": mid}, include=["embeddings"])
    assert len(got["embeddings"][0]) == 8
    assert db.count_transcripts() == (1, 0)
    # Same bundle again: nothing to do.
    assert T.import_bundle(vs, db, bundle)["unchanged"] == 1


def test_bundle_from_another_model_stores_text_only(db, tmp_path):
    src_db = Database(db_path=tmp_path / "other.db")
    mid, _ = T.add_text(src_db, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    buf = io.StringIO()
    T.write_bundle([src_db.get_transcript(mid)], buf, _FakeEF())
    bundle = tmp_path / "b.jsonl"
    bundle.write_text(buf.getvalue())
    vs = VectorSearch(chroma_path=tmp_path / "chroma")   # local default EF, not fake-ef
    stats = T.import_bundle(vs, db, bundle)
    assert not stats["compatible"] and stats["pending"] == 1
    assert db.get_transcript(mid) is not None
    assert vs.collection.count() == 0


def test_search_falls_back_to_keywords_when_the_query_cannot_be_embedded(db, vs, monkeypatch):
    mid, _ = T.add_text(db, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    T.index_row(vs, db, db.get_transcript(mid))

    def no_key(**_kw):
        raise RuntimeError("needs a Gemini API key")
    monkeypatch.setattr(vs.collection, "query", no_key)
    hits = vs.search("annual pricing revenue", hybrid=True, mmr=True)
    assert hits and hits[0]["metadata"]["meeting_id"] == mid
    assert vs.search("annual pricing", where={"meeting_id": "someone-else"}) == []
    assert vs.search("annual pricing", where={"$and": [{"meeting_id": mid},
                                                       {"start_ts_unix": {"$gte": 0}}]})


def test_gemini_ef_without_a_key_constructs_and_fails_only_when_used(monkeypatch):
    pytest.importorskip("google.genai")
    from context_orchestrator import search
    ef = search._build_gemini_embedding_function("gemini-embedding-001")
    with pytest.raises(RuntimeError, match="needs a Gemini API key"):
        ef(["hello"])


def test_cli_add_and_show_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "chroma"))
    f = tmp_path / "ep.vtt"
    f.write_text(VTT)
    assert T.main(["add", str(f), "--title", "Ep 12 Jane", "--started-at", "2026-09-28T14:00",
                   "--source", "https://example.com/ep12.vtt"]) == 0
    out = capsys.readouterr().out
    assert "stored 2026-09-28-1400-ep-12-jane" in out and "indexed" in out
    assert T.main(["show", "2026-09-28-1400-ep-12-jane"]) == 0
    assert "[14:00:05] Jane Doe:" in capsys.readouterr().out
    assert T.main(["add", str(f)]) == 0
    assert "already stored" in capsys.readouterr().out


def test_keyless_round_trip_export_embed_elsewhere_import(tmp_path, monkeypatch):
    """Work machine (no key) → export pending → other machine embeds → import."""
    work = Database(db_path=tmp_path / "work.db")
    mid, _ = T.add_text(work, VTT, title="Ep 12 Jane", started_at="2026-09-28T14:00:00")
    pending = tmp_path / "pending.jsonl"
    with pending.open("w") as out:
        assert T.write_text_bundle(work.transcripts_to_index(time.time()), out) == 1
    assert T.is_bundle(pending)

    # Other machine: embed straight from the text-only bundle.
    bundle = tmp_path / "bundle.jsonl"
    with bundle.open("w") as out:
        T.write_bundle(T.read_bundle_rows(pending), out, _FakeEF())

    vs = VectorSearch(chroma_path=tmp_path / "chroma")
    monkeypatch.setattr(T, "_ef_identity", lambda ef: "fake-ef")
    stats = T.import_bundle(vs, work, bundle)
    assert stats["vectors_loaded"] == 1
    assert work.get_transcript(mid)["title"] == "Ep 12 Jane"
    assert work.count_transcripts() == (1, 0)


def test_text_only_bundle_import_just_stores(db, tmp_path):
    src = Database(db_path=tmp_path / "src.db")
    T.add_text(src, "[10:00:01] Host: some words about the roadmap", started_at="2026-09-28T10:00")
    f = tmp_path / "t.jsonl"
    with f.open("w") as out:
        T.write_text_bundle([src.get_transcript(r["meeting_id"]) for r in src.list_transcripts()], out)
    vs = VectorSearch(chroma_path=tmp_path / "chroma")
    stats = T.import_bundle(vs, db, f)
    assert stats["text_only"] and stats["stored"] == 1 and stats["pending"] == 1


# ---- full-text search + embedding choice -----------------------------------

def test_fts_finds_transcripts_immediately_including_raw_appends(db):
    T.add_text(db, "[10:00:01] Jane: we decided to move Acme to annual billing",
               title="Pricing sync", started_at="2026-09-28T10:00")
    # meeting-capture appends straight to the table; triggers keep FTS current.
    db.conn.execute("INSERT INTO transcripts (meeting_id, title, source, started_at, body, created_at, updated_at) "
                    "VALUES ('meeting-2026-09-29T09-00-00', 'meeting-2026-09-29T09-00-00', 'meeting-capture', "
                    "'2026-09-29T09:00:00', '# Meeting\n', 1, 1)")
    db.conn.execute("UPDATE transcripts SET body = body || ? WHERE meeting_id = 'meeting-2026-09-29T09-00-00'",
                    ("[09:00:05] **Them:** the Zephyr launch slips to November\n",))
    db.conn.commit()
    hits = db.search_text("decide annual billing")          # stemming: decide ~ decided
    assert hits and hits[0]["title"] == "Pricing sync"
    z = db.search_text("zephyr")
    assert z and z[0]["meeting_id"] == "meeting-2026-09-29T09-00-00" and "November" in z[0]["text"]
    assert db.search_text("zephyr", after="2026-09-30T00:00:00") == []
    assert db.search_text("zephyr", meeting_id="other") == []
    assert db.search_text('"; DROP TABLE x; --') == [] or True   # never a syntax error


def test_fts_backfills_existing_rows_and_covers_knowledge_and_sources(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)   # a DB from before FTS existed
    old.executescript("CREATE TABLE repo_knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, repo_url TEXT NOT NULL, "
                      "insight TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));"
                      "INSERT INTO repo_knowledge (repo_url, insight) VALUES ('r', 'run the flaky xcodebuild retry loop');")
    old.commit(); old.close()
    db = Database(db_path=path)
    assert db.search_text("xcodebuild")[0]["kind"] == "repo_knowledge"
    t = db.create_task("launch", project="p")
    db.add_source(t["id"], "text", "Blink camera order ships Thursday", notes="")
    assert db.search_text("blink camera", project="p")[0]["kind"] == "source"
    assert db.search_text("blink camera", project="other") == []


def test_embeddings_none_means_no_vector_index_and_fts_search(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "none")
    vs = VectorSearch(chroma_path=tmp_path / "chroma")
    assert not vs.enabled and vs.search("x") == [] and vs.count() == 0
    db = Database(db_path=tmp_path / "c.db")
    mid, _ = T.add_text(db, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    assert T.index_row(vs, db, db.get_transcript(mid)) == 0
    assert db.count_transcripts() == (1, 0)
    assert db.get_transcript(mid)["indexed_with"] == "none"


def test_switching_models_uses_a_new_collection_and_reindexes(tmp_path, monkeypatch):
    chroma = tmp_path / "chroma"
    db = Database(db_path=tmp_path / "c.db")
    mid, _ = T.add_text(db, VTT, title="Jane", started_at="2026-09-28T14:00:00")
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    local = VectorSearch(chroma_path=chroma)
    assert local.collection.name == "context"               # first model keeps the old name
    assert T.index_pending(local, db, settle_seconds=0, now=time.time() + 5) == [mid]
    assert T.index_pending(local, db, settle_seconds=0, now=time.time() + 5) == []

    monkeypatch.setattr(local, "identity", "other-model")    # a second model (no download in tests)
    monkeypatch.setattr(VectorSearch, "_connect", lambda self: None)
    other = VectorSearch.__new__(VectorSearch)
    other.chroma_path, other.enabled, other.identity = chroma, True, "other-model"
    assert other._collection_name("other-model") == "context-other-model"
    assert other._collection_name("default") == "context"    # switching back reuses the old one
    # Rows embedded with "default" are pending for "other-model" only.
    assert [r["meeting_id"] for r in db.transcripts_to_index(time.time() + 5, "other-model")] == [mid]
    assert db.transcripts_to_index(time.time() + 5, "default") == []


def test_contorch_memory_embeddings_writes_env_file(tmp_path, monkeypatch):
    from context_orchestrator import settings
    env = tmp_path / "env"
    env.write_text("# my settings\nCO_RERANK_MODEL=gemini-flash-latest\nCO_EMBEDDING_MODEL=gemini-embedding-001\n")
    monkeypatch.delenv("CO_EMBEDDING_MODEL", raising=False)
    msg = settings.set_embeddings("none", env_file=env)
    assert "keyword" in msg
    text = env.read_text()
    assert "CO_EMBEDDING_MODEL=none" in text and "CO_RERANK_MODEL=gemini-flash-latest" in text
    assert text.count("CO_EMBEDDING_MODEL") == 1 and text.startswith("# my settings")
    with pytest.raises(ValueError):
        settings.set_embeddings("bogus", env_file=env)



def test_failed_backup_deletes_nothing(db, tmp_path, monkeypatch):
    d = tmp_path / "transcripts"; d.mkdir()
    import os
    for i in range(3):
        f = d / f"meeting-2026-05-0{i+1}T10-00-00.md"
        f.write_text(f"[10:00:01] **Me:** meeting number {i}")
        os.utime(f, (time.time() - 600, time.time() - 600))
    real = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda self, name: b"corrupted")   # verify fails
    with pytest.raises(T.BackupError):
        T.import_path(db, d, delete=True, backup_dir=tmp_path / "backups")
    monkeypatch.setattr(zipfile.ZipFile, "read", real)
    assert len(list(d.glob("*.md"))) == 3, "no file may be deleted when the backup fails"
    assert list((tmp_path / "backups").glob("*.zip")) == []          # no half-written backup left
    assert db.count_transcripts()[0] == 3                              # the import itself happened


def test_cli_rm_keeps_the_text_in_a_backup(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CO_DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setenv("CO_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(T, "BACKUP_DIR", tmp_path / "backups")
    f = tmp_path / "ep.txt"; f.write_text("[14:00:01] Host: the secret number is 42")
    T.main(["add", str(f), "--title", "Ep", "--started-at", "2026-09-28T14:00"])
    capsys.readouterr()
    assert T.main(["rm", "2026-09-28-1400-ep"]) == 0
    assert "kept in" in capsys.readouterr().out
    with zipfile.ZipFile(tmp_path / "backups" / "deleted-transcripts.zip") as z:
        (name,) = z.namelist()
        assert name.startswith("2026-09-28-1400-ep-deleted-")
        assert b"the secret number is 42" in z.read(name)
    assert T.main(["show", "2026-09-28-1400-ep"]) == 1
