import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from context_orchestrator import watcher
from context_orchestrator.search import VectorSearch


@pytest.fixture
def vs():
    return VectorSearch(chroma_path=Path(tempfile.mkdtemp()))


@pytest.fixture
def db(tmp_path):
    from context_orchestrator.db import Database
    return Database(db_path=tmp_path / "context.db")


@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    return d


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "state.json"


def _write_old(path: Path, text: str, age_seconds: float = 60.0) -> None:
    path.write_text(text, encoding="utf-8")
    past = time.time() - age_seconds
    os.utime(path, (past, past))


M1 = "meeting-2026-09-01T10-00-00"


def test_scan_imports_file_into_db_and_indexes(vs, db, watch_dir, state_file):
    f = watch_dir / f"{M1}.md"
    _write_old(f, "# Meeting\n\n[10:00:05] **Them:** Discussion of the new auth flow with the security team.\n")

    state = watcher.load_state(state_file)
    indexed = watcher.scan_once(vs, watch_dir, state, db=db)

    assert indexed == [M1]
    assert "auth flow" in db.get_transcript(M1)["body"]
    assert vs.count() == 1
    assert str(f) in state
    assert f.exists(), "the watcher imports; only `import --delete` removes files"


def test_scan_skips_unchanged_file(vs, db, watch_dir, state_file):
    _write_old(watch_dir / f"{M1}.md", "[10:00:01] First pass content here, long enough.")
    state = watcher.load_state(state_file)
    watcher.scan_once(vs, watch_dir, state, db=db)
    count_after_first = vs.count()
    assert watcher.scan_once(vs, watch_dir, state, db=db) == []
    assert vs.count() == count_after_first


def test_scan_reindexes_modified_file_and_drops_old_chunks(vs, db, watch_dir, state_file):
    f = watch_dir / f"{M1}.md"
    _write_old(f, " ".join(f"token{i}" for i in range(1200)), age_seconds=120.0)
    state = watcher.load_state(state_file)
    watcher.scan_once(vs, watch_dir, state, db=db)
    assert vs.count() >= 3

    _write_old(f, "Sprint planning notes covering the database migration timeline.", age_seconds=61.0)
    assert watcher.scan_once(vs, watch_dir, state, db=db) == [M1]
    assert vs.count() == 1, "old chunks must be dropped"


def test_scan_skips_recently_modified_file(vs, db, watch_dir, state_file):
    (watch_dir / f"{M1}.md").write_text("just written, still being appended to")
    state = watcher.load_state(state_file)
    assert watcher.scan_once(vs, watch_dir, state, settle_seconds=10.0, db=db) == []
    assert vs.count() == 0
    assert db.get_transcript(M1) is None


def test_scan_indexes_rows_written_straight_to_the_db(vs, db, watch_dir, state_file):
    """meeting-capture appends to the table; no file ever exists."""
    db.put_transcript(M1, "[10:00:00] **Me:** we agreed to raise prices in March for all tiers",
                      now=time.time() - 120)
    assert watcher.scan_once(vs, watch_dir, {}, db=db) == [M1]
    hit = vs.collection.get(where={"meeting_id": M1}, include=["metadatas"])["metadatas"][0]
    assert hit["type"] == "transcript" and hit["chunk_type"] == "speech"
    assert watcher.scan_once(vs, watch_dir, {}, db=db) == []


def test_scan_waits_for_a_live_meeting_to_settle(vs, db, watch_dir):
    db.put_transcript(M1, "[10:00:00] **Me:** still talking about the roadmap here")
    assert watcher.scan_once(vs, watch_dir, {}, settle_seconds=60.0, db=db) == []


def test_state_roundtrip(state_file):
    state = {"a.md": 1.0, "b.md": 2.5}
    watcher.save_state(state, state_file)
    loaded = watcher.load_state(state_file)
    assert loaded == state


def test_load_state_returns_empty_when_missing(tmp_path):
    missing = tmp_path / "nope.json"
    assert watcher.load_state(missing) == {}


def test_load_state_returns_empty_on_corrupt_json(state_file):
    state_file.write_text("not json at all {{")
    assert watcher.load_state(state_file) == {}


def test_plist_payload_round_trips():
    import plistlib
    payload = watcher._plist_payload("/usr/bin/python3")
    parsed = plistlib.loads(payload)
    assert parsed["Label"] == watcher.LAUNCHD_LABEL
    assert parsed["ProgramArguments"][0] == "/usr/bin/python3"
    assert parsed["ProgramArguments"][-2:] == ["context_orchestrator.watcher", "run"]
    assert parsed["RunAtLoad"] is True


def test_main_no_subcommand_defaults_to_run(monkeypatch):
    called = {}
    def fake_loop(watch_dir, interval):
        called["watch_dir"] = watch_dir
        called["interval"] = interval
        raise KeyboardInterrupt
    monkeypatch.setattr(watcher, "watch_loop", fake_loop)
    try:
        watcher.main([])
    except KeyboardInterrupt:
        pass
    assert called["watch_dir"] == watcher.TRANSCRIPT_DIR
    assert called["interval"] == watcher.DEFAULT_INTERVAL


# ---- on-demand indexing (catch_up) — the default, daemon-free path

def test_catch_up_indexes_and_persists_state(vs, db, watch_dir, state_file, tmp_path):
    _write_old(watch_dir / "meeting-a.md", "# Meeting\n**Them:** ship the retry fix behind a flag")
    lock = tmp_path / "index.lock"
    indexed = watcher.catch_up(vs, watch_dir, state_file, lock_file=lock, db=db)
    assert len(indexed) == 1 and indexed[0].endswith("-meeting-a")
    assert str(watch_dir / "meeting-a.md") in json.loads(state_file.read_text())
    # Second pass: nothing new.
    assert watcher.catch_up(vs, watch_dir, state_file, lock_file=lock, db=db) == []


def test_catch_up_returns_none_while_another_process_indexes(vs, db, watch_dir, state_file, tmp_path):
    import fcntl
    _write_old(watch_dir / "meeting-b.md", "# Meeting\n**Me:** hello")
    lock = tmp_path / "index.lock"
    with open(lock, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)          # "another MCP server" is indexing
        assert watcher.catch_up(vs, watch_dir, state_file, lock_file=lock, db=db) is None
    # Lock released → this process picks it up.
    got = watcher.catch_up(vs, watch_dir, state_file, lock_file=lock, db=db)
    assert len(got) == 1 and got[0].endswith("-meeting-b")


def test_catch_up_rereads_state_written_by_another_process(vs, db, watch_dir, state_file, tmp_path):
    f = watch_dir / "meeting-c.md"
    _write_old(f, "# Meeting\n**Them:** already handled elsewhere")
    watcher.save_state({str(f): f.stat().st_mtime}, state_file)   # someone else indexed it
    assert watcher.catch_up(vs, watch_dir, state_file, lock_file=tmp_path / "l", db=db) == []


def test_catch_up_skips_a_meeting_still_being_written(vs, db, watch_dir, state_file, tmp_path):
    (watch_dir / "meeting-live.md").write_text("# Meeting\n**Me:** still talking")
    assert watcher.catch_up(vs, watch_dir, state_file, lock_file=tmp_path / "l", db=db) == []


def test_save_state_is_atomic_and_leaves_no_tmp(state_file):
    watcher.save_state({"a": 1.0}, state_file)
    assert json.loads(state_file.read_text()) == {"a": 1.0}
    assert list(state_file.parent.glob("*.tmp")) == []
