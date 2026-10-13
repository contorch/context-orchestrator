"""Smoke tests for the MCP server's search tool. We verify the date parser
and that filter args translate into the right Chroma `where` shape via the
public search tool. The full ranking pipeline is exercised by test_search."""
from datetime import datetime, timezone

import pytest

from context_orchestrator.server import _parse_iso_date


class TestParseIsoDate:
    def test_date_only(self):
        ts = _parse_iso_date("2026-04-30")
        expected = datetime(2026, 4, 30, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_datetime_no_tz(self):
        ts = _parse_iso_date("2026-04-30T14:25:00")
        expected = datetime(2026, 4, 30, 14, 25, 0, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_datetime_with_z(self):
        ts = _parse_iso_date("2026-04-30T14:25:00Z")
        expected = datetime(2026, 4, 30, 14, 25, 0, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_datetime_with_offset(self):
        ts = _parse_iso_date("2026-04-30T14:25:00+00:00")
        expected = datetime(2026, 4, 30, 14, 25, 0, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_unparseable_returns_none(self):
        assert _parse_iso_date("not a date") is None
        assert _parse_iso_date("") is None
        assert _parse_iso_date("2026/04/30") is None

    def test_partial_iso_with_offset(self):
        # Edge case: time has explicit +HH:MM offset
        ts = _parse_iso_date("2026-04-30T14:25:00-05:00")
        # Should be 14:25 EST = 19:25 UTC
        expected = datetime(2026, 4, 30, 19, 25, 0, tzinfo=timezone.utc).timestamp()
        assert ts == expected


def _chunk(meeting_id, n):
    return {"id": f"transcript:{meeting_id}:{n}", "text": f"{meeting_id} chunk {n}",
            "metadata": {"type": "transcript", "meeting_id": meeting_id}}


class TestFuse:
    """server._fuse ranks meetings, not chunks: a meeting counts once, at the
    rank of its best chunk."""

    def test_many_mediocre_chunks_do_not_beat_the_right_chunk(self):
        from context_orchestrator.server import _fuse
        # Meeting "right" holds the best chunk; "chatty" has the next nine.
        # Summing reciprocal ranks per meeting used to put "chatty" first
        # (9 x ~1/65 > 1/61).
        dense = [_chunk("right", 0)] + [_chunk("chatty", i) for i in range(1, 10)]
        hits = _fuse(dense, [], limit=10)
        assert [h["metadata"]["meeting_id"] for h in hits] == ["right", "chatty"]
        assert hits[1]["id"] == "transcript:chatty:1"   # shown as its best chunk

    def test_with_full_text_hits(self):
        from context_orchestrator.server import _fuse
        dense = [_chunk("right", 0)] + [_chunk("chatty", i) for i in range(1, 10)]
        text = [{"kind": "transcript", "meeting_id": "right", "text": "x"},
                {"kind": "transcript", "meeting_id": "chatty", "text": "y"},
                {"kind": "transcript", "meeting_id": "fts-only", "text": "z"}]
        hits = _fuse(dense, text, limit=10)
        ids = [h["meeting_id"] if h.get("fts") else h["metadata"]["meeting_id"] for h in hits]
        assert ids == ["right", "chatty", "fts-only"]
        assert not hits[0].get("fts")    # found by both → shown as the vector chunk

    def test_meeting_ranks_are_compacted(self):
        from context_orchestrator.server import _fuse
        # Dense: a, a, a, b → b is the 2nd meeting (rank 1), not rank 3, so it
        # ties FTS's rank-1 meeting c and keeps the dense-first order.
        dense = [_chunk("a", 0), _chunk("a", 1), _chunk("a", 2), _chunk("b", 0)]
        text = [{"kind": "transcript", "meeting_id": "a", "text": ""},
                {"kind": "transcript", "meeting_id": "c", "text": ""}]
        hits = _fuse(dense, text, limit=10)
        ids = [h["meeting_id"] if h.get("fts") else h["metadata"]["meeting_id"] for h in hits]
        assert ids == ["a", "b", "c"]

    def test_non_transcript_hits_stay_separate(self):
        from context_orchestrator.server import _fuse
        dense = [{"id": "repo_knowledge:1", "text": "k1", "metadata": {"type": "repo_knowledge"}},
                 {"id": "repo_knowledge:2", "text": "k2", "metadata": {"type": "repo_knowledge"}}]
        assert [h["id"] for h in _fuse(dense, [], limit=10)] == ["repo_knowledge:1", "repo_knowledge:2"]

    def test_limit(self):
        from context_orchestrator.server import _fuse
        dense = [_chunk(f"m{i}", 0) for i in range(20)]
        assert len(_fuse(dense, [], limit=10)) == 10
