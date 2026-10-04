"""Transcripts live in SQLite; Chroma is the search index built from them.

There are no transcript files. The full text of every meeting / podcast is a
row in the `transcripts` table of context.db (see db.py). meeting-capture
appends to that row while a meeting runs; `add_transcript` (MCP) and
`contorch-transcripts import` add finished ones. Indexing reads the row,
chunks it exactly as the old .md files were chunked, and upserts the chunks.

Because the text is stored before anything is embedded, a failing embedding
call (bad key, quota, outage) loses nothing: the row stays "pending" and is
indexed on a later pass.

Embedding bundles: `contorch-transcripts embed` chunks + embeds transcripts on
any machine with a Gemini key and writes a JSONL bundle; `import` on this
machine loads the vectors straight into Chroma without calling the API. The
bundle records the embedding function's name and dimension, and import
refuses vectors that don't match this collection (mixed models = garbage
search results).

    contorch-transcripts add     <file|-> --title T --started-at ISO --source URL
    contorch-transcripts import  <file|dir|zip|bundle.jsonl> [--delete]
    contorch-transcripts export  pending.jsonl --pending        (keyless machine)
    contorch-transcripts embed   <file|dir|zip|pending.jsonl> -o bundle.jsonl
    contorch-transcripts list | show ID | rm ID | reindex [ID]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .chunking import chunk_transcript, is_hallucination
from .corrections import load_corrections

log = logging.getLogger("context-orchestrator.transcripts")

# A meeting still being appended to is left alone until it has been quiet this
# long — re-embedding on every appended line is O(n²) in meeting length.
SETTLE_SECONDS = 60.0
BUNDLE_FORMAT = "contorch-transcript-bundle"
BUNDLE_VERSION = 1
EMBED_BATCH = 100  # Gemini batchEmbedContents limit
TEXT_SUFFIXES = {".md", ".txt", ".vtt", ".srt"}

_MEETING_ID_RE = re.compile(r"^meeting-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$")
_DATED_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}-")
_CUE_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*[\d:.,]+"
)
_VOICE_RE = re.compile(r"<v\s+([^>]+)>")
_TAG_RE = re.compile(r"<[^>]+>")


# ---- ids, hashing, normalisation ------------------------------------------

def content_sha(body: str) -> str:
    return hashlib.sha256(body.strip().encode("utf-8")).hexdigest()


def slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "transcript"


def make_meeting_id(title: str, started_at: dt.datetime) -> str:
    """`YYYY-MM-DD-HHMM-<slug>` — the dated form the chunker already parses,
    so timestamped lines get time-window metadata."""
    return f"{started_at:%Y-%m-%d-%H%M}-{slugify(title)}"


def _id_time(meeting_id: str) -> Optional[dt.datetime]:
    """Start time encoded in a meeting-capture id (local clock)."""
    if _MEETING_ID_RE.match(meeting_id):
        return dt.datetime.strptime(meeting_id[len("meeting-"):], "%Y-%m-%dT%H-%M-%S")
    return None


def is_dated_id(meeting_id: str) -> bool:
    return bool(_MEETING_ID_RE.match(meeting_id) or _DATED_ID_RE.match(meeting_id))


def parse_started_at(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def looks_like_captions(text: str) -> bool:
    head = text.lstrip()[:2000]
    return head.startswith("WEBVTT") or bool(re.search(r"^\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*-->", head, re.M))


def captions_to_lines(text: str, started_at: dt.datetime) -> str:
    """WebVTT / SRT → `[HH:MM:SS] Speaker: text` lines on the wall clock
    (start + cue offset), the format the chunker timestamps."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _CUE_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        h, mi, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
        wall = started_at + dt.timedelta(hours=h, minutes=mi, seconds=s)
        i += 1
        parts: list[str] = []
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            v = _VOICE_RE.search(line)
            line = _TAG_RE.sub("", line).strip()
            if v and line:
                line = f"{v.group(1).strip()}: {line}"
            if line:
                parts.append(line)
            i += 1
        if parts:
            out.append(f"[{wall:%H:%M:%S}] {' '.join(parts)}")
    return "\n\n".join(out) + "\n"


def normalise(text: str, started_at: dt.datetime) -> str:
    return captions_to_lines(text, started_at) if looks_like_captions(text) else text


# ---- chunking + indexing ----------------------------------------------------

def chunks_for(row: dict) -> list[tuple[str, str, dict]]:
    """(doc_id, text, metadata) for one transcript row. Ids and metadata match
    what file-based indexing produced, so re-indexing a migrated meeting
    overwrites its old chunks in place."""
    mid = row["meeting_id"]
    pairs = chunk_transcript(row["body"], f"{mid}.md", corrections=load_corrections())
    out = []
    for i, (text, meta) in enumerate(pairs):
        if is_hallucination(text)[0]:
            continue
        out.append((f"transcript:{mid}.md:{i}", text, {
            "type": "transcript",
            "meeting_id": mid,
            "title": row.get("title") or "",
            "source": row.get("source") or "",
            "chunk_index": i,
            "total_chunks": len(pairs),
            **meta,
        }))
    return out


def _replace_chunks(vs, meeting_id: str, ids: list[str], docs: list[str], metas: list[dict],
                    embeddings: Optional[list] = None) -> None:
    """Upsert the new chunks first, then drop leftovers. If embedding fails
    (no key, quota) nothing is written and the meeting's previous chunks stay
    searchable. Vectors are computed before the Chroma session lock is taken;
    the session only writes."""
    if not vs.enabled:
        return
    if ids and embeddings is None:
        embeddings = vs.embed_documents(docs)
    with vs.session(write=True) as col:
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        existing = col.get(where={"meeting_id": meeting_id}, include=[])["ids"]
        stale = sorted(set(existing) - set(ids))
        if stale:
            col.delete(ids=stale)
    vs.invalidate_bm25()


def index_row(vs, db, row: dict) -> int:
    """Embed one transcript into the current model's collection. With
    embeddings off there is nothing to embed (full-text search already has
    it); the row is just marked as done for "none"."""
    if not vs.enabled:
        db.mark_transcript_indexed(row["meeting_id"], row["updated_at"], "none")
        return 0
    chunks = chunks_for(row)
    _replace_chunks(vs, row["meeting_id"], [c[0] for c in chunks], [c[1] for c in chunks],
                    [c[2] for c in chunks])
    db.mark_transcript_indexed(row["meeting_id"], row["updated_at"], vs.identity)
    return len(chunks)


def index_pending(vs, db, settle_seconds: float = SETTLE_SECONDS,
                  now: Optional[float] = None) -> list[str]:
    """Index every transcript that changed and has settled. Stops after three
    failures in a row — that's a dead key or network, not a bad transcript."""
    now = time.time() if now is None else now
    done: list[str] = []
    failures = 0
    for row in db.transcripts_to_index(now - settle_seconds, vs.identity):
        try:
            index_row(vs, db, row)
        except Exception as exc:
            log.warning("indexing %s failed (kept in the database, retried later): %s",
                        row["meeting_id"], str(exc)[:200])
            failures += 1
            if failures >= 3:
                break
            continue
        failures = 0
        done.append(row["meeting_id"])
    return done


def remove_from_index(vs, meeting_id: str) -> None:
    if not vs.enabled:
        return
    with vs.session(write=True) as col:
        ids = col.get(where={"meeting_id": meeting_id}, include=[])["ids"]
        if ids:
            col.delete(ids=ids)
    vs.invalidate_bm25()


# ---- adding transcripts -----------------------------------------------------

def add_text(db, text: str, title: str = "", started_at: str = "", source: str = "",
             meeting_id: str = "", fallback_time: Optional[dt.datetime] = None,
             updated_at: Optional[float] = None) -> tuple[str, bool]:
    """Store one transcript. Returns (meeting_id, created). An identical body
    already stored (under any id) is not stored twice."""
    started = parse_started_at(started_at) or fallback_time or dt.datetime.now()
    body = normalise(text, started)
    if not body.strip():
        raise ValueError("transcript is empty")
    # Hash what was given, not the normalised body: the same caption file
    # added with a different start time is still the same transcript.
    sha = content_sha(text)
    dup = db.find_transcript_by_sha(sha)
    if dup:
        return dup["meeting_id"], False
    mid = meeting_id or make_meeting_id(title or "transcript", started)
    if not meeting_id:
        base, n = mid, 2
        while db.get_transcript(mid):  # same minute + title, different content
            mid, n = f"{base}-{n}", n + 1
    db.put_transcript(mid, body, title=title, source=source,
                      started_at=started.isoformat(timespec="seconds"), content_sha=sha,
                      now=updated_at)
    return mid, True


def _meeting_id_for_file(name: str, mtime: dt.datetime) -> tuple[str, str]:
    """(meeting_id, title) for an imported file. meeting-capture / dated names
    keep their stem so existing index entries line up."""
    stem = Path(name).stem
    if is_dated_id(stem):
        return stem, stem
    return make_meeting_id(stem, mtime), stem


def add_file_text(db, name: str, text: str, mtime: dt.datetime, source: str) -> tuple[str, bool]:
    """A file's modification time is when its transcript last changed, so a
    finished meeting imported now is indexable at once rather than after the
    live-meeting settle window."""
    changed_at = mtime.timestamp()
    mid, title = _meeting_id_for_file(name, mtime)
    if is_dated_id(Path(name).stem):
        # Known id: replace (a newer copy of the same meeting wins).
        started = _id_time(mid) or mtime
        body = normalise(text, started)
        changed = db.put_transcript(mid, body, title=title, source=source,
                                    started_at=started.isoformat(timespec="seconds"),
                                    content_sha=content_sha(body), now=changed_at)
        return mid, changed
    return add_text(db, text, title=title, source=source, meeting_id="",
                    fallback_time=mtime, updated_at=changed_at)


def iter_text_files(path: Path) -> Iterator[tuple[str, str, dt.datetime, Optional[Path]]]:
    """(name, text, mtime, on-disk path or None) for every transcript-like file
    in a file, directory, or zip. Zip members are read in memory, never
    extracted."""
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES and not f.name.startswith("."):
                yield f.name, f.read_text(encoding="utf-8", errors="replace"), \
                    dt.datetime.fromtimestamp(f.stat().st_mtime), f
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                name = Path(info.filename).name
                if info.is_dir() or name.startswith(".") or "__MACOSX" in info.filename:
                    continue
                if Path(name).suffix.lower() not in TEXT_SUFFIXES:
                    continue
                text = z.read(info).decode("utf-8", errors="replace")
                yield name, text, dt.datetime(*info.date_time), None
    elif path.is_file():
        yield path.name, path.read_text(encoding="utf-8", errors="replace"), \
            dt.datetime.fromtimestamp(path.stat().st_mtime), path


# ---- embedding bundles ------------------------------------------------------

def _ef_identity(ef) -> str:
    from .search import ef_identity
    return ef_identity(ef)


def write_bundle(rows: Iterable[dict], out: io.TextIOBase, ef) -> int:
    """Chunk + embed rows with `ef` and write a JSONL bundle. Returns the
    number of transcripts written."""
    header_written = False
    n = 0
    for row in rows:
        chunks = chunks_for(row)
        vectors: list[list[float]] = []
        texts = [c[1] for c in chunks]
        for i in range(0, len(texts), EMBED_BATCH):
            vectors.extend([float(x) for x in v] for v in ef.embed_documents(texts[i:i + EMBED_BATCH]))
        if not header_written:
            dim = len(vectors[0]) if vectors else 0
            out.write(json.dumps({"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
                                  "embedding_function": _ef_identity(ef), "dim": dim}) + "\n")
            header_written = True
        out.write(json.dumps({
            "meeting_id": row["meeting_id"], "title": row.get("title", ""),
            "source": row.get("source", ""), "started_at": row.get("started_at", ""),
            "body": row["body"],
            "chunks": [{"id": c[0], "text": c[1], "metadata": c[2], "embedding": v}
                       for c, v in zip(chunks, vectors)],
        }) + "\n")
        n += 1
    return n


def write_text_bundle(rows: Iterable[dict], out: io.TextIOBase) -> int:
    """Transcripts without vectors — to carry to a machine that can embed
    them (`embed` accepts this file) and bring back as a full bundle."""
    out.write(json.dumps({"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
                          "embedding_function": None, "dim": 0}) + "\n")
    n = 0
    for row in rows:
        out.write(json.dumps({k: row.get(k, "") for k in
                              ("meeting_id", "title", "source", "started_at", "body")}) + "\n")
        n += 1
    return n


def read_bundle_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.strip():
                yield json.loads(line)


def is_bundle(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in (".jsonl", ".json"):
        return False
    with path.open(encoding="utf-8") as f:
        try:
            return json.loads(f.readline()).get("format") == BUNDLE_FORMAT
        except (ValueError, AttributeError):
            return False


def _collection_dim(vs) -> Optional[int]:
    if not vs.enabled:
        return None
    with vs.session() as col:
        if col.count() == 0:
            return None
        got = col.get(limit=1, include=["embeddings"]).get("embeddings")
    return len(got[0]) if got is not None and len(got) else None


def import_bundle(vs, db, path: Path) -> dict:
    """Store each transcript's text, and load its vectors when they were made
    by the same embedding function (and dimension) as this collection.
    Otherwise the text is stored and left pending for normal indexing."""
    stats = {"stored": 0, "unchanged": 0, "vectors_loaded": 0, "pending": 0}
    with path.open(encoding="utf-8") as f:
        header = json.loads(f.readline())
        local_ef = _ef_identity(vs.embedding_function) if vs.enabled else "none"
        local_dim = _collection_dim(vs)
        text_only = header.get("embedding_function") is None
        compatible = not text_only and header.get("embedding_function") == local_ef and (
            local_dim is None or header.get("dim") == local_dim)
        stats["text_only"] = text_only
        if not compatible and not text_only:
            log.warning("bundle vectors are %s/%sd but this index uses %s/%sd — storing the "
                        "text only; it will be embedded here", header.get("embedding_function"),
                        header.get("dim"), local_ef, local_dim)
        stats["compatible"] = compatible
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mid = rec["meeting_id"]
            changed = db.put_transcript(mid, rec["body"], title=rec.get("title", ""),
                                        source=rec.get("source", ""),
                                        started_at=rec.get("started_at", ""),
                                        content_sha=content_sha(rec["body"]))
            stats["stored" if changed else "unchanged"] += 1
            row = db.get_transcript(mid)
            if (row["indexed_at"] is not None and row["indexed_at"] >= row["updated_at"]
                    and row.get("indexed_with") == vs.identity):
                continue  # already indexed from this exact text, with this model
            if compatible and rec.get("chunks") is not None:
                ch = rec["chunks"]
                _replace_chunks(vs, mid, [c["id"] for c in ch], [c["text"] for c in ch],
                                [c["metadata"] for c in ch], [c["embedding"] for c in ch])
                db.mark_transcript_indexed(mid, row["updated_at"], vs.identity)
                stats["vectors_loaded"] += 1
            else:
                stats["pending"] += 1
    return stats


# ---- CLI --------------------------------------------------------------------

def _open_db():
    from .db import Database
    import os
    p = os.environ.get("CO_DB_PATH")
    return Database(db_path=Path(p) if p else None)


def _open_vs():
    from .search import VectorSearch
    import os
    p = os.environ.get("CO_CHROMA_PATH")
    return VectorSearch(chroma_path=Path(p) if p else None)


BACKUP_DIR = Path.home() / ".context-orchestrator" / "backups"


class BackupError(RuntimeError):
    """The backup zip could not be written or didn't verify — nothing deleted."""


def backup_files(files: list[Path], root: Path, backup_dir: Optional[Path] = None,
                 label: str = "transcripts") -> Path:
    """Zip `files` (paths kept relative to `root`, with modification times),
    then re-open the archive and compare every member byte-for-byte with the
    file on disk. Returns the zip path; raises BackupError on any mismatch, so
    callers delete only after this returns."""
    backup_dir = backup_dir or BACKUP_DIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = backup_dir / f"{label}-{stamp}.zip"
    n = 2
    while dest.exists():
        dest = backup_dir / f"{label}-{stamp}-{n}.zip"
        n += 1
    tmp = dest.with_suffix(".zip.partial")
    names: dict[Path, str] = {}
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in files:
                try:
                    arc = str(f.relative_to(root)) if root.is_dir() else f.name
                except ValueError:
                    arc = f.name
                if arc in names.values():
                    arc = f"{len(names)}-{arc}"
                z.write(f, arc)
                names[f] = arc
        with zipfile.ZipFile(tmp) as z:
            bad = z.testzip()
            if bad is not None:
                raise BackupError(f"backup zip is corrupt at {bad}")
            for f, arc in names.items():
                if z.read(arc) != f.read_bytes():
                    raise BackupError(f"backup of {f} does not match the file")
        tmp.chmod(0o600)   # transcripts are private
        os.replace(tmp, dest)
    except BackupError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise BackupError(f"could not write backup {dest}: {exc}") from exc
    return dest


def import_path(db, path: Path, delete: bool = False,
                settle_seconds: float = SETTLE_SECONDS, backup_dir: Optional[Path] = None) -> dict:
    """Import a file, directory or zip into the database. With delete=True,
    the files whose text is verifiably stored are first zipped into
    `backup_dir` (verified byte-for-byte) and only then removed; if the backup
    fails nothing is deleted. Files touched in the last `settle_seconds` (a
    meeting still being written by an older meeting-capture) are skipped."""
    stats = {"stored": 0, "unchanged": 0, "deleted": 0, "skipped_live": 0, "backup": ""}
    now = time.time()
    to_delete: list[Path] = []
    for name, text, mtime, disk in iter_text_files(path):
        if disk is not None and now - disk.stat().st_mtime < settle_seconds:
            stats["skipped_live"] += 1
            continue
        mid, changed = add_file_text(db, name, text, mtime, source=f"import:{path.name}")
        stats["stored" if changed else "unchanged"] += 1
        if delete and disk is not None:
            row = db.get_transcript(mid)
            if row and row["body"].strip():
                to_delete.append(disk)
    if to_delete:
        stats["backup"] = str(backup_files(to_delete, path, backup_dir))
        for f in to_delete:
            f.unlink()
            stats["deleted"] += 1
    return stats


def _cmd_import(args) -> int:
    db = _open_db()
    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1
    if is_bundle(path):
        stats = import_bundle(_open_vs(), db, path)
        print(f"bundle: {stats['stored']} stored, {stats['unchanged']} unchanged, "
              f"{stats['vectors_loaded']} loaded with their vectors, {stats['pending']} left to embed here")
        if not stats["compatible"] and not stats["text_only"]:
            print("  vectors skipped: they were made with a different embedding model or size", file=sys.stderr)
        return 0
    try:
        stats = import_path(db, path, delete=args.delete)
    except BackupError as exc:
        print(f"nothing deleted: {exc}", file=sys.stderr)
        return 1
    print(f"{stats['stored']} stored, {stats['unchanged']} already there"
          + (f", {stats['deleted']} file(s) deleted" if args.delete else "")
          + (f"\nbackup of the deleted files: {stats['backup']}" if stats["backup"] else "")
          + (f", {stats['skipped_live']} skipped (modified in the last minute)" if stats["skipped_live"] else ""))
    if not args.no_index:
        vs = _open_vs()
        done = index_pending(vs, db, settle_seconds=0)
        total, pending = db.count_transcripts()
        print(f"indexed {len(done)}; {pending} of {total} transcript(s) still waiting to be embedded")
    return 0


def _cmd_add(args) -> int:
    """One transcript with a title / start time, from a file or stdin (-)."""
    text = sys.stdin.read() if args.path == "-" else \
        Path(args.path).expanduser().read_text(encoding="utf-8", errors="replace")
    db = _open_db()
    try:
        mid, created = add_text(db, text, title=args.title, started_at=args.started_at,
                                source=args.source)
    except ValueError as e:
        print(f"not stored: {e}", file=sys.stderr)
        return 1
    if not created:
        print(f"already stored as {mid} (identical text)")
        return 0
    row = db.get_transcript(mid)
    try:
        n = index_row(_open_vs(), db, row)
        status = f"indexed ({n} chunks)"
    except Exception as e:
        status = f"stored; indexing pending ({str(e)[:120]})"
    print(f"stored {mid} ({len(row['body'].split())} words) — {status}")
    return 0


def _cmd_embed(args) -> int:
    """Chunk + embed files into a bundle without touching any database."""
    from .search import _build_embedding_function
    ef = _build_embedding_function()
    if ef is None:
        print("no embedding model configured: set a Gemini key (GOOGLE_API_KEY or "
              "~/.config/google/key) or CO_EMBEDDING_MODEL", file=sys.stderr)
        return 1
    path = Path(args.path).expanduser()

    def rows():
        if is_bundle(path):
            yield from read_bundle_rows(path)
            return
        seen = set()
        for name, text, mtime, _disk in iter_text_files(path):
            mid, title = _meeting_id_for_file(name, mtime)
            body = normalise(text, mtime)
            if not body.strip() or mid in seen:
                continue
            seen.add(mid)
            yield {"meeting_id": mid, "title": title, "source": f"import:{path.name}",
                   "started_at": mtime.isoformat(timespec="seconds"), "body": body}

    with open(args.output, "w", encoding="utf-8") as out:
        n = write_bundle(rows(), out, ef)
    print(f"wrote {n} transcript(s) with embeddings ({_ef_identity(ef)}) to {args.output}")
    return 0


def _cmd_export(args) -> int:
    db = _open_db()
    if args.meeting_ids:
        rows = [db.get_transcript(m) for m in args.meeting_ids]
        missing = [m for m, r in zip(args.meeting_ids, rows) if r is None]
        if missing:
            print(f"no transcript {', '.join(missing)}", file=sys.stderr)
            return 1
    elif args.pending:
        rows = db.transcripts_to_index(time.time())
    else:
        rows = [db.get_transcript(r["meeting_id"]) for r in db.list_transcripts(1_000_000)]
    with open(args.output, "w", encoding="utf-8") as out:
        n = write_text_bundle(rows, out)
    print(f"wrote {n} transcript(s) (text only) to {args.output} — on a machine with a Gemini key run "
          f"`contorch-transcripts embed {Path(args.output).name} -o bundle.jsonl`, then import bundle.jsonl here")
    return 0


def _cmd_list(args) -> int:
    for r in _open_db().list_transcripts(args.limit):
        state = "indexed" if r["indexed_at"] and r["indexed_at"] >= r["updated_at"] else "pending"
        print(f"{r['meeting_id']}\t{r['chars']:>7} chars\t{state}\t{r['title']}")
    return 0


def _cmd_show(args) -> int:
    row = _open_db().get_transcript(args.meeting_id)
    if not row:
        print(f"no transcript {args.meeting_id}", file=sys.stderr)
        return 1
    sys.stdout.write(row["body"])
    return 0


def backup_transcript(row: dict, backup_dir: Optional[Path] = None) -> Path:
    """Append a transcript's full text (plus its metadata) to
    backups/deleted-transcripts.zip before it is deleted; verified by reading
    it back."""
    backup_dir = backup_dir or BACKUP_DIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / "deleted-transcripts.zip"
    arc = f"{row['meeting_id']}-deleted-{time.strftime('%Y%m%d-%H%M%S')}.md"
    meta = {k: row.get(k, "") for k in ("meeting_id", "title", "source", "started_at")}
    data = f"<!-- {json.dumps(meta)} -->\n{row['body']}".encode("utf-8")
    with zipfile.ZipFile(dest, "a", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(arc, data)
    with zipfile.ZipFile(dest) as z:
        if z.read(arc) != data:
            raise BackupError(f"backup of {row['meeting_id']} did not verify")
    dest.chmod(0o600)
    return dest


def _cmd_rm(args) -> int:
    db = _open_db()
    row = db.get_transcript(args.meeting_id)
    if not row:
        print(f"no transcript {args.meeting_id}", file=sys.stderr)
        return 1
    try:
        dest = backup_transcript(row)
    except (BackupError, OSError) as exc:
        print(f"not deleted — backup failed: {exc}", file=sys.stderr)
        return 1
    db.delete_transcript(args.meeting_id)
    remove_from_index(_open_vs(), args.meeting_id)
    print(f"deleted {args.meeting_id} (text kept in {dest})")
    return 0


def _cmd_reindex(args) -> int:
    db, vs = _open_db(), _open_vs()
    if args.meeting_id:
        row = db.get_transcript(args.meeting_id)
        if not row:
            print(f"no transcript {args.meeting_id}", file=sys.stderr)
            return 1
        print(f"{index_row(vs, db, row)} chunk(s) indexed")
        return 0
    done = index_pending(vs, db, settle_seconds=0)
    total, pending = db.count_transcripts()
    print(f"indexed {len(done)}; {pending} of {total} still pending")
    return 0 if pending == 0 else 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="contorch-transcripts",
                                description="Transcripts stored in the context-orchestrator database.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("import", help="add .md/.txt/.vtt/.srt files, a folder, a zip, or an embedding bundle")
    s.add_argument("path")
    s.add_argument("--delete", action="store_true", help="delete each source file once it is stored")
    s.add_argument("--no-index", action="store_true", help="store only; embed on the next search")
    s.set_defaults(func=_cmd_import)
    s = sub.add_parser("add", help="add one transcript with a title and start time")
    s.add_argument("path", help="file, or - for stdin")
    s.add_argument("--title", default="")
    s.add_argument("--started-at", default="", help="ISO local time, e.g. 2026-09-28T14:00")
    s.add_argument("--source", default="", help="where it came from, e.g. the URL")
    s.set_defaults(func=_cmd_add)
    s = sub.add_parser("embed", help="make an embedding bundle (run on another machine)")
    s.add_argument("path")
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(func=_cmd_embed)
    s = sub.add_parser("export", help="write transcripts (text only) to a .jsonl for embedding elsewhere")
    s.add_argument("output")
    s.add_argument("meeting_ids", nargs="*")
    s.add_argument("--pending", action="store_true", help="only transcripts not yet embedded")
    s.set_defaults(func=_cmd_export)
    s = sub.add_parser("list")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=_cmd_list)
    s = sub.add_parser("show", help="print a transcript")
    s.add_argument("meeting_id")
    s.set_defaults(func=_cmd_show)
    s = sub.add_parser("rm", help="delete a transcript and its index entries")
    s.add_argument("meeting_id")
    s.set_defaults(func=_cmd_rm)
    s = sub.add_parser("reindex", help="embed pending transcripts (or one, forced)")
    s.add_argument("meeting_id", nargs="?")
    s.set_defaults(func=_cmd_reindex)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s", stream=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
