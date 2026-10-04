"""`contorch-memory` — how the memory module is configured.

    contorch-memory status
    contorch-memory embeddings                  # show the current choice
    contorch-memory embeddings gemini|local|none
    contorch-memory claude install|uninstall|status [--channel app|brew|dev] [--json]
    contorch-memory status --json [--deep] | selftest --json | where --json
    contorch-memory backup --to DIR [--stop-server] --json | restore --from DIR --json
    contorch-memory index migrate --in-process [--backup-dir DIR] --json

The embedding choice decides how search understands a question:
  gemini — Google's gemini-embedding-001: best at paraphrases; needs a key
           (~/.config/google/key) and network for every search and every new
           transcript.
  local  — all-MiniLM-L6-v2 running on this Mac: no key, no network; ~80 MB
           model downloaded once; weaker than Gemini on paraphrases.
  none   — no embeddings: keyword search (SQLite full-text, stemmed) only.
Full-text search is always on, whatever the choice.

The choice is written to ~/.context-orchestrator/env, which every entry point
reads (MCP server, CLIs). Each model has its own vector collection; switching
re-embeds transcripts from the stored text on the next search (or
`contorch-transcripts reindex`), and switching back reuses the old vectors.
Machines that share embedding bundles must use the same choice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import ENV_FILE
from .search import EMBEDDING_MODEL_ENV, _resolve_gemini_api_key, embedding_choice

CHOICES = {"gemini": "gemini-embedding-001", "local": "local", "none": "none"}


def set_env_value(key: str, value: str, path: Path = ENV_FILE) -> None:
    """Set KEY=value in the env file, keeping every other line. Several
    writers (setup, the menu bar, CLIs) may do this at once: an flock on
    `<file>.lock` serialises read-modify-write, and the new file replaces the
    old one atomically, so a reader never sees half a file (lab E8: 7-18 of 20
    concurrent writes were lost without this)."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            lines = []
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                pass
            out, done = [], False
            for line in lines:
                k = line.split("=", 1)[0].strip().removeprefix("export ").strip()
                if k == key and not line.lstrip().startswith("#"):
                    if not done:
                        out.append(f"{key}={value}")
                        done = True
                    continue
                out.append(line)
            if not done:
                out.append(f"{key}={value}")
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
            try:
                os.chmod(tmp, path.stat().st_mode & 0o777)
            except FileNotFoundError:
                pass
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def describe(choice: str) -> str:
    if choice == "none":
        return "none — keyword (full-text) search only"
    if choice == "local":
        return "local — all-MiniLM-L6-v2 on this Mac (no key, no network)"
    if choice.startswith("gemini"):
        key = "key found" if _resolve_gemini_api_key() else "NO KEY — add one to ~/.config/google/key"
        return f"gemini — {choice} ({key})"
    if not choice:
        return ("auto — Gemini when a key is present, otherwise local "
                f"(currently {'Gemini' if _resolve_gemini_api_key() else 'local'})")
    return f"{choice} (sentence-transformers)"


def set_embeddings(name: str, env_file: Path = ENV_FILE) -> str:
    if name not in CHOICES:
        raise ValueError(f"choose one of: {', '.join(CHOICES)}")
    set_env_value(EMBEDDING_MODEL_ENV, CHOICES[name], env_file)
    os.environ[EMBEDDING_MODEL_ENV] = CHOICES[name]
    msg = [f"embeddings: {describe(embedding_choice())}"]
    if name == "gemini" and not _resolve_gemini_api_key():
        msg.append("  Search stays keyword-only until a Gemini key is in ~/.config/google/key.")
    if name == "local":
        msg.append("  The model (~80 MB) downloads on first use.")
    if name != "none":
        msg.append("  Transcripts are re-embedded from the stored text on the next search "
                   "(or now: contorch-transcripts reindex). Restart Claude Code to apply.")
    else:
        msg.append("  Restart Claude Code to apply. Existing vectors are kept for switching back.")
    return "\n".join(msg)


def _cmd_claude(args) -> int:
    from . import claude_install as ci
    from .jsonout import emit, error, reserved_stdout
    backup = Path(args.backup_dir).expanduser() if args.backup_dir else None

    def run():
        if args.action == "install":
            return ci.install(args.channel, hook=not args.no_hook, backup_dir=backup)
        if args.action == "uninstall":
            return ci.uninstall(args.channel, backup_dir=backup)
        return ci.status(args.channel)
    if args.json:
        with reserved_stdout() as out:
            try:
                doc = run()
            except Exception as exc:   # one document, whatever happens
                doc = {"schema": ci.SCHEMA, "ok": False, "action": args.action,
                       "error": error("internal", f"{type(exc).__name__}: {exc}")}
            print(ci.describe(doc) if "mcp" in doc else doc["error"]["message"], file=sys.stderr)
            emit(doc, out)
    else:
        doc = run()
        print(ci.describe(doc))
    return 0 if (doc.get("ok") or args.action == "status") else 1


def _emit_json(build, schema: str, action: str, describe=None) -> dict:
    """Run `build()` with stdout reserved for one JSON document; any failure
    becomes {schema, ok: false, error{code, message}}."""
    from .backup import DataOpError
    from .jsonout import emit, error, reserved_stdout
    with reserved_stdout() as out:
        try:
            doc = build()
        except DataOpError as exc:
            doc = {"schema": schema, "ok": False, "action": action, "error": exc.as_json()}
        except Exception as exc:
            doc = {"schema": schema, "ok": False, "action": action,
                   "error": error("internal", f"{type(exc).__name__}: {exc}"[:500])}
        if describe:
            try:
                print(describe(doc), file=sys.stderr)
            except Exception:
                pass
        emit(doc, out)
    return doc


def _cmd_data(args) -> int:
    """status --json, selftest, where, backup, restore, index migrate."""
    from . import backup as B
    from . import memstatus, selftest
    from .backup import DataOpError
    if args.cmd == "status":
        build, schema, desc = (lambda: memstatus.status(deep=args.deep)), memstatus.STATUS_SCHEMA, memstatus.describe
    elif args.cmd == "selftest":
        build, schema, desc = selftest.run, selftest.SCHEMA, lambda d: (
            f"selftest: {'ok' if d['ok'] else 'FAILED'} at {d.get('stage')} in {d.get('ms')} ms"
            + (f" — {d['error']['code']}: {d['error']['message']}" if d.get("error") else ""))
    elif args.cmd == "where":
        build, schema, desc = (lambda: memstatus.where(args.channel)), memstatus.WHERE_SCHEMA, None
    elif args.cmd == "backup":
        build, schema, desc = (lambda: B.backup(Path(args.to), stop=args.stop_server)), B.SCHEMA, B.describe
    elif args.cmd == "restore":
        build, schema, desc = (lambda: B.restore(Path(args.frm), stop=args.stop_server)), B.SCHEMA, B.describe
    else:  # index migrate
        build, schema, desc = (lambda: B.migrate_in_process(
            Path(args.backup_dir) if args.backup_dir else None)), B.SCHEMA, B.describe
    if args.json:
        doc = _emit_json(build, schema, args.cmd, desc)
    else:
        try:
            doc = build()
        except DataOpError as exc:
            print(f"{args.cmd}: failed — {exc.code}: {exc}", file=sys.stderr)
            return 1
        print(desc(doc) if desc else json.dumps(doc, indent=2))
    return 0 if doc.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="contorch-memory", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embeddings", help="show or set the embedding model: gemini, local, none")
    e.add_argument("choice", nargs="?", choices=list(CHOICES))
    st = sub.add_parser("status", help="embedding choice, transcripts stored, pending, index size")
    st.add_argument("--json", action="store_true", help="contorch-memory.status/1 on stdout (no chromadb import)")
    st.add_argument("--deep", action="store_true", help="--json: also open the index with chromadb and count")
    t = sub.add_parser("selftest", help="write, search and delete a marker document end to end")
    t.add_argument("--json", action="store_true")
    w = sub.add_parser("where", help="absolute paths of this install's commands and files")
    w.add_argument("--channel", choices=("app", "brew", "dev"), default=None)
    w.add_argument("--json", action="store_true")
    b = sub.add_parser("backup", help="verified copy of context.db and the vector index")
    b.add_argument("--to", required=True, help="an empty or new directory")
    b.add_argument("--stop-server", action="store_true", help="stop a running chroma server for the copy")
    b.add_argument("--json", action="store_true")
    r = sub.add_parser("restore", help="put a backup's vector index back (context.db is never rolled back)")
    r.add_argument("--from", dest="frm", required=True)
    r.add_argument("--stop-server", action="store_true")
    r.add_argument("--json", action="store_true")
    ix = sub.add_parser("index", help="vector index location")
    ixs = ix.add_subparsers(dest="index_cmd", required=True)
    m = ixs.add_parser("migrate", help="retire the chroma server: the index is opened in-process")
    m.add_argument("--in-process", action="store_true", required=True)
    m.add_argument("--backup-dir")
    m.add_argument("--json", action="store_true")
    c = sub.add_parser("claude", help="connect Claude Code: MCP server, hook, CLAUDE.md block, transcripts skill")
    c.add_argument("action", choices=("install", "uninstall", "status"))
    c.add_argument("--channel", choices=("app", "brew", "dev"), default=None,
                   help="which install this is (default: $CONTORCH_CHANNEL, else dev)")
    c.add_argument("--no-hook", action="store_true", help="install: leave the auto-context hook out")
    c.add_argument("--backup-dir", help="copy settings.json / CLAUDE.md here before changing them")
    c.add_argument("--json", action="store_true", help="one JSON document on stdout")
    args = p.parse_args(argv)

    if args.cmd == "claude":
        return _cmd_claude(args)
    if args.cmd in ("selftest", "where", "backup", "restore", "index") or (
            args.cmd == "status" and (args.json or args.deep)):
        return _cmd_data(args)

    if args.cmd == "embeddings":
        if args.choice is None:
            print(describe(embedding_choice()))
            return 0
        print(set_embeddings(args.choice))
        return 0

    from .transcripts import _open_db, _open_vs
    db = _open_db()
    vs = _open_vs()
    total, _ = db.count_transcripts()
    pending = len(db.transcripts_to_index(float("inf"), vs.identity))
    print(f"embeddings:  {describe(embedding_choice())}")
    print(f"transcripts: {total} stored, {pending} not yet embedded with this model")
    print(f"vector index: {vs.count()} chunks" + (f" (collection {vs.collection_name})" if vs.enabled else ""))
    print("full-text:   on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
