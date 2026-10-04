"""`contorch-memory` — how the memory module is configured.

    contorch-memory status
    contorch-memory embeddings                  # show the current choice
    contorch-memory embeddings gemini|local|none

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
import os
import sys
from pathlib import Path

from . import ENV_FILE
from .search import EMBEDDING_MODEL_ENV, _resolve_gemini_api_key, embedding_choice

CHOICES = {"gemini": "gemini-embedding-001", "local": "local", "none": "none"}


def set_env_value(key: str, value: str, path: Path = ENV_FILE) -> None:
    """Set KEY=value in the env file, keeping every other line."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="contorch-memory", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embeddings", help="show or set the embedding model: gemini, local, none")
    e.add_argument("choice", nargs="?", choices=list(CHOICES))
    sub.add_parser("status", help="embedding choice, transcripts stored, pending, index size")
    args = p.parse_args(argv)

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
