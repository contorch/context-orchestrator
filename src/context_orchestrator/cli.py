#!/usr/bin/env python3
"""save-transcript CLI — store a clipboard transcript in the database and index it."""

import argparse
import subprocess
import sys
from pathlib import Path

from context_orchestrator import transcripts


def get_clipboard() -> str:
    """Get text from macOS clipboard."""
    result = subprocess.run(["pbpaste"], capture_output=True, text=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Save clipboard transcript and index for search"
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="meeting",
        help="Short name for the transcript (e.g., 'sprint-planning', 'auth-discussion')",
    )
    parser.add_argument(
        "--file",
        help="Store an existing file (.md/.txt/.vtt/.srt) instead of the clipboard",
    )
    args = parser.parse_args()

    db = transcripts._open_db()
    if args.file:
        file_path = Path(args.file).expanduser().resolve()
        if not file_path.exists():
            print(f"File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        stats = transcripts.import_path(db, file_path, settle_seconds=0)
        print(f"Stored: {file_path.name} ({stats['stored']} new, {stats['unchanged']} unchanged)")
    else:
        text = get_clipboard()
        if len(text.strip().split("\n")) < 2:
            print("Clipboard looks empty or too short. Copy the transcript first.", file=sys.stderr)
            sys.exit(1)
        meeting_id, created = transcripts.add_text(db, text, title=args.name, source="clipboard")
        print(f"{'Stored' if created else 'Already stored'}: {meeting_id} ({len(text.split())} words)")

    # Index in ChromaDB now; if embedding fails the text is safe in the
    # database and the next search() picks it up.
    done = transcripts.index_pending(transcripts._open_vs(), db, settle_seconds=0)
    print(f"Indexed {len(done)} transcript(s). Searchable via search().")


if __name__ == "__main__":
    main()
