"""Contorch JSON output rules: one JSON document (or JSON Lines ending in
{"event": "result"}) on stdout, human text on stderr, `schema` + `ok` +
`error{code, message}`. While a --json command runs, fd 1 points at stderr,
so stray prints and native libraries can't corrupt the document."""
import contextlib
import json
import os
import sys


@contextlib.contextmanager
def reserved_stdout():
    """Yield a writer for the real stdout; everything else goes to stderr."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    real, sys.stdout = sys.stdout, sys.stderr
    out = os.fdopen(os.dup(saved), "w", encoding="utf-8")
    try:
        yield out
    finally:
        out.flush()
        out.close()
        sys.stdout = real
        os.dup2(saved, 1)
        os.close(saved)


def emit(doc: dict, out=None) -> None:
    out = out or sys.stdout
    out.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
    out.flush()


def error(code: str, message: str, **detail) -> dict:
    return {"code": code, "message": message, **detail}
