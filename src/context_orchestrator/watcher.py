"""Indexes new or modified .md files in ~/transcripts/.

Two ways in, one implementation (`catch_up`):
  * on demand — the MCP server calls catch_up() when Claude Code starts it and
    before searches. This is the default: no background daemon.
  * `transcript-watcher run` — the old always-on poll loop, still available
    for anyone who wants it (and `install` still writes its launchd agent).

Designed to pair with meeting-capture (https://github.com/contorch/meeting-capture),
which writes transcripts continuously while a meeting runs. Files get appended to
in flight, so the watcher must reindex when mtime advances — not just on first sight.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

from .cli import index_transcript
from .search import VectorSearch

TRANSCRIPT_DIR = Path.home() / "transcripts"
STATE_DIR = Path.home() / ".context-orchestrator"
STATE_FILE = STATE_DIR / "watcher_state.json"
LOG_FILE = STATE_DIR / "watcher.log"
# Serialises indexing across processes: several Claude Code sessions each run
# their own MCP server, and an old watcher agent may still be around.
LOCK_FILE = STATE_DIR / "index.lock"
DEFAULT_INTERVAL = 5.0
# How long a file must be quiet (no mtime advance) before we re-index it.
#
# meeting-capture appends a new chunk to its active transcript every ~10-20s
# during a live meeting. With a small settle window the watcher catches the
# file in its short quiet stretch after every append and re-chunks + re-embeds
# the ENTIRE file each time, producing O(N²) Gemini embedding calls in the
# length of the meeting (verified 5/8: one 2.5h meeting re-indexed 457 times,
# 8175 embed calls, ~6M tokens — hit 88% of the Embedding-1 TPM cap).
#
# 60s is comfortably longer than the longest natural gap between meeting-
# capture chunks, so live transcripts wait until the meeting actually ends.
# Tradeoff: ~60s delay between meeting-end and the transcript being
# searchable. Acceptable for this workload.
SETTLE_SECONDS = 60.0

LAUNCHD_LABEL = "com.contorch.transcript-watcher"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
# Pre-rebrand label (com.stirredo.*): retired automatically by `install`.
LEGACY_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.stirredo.transcript-watcher.plist"

log = logging.getLogger("context-orchestrator.watcher")


def load_state(state_file: Path = STATE_FILE) -> dict[str, float]:
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, float], state_file: Path = STATE_FILE) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: another process may read it at any moment.
    tmp = state_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, state_file)


def scan_once(
    vs: VectorSearch,
    watch_dir: Path,
    state: dict[str, float],
    settle_seconds: float = SETTLE_SECONDS,
) -> list[Path]:
    """Reindex any .md file in watch_dir whose mtime has advanced.

    Files modified within the last `settle_seconds` are skipped on this pass to
    avoid indexing mid-write. They'll be picked up on the next scan.
    """
    if not watch_dir.exists():
        return []
    now = time.time()
    indexed: list[Path] = []
    for f in sorted(watch_dir.glob("*.md")):
        try:
            mtime = f.stat().st_mtime
        except FileNotFoundError:
            continue
        key = str(f)
        if mtime <= state.get(key, 0.0):
            continue
        if (now - mtime) < settle_seconds:
            continue
        try:
            vs.collection.delete(where={"file_path": key})
        except Exception:
            log.exception("failed to clear old chunks for %s", key)
        try:
            index_transcript(vs, f)
        except Exception:
            log.exception("failed to index %s", key)
            continue
        state[key] = mtime
        indexed.append(f)
    return indexed


def catch_up(
    vs: VectorSearch,
    watch_dir: Path = TRANSCRIPT_DIR,
    state_file: Path = STATE_FILE,
    settle_seconds: float = SETTLE_SECONDS,
    lock_file: Path = LOCK_FILE,
) -> list[Path] | None:
    """One indexing pass that is safe to run from several processes at once.

    Returns the files indexed, or None if another process is already indexing
    (the caller just proceeds with the index as it is — it never waits).
    State is re-read inside the lock because another process may have
    advanced it since we last looked.
    """
    import fcntl

    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            state = load_state(state_file)
            indexed = scan_once(vs, watch_dir, state, settle_seconds)
            if indexed:
                save_state(state, state_file)
            return indexed
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def watch_loop(
    watch_dir: Path = TRANSCRIPT_DIR,
    interval: float = DEFAULT_INTERVAL,
    state_file: Path = STATE_FILE,
) -> None:
    vs = VectorSearch()
    log.info("watching %s every %.1fs", watch_dir, interval)
    while True:
        try:
            for p in catch_up(vs, watch_dir, state_file) or []:
                log.info("indexed %s", p.name)
        except Exception:
            log.exception("scan failed")
        time.sleep(interval)


def _plist_payload(python_exe: str) -> bytes:
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [python_exe, "-m", "context_orchestrator.watcher", "run"],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False, "Crashed": True},
        "StandardOutPath": str(LOG_FILE),
        "StandardErrorPath": str(LOG_FILE),
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        },
        "ProcessType": "Background",
    }
    return plistlib.dumps(payload)


def cmd_run(args) -> int:
    watch_loop(Path(args.dir).expanduser(), args.interval)
    return 0


def cmd_once(args) -> int:
    vs = VectorSearch()
    indexed = catch_up(vs, Path(args.dir).expanduser())
    if indexed is None:
        log.info("another process is indexing right now — nothing to do")
    else:
        log.info("indexed %d file(s)", len(indexed))
    return 0


def cmd_install(_args) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
    if LEGACY_PLIST.exists():
        subprocess.run(["launchctl", "unload", "-w", str(LEGACY_PLIST)],
                       check=False, stderr=subprocess.DEVNULL)
        LEGACY_PLIST.unlink()
        print(f"retired legacy agent {LEGACY_PLIST.name}")
    LAUNCHD_PLIST.write_bytes(_plist_payload(sys.executable))
    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST)], check=False, stderr=subprocess.DEVNULL)
    subprocess.run(["launchctl", "load", "-w", str(LAUNCHD_PLIST)], check=False)
    print(f"installed launchd agent at {LAUNCHD_PLIST}")
    print("watcher will auto-start at login.")
    return 0


def cmd_uninstall(_args) -> int:
    if not LAUNCHD_PLIST.exists():
        print("launchd agent not installed")
        return 0
    subprocess.run(["launchctl", "unload", "-w", str(LAUNCHD_PLIST)], check=False)
    LAUNCHD_PLIST.unlink()
    print(f"removed {LAUNCHD_PLIST}")
    return 0


def cmd_status(_args) -> int:
    print(f"transcript-watcher")
    print(f"  watching:   {TRANSCRIPT_DIR}")
    print(f"  state file: {STATE_FILE}")
    print(f"  log file:   {LOG_FILE}")
    print(f"  launchd:    {'installed' if LAUNCHD_PLIST.exists() else 'not installed'}")
    if LAUNCHD_PLIST.exists():
        result = subprocess.run(
            ["launchctl", "list", LAUNCHD_LABEL],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  loaded:     yes")
        else:
            print(f"  loaded:     no")
    return 0


def cmd_doctor(_args) -> int:
    """Health check for the transcript-watcher side of the pipeline."""
    failures = 0

    def _ok(label, value=""):
        suffix = f" — {value}" if value else ""
        print(f"  ✓ {label}{suffix}")

    def _fail(label, hint):
        nonlocal failures
        failures += 1
        print(f"  ✗ {label}")
        print(f"      → {hint}")

    print("transcript-watcher — doctor\n")

    print("Paths:")
    if TRANSCRIPT_DIR.exists():
        n = len(list(TRANSCRIPT_DIR.glob("*.md")))
        _ok("watch dir", f"{TRANSCRIPT_DIR} ({n} files)")
    else:
        _fail("watch dir missing", f"mkdir -p {TRANSCRIPT_DIR}")
    if STATE_DIR.exists():
        _ok("state dir", str(STATE_DIR))
    else:
        _fail("state dir missing", f"mkdir -p {STATE_DIR}")
    state = load_state()
    print(f"  · state file tracks {len(state)} indexed file(s)")

    print("\nChroma server:")
    from . import chroma_daemon
    if chroma_daemon.LAUNCHD_PLIST.exists():
        _ok("launchd plist installed", str(chroma_daemon.LAUNCHD_PLIST))
    else:
        _fail("chroma launchd plist not installed", "context-orchestrator-chroma install")
    if chroma_daemon.is_listening():
        _ok(f"server listening", f"{chroma_daemon.DEFAULT_HOST}:{chroma_daemon.DEFAULT_PORT}")
        ok, msg = chroma_daemon.heartbeat()
        if ok:
            _ok("heartbeat", msg)
        else:
            _fail("heartbeat failed", msg)
    else:
        _fail("chroma server not listening",
              f"launchctl load -w {chroma_daemon.LAUNCHD_PLIST}")

    print("\nVector index:")
    try:
        vs = VectorSearch()
        _ok(f"ChromaDB connected", f"{vs.count()} documents in collection")
    except Exception as exc:
        _fail("ChromaDB connection failed", f"{exc}")

    print("\nDaemon:")
    if LAUNCHD_PLIST.exists():
        _ok("launchd plist installed", str(LAUNCHD_PLIST))
        result = subprocess.run(
            ["launchctl", "list", LAUNCHD_LABEL], capture_output=True, text=True
        )
        if result.returncode == 0:
            _ok("launchd service loaded")
        else:
            _fail("launchd service not loaded", f"launchctl load -w {LAUNCHD_PLIST}")
    else:
        _ok("no watcher daemon (default)",
            "transcripts are indexed on demand by the MCP server; `transcript-watcher install` for always-on")

    print("\nUpstream (meeting-capture):")
    mc_log = Path.home() / ".meeting-capture" / "daemon.log"
    if mc_log.exists():
        size_kb = mc_log.stat().st_size / 1024
        _ok("meeting-capture daemon log present", f"{mc_log} ({size_kb:.1f} KB)")
    else:
        print(f"  · meeting-capture not detected (log {mc_log} missing). That's fine — watcher works with any source of *.md files in {TRANSCRIPT_DIR}.")

    print("\nManual gates (cannot be checked from code):")
    print("  ?  Claude Code restarted since context-orchestrator install (so MCP server is live)")

    print()
    if failures == 0:
        print("All automatic checks passed.")
        return 0
    else:
        print(f"{failures} issue(s). Fix and re-run.")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcript-watcher",
        description="Watch ~/transcripts/ and auto-index new or modified files.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="watch loop (default)")
    p_run.add_argument("--dir", default=str(TRANSCRIPT_DIR))
    p_run.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    p_run.set_defaults(func=cmd_run)

    p_once = sub.add_parser("once", help="index anything new once and exit")
    p_once.add_argument("--dir", default=str(TRANSCRIPT_DIR))
    p_once.set_defaults(func=cmd_once)

    sub.add_parser("install", help="install launchd auto-start agent").set_defaults(func=cmd_install)
    sub.add_parser("uninstall", help="remove launchd agent").set_defaults(func=cmd_uninstall)
    sub.add_parser("status", help="show watcher status").set_defaults(func=cmd_status)
    sub.add_parser("doctor", help="full health check").set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    if not args.cmd:
        args = parser.parse_args(["run"])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
