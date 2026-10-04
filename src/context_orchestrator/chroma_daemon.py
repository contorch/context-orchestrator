"""Manages the local chroma server (launchd agent on macOS).

Running chroma as an HTTP server lets the watcher daemon, the MCP server, and
any one-shot CLIs (save-transcript, transcript-watcher once) share a single
index without fighting over the SQLite handle.

The on-disk format is identical to PersistentClient — switching is zero-
migration. The default port is 8765 (avoids the very common 8000).
"""
from __future__ import annotations

import argparse
import os
import plistlib
import socket
import sys
from pathlib import Path

from .launchd import launchctl

# chromadb is imported where it is used: `contorch-memory status` and the
# claude installer read LAUNCHD_PLIST without loading Chroma.

CHROMA_PATH = Path.home() / ".context-orchestrator" / "chroma"
LOG_DIR = Path.home() / ".context-orchestrator"
LOG_FILE = LOG_DIR / "chroma-daemon.log"

LAUNCHD_LABEL = "com.contorch.context-orchestrator-chroma"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
# Pre-rebrand label (com.stirredo.*): retired automatically by `install`.
LEGACY_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.stirredo.context-orchestrator-chroma.plist"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _chroma_cli(python_exe: str) -> str:
    """Locate the `chroma` CLI shipped with the chromadb package."""
    candidate = Path(python_exe).parent / "chroma"
    if candidate.exists():
        return str(candidate)
    return "chroma"


def _plist_payload(python_exe: str, host: str, port: int, path: Path) -> bytes:
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [
            _chroma_cli(python_exe), "run",
            "--path", str(path),
            "--host", host,
            "--port", str(port),
        ],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False, "Crashed": True},
        "StandardOutPath": str(LOG_FILE),
        "StandardErrorPath": str(LOG_FILE),
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
            "ANONYMIZED_TELEMETRY": "False",
        },
        "ProcessType": "Background",
    }
    return plistlib.dumps(payload)


def is_listening(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 0.5) -> bool:
    """True if a TCP socket on host:port accepts connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def heartbeat(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> tuple[bool, str]:
    """Round-trip a count() against the chroma server. Returns (ok, message)."""
    try:
        import chromadb
        client = chromadb.HttpClient(host=host, port=port)
        col = client.get_or_create_collection("context")
        return True, f"{col.count()} documents"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def cmd_install(args) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
    if LEGACY_PLIST.exists():
        launchctl("unload", "-w", str(LEGACY_PLIST), quiet=True)
        LEGACY_PLIST.unlink()
        print(f"retired legacy agent {LEGACY_PLIST.name}")
    LAUNCHD_PLIST.write_bytes(
        _plist_payload(sys.executable, args.host, args.port, CHROMA_PATH)
    )
    launchctl("unload", str(LAUNCHD_PLIST), quiet=True)
    launchctl("load", "-w", str(LAUNCHD_PLIST))
    print(f"installed launchd agent at {LAUNCHD_PLIST}")
    print(f"chroma server: http://{args.host}:{args.port} (path: {CHROMA_PATH})")
    return 0


def cmd_uninstall(_args) -> int:
    if not LAUNCHD_PLIST.exists():
        print("launchd agent not installed")
        return 0
    launchctl("unload", "-w", str(LAUNCHD_PLIST))
    LAUNCHD_PLIST.unlink()
    print(f"removed {LAUNCHD_PLIST}")
    return 0


def cmd_status(args) -> int:
    print("context-orchestrator-chroma")
    print(f"  path:       {CHROMA_PATH}")
    print(f"  log file:   {LOG_FILE}")
    print(f"  launchd:    {'installed' if LAUNCHD_PLIST.exists() else 'not installed'}")
    if LAUNCHD_PLIST.exists():
        loaded = launchctl("list", LAUNCHD_LABEL, capture=True).returncode == 0
        print(f"  loaded:     {'yes' if loaded else 'no'}")
    listening = is_listening(args.host, args.port)
    print(f"  listening:  {'yes' if listening else 'no'} ({args.host}:{args.port})")
    if listening:
        ok, msg = heartbeat(args.host, args.port)
        print(f"  heartbeat:  {'ok' if ok else 'fail'} — {msg}")
    return 0 if listening else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="context-orchestrator-chroma",
        description="Manage the local chroma HTTP server.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_install = sub.add_parser("install", help="install + start launchd agent")
    p_install.add_argument("--host", default=DEFAULT_HOST)
    p_install.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_install.set_defaults(func=cmd_install)

    sub.add_parser("uninstall", help="stop + remove launchd agent").set_defaults(func=cmd_uninstall)

    p_status = sub.add_parser("status", help="show daemon status")
    p_status.add_argument("--host", default=DEFAULT_HOST)
    p_status.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
