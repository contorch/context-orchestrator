"""Every launchctl call in context-orchestrator goes through here.

launchd labels are per user, not per HOME: an install run from a scratch
HOME (tests, labs) still replaces the user's real agent of the same label.
CO_NO_LAUNCHCTL=1 (set by the test suite) makes any attempt an error instead.
"""
from __future__ import annotations

import os
import subprocess


class LaunchctlDisabled(RuntimeError):
    pass


def launchctl(*args: str, capture: bool = False, quiet: bool = False) -> subprocess.CompletedProcess:
    if os.environ.get("CO_NO_LAUNCHCTL") == "1":
        raise LaunchctlDisabled(f"launchctl {' '.join(args)} refused (CO_NO_LAUNCHCTL=1)")
    kw: dict = {"check": False}
    if capture:
        kw.update(capture_output=True, text=True)
    elif quiet:
        kw.update(stderr=subprocess.DEVNULL)
    return subprocess.run(["launchctl", *args], **kw)


def domain() -> str:
    return f"gui/{os.getuid()}"
