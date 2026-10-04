"""The bash channel-guard reader (scripts/contorch_channel_guard.sh) against
pipeline-monitor's contract fixtures (vendored, pinned in PIN)."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "channel_guard"


def test_vendored_fixtures_match_the_pin():
    lines = [l.split() for l in (FIX / "PIN").read_text().splitlines()
             if l and not l.startswith(("#", "commit"))]
    pinned = {name: sha for sha, name in lines}
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FIX.glob("*.json")}
    assert pinned == files and len(files) >= 15


def test_every_fixture_under_bash_3_2():
    r = subprocess.run(["/bin/bash", str(ROOT / "tests" / "bootstrap_guard_test.sh")],
                       capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert " 0 failed" in r.stdout
