"""Claude Code integration — the one owner (`contorch-memory claude …`).

    contorch-memory claude install   [--channel app|brew|dev] [--no-hook] [--backup-dir D] [--json]
    contorch-memory claude uninstall [--channel …] [--backup-dir D] [--json]
    contorch-memory claude status    [--channel …] [--json]

What Contorch puts into Claude Code, and nothing else:

* the MCP server entry `context-orchestrator` (user scope, via `claude mcp
  add/remove`), pointing at this install's own `contorch-mcp`, keeping only the
  allowed CO_* env keys (CO_CHROMA_HOST/PORT are dropped when the index is
  in-process);
* one UserPromptSubmit hook entry, guarded and tagged:
      [ -x '<p>' ] && '<p>'; exit 0 # contorch-hook:<channel>
  (silent when the file is gone, e.g. a trashed app; quoted for spaces), with
  timeout HOOK_TIMEOUT_S;
* one marked block in CLAUDE.md (<!-- contorch --> … <!-- /contorch -->);
* the `transcripts` skill, rendered with this install's absolute
  `contorch-transcripts` path, stamped `.contorch-skill.json`.

Things written by earlier installers (the copied ~/.claude/hooks/auto-context.py,
the CONTEXT-ORCHESTRATOR / auto-context-section CLAUDE.md blocks, a curl'ed
SKILL.md) are replaced only when their sha256 is one we shipped
(templates/legacy_hashes.json); anything else is the user's and is kept.
Uninstall removes only tagged / marked / stamped / known items.

Claude Code's files follow CLAUDE_CONFIG_DIR (default ~/.claude; the user-scope
MCP entries live in $CLAUDE_CONFIG_DIR/.claude.json, else ~/.claude.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = "contorch-memory.claude/1"
CHANNELS = ("app", "brew", "dev")
MCP_NAME = "context-orchestrator"
HOOK_TAG = "contorch-hook:"
HOOK_TIMEOUT_S = 10                 # Claude Code kills the hook after this
MD_BEGIN, MD_END = "<!-- contorch -->", "<!-- /contorch -->"
LEGACY_MD_MARKERS = (
    ("<!-- BEGIN CONTEXT-ORCHESTRATOR -->", "<!-- END CONTEXT-ORCHESTRATOR -->"),
    ("<!-- BEGIN auto-context-section", "<!-- END auto-context-section -->"),
)
LEGACY_HOOK_NAMES = ("auto-context.py",)
SKILL_NAME = "transcripts"
SKILL_STAMP = ".contorch-skill.json"
# CO_* keys that mean something to this version's MCP server.
ALLOWED_ENV = ("CO_EMBEDDING_MODEL", "CO_RERANK_MODEL", "CO_DB_PATH", "CO_CHROMA_PATH",
               "CO_CHROMA_HOST", "CO_CHROMA_PORT")
SERVER_ONLY_ENV = ("CO_CHROMA_HOST", "CO_CHROMA_PORT")
MANAGED_DIR = Path("/Library/Application Support/ClaudeCode")

TEMPLATES = Path(__file__).resolve().parent / "templates"


class ClaudeInstallError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- locations

def channel() -> str:
    """$CONTORCH_CHANNEL; unset or unknown = dev."""
    c = os.environ.get("CONTORCH_CHANNEL", "")
    return c if c in CHANNELS else "dev"


@dataclass
class ClaudePaths:
    config_dir: Path
    claude_json: Path
    managed_dir: Path

    @classmethod
    def detect(cls) -> "ClaudePaths":
        cfg = os.environ.get("CLAUDE_CONFIG_DIR")
        config_dir = Path(cfg).expanduser() if cfg else Path.home() / ".claude"
        claude_json = config_dir / ".claude.json" if cfg else Path.home() / ".claude.json"
        managed = Path(os.environ.get("CO_CLAUDE_MANAGED_DIR") or MANAGED_DIR)
        return cls(config_dir, claude_json, managed)

    @property
    def settings(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def claude_md(self) -> Path:
        return self.config_dir / "CLAUDE.md"

    @property
    def skill_dir(self) -> Path:
        return self.config_dir / "skills" / SKILL_NAME

    @property
    def legacy_hook(self) -> Path:
        return self.config_dir / "hooks" / "auto-context.py"


def _brew_bin(name: str) -> Optional[Path]:
    """A Homebrew wrapper for one of our commands (stable across upgrades and
    Python bumps; it rebuilds the venv itself), if this install is brew's."""
    prefixes = [os.environ.get("HOMEBREW_PREFIX"), "/opt/homebrew", "/usr/local"]
    for pre in filter(None, prefixes):
        p = Path(pre) / "bin" / name
        try:
            if p.exists() and "/Cellar/context-orchestrator/" in os.path.realpath(p):
                return p
        except OSError:
            continue
    return None


def entry_points(chan: str) -> dict:
    """Absolute paths of this install's commands, as Claude Code should run
    them. Own interpreter's bin dir; in the brew channel the brew wrapper when
    it exists."""
    bindir = Path(sys.executable).parent
    out = {}
    for key, name in (("mcp", "contorch-mcp"), ("hook", "contorch-hook"),
                      ("transcripts", "contorch-transcripts"), ("memory", "contorch-memory"),
                      ("chroma_cli", "context-orchestrator-chroma")):
        p = bindir / name
        if chan == "brew":
            p = _brew_bin(name) or p
        out[key] = str(p)
    return out


def find_claude() -> Optional[str]:
    """Claude Code's CLI. A GUI app's PATH is /usr/bin:/bin:/usr/sbin:/sbin, so
    also look where its installers put it, then ask the login shell."""
    env = os.environ.get("CO_CLAUDE_BIN")
    if env:
        return env if Path(env).exists() else None
    found = shutil.which("claude")
    if found:
        return found
    home = Path.home()
    for p in (home / ".local" / "bin" / "claude", home / ".claude" / "local" / "claude",
              Path("/opt/homebrew/bin/claude"), Path("/usr/local/bin/claude")):
        if p.exists():
            return str(p)
    try:
        r = subprocess.run([os.environ.get("SHELL", "/bin/zsh"), "-lc", "command -v claude"],
                           capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
        p = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
        return p if p and Path(p).exists() else None
    except Exception:
        return None


# ---------------------------------------------------------------- helpers

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def legacy_hashes() -> dict:
    return json.loads((TEMPLATES / "legacy_hashes.json").read_text())


def _load_json(path: Path) -> Optional[dict]:
    """{} for a missing/empty file, None for invalid JSON (never overwritten)."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except OSError:
        return None
    if not text.strip():
        return {}
    try:
        d = json.loads(text)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _backup(path: Path, backup_dir: Optional[Path]) -> None:
    if backup_dir is None or not path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / path.name
    n = 2
    while dest.exists():
        dest = backup_dir / f"{path.name}.{n}"
        n += 1
    shutil.copy2(path, dest)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.contorch-{os.getpid()}.tmp")
    tmp.write_text(text)
    try:
        os.chmod(tmp, path.stat().st_mode & 0o777)
    except FileNotFoundError:
        pass
    os.replace(tmp, path)


def _in_process_index() -> bool:
    """Will the MCP server use the in-process index (no chroma server)?"""
    from .chroma_daemon import LAUNCHD_PLIST
    return not LAUNCHD_PLIST.exists()


# ---------------------------------------------------------------- managed settings

def managed_policy(paths: ClaudePaths) -> dict:
    """What an administrator's managed settings forbid (macOS:
    /Library/Application Support/ClaudeCode/managed-settings.json and
    managed-mcp.json)."""
    reasons = []
    s = _load_json(paths.managed_dir / "managed-settings.json") or {}
    if s.get("disableAllHooks") is True:
        reasons.append("hooks: managed settings disable all hooks")
    elif s.get("allowManagedHooksOnly") is True:
        reasons.append("hooks: managed settings allow only managed hooks")
    allowed = s.get("allowedMcpServers")
    if isinstance(allowed, list) and not any(
            isinstance(e, dict) and e.get("serverName") == MCP_NAME for e in allowed):
        reasons.append(f"mcp: {MCP_NAME} is not in allowedMcpServers")
    denied = s.get("deniedMcpServers")
    if isinstance(denied, list) and any(
            isinstance(e, dict) and e.get("serverName") == MCP_NAME for e in denied):
        reasons.append(f"mcp: {MCP_NAME} is in deniedMcpServers")
    if (paths.managed_dir / "managed-mcp.json").exists():
        reasons.append("mcp: managed-mcp.json takes exclusive control of MCP servers")
    return {"blocked": bool(reasons), "reasons": reasons,
            "mcp_blocked": any(r.startswith("mcp:") for r in reasons),
            "hook_blocked": any(r.startswith("hooks:") for r in reasons)}


# ---------------------------------------------------------------- MCP entry

def _mcp_entry(paths: ClaudePaths) -> Optional[dict]:
    d = _load_json(paths.claude_json) or {}
    srv = (d.get("mcpServers") or {}).get(MCP_NAME)
    return srv if isinstance(srv, dict) else None


def _is_our_mcp(entry: Optional[dict]) -> bool:
    if not entry:
        return False
    cmd = entry.get("command") or ""
    args = " ".join(entry.get("args") or [])
    return Path(cmd).name == "contorch-mcp" or "context_orchestrator.server" in args


def desired_mcp_env(current: Optional[dict]) -> dict:
    env = dict((current or {}).get("env") or {})
    keep = {k: v for k, v in env.items() if k in ALLOWED_ENV}
    if _in_process_index():
        for k in SERVER_ONLY_ENV:
            keep.pop(k, None)
    return keep


def mcp_status(paths: ClaudePaths, eps: dict) -> dict:
    cur = _mcp_entry(paths)
    if not cur:
        return {"present": False, "path": None, "matches": False, "env_keys": []}
    want_env = desired_mcp_env(cur)
    env = cur.get("env") or {}
    matches = (cur.get("command") == eps["mcp"] and not (cur.get("args") or [])
               and env == want_env and cur.get("type", "stdio") == "stdio")
    return {"present": True, "path": cur.get("command"), "matches": matches,
            "env_keys": sorted(env)}


def _run_claude(claude: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([claude, *args], capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL)


def mcp_install(paths: ClaudePaths, eps: dict, claude: Optional[str]) -> tuple[dict, list, list]:
    """(status, changed, todo)"""
    st = mcp_status(paths, eps)
    if st["matches"]:
        return st, [], []
    if not claude:
        return st, [], [f"Install Claude Code, then run `{eps['memory']} claude install` "
                        f"(MCP server {eps['mcp']})"]
    prev = _mcp_entry(paths)
    env = desired_mcp_env(prev)
    # `claude mcp add` of an existing name fails, so remove first; the name
    # must come before the variadic -e options.
    if prev:
        _run_claude(claude, ["mcp", "remove", "--scope", "user", MCP_NAME])
    argv = ["mcp", "add", "--scope", "user", MCP_NAME]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    r = _run_claude(claude, argv + ["--", eps["mcp"]])
    if r.returncode != 0:
        if prev:   # never leave Claude Code with no server
            _run_claude(claude, ["mcp", "add-json", "--scope", "user", MCP_NAME, json.dumps(prev)])
        raise ClaudeInstallError("mcp_failed", "claude mcp add failed: "
                                 + (r.stderr or r.stdout).strip()[-300:])
    return mcp_status(paths, eps), ["mcp"], []


def mcp_uninstall(paths: ClaudePaths, claude: Optional[str]) -> tuple[list, list]:
    cur = _mcp_entry(paths)
    if not cur:
        return [], []
    if not _is_our_mcp(cur):
        return [], [f"`{MCP_NAME}` in Claude Code points at {cur.get('command')!r}, not Contorch; left alone"]
    if not claude:
        return [], [f"Install or locate Claude Code, then: claude mcp remove --scope user {MCP_NAME}"]
    _run_claude(claude, ["mcp", "remove", "--scope", "user", MCP_NAME])
    return (["mcp"] if not _mcp_entry(paths) else []), []


# ---------------------------------------------------------------- hook

def hook_command(hook_path: str, chan: str) -> str:
    """The guarded hook command (Claude Code runs it with /bin/sh). Lab-proven:
    an unguarded missing path errors on every prompt and an unquoted path with
    a space fails; this form is silent when the file is gone."""
    q = shlex.quote(str(hook_path))
    return f"[ -x {q} ] && {q}; exit 0 # {HOOK_TAG}{chan}"


def _first_token(command: str) -> str:
    try:
        return shlex.split(command)[0]
    except (ValueError, IndexError):
        return (command or "").split(" ")[0]


def _is_tagged(command: str) -> bool:
    return bool(re.search(re.escape(HOOK_TAG) + r"\w+\s*$", command or ""))


def _legacy_hook_path(command: str) -> str:
    """The script an old hook entry runs. install-claude-context.sh wrote the
    bare path, unquoted, so a path with a space is the whole command."""
    c = (command or "").strip()
    if c.endswith(LEGACY_HOOK_NAMES) and not c.startswith(("'", '"')):
        return c
    return _first_token(c)


def _legacy_hook_state(paths: ClaudePaths, command: str) -> Optional[str]:
    """For a command running an old auto-context.py: "ours" (missing file, a
    symlink, or a copy we shipped) or "custom". None if it isn't one."""
    if _is_tagged(command):
        return None
    tok = os.path.expanduser(_legacy_hook_path(command))
    if Path(tok).name not in LEGACY_HOOK_NAMES:
        return None
    p = Path(tok)
    if p.is_symlink() or not p.exists():
        return "ours"
    try:
        return "ours" if _sha(p.read_bytes()) in legacy_hashes()["auto-context.py"] else "custom"
    except OSError:
        return "custom"


def _hook_entries(settings: dict) -> list[tuple[dict, dict]]:
    out = []
    for block in ((settings.get("hooks") or {}).get("UserPromptSubmit") or []):
        for h in block.get("hooks") or []:
            if h.get("type") == "command":
                out.append((block, h))
    return out


def hook_status(paths: ClaudePaths, eps: dict, chan: str) -> dict:
    s = _load_json(paths.settings)
    if s is None:
        return {"present": False, "matches": False, "legacy_copy": "none", "settings_invalid": True}
    want = hook_command(eps["hook"], chan)
    tagged = [h for _b, h in _hook_entries(s) if _is_tagged(h.get("command", ""))]
    legacy = [_legacy_hook_state(paths, h.get("command", "")) for _b, h in _hook_entries(s)]
    legacy = [x for x in legacy if x]
    present = bool(tagged)
    matches = (len(tagged) == 1 and tagged[0].get("command") == want
               and tagged[0].get("timeout") == HOOK_TIMEOUT_S and not legacy)
    state = "kept_custom" if "custom" in legacy else ("found" if legacy else "none")
    if not legacy and paths.legacy_hook.exists() and not paths.legacy_hook.is_symlink():
        try:
            known = _sha(paths.legacy_hook.read_bytes()) in legacy_hashes()["auto-context.py"]
        except OSError:
            known = False
        state = "found" if known else "none"
    return {"present": present, "matches": matches, "legacy_copy": state,
            "command": tagged[0].get("command") if tagged else None}


def _edit_hooks(paths: ClaudePaths, s: dict, add: Optional[str], drop) -> list[str]:
    hooks = s.setdefault("hooks", {})
    blocks = hooks.get("UserPromptSubmit") or []
    removed = []
    for b in blocks:
        keep = []
        for h in b.get("hooks") or []:
            if h.get("type") == "command" and drop(h.get("command", "")):
                removed.append(h.get("command", ""))
            else:
                keep.append(h)
        b["hooks"] = keep
    blocks = [b for b in blocks if b.get("hooks")]
    if add:
        blocks.append({"matcher": "*", "hooks": [{"type": "command", "command": add,
                                                  "timeout": HOOK_TIMEOUT_S}]})
    if blocks:
        hooks["UserPromptSubmit"] = blocks
    else:
        hooks.pop("UserPromptSubmit", None)
    if not hooks:
        s.pop("hooks", None)
    return removed


def _remove_legacy_hook_file(paths: ClaudePaths) -> bool:
    p = paths.legacy_hook
    if p.is_symlink():
        p.unlink()
        return True
    if p.exists():
        try:
            if _sha(p.read_bytes()) in legacy_hashes()["auto-context.py"]:
                p.unlink()
                return True
        except OSError:
            pass
    return False


def hook_install(paths: ClaudePaths, eps: dict, chan: str, backup_dir: Optional[Path]
                 ) -> tuple[dict, list, list]:
    st = hook_status(paths, eps, chan)
    todo: list = []
    if st.get("settings_invalid"):
        return st, [], [f"{paths.settings} is not valid JSON; fix it, then run claude install again"]
    s = _load_json(paths.settings)
    legacy_custom = any(_legacy_hook_state(paths, h.get("command", "")) == "custom"
                        for _b, h in _hook_entries(s))
    changed = []
    if legacy_custom:
        # A customised copy of the old hook stays, and so does its entry:
        # adding ours would inject the context twice.
        todo.append(f"Your customised {paths.legacy_hook} is kept; delete it (and its settings "
                    f"entry) to switch to contorch-hook, then run claude install again")
        if _remove_legacy_hook_file(paths):
            changed.append("legacy_hook_file")
        st = hook_status(paths, eps, chan)
        st["legacy_copy"] = "kept_custom"
        return st, changed, todo
    if not st["matches"]:
        want = hook_command(eps["hook"], chan)
        _backup(paths.settings, backup_dir)
        removed = _edit_hooks(paths, s, want, lambda cmd: _is_tagged(cmd)
                              or _legacy_hook_state(paths, cmd) == "ours")
        _write_atomic(paths.settings, json.dumps(s, indent=2) + "\n")
        changed.append("hook")
        if any(not _is_tagged(c) for c in removed):
            st_legacy = "replaced"
        else:
            st_legacy = "none"
    else:
        st_legacy = "none"
    if _remove_legacy_hook_file(paths):
        changed.append("legacy_hook_file")
        st_legacy = "replaced"
    st = hook_status(paths, eps, chan)
    st["legacy_copy"] = st_legacy
    if not Path(eps["hook"]).exists():
        todo.append(f"{eps['hook']} is missing — reinstall context-orchestrator (the hook entry is "
                    "guarded, so prompts are unaffected)")
    return st, changed, todo


def hook_uninstall(paths: ClaudePaths, backup_dir: Optional[Path]) -> tuple[list, list]:
    s = _load_json(paths.settings)
    changed = []
    if s is None:
        return [], [f"{paths.settings} is not valid JSON; hook entries not removed"]
    if any(_is_tagged(h.get("command", "")) or _legacy_hook_state(paths, h.get("command", "")) == "ours"
           for _b, h in _hook_entries(s)):
        _backup(paths.settings, backup_dir)
        _edit_hooks(paths, s, None, lambda cmd: _is_tagged(cmd)
                    or _legacy_hook_state(paths, cmd) == "ours")
        _write_atomic(paths.settings, json.dumps(s, indent=2) + "\n")
        changed.append("hook")
    if _remove_legacy_hook_file(paths):
        changed.append("legacy_hook_file")
    return changed, []


# ---------------------------------------------------------------- CLAUDE.md

def md_template() -> str:
    return (TEMPLATES / "claude-md-block.md").read_text().strip()


def _block_spans(text: str, begin: str, end: str) -> list[tuple[int, int]]:
    """(start, stop) of each begin…end block, end line included."""
    spans = []
    pos = 0
    while True:
        i = text.find(begin, pos)
        if i < 0:
            return spans
        j = text.find(end, i)
        if j < 0:
            return spans
        k = text.find("\n", j)
        stop = len(text) if k < 0 else k
        spans.append((i, stop))
        pos = stop


def _md_parts(text: str) -> dict:
    known = set(legacy_hashes()["claude_md_block"])
    ours = _block_spans(text, MD_BEGIN, MD_END)
    legacy = []
    for b, e in LEGACY_MD_MARKERS:
        for i, j in _block_spans(text, b, e):
            legacy.append((i, j, _sha(text[i:j].strip().encode()) in known))
    inner = md_template()[len(MD_BEGIN):-len(MD_END)].strip()
    rest = text
    for i, j in sorted([(i, j) for i, j in ours] + [(i, j) for i, j, _ in legacy], reverse=True):
        rest = rest[:i] + rest[j:]
    low = rest.lower()
    return {"ours": ours, "legacy": sorted(legacy), "unmarked_exact": inner in rest,
            "user_guidance": "context-orchestrator" in low or "context orchestrator" in low}


def claude_md_status(paths: ClaudePaths) -> dict:
    try:
        text = paths.claude_md.read_text()
    except FileNotFoundError:
        text = ""
    p = _md_parts(text)
    present = bool(p["ours"])
    matches = (len(p["ours"]) == 1 and text[p["ours"][0][0]:p["ours"][0][1]].strip() == md_template()
               and not any(known for _i, _j, known in p["legacy"]))
    legacy = "kept_custom" if any(not k for _i, _j, k in p["legacy"]) else \
        ("found" if p["legacy"] else "none")
    return {"present": present, "matches": matches, "legacy_block": legacy,
            "user_guidance": p["user_guidance"]}


def claude_md_install(paths: ClaudePaths, backup_dir: Optional[Path]) -> tuple[dict, list, list]:
    st = claude_md_status(paths)
    if st["matches"]:
        return st, [], []
    try:
        text = paths.claude_md.read_text()
    except FileNotFoundError:
        text = ""
    p = _md_parts(text)
    block = md_template()
    todo = []
    custom_legacy = [s for s in p["legacy"] if not s[2]]
    # Drop known legacy blocks and extra copies of ours, remember where the
    # first one was, and put exactly one current block there.
    cuts = [(i, j) for i, j, known in p["legacy"] if known] + p["ours"]
    at = min((i for i, _j in cuts), default=None)
    new = text
    for i, j in sorted(cuts, reverse=True):
        new = new[:i] + new[j:]
        if at is not None and i < at:
            at -= (j - i)
    if at is not None:
        new = new[:at] + block + new[at:]
    elif custom_legacy:
        todo.append(f"{paths.claude_md} has an edited older Contorch block; kept as is "
                    "(delete it and run claude install again to get the current one)")
    elif p["unmarked_exact"]:
        inner = block[len(MD_BEGIN):-len(MD_END)].strip()
        new = new.replace(inner, block, 1)       # adopt today's unmarked copy
    elif p["user_guidance"]:
        todo.append(f"{paths.claude_md} already has your own context-orchestrator guidance; "
                    "left as is (no second copy added)")
    else:
        sep = "" if not new or new.endswith("\n\n") else ("\n" if new.endswith("\n") else "\n\n")
        new = new + sep + block + "\n"
    new = re.sub(r"\n{3,}", "\n\n", new)
    if new != text:
        _backup(paths.claude_md, backup_dir)
        _write_atomic(paths.claude_md, new)
        return claude_md_status(paths), ["claude_md"], todo
    return claude_md_status(paths), [], todo


def claude_md_uninstall(paths: ClaudePaths, backup_dir: Optional[Path]) -> tuple[list, list]:
    try:
        text = paths.claude_md.read_text()
    except FileNotFoundError:
        return [], []
    p = _md_parts(text)
    cuts = p["ours"] + [(i, j) for i, j, known in p["legacy"] if known]
    if not cuts:
        return [], []
    new = text
    for i, j in sorted(cuts, reverse=True):
        new = new[:i] + new[j:]
    new = re.sub(r"\n{3,}", "\n\n", new).strip("\n")
    _backup(paths.claude_md, backup_dir)
    _write_atomic(paths.claude_md, (new + "\n") if new else "")
    return ["claude_md"], []


# ---------------------------------------------------------------- skill

def render_skill(eps: dict) -> str:
    return (TEMPLATES / "transcripts-skill.md.in").read_text().replace(
        "{{CONTORCH_TRANSCRIPTS}}", shlex.quote(eps["transcripts"]))


def _skill_owner(paths: ClaudePaths) -> str:
    """"none" | "symlink" | "stamped" | "legacy" (a copy we shipped) | "user"."""
    d = paths.skill_dir
    if d.is_symlink():
        return "symlink"
    if not d.exists():
        return "none"
    if (d / SKILL_STAMP).exists():
        return "stamped"
    try:
        if _sha((d / "SKILL.md").read_bytes()) in legacy_hashes()["transcripts_skill"]:
            return "legacy"
    except OSError:
        pass
    return "user"


def skill_status(paths: ClaudePaths, eps: dict, chan: str) -> dict:
    owner = _skill_owner(paths)
    present = owner == "stamped"
    matches = False
    if present:
        try:
            stamp = json.loads((paths.skill_dir / SKILL_STAMP).read_text())
            matches = ((paths.skill_dir / "SKILL.md").read_text() == render_skill(eps)
                       and stamp.get("channel") == chan)
        except (OSError, ValueError):
            matches = False
    return {"present": present, "matches": matches, "path": str(paths.skill_dir),
            "user_copy": owner == "user"}


def skill_install(paths: ClaudePaths, eps: dict, chan: str) -> tuple[dict, list, list]:
    st = skill_status(paths, eps, chan)
    if st["matches"]:
        return st, [], []
    owner = _skill_owner(paths)
    if owner == "user":
        return st, [], [f"{paths.skill_dir} is your own copy; kept (delete it and run claude "
                        "install again to get the managed one)"]
    d = paths.skill_dir
    if owner == "symlink":
        d.unlink()
    d.mkdir(parents=True, exist_ok=True)
    _write_atomic(d / "SKILL.md", render_skill(eps))
    _write_atomic(d / SKILL_STAMP, json.dumps({"owner": "context-orchestrator", "channel": chan,
                                               "transcripts": eps["transcripts"],
                                               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                              indent=2) + "\n")
    return skill_status(paths, eps, chan), ["skill"], []


def skill_uninstall(paths: ClaudePaths) -> tuple[list, list]:
    owner = _skill_owner(paths)
    d = paths.skill_dir
    if owner == "symlink":
        d.unlink()
        return ["skill"], []
    if owner in ("stamped", "legacy"):
        for f in ("SKILL.md", SKILL_STAMP):
            (d / f).unlink(missing_ok=True)
        try:
            d.rmdir()
        except OSError:
            return ["skill"], [f"{d} had other files; they were kept"]
        return ["skill"], []
    return [], []


# ---------------------------------------------------------------- verbs

def _doc(ok: bool, chan: str, eps: dict, paths: ClaudePaths, mcp: dict, hook: dict, md: dict,
         skill: dict, policy: dict, todo: list, changed: list, action: str,
         error: Optional[dict] = None) -> dict:
    doc = {"schema": SCHEMA, "ok": ok, "action": action, "channel": chan,
           "mcp": mcp, "hook": hook, "claude_md": md, "skill": skill,
           "blocked_by_managed_settings": policy["blocked"], "managed_reasons": policy["reasons"],
           "todo": todo, "changed": changed,
           "paths": {"config_dir": str(paths.config_dir), "claude_json": str(paths.claude_json),
                     **{k: v for k, v in eps.items()}}}
    if error:
        doc["error"] = error
    return doc


def status(chan: Optional[str] = None, paths: Optional[ClaudePaths] = None) -> dict:
    chan = chan or channel()
    paths = paths or ClaudePaths.detect()
    eps = entry_points(chan)
    policy = managed_policy(paths)
    mcp, hook = mcp_status(paths, eps), hook_status(paths, eps, chan)
    md, skill = claude_md_status(paths), skill_status(paths, eps, chan)
    todo = [f"blocked by managed settings — {r}" for r in policy["reasons"]]
    ok = mcp["matches"] and hook["matches"] and md["matches"] and skill["matches"]
    return _doc(ok, chan, eps, paths, mcp, hook, md, skill, policy, todo, [], "status")


def install(chan: Optional[str] = None, hook: bool = True, backup_dir: Optional[Path] = None,
            paths: Optional[ClaudePaths] = None) -> dict:
    chan = chan or channel()
    paths = paths or ClaudePaths.detect()
    eps = entry_points(chan)
    policy = managed_policy(paths)
    todo = [f"blocked by managed settings — {r}" for r in policy["reasons"]]
    changed: list = []
    error = None
    mcp_st = mcp_status(paths, eps)
    if not policy["mcp_blocked"]:
        try:
            mcp_st, ch, td = mcp_install(paths, eps, find_claude())
            changed += ch
            todo += td
        except ClaudeInstallError as exc:
            error = {"code": exc.code, "message": str(exc)}
    if hook and not policy["hook_blocked"]:
        hook_st, ch, td = hook_install(paths, eps, chan, backup_dir)
        changed += ch
        todo += td
    else:
        hook_st = hook_status(paths, eps, chan)
    md_st, ch, td = claude_md_install(paths, backup_dir)
    changed += ch
    todo += td
    skill_st, ch, td = skill_install(paths, eps, chan)
    changed += ch
    todo += td
    return _doc(error is None, chan, eps, paths, mcp_st, hook_st, md_st, skill_st, policy,
                todo, changed, "install", error)


def uninstall(chan: Optional[str] = None, backup_dir: Optional[Path] = None,
              paths: Optional[ClaudePaths] = None) -> dict:
    chan = chan or channel()
    paths = paths or ClaudePaths.detect()
    eps = entry_points(chan)
    policy = managed_policy(paths)
    changed, todo = [], []
    for ch, td in (mcp_uninstall(paths, find_claude()), hook_uninstall(paths, backup_dir),
                   claude_md_uninstall(paths, backup_dir), skill_uninstall(paths)):
        changed += ch
        todo += td
    return _doc(True, chan, eps, paths, mcp_status(paths, eps), hook_status(paths, eps, chan),
                claude_md_status(paths), skill_status(paths, eps, chan), policy, todo, changed,
                "uninstall")


def describe(doc: dict) -> str:
    """Human summary (stderr / text mode)."""
    def mark(part):
        return "✓" if part.get("matches") else ("·" if not part.get("present") else "!")
    lines = [f"Claude Code integration ({doc['channel']}, {doc['action']}):",
             f"  {mark(doc['mcp'])} MCP server   {doc['mcp'].get('path') or 'not registered'}",
             f"  {mark(doc['hook'])} hook         {'installed' if doc['hook']['present'] else 'not installed'}"
             + (f" (old copy: {doc['hook']['legacy_copy']})" if doc['hook'].get('legacy_copy') not in (None, 'none') else ""),
             f"  {mark(doc['claude_md'])} CLAUDE.md    {'block present' if doc['claude_md']['present'] else 'no Contorch block'}",
             f"  {mark(doc['skill'])} skill        {doc['skill']['path'] if doc['skill']['present'] else 'not installed'}"]
    if doc["changed"]:
        lines.append("  changed: " + ", ".join(doc["changed"]))
    for t in doc["todo"]:
        lines.append("  todo: " + t)
    if doc.get("error"):
        lines.append(f"  error: {doc['error']['message']}")
    return "\n".join(lines)
