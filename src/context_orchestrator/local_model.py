"""EmbeddingGemma 2 — the on-device embedding model ("local").

What runs: Google's EmbeddingGemma 2 text graph (Apache-2.0), quantized by
Contorch to int8 matmuls + a 4-bit token-embedding table (~243 MB; cosine
0.993 min / 0.999 mean against the fp32 reference), executed with
onnxruntime only — no torch, no transformers. The tokenizer is the model's
own sentencepiece file; ids are wrapped `[2] + ids + [1]` (<bos> … <eos>),
identical to the Hugging Face tokenizer.

Files are not shipped in the wheel. They are downloaded once, on first use,
from a pinned GitHub release asset of contorch/context-orchestrator
(RELEASE_TAG), checked against the sizes and SHA-256 hashes below, into
`~/.context-orchestrator/models/<RELEASE_TAG>/` — or `$CO_MODELS_DIR/<tag>/`
(Contorch.app points that at its Application Support folder). A download
writes `<file>.part`, resumes it with an HTTP Range request after an
interruption, verifies it, and only then renames it into place; one process
downloads at a time (flock), the others wait for it. `verified.json` marks
the set complete; until it exists the model is "not ready" and nothing loads
it (search stays on full-text, or MiniLM for "local").

Progress for `contorch-memory status --json` is in `download.json` next to
the files (state(), below — cheap: no onnxruntime import).
"""
from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("context-orchestrator")

MODEL_NAME = "embeddinggemma-2-int8-768"      # ef.name(): collection + indexed_with
RELEASE_TAG = "models-eg2-int8-v1"
DEFAULT_BASE_URL = ("https://github.com/contorch/context-orchestrator/releases/download/"
                    + RELEASE_TAG)
BASE_URL_ENV = "CO_MODEL_BASE_URL"      # a mirror; hashes are checked whatever the source
MODELS_DIR_ENV = "CO_MODELS_DIR"        # Contorch.app: ~/Library/Application Support/Contorch/models
AUTO_DOWNLOAD_ENV = "CO_MODEL_DOWNLOAD"  # "off": never download in the background

ONNX_FILE = "eg2-mixed.onnx"
TOKENIZER_FILE = "tokenizer.model"
# name -> (size in bytes, sha256). The .onnx graph refers to its weights as
# "eg2-mixed.onnx_data" in the same folder, so the names are part of the pin.
FILES: dict[str, tuple[int, str]] = {
    "eg2-mixed.onnx": (466323, "baa00e2018d549df76b756c310b3f08b8f9d1215cc294b530e02b348a04969a1"),
    "eg2-mixed.onnx_data": (238149632, "89005c0e96b89303495eab1747ae3915b4509068a1c8ccfb5161ff4a6a2fd289"),
    "tokenizer.model": (4689013, "e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03"),
}
LICENSE_FILES = ("LICENSE", "NOTICE")   # shipped in the package, copied next to the weights

DOC_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "
MAX_TOKENS = 2048          # incl. <bos>/<eos>; longer text is truncated
BOS, EOS = 2, 1
DIM = 768
MEDIA_FEATURE_DIM = 512    # the text graph still takes (0, 512) image/video/audio inputs

STATE_FILE = "download.json"
VERIFIED_FILE = "verified.json"
LOCK_FILE = ".download.lock"
RETRY_AFTER_FAILURE_S = 15 * 60
PROGRESS_EVERY_S = 1.0
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


# ---- where ---------------------------------------------------------------------

def models_dir() -> Path:
    v = os.environ.get(MODELS_DIR_ENV)
    return Path(v).expanduser() if v else Path.home() / ".context-orchestrator" / "models"


def model_dir() -> Path:
    return models_dir() / RELEASE_TAG


def base_url() -> str:
    return (os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


def auto_download_enabled() -> bool:
    return (os.environ.get(AUTO_DOWNLOAD_ENV) or "").strip().lower() not in ("off", "0", "no", "false")


def total_bytes() -> int:
    return sum(size for size, _ in FILES.values())


def _pins() -> dict:
    return {name: {"size": size, "sha256": sha} for name, (size, sha) in FILES.items()}


# ---- state (cheap: file stats + two small JSON files) -------------------------

def is_ready(directory: Optional[Path] = None) -> bool:
    """All files present at their pinned sizes and a verified.json that pins
    exactly these hashes. (Hashing happens once, when the download finishes.)"""
    d = directory or model_dir()
    try:
        marker = json.loads((d / VERIFIED_FILE).read_text())
    except (OSError, ValueError):
        return False
    if marker.get("files") != _pins():
        return False
    for name, (size, _) in FILES.items():
        try:
            if (d / name).stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def _read_state(d: Path) -> dict:
    try:
        return json.loads((d / STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}


def _write_state(d: Path, **doc) -> None:
    doc = {"schema": "contorch-model-download/1", "model": MODEL_NAME, "tag": RELEASE_TAG,
           "updated_at": time.time(), **doc}
    tmp = d / f".{STATE_FILE}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, d / STATE_FILE)
    except OSError as exc:
        logger.debug("could not write %s: %s", d / STATE_FILE, exc)


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _bytes_on_disk(d: Path) -> int:
    done = 0
    for name, (size, _) in FILES.items():
        for p in (d / name, d / (name + ".part")):
            try:
                done += min(p.stat().st_size, size)
                break
            except OSError:
                continue
    return done


def state() -> dict:
    """{model, state, bytes_done, bytes_total, percent, error, dir}.
    state: "verified" | "downloading" | "partial" (interrupted; resumes) |
    "failed" | "absent"."""
    d = model_dir()
    total = total_bytes()
    out = {"model": MODEL_NAME, "release": RELEASE_TAG, "dir": str(d), "bytes_total": total,
           "auto_download": auto_download_enabled()}
    if is_ready(d):
        return {**out, "state": "verified", "bytes_done": total, "percent": 100.0, "error": None}
    st = _read_state(d)
    done = _bytes_on_disk(d) if d.exists() else 0
    if st.get("state") == "downloading" and _pid_alive(st.get("pid")):
        done = max(done, int(st.get("bytes_done") or 0))
        s = "downloading"
    elif st.get("state") == "failed":
        s = "failed"
    elif done:
        s = "partial"
    else:
        s = "absent"
    return {**out, "state": s, "bytes_done": done, "percent": round(100.0 * done / total, 1),
            "error": st.get("error") if s == "failed" else None,
            **({"failed_at": st.get("updated_at")} if s == "failed" else {})}


def recently_failed(now: Optional[float] = None) -> bool:
    st = _read_state(model_dir())
    if st.get("state") != "failed":
        return False
    return ((now or time.time()) - float(st.get("updated_at") or 0)) < RETRY_AFTER_FAILURE_S


# ---- download --------------------------------------------------------------------

def _sha256_file(path: Path, h=None):
    h = h or hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h


def _fetch(url: str, part: Path, size: int, sha: str, on_bytes: Callable[[int], None],
           timeout: float) -> None:
    """Download `url` into `part`, resuming what is already there. Raises
    DownloadError on HTTP errors, short reads, size or hash mismatch (and
    removes a .part that can't be right)."""
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    h = hashlib.sha256()
    if have:
        _sha256_file(part, h)
    if have < size:
        req = urllib.request.Request(url, headers={"User-Agent": "context-orchestrator"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:        # nothing left to send: verify what we have
                resp = None
            else:
                raise DownloadError(f"{url}: HTTP {exc.code} {exc.reason}") from exc
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise DownloadError(f"{url}: {getattr(exc, 'reason', exc)}") from exc
        if resp is not None:
            with resp:
                if have and resp.status != 206:   # server ignored Range: start over
                    have, h = 0, hashlib.sha256()
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                on_bytes(have)
                with open(part, mode) as fh:
                    while True:
                        try:
                            block = resp.read(CHUNK)
                        except (OSError, http.client.HTTPException) as exc:
                            raise DownloadError(f"{url}: {exc}") from exc
                        if not block:
                            break
                        if have + len(block) > size:
                            fh.close()
                            part.unlink()
                            raise DownloadError(f"{part.name}: server sent more than {size} bytes")
                        fh.write(block)
                        h.update(block)
                        have += len(block)
                        on_bytes(have)
    if have != size:
        raise DownloadError(f"{part.name}: got {have} of {size} bytes (will resume)")
    if h.hexdigest() != sha:
        part.unlink()
        raise DownloadError(f"{part.name[:-5]}: checksum mismatch (expected {sha[:12]}…, "
                            f"got {h.hexdigest()[:12]}…) — removed, will download again")


def _install_licenses(d: Path) -> None:
    src = Path(__file__).parent / "model_licenses" / "embeddinggemma-2"
    for name in LICENSE_FILES:
        try:
            shutil.copyfile(src / name, d / name)
        except OSError as exc:
            logger.warning("could not copy %s next to the model: %s", name, exc)


@contextlib.contextmanager
def _download_lock(d: Path):
    import fcntl
    with open(d / LOCK_FILE, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def download(progress: Optional[Callable[[dict], None]] = None, timeout: float = 60.0) -> Path:
    """Fetch, verify and install every file; return the model folder. Safe to
    call from several processes: one downloads, the rest wait on the lock and
    then find the files verified. Raises DownloadError (state "failed")."""
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    if is_ready(d):
        return d
    with _download_lock(d):
        if is_ready(d):                     # another process finished while we waited
            return d
        total = total_bytes()
        started = time.time()
        last = [0.0]
        done_before = [0]

        def report(file_bytes: int, name: str, force: bool = False):
            now = time.monotonic()
            if not force and now - last[0] < PROGRESS_EVERY_S:
                return
            last[0] = now
            done = done_before[0] + file_bytes
            doc = {"state": "downloading", "pid": os.getpid(), "file": name, "bytes_done": done,
                   "bytes_total": total, "started_at": started}
            _write_state(d, **doc)
            if progress:
                progress({**doc, "percent": round(100.0 * done / total, 1)})

        try:
            for name, (size, sha) in FILES.items():
                final = d / name
                if final.exists() and final.stat().st_size == size \
                        and _sha256_file(final).hexdigest() == sha:
                    done_before[0] += size
                    continue
                part = d / (name + ".part")
                report(0, name, force=True)
                _fetch(f"{base_url()}/{name}", part, size, sha,
                       lambda n, name=name: report(n, name), timeout)
                os.replace(part, final)            # atomic: final only ever holds verified bytes
                done_before[0] += size
            _install_licenses(d)
            marker = {"schema": "contorch-model/1", "model": MODEL_NAME, "tag": RELEASE_TAG,
                      "files": _pins(), "verified_at": time.time()}
            tmp = d / f".{VERIFIED_FILE}.tmp"
            tmp.write_text(json.dumps(marker, indent=1))
            os.replace(tmp, d / VERIFIED_FILE)
            _write_state(d, state="verified", bytes_done=total, bytes_total=total)
            if progress:
                progress({"state": "verified", "bytes_done": total, "bytes_total": total, "percent": 100.0})
            logger.info("on-device model %s verified in %s", MODEL_NAME, d)
            return d
        except Exception as exc:
            msg = str(exc)[:300] if isinstance(exc, DownloadError) else f"{type(exc).__name__}: {exc}"[:300]
            _write_state(d, state="failed", error=msg, bytes_done=_bytes_on_disk(d), bytes_total=total)
            if isinstance(exc, DownloadError):
                raise
            raise DownloadError(msg) from exc


_bg_lock = threading.Lock()
_bg_thread: Optional[threading.Thread] = None


def ensure_downloading(on_ready: Optional[Callable[[], None]] = None) -> bool:
    """Start a background download if the model isn't ready (at most one
    thread per process; none while a recent failure is backing off, or with
    CO_MODEL_DOWNLOAD=off). Returns is_ready()."""
    global _bg_thread
    if is_ready():
        return True
    if not auto_download_enabled() or recently_failed():
        return False
    with _bg_lock:
        if _bg_thread is not None and _bg_thread.is_alive():
            return False

        def run():
            try:
                download()
            except Exception as exc:
                logger.warning("on-device model download failed (search keeps working; "
                               "retried in %d min): %s", RETRY_AFTER_FAILURE_S // 60, exc)
                return
            if on_ready:
                try:
                    on_ready()
                except Exception:
                    logger.exception("after the model download")
        _bg_thread = threading.Thread(target=run, name="eg2-download", daemon=True)
        _bg_thread.start()
    return False


# ---- inference -------------------------------------------------------------------

_session_lock = threading.Lock()
_SESSION = None
_SP = None
_INPUTS: list[str] = []


def _load(directory: Optional[Path] = None):
    """One onnxruntime session + tokenizer per process (whatever the number of
    VectorSearch objects); released at exit."""
    global _SESSION, _SP, _INPUTS
    with _session_lock:
        if _SESSION is None:
            d = directory or model_dir()
            if not is_ready(d):
                raise RuntimeError(f"the on-device model isn't downloaded and verified yet ({d})")
            import atexit
            import onnxruntime as ort
            import sentencepiece as spm
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            _SP = spm.SentencePieceProcessor(model_file=str(d / TOKENIZER_FILE))
            _SESSION = ort.InferenceSession(str(d / ONNX_FILE), so, providers=["CPUExecutionProvider"])
            _INPUTS = [i.name for i in _SESSION.get_inputs()]
            atexit.register(release)
        return _SESSION, _SP


def release() -> None:
    """Drop the session before interpreter teardown so onnxruntime's worker
    threads are joined while the runtime is intact (see search._release_default_ef)."""
    global _SESSION, _SP
    with _session_lock:
        _SESSION = None
        _SP = None
    import gc
    gc.collect()


def token_ids(text: str, sp=None) -> list[int]:
    """[<bos>] + sentencepiece ids + [<eos>], capped at MAX_TOKENS."""
    sp = sp or _load()[1]
    return [BOS] + sp.encode(text)[:MAX_TOKENS - 2] + [EOS]


def embed_one(text: str):
    """Normalised 768-d vector for one already-prefixed text (batch size 1 is
    the fastest and caps memory)."""
    import numpy as np
    sess, sp = _load()
    ids = np.asarray([token_ids(text, sp)], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    empty = np.zeros((0, MEDIA_FEATURE_DIM), dtype=np.float32)
    for name in ("image_features", "video_features", "audio_features"):
        if name in _INPUTS:
            feed[name] = empty
    vec = sess.run(["sentence_embedding"], feed)[0][0].astype(np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


class EmbeddingGemmaEF:
    """Chroma EmbeddingFunction (same protocol as the Gemini one): documents
    and queries get EmbeddingGemma's retrieval prompts."""

    def name(self) -> str:
        return MODEL_NAME

    def __call__(self, input):
        return self.embed_documents(input)

    def embed_documents(self, input):
        texts = input if isinstance(input, list) else [input]
        return [embed_one(DOC_PREFIX + t) for t in texts]

    def embed_query(self, input):
        texts = input if isinstance(input, list) else [input]
        return [embed_one(QUERY_PREFIX + t) for t in texts]
