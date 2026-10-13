"""EmbeddingGemma 2 as the on-device model: choice resolution, collection
naming, the download (verify / resume / atomic install) against a local fake
server, and — when the real files are available — tokenizer parity and
embedding fidelity.

Real-model tests need the published release files in a folder:
    CO_TEST_EG2_RELEASE_DIR=/path/with/eg2-mixed.onnx,eg2-mixed.onnx_data,tokenizer.model
(`gh release download models-eg2-int8-v1 -D DIR`), or only the tokenizer:
    CO_TEST_EG2_TOKENIZER=/path/to/tokenizer.model
Fixtures are synthetic (tests/fixtures/eg2/texts.json); the real-corpus
benchmark lives in test_local_model_bench.py and runs locally only.
"""
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from context_orchestrator import local_model as lm
from context_orchestrator import memstatus, search, settings

FIX = Path(__file__).parent / "fixtures" / "eg2"
SRC = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path / "models" / lm.RELEASE_TAG


def _ready(monkeypatch, ready: bool):
    monkeypatch.setattr(lm, "is_ready", lambda directory=None: ready)


def _gemini(monkeypatch, available: bool):
    monkeypatch.setattr(search, "_gemini_available", lambda: available)


# ---- choice resolution ---------------------------------------------------------

@pytest.mark.parametrize("raw,choice", [
    ("", ""), ("local", "local"), ("off", "local"), ("default", "local"),
    ("eg2", "embeddinggemma-2"), ("EmbeddingGemma-2", "embeddinggemma-2"),
    ("local-gemma", "embeddinggemma-2"), ("embeddinggemma-2-int8-768", "embeddinggemma-2"),
    ("minilm", "minilm"), ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),   # still a sentence-transformers name
    ("gemini", "gemini-embedding-001"), ("none", "none"), ("keyword", "none"),
])
def test_embedding_choice_aliases(monkeypatch, raw, choice):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", raw)
    assert search.embedding_choice() == choice


@pytest.mark.parametrize("choice,gemini,ready,backend,identity", [
    ("local", False, False, "minilm", "default"),
    ("local", False, True, "eg2", "embeddinggemma-2-int8-768"),
    ("local", True, True, "eg2", "embeddinggemma-2-int8-768"),     # an explicit local ignores the key
    ("", False, False, "minilm", "default"),                     # unset, no key = local
    ("", False, True, "eg2", "embeddinggemma-2-int8-768"),
    ("", True, True, "gemini-embedding-001", "gemini-gemini-embedding-001"),   # key = Gemini, as before
    ("embeddinggemma-2", False, False, "eg2-pending", "none"),   # keyword search until verified
    ("embeddinggemma-2", True, True, "eg2", "embeddinggemma-2-int8-768"),
    ("minilm", False, True, "minilm", "default"),
    ("none", False, True, "none", "none"),
    ("gemini-embedding-001", False, True, "gemini-embedding-001", "gemini-gemini-embedding-001"),
])
def test_resolve_backend(monkeypatch, choice, gemini, ready, backend, identity):
    _gemini(monkeypatch, gemini)
    _ready(monkeypatch, ready)
    assert search.resolve_backend(choice) == backend
    assert search.backend_identity(backend) == identity
    assert memstatus.effective_identity(choice) == identity


def test_wants_local_model(monkeypatch):
    _gemini(monkeypatch, True)
    assert search.wants_local_model("local") and search.wants_local_model("embeddinggemma-2")
    assert not search.wants_local_model("") and not search.wants_local_model("minilm")
    _gemini(monkeypatch, False)
    assert search.wants_local_model("")


def test_build_embedding_function(monkeypatch, models):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    assert search._build_embedding_function() is None            # MiniLM until verified
    _ready(monkeypatch, True)
    ef = search._build_embedding_function()
    assert isinstance(ef, lm.EmbeddingGemmaEF) and ef.name() == "embeddinggemma-2-int8-768"
    assert search.ef_identity(ef) == lm.MODEL_NAME


def test_explicit_eg2_is_keyword_search_until_verified(monkeypatch, tmp_path, models):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "eg2")
    vs = search.VectorSearch(chroma_path=tmp_path / "chroma")
    assert not vs.enabled and vs.waiting_for_model and vs.identity == "none"
    assert vs.search("anything") == [] and vs.count() == 0


def test_collection_naming(monkeypatch, tmp_path, models):
    """One collection per model: the first keeps "context", EG2 gets its own."""
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    monkeypatch.setattr(search, "_build_embedding_function", lambda: None)
    first = search.VectorSearch(chroma_path=tmp_path / "chroma")
    assert first.identity == "default" and first.collection_name == "context"
    monkeypatch.undo()
    monkeypatch.setenv("CO_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    _ready(monkeypatch, True)
    eg2 = search.VectorSearch(chroma_path=tmp_path / "chroma")     # empty collection: no embedding at init
    assert eg2.identity == lm.MODEL_NAME
    assert eg2.collection_name == "context-embeddinggemma-2-int8-768"
    mapping = json.loads((tmp_path / "chroma" / "contorch-collections.json").read_text())
    assert mapping == {"default": "context", lm.MODEL_NAME: "context-embeddinggemma-2-int8-768"}
    # memstatus finds the same collection without chromadb
    assert memstatus._collection_name(tmp_path / "chroma", lm.MODEL_NAME) == "context-embeddinggemma-2-int8-768"


def test_fresh_install_eg2_first_takes_context(monkeypatch, tmp_path, models):
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    _ready(monkeypatch, True)
    assert search.VectorSearch(chroma_path=tmp_path / "chroma").collection_name == "context"


# ---- contorch-memory embeddings / status ------------------------------------------

@pytest.mark.parametrize("arg,stored", [("eg2", "embeddinggemma-2"), ("embeddinggemma-2", "embeddinggemma-2"),
                                        ("local-gemma", "embeddinggemma-2"), ("minilm", "minilm"),
                                        ("local", "local")])
def test_set_embeddings(tmp_path, arg, stored, models):
    env = tmp_path / "env"
    out = settings.set_embeddings(arg, env)
    assert f"CO_EMBEDDING_MODEL={stored}" in env.read_text().splitlines()
    if stored != "minilm":
        assert "contorch-memory model download" in out


def test_describe(monkeypatch, models):
    assert "MiniLM" in settings.describe("local") and "not downloaded" in settings.describe("local")
    assert "keyword search until then" in settings.describe("embeddinggemma-2")
    _ready(monkeypatch, True)
    monkeypatch.setattr(lm, "state", lambda: {"state": "verified"})
    assert settings.describe("local").startswith("local — on this Mac, no key, no network: EmbeddingGemma 2")
    assert "until" not in settings.describe("embeddinggemma-2")


def test_cli_accepts_new_choices():
    for c in ("eg2", "embeddinggemma-2", "local-gemma", "minilm"):
        assert c in settings.CHOICES


def test_status_json_reports_the_download_without_heavy_imports(tmp_path, models):
    models.mkdir(parents=True)
    (models / "eg2-mixed.onnx_data.part").write_bytes(b"x" * 1000)
    env = dict(os.environ, PYTHONPATH=str(SRC), CO_DB_PATH=str(tmp_path / "c.db"),
               CO_CHROMA_PATH=str(tmp_path / "chroma"), CO_EMBEDDING_MODEL="local",
               CO_MODELS_DIR=str(tmp_path / "models"))
    r = subprocess.run([sys.executable, "-c",
                        "import sys; from context_orchestrator.settings import main; rc = main(['status', '--json']);"
                        "print([m for m in ('chromadb', 'onnxruntime', 'sentencepiece') if m in sys.modules], file=sys.stderr);"
                        "sys.exit(rc)"], env=env, capture_output=True, text=True, timeout=120)
    doc = json.loads(r.stdout)
    assert r.stderr.strip().endswith("[]"), r.stderr
    assert doc["embeddings"] == "local" and doc["embedding_identity"] == "default"
    assert doc["embedding_target"] == lm.MODEL_NAME
    assert doc["local_model"]["state"] == "partial" and doc["local_model"]["bytes_done"] == 1000
    assert doc["local_model"]["bytes_total"] == lm.total_bytes()


# ---- download against a local fake server -------------------------------------------

class _Server:
    """Serves `files`; honours Range unless `ranges=False`; `cut_after[name]`
    drops the connection after that many bytes on the next request."""

    def __init__(self, files):
        self.files = dict(files)
        self.ranges = True
        self.cut_after: dict[str, int] = {}
        self.requests: list[tuple[str, str]] = []
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.rsplit("/", 1)[-1]
                rng = self.headers.get("Range") or ""
                srv.requests.append((name, rng))
                body = srv.files.get(name)
                if body is None:
                    self.send_error(404)
                    return
                start = 0
                if rng and srv.ranges:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(body):
                        self.send_response(416)
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                data = body[start:]
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                cut = srv.cut_after.pop(name, None)
                self.wfile.write(data if cut is None else data[:cut])
                if cut is not None:
                    self.wfile.flush()
                    self.connection.shutdown(2)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/dl"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake(monkeypatch, models):
    files = {"eg2-mixed.onnx": os.urandom(20_000), "eg2-mixed.onnx_data": os.urandom(300_000),
             "tokenizer.model": os.urandom(5_000)}
    monkeypatch.setattr(lm, "FILES", {n: (len(b), hashlib.sha256(b).hexdigest()) for n, b in files.items()})
    monkeypatch.setattr(lm, "CHUNK", 4096)
    monkeypatch.setattr(lm, "PROGRESS_EVERY_S", 0.0)
    srv = _Server(files)
    monkeypatch.setenv("CO_MODEL_BASE_URL", srv.url)
    yield srv
    srv.close()


def test_download_verifies_and_installs(fake, models):
    seen = []
    assert lm.state()["state"] == "absent"
    d = lm.download(progress=seen.append)
    assert d == models and lm.is_ready()
    for name, body in fake.files.items():
        assert (models / name).read_bytes() == body and not (models / (name + ".part")).exists()
    assert (models / "LICENSE").read_text().startswith("\n                                 Apache License")
    assert "EmbeddingGemma 2" in (models / "NOTICE").read_text()
    assert json.loads((models / "verified.json").read_text())["files"] == lm._pins()
    st = lm.state()
    assert st["state"] == "verified" and st["percent"] == 100.0
    assert seen[-1]["state"] == "verified" and any(p["state"] == "downloading" for p in seen)
    n = len(fake.requests)
    lm.download()                                       # already verified: no network
    assert len(fake.requests) == n


def test_interrupted_download_resumes_with_range(fake, models):
    fake.cut_after["eg2-mixed.onnx_data"] = 100_000
    with pytest.raises(lm.DownloadError):
        lm.download()
    part = models / "eg2-mixed.onnx_data.part"
    have = part.stat().st_size
    assert 0 < have < 300_000 and not (models / "eg2-mixed.onnx_data").exists()   # never a half file in place
    assert not lm.is_ready()
    st = lm.state()
    assert st["state"] == "failed" and "eg2-mixed.onnx_data" in st["error"]
    assert lm.recently_failed()
    lm.download()
    assert ("eg2-mixed.onnx_data", f"bytes={have}-") in fake.requests
    assert (models / "eg2-mixed.onnx_data").read_bytes() == fake.files["eg2-mixed.onnx_data"]
    assert lm.is_ready() and not lm.recently_failed()
    # the file finished before the cut wasn't fetched again
    assert [r for r in fake.requests if r[0] == "eg2-mixed.onnx"] == [("eg2-mixed.onnx", "")]


def test_server_without_range_support_restarts_the_file(fake, models):
    fake.cut_after["eg2-mixed.onnx_data"] = 50_000
    with pytest.raises(lm.DownloadError):
        lm.download()
    fake.ranges = False
    lm.download()
    assert (models / "eg2-mixed.onnx_data").read_bytes() == fake.files["eg2-mixed.onnx_data"]


def test_checksum_mismatch_is_rejected_and_removed(fake, models):
    fake.files["tokenizer.model"] = b"y" * 5_000          # right size, wrong bytes
    with pytest.raises(lm.DownloadError, match="checksum mismatch"):
        lm.download()
    assert not (models / "tokenizer.model").exists() and not (models / "tokenizer.model.part").exists()
    assert not lm.is_ready() and lm.state()["state"] == "failed"


def test_missing_asset_fails_cleanly(fake, models):
    del fake.files["eg2-mixed.onnx"]
    with pytest.raises(lm.DownloadError, match="HTTP 404"):
        lm.download()
    assert lm.state()["state"] == "failed"


def test_ready_needs_the_marker_and_the_sizes(fake, models):
    lm.download()
    with open(models / "eg2-mixed.onnx_data", "ab") as fh:
        fh.write(b"!")
    assert not lm.is_ready()
    lm.download()                                          # re-verifies, replaces the bad file
    assert lm.is_ready()
    (models / "verified.json").unlink()
    assert not lm.is_ready()


def test_two_downloaders_fetch_once(fake, models):
    errs = []

    def run():
        try:
            lm.download()
        except Exception as exc:   # pragma: no cover
            errs.append(exc)
    ts = [threading.Thread(target=run) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not errs and lm.is_ready()
    assert sorted(n for n, _ in fake.requests) == sorted(fake.files)


def test_background_download_respects_off_and_backoff(fake, monkeypatch, models):
    assert os.environ.get("CO_MODEL_DOWNLOAD") == "off"      # conftest: never in tests by default
    assert lm.ensure_downloading() is False and fake.requests == []
    monkeypatch.setenv("CO_MODEL_DOWNLOAD", "on")
    done = threading.Event()
    lm.ensure_downloading(on_ready=done.set)
    assert done.wait(30) and lm.is_ready()
    assert lm.ensure_downloading() is True


# ---- the MCP server switches once the model is verified --------------------------

class _NamedEF:
    def __init__(self, name):
        self._n = name

    def name(self):
        return self._n

    def __call__(self, input):
        return [[float((sum(map(ord, t)) >> i) & 1) + 0.01 for i in range(8)] for t in input]

    def embed_query(self, input):
        return self(input if isinstance(input, list) else [input])


def test_server_reembeds_in_the_background_then_switches(monkeypatch, tmp_path, models):
    from context_orchestrator import server, transcripts as T
    from context_orchestrator.db import Database
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "local")
    db = Database(db_path=tmp_path / "c.db")
    monkeypatch.setattr(server, "db", db)
    monkeypatch.setattr(server, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _NamedEF("default"))
    old = search.VectorSearch(chroma_path=tmp_path / "chroma")
    monkeypatch.setattr(server, "vs", old)
    for i in range(3):
        mid, _ = T.add_text(db, f"[10:00:0{i}] Jane: meeting {i} about the zebra migration",
                            started_at=f"2026-09-2{i}T10:00")
        T.index_row(old, db, db.get_transcript(mid))
    db.conn.execute("UPDATE transcripts SET updated_at = updated_at - 120")
    db.conn.commit()
    server._maybe_switch_model()                       # not verified yet: nothing happens
    assert server._switch_thread is None or not server._switch_thread.is_alive()
    assert server.vs is old
    _ready(monkeypatch, True)
    monkeypatch.setattr(search, "_build_embedding_function", lambda: _NamedEF(lm.MODEL_NAME))
    monkeypatch.setattr(server, "_switch_failed_at", None)
    assert server._switch_model(poll_s=0.01) is True
    assert server.vs is not old and server.vs.identity == lm.MODEL_NAME
    assert server.vs.collection_name == "context-embeddinggemma-2-int8-768"
    assert server.vs.count() >= 3
    assert {r["indexed_with"] for r in db.conn.execute("SELECT indexed_with FROM transcripts")} == {lm.MODEL_NAME}
    assert old.count() >= 3                             # the MiniLM vectors stay for switching back


# ---- real model (skipped unless the files are available) -------------------------

def _release_dir():
    d = os.environ.get("CO_TEST_EG2_RELEASE_DIR")
    return Path(d).expanduser() if d and Path(d, lm.ONNX_FILE).exists() else None


def _tokenizer_path():
    t = os.environ.get("CO_TEST_EG2_TOKENIZER")
    if t and Path(t).exists():
        return Path(t)
    d = _release_dir()
    return d / lm.TOKENIZER_FILE if d else None


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """The real files installed the way users get them (download() from a
    file:// mirror of the release: verified, renamed into place)."""
    rel = _release_dir()
    if rel is None:
        pytest.skip("set CO_TEST_EG2_RELEASE_DIR to the models-eg2-int8-v1 release files")
    root = tmp_path_factory.mktemp("eg2-models")
    saved = {k: os.environ.get(k) for k in ("CO_MODELS_DIR", "CO_MODEL_BASE_URL")}
    os.environ["CO_MODELS_DIR"] = str(root)
    os.environ["CO_MODEL_BASE_URL"] = rel.as_uri()
    lm.download()
    yield root / lm.RELEASE_TAG
    lm.release()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_tokenizer_parity_with_hugging_face():
    tok = _tokenizer_path()
    if tok is None:
        pytest.skip("set CO_TEST_EG2_TOKENIZER (or CO_TEST_EG2_RELEASE_DIR)")
    assert hashlib.sha256(tok.read_bytes()).hexdigest() == lm.FILES["tokenizer.model"][1]
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    cases = json.loads((FIX / "tokenizer_parity.json").read_text())["cases"]
    assert len(cases) >= 20
    for c in cases:
        assert lm.token_ids(c["text"], sp) == c["ids"], c["text"]
    # Literal "<bos>"/"<eos>" in a transcript stay text (HF would inject the
    # special tokens): no special id but the wrapping ones.
    for t in json.loads((FIX / "texts.json").read_text())["literal_specials"]:
        ids = lm.token_ids(t, sp)
        assert ids[0] == lm.BOS and ids[-1] == lm.EOS and not {0, 1, 2} & set(ids[1:-1])
    long = "zebra migration " * 3000
    ids = lm.token_ids(long, sp)
    assert len(ids) == lm.MAX_TOKENS and ids[0] == lm.BOS and ids[-1] == lm.EOS


def test_embedding_fidelity_vs_reference(installed):
    import numpy as np
    texts = json.loads((FIX / "texts.json").read_text())
    ref = np.load(FIX / "reference.npz")
    ef = lm.EmbeddingGemmaEF()
    D = np.stack(ef.embed_documents([d["text"] for d in texts["corpus"]]))
    Q = np.stack(ef.embed_query([q["query"] for q in texts["queries"]]))
    assert D.shape == (10, 768) and Q.dtype == np.float32
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0, atol=1e-4)
    for X, R in ((D, ref["D"]), (Q, ref["Q"])):
        assert float((X * R).sum(1).min()) > 0.999


def test_synthetic_retrieval_through_vectorsearch(installed, monkeypatch, tmp_path):
    monkeypatch.setenv("CO_MODELS_DIR", str(installed.parent))
    monkeypatch.setenv("CO_EMBEDDING_MODEL", "eg2")
    texts = json.loads((FIX / "texts.json").read_text())
    vs = search.VectorSearch(chroma_path=tmp_path / "chroma")
    assert vs.identity == lm.MODEL_NAME and vs.collection_name == "context"
    vs.upsert([d["id"] for d in texts["corpus"]], [d["text"] for d in texts["corpus"]],
              [{"type": "x"}] * len(texts["corpus"]))
    rr = []
    for q in texts["queries"]:
        ids = [h["id"] for h in vs.search(q["query"], n_results=10)]
        rr.append(1 / (ids.index(q["relevant"]) + 1) if q["relevant"] in ids else 0)
    assert sum(rr) / len(rr) >= 0.8, rr
