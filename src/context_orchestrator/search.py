import contextlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from .chroma_lock import DEFAULT_TIMEOUT_S, LockTimeout, lock_path_for, session_lock

logger = logging.getLogger("context-orchestrator")

DEFAULT_CHROMA_PATH = Path.home() / ".context-orchestrator" / "chroma"
DEFAULT_CHROMA_HOST = "127.0.0.1"
DEFAULT_CHROMA_PORT = 8765

# Optional embedding-model override. When unset, Chroma's default
# all-MiniLM-L6-v2 is used (384d, 256-token cap, no extra deps).
#
# Two upgrade paths:
#   1) Local sentence-transformers (e.g. nomic — 768d, 8k context):
#        pip install -e '.[embeddings]'
#        export CO_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
#   2) Hosted Gemini (3072d, top-MTEB, asymmetric retrieval):
#        pip install -e '.[embeddings-gemini]'
#        export CO_EMBEDDING_MODEL=gemini-embedding-001
#        # API key from GOOGLE_API_KEY/GEMINI_API_KEY env or
#        # ~/.config/google/key (mode 600)
#
# Switching invalidates existing vectors — wipe the collection /
# chroma path first so HNSW dimensions match.
EMBEDDING_MODEL_ENV = "CO_EMBEDDING_MODEL"
GEMINI_KEY_FILE = Path.home() / ".config" / "google" / "key"

# MMR re-rank knobs. Lambda 1.0 = pure relevance (no diversity); 0.0 = pure
# diversity (ignore relevance). 0.7 is empirically a good balance for
# transcript-heavy corpora — keeps top-3 strictly on-topic, then opens up.
DEFAULT_MMR_LAMBDA = 0.7
# How many candidates to fetch from Chroma before MMR re-ranking. 3-5x the
# requested n_results gives MMR room to spread.
MMR_CANDIDATE_MULTIPLIER = 3
MMR_CANDIDATE_MIN = 30

# RRF (reciprocal rank fusion) combines dense and BM25 rankings.
# Per the canonical RRF paper, k=60 is a robust default.
DEFAULT_RRF_K = 60
# How many candidates each retriever should pull before fusion.
HYBRID_FETCH_PER_RETRIEVER = 50

# Optional LLM re-rank. When `rerank=True` is passed to `search()` (or the
# MCP tool exposes it), top-N candidates are sent to the configured LLM
# for relevance scoring; results are re-ordered by score.
#
# Default model is read from CO_RERANK_MODEL. If unset, no LLM rerank is
# performed even when `rerank=True` is requested (returns base ranking).
RERANK_MODEL_ENV = "CO_RERANK_MODEL"
# How many candidates to pull through the LLM. Bigger = higher quality
# top-K but more tokens spent. 30 matched empirical sweet-spot in eval.
RERANK_FETCH = 30


def _cosine(a, b) -> float:
    """Cosine similarity between two embedding vectors."""
    import numpy as np  # local import to keep search.py importable without numpy
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _resolve_gemini_api_key() -> Optional[str]:
    """Look up the Gemini API key in env first, then ~/.config/google/key."""
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    if GEMINI_KEY_FILE.exists():
        try:
            return GEMINI_KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return None


def embedding_choice() -> str:
    """The configured embedding model, normalised:
      "none"                 → no embeddings; search is full-text (FTS5) only
      "local"                → Chroma's built-in all-MiniLM-L6-v2 (offline, 384d)
      "gemini-embedding-001" → Gemini (needs a key; "gemini" is an alias)
      any other name         → a sentence-transformers model
      ""                     → unset: Gemini if a key + google-genai exist, else local
    Set with `contorch-memory embeddings gemini|local|none`
    (~/.context-orchestrator/env)."""
    v = (os.environ.get(EMBEDDING_MODEL_ENV) or "").strip()
    low = v.lower()
    if low in ("none", "fts", "keyword"):
        return "none"
    if low in ("off", "default", "local"):   # "off" meant "not Gemini" before 0.4
        return "local"
    if low == "gemini":
        return "gemini-embedding-001"
    return v


def _build_embedding_function():
    """Construct a Chroma EmbeddingFunction for the user-configured model.

    Resolution order:
      1. CO_EMBEDDING_MODEL set to a model name → use it.
      2. CO_EMBEDDING_MODEL set to "off" / "none" / "default" → force the
         local Chroma default (returns None).
      3. CO_EMBEDDING_MODEL unset → AUTO-DETECT: if a Gemini API key is
         resolvable AND google-genai is importable, default to
         gemini-embedding-001. Otherwise fall back to local default.

    The auto-detect is what lets the watcher (with explicit env in its
    plist) and the MCP server (which inherits whatever the spawning client
    happens to pass) converge to the same EF without manual sync — as
    long as the key file is present, both will pick Gemini. Setting
    CO_EMBEDDING_MODEL=off explicitly opts out.
    """
    model_name = embedding_choice()
    if model_name == "none":
        raise RuntimeError("embeddings are turned off (CO_EMBEDDING_MODEL=none)")
    if model_name == "local":
        logger.info(f"{EMBEDDING_MODEL_ENV}=local: chroma default (all-MiniLM-L6-v2, 384d)")
        return None
    if not model_name:
        # Auto-detect Gemini availability
        try:
            import google.genai  # noqa: F401
            if _resolve_gemini_api_key():
                model_name = "gemini-embedding-001"
                logger.info(
                    f"{EMBEDDING_MODEL_ENV} unset; auto-detected Gemini "
                    f"(key present + google-genai installed) → {model_name}. "
                    f"Set {EMBEDDING_MODEL_ENV}=off to force local default."
                )
        except ImportError:
            pass
    if not model_name:
        logger.info(
            f"{EMBEDDING_MODEL_ENV} unset and Gemini unavailable; "
            "using chroma default (local 384d)"
        )
        return None
    if model_name.startswith("gemini-"):
        return _build_gemini_embedding_function(model_name)
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as e:
        raise RuntimeError(
            f"{EMBEDDING_MODEL_ENV}={model_name} requires the 'embeddings' "
            "extra. Install with: pip install -e '.[embeddings]'"
        ) from e
    # trust_remote_code is required for nomic-embed and similar custom-arch
    # models. It's safe here because the model name is user-controlled and
    # they explicitly opted in via env var.
    try:
        return SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            trust_remote_code=True,
        )
    except ValueError as e:
        # Newer chromadb imports sentence_transformers lazily and raises
        # ValueError at construction rather than ImportError at import.
        if "sentence_transformers" not in str(e):
            raise
        raise RuntimeError(
            f"{EMBEDDING_MODEL_ENV}={model_name} requires the 'embeddings' "
            "extra. Install with: pip install -e '.[embeddings]'"
        ) from e


def _build_gemini_embedding_function(model_name: str):
    """Construct a Chroma-compatible EmbeddingFunction backed by Gemini.

    Uses asymmetric task_types: RETRIEVAL_DOCUMENT for indexing,
    RETRIEVAL_QUERY for query-time. Gemini's embedding API benefits
    measurably from this separation.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            f"{EMBEDDING_MODEL_ENV}={model_name} requires the "
            "'embeddings-gemini' extra. Install with: "
            "pip install -e '.[embeddings-gemini]'"
        ) from e
    api_key = _resolve_gemini_api_key()
    # No key is a supported setup (a machine that only imports embedding
    # bundles made elsewhere): the server still starts, vectors come from the
    # bundles, and search falls back to keyword matching. Only an actual
    # embedding call reports the missing key.
    client = genai.Client(api_key=api_key) if api_key else None

    import numpy as np

    def _embed(texts, task_type: str):
        if client is None:
            raise RuntimeError(
                f"{EMBEDDING_MODEL_ENV}={model_name} needs a Gemini API key to embed. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY, or write the key to "
                f"{GEMINI_KEY_FILE} (mode 600)."
            )
        result = client.models.embed_content(
            model=model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [np.asarray(e.values, dtype=np.float32) for e in result.embeddings]

    class _GeminiEF:
        """Chroma EmbeddingFunction protocol. Newer Chroma (>=1.x) calls
        embed_documents at insert and embed_query at query time; older
        versions just call __call__. Implement all three for compatibility."""
        def name(self) -> str:
            return f"gemini-{model_name}"

        def __call__(self, input):
            return _embed(input, "RETRIEVAL_DOCUMENT")

        def embed_documents(self, input):
            return _embed(input, "RETRIEVAL_DOCUMENT")

        def embed_query(self, input):
            texts = input if isinstance(input, list) else [input]
            return _embed(texts, "RETRIEVAL_QUERY")

    return _GeminiEF()


def _llm_rerank(query: str, candidates: list[dict], n_results: int,
                model: str) -> list[dict]:
    """Re-rank `candidates` by sending (query, chunk) pairs to an LLM and
    sorting by the LLM's relevance score (0-10 scale).

    Currently supports model names starting with "gemini-" (Google).
    Returns up to `n_results` candidates ordered by score desc. On any
    error (parse failure, API error, missing extra), logs a warning and
    falls back to the input ordering — never raises so callers don't have
    to wrap.
    """
    if not candidates:
        return []
    if model.startswith("gemini-"):
        return _llm_rerank_gemini(query, candidates, n_results, model)
    logger.warning(f"unknown rerank model {model!r}; returning base ranking")
    return candidates[:n_results]


def _llm_rerank_gemini(query: str, candidates: list[dict], n_results: int,
                       model: str) -> list[dict]:
    """Gemini-backed implementation of the LLM rerank step. Soft-fails to
    base ordering on any error."""
    import json as _json
    import re as _re
    try:
        from google import genai
    except ImportError:
        logger.warning(
            f"rerank model {model} requires the 'embeddings-gemini' extra; "
            "falling back to base ranking"
        )
        return candidates[:n_results]
    api_key = _resolve_gemini_api_key()
    if not api_key:
        logger.warning(
            "no Gemini API key found for rerank; falling back to base ranking"
        )
        return candidates[:n_results]

    client = genai.Client(api_key=api_key)
    fetch_n = min(len(candidates), RERANK_FETCH)
    block = "\n".join(
        f"[{i}] {(c.get('text') or '')[:240].replace(chr(10), ' ')}"
        for i, c in enumerate(candidates[:fetch_n])
    )
    prompt = (
        f"Re-rank these search candidates for the query:\n\nQUERY: {query}\n\n"
        f"CANDIDATES:\n{block}\n\n"
        "For each candidate, score 0-10 for how directly it answers the "
        "query (10=direct answer, 7-9=strongly relevant, 4-6=adjacent, "
        "0-3=off-topic or noise). Return ONLY a JSON array like "
        '[{"i": 0, "score": 8}, ...]. No prose, no markdown.'
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        text = resp.text or ""
    except Exception as e:
        logger.warning(f"rerank API call failed ({e}); falling back to base ranking")
        return candidates[:n_results]

    try:
        m = _re.search(r"\[.*\]", text, _re.DOTALL)
        if not m:
            raise ValueError("no JSON array in response")
        decisions = _json.loads(m.group(0))
        scores = {int(d["i"]): float(d["score"]) for d in decisions if "i" in d}
    except Exception as e:
        logger.warning(f"rerank parse failed ({e}); falling back to base ranking")
        return candidates[:n_results]

    # Reorder by score; tied/missing candidates keep their original position
    indexed = list(enumerate(candidates[:fetch_n]))
    indexed.sort(key=lambda x: (-scores.get(x[0], -1.0), x[0]))
    return [c for _, c in indexed[:n_results]]


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Good enough for technical proper-noun queries
    where exact matches dominate."""
    return re.findall(r"\w+", text.lower())


def _rrf_fuse(rankings: list[list[str]], k: int = DEFAULT_RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Each input is an ordered list of doc ids.
    Output is sorted by sum(1 / (k + rank)) descending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _mmr_select(candidates: list[dict], k: int, lam: float) -> list[dict]:
    """Maximal Marginal Relevance re-ranking.

    Each candidate is a dict with at least 'sim_to_query' (float in [0,1])
    and 'embedding' (list[float]). Returns up to k candidates ordered by
    `lam * sim_to_query - (1 - lam) * max(sim_to_already_selected)`.
    """
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        best_i, best_score = remaining[0], -float("inf")
        for ri in remaining:
            relev = candidates[ri]["sim_to_query"]
            penalty = 0.0
            if selected:
                penalty = max(
                    _cosine(candidates[ri]["embedding"], candidates[s]["embedding"])
                    for s in selected
                )
            score = lam * relev - (1 - lam) * penalty
            if score > best_score:
                best_score, best_i = score, ri
        selected.append(best_i)
        remaining.remove(best_i)
    return [candidates[i] for i in selected]


def _embedded_path_if_no_server() -> Optional[Path]:
    """Which Chroma to use when the caller didn't say.

    CO_CHROMA_PATH → that folder, in-process. CO_CHROMA_HOST/PORT set, or the
    chroma launchd server installed → the server (HTTP). Otherwise — a
    lightweight install with no server — the default folder, in-process.
    Several processes (one MCP server per Claude Code session, the CLI) can
    open the same folder at once.
    """
    env_path = os.environ.get("CO_CHROMA_PATH")
    if env_path:
        return Path(env_path).expanduser()
    if os.environ.get("CO_CHROMA_HOST") or os.environ.get("CO_CHROMA_PORT"):
        return None
    from .chroma_daemon import LAUNCHD_PLIST
    return None if LAUNCHD_PLIST.exists() else DEFAULT_CHROMA_PATH


def ef_identity(ef) -> str:
    """Stable name of an embedding function — tags bundles, names the
    collection, and records which model indexed each transcript."""
    return ef.name() if ef is not None and hasattr(ef, "name") else "default"


def _matches_where(meta: dict, where: dict) -> bool:
    """Evaluate the subset of Chroma `where` syntax that search() produces."""
    if "$and" in where:
        return all(_matches_where(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(_matches_where(meta, w) for w in where["$or"])
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, ref in cond.items():
                if val is None:
                    return False
                if op == "$gte" and not val >= ref: return False
                if op == "$lte" and not val <= ref: return False
                if op == "$gt" and not val > ref: return False
                if op == "$lt" and not val < ref: return False
                if op == "$eq" and val != ref: return False
                if op == "$ne" and val == ref: return False
        elif val != cond:
            return False
    return True


INDEX_STAMP_FILE = "contorch-index.json"


def chromadb_version() -> Optional[str]:
    """The chromadb installed for THIS interpreter, from its dist-info (no
    import of chromadb itself)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("chromadb")
    except Exception:  # PackageNotFoundError, or a broken dist-info
        return None


def read_index_stamp(chroma_path: Path) -> dict:
    """`<chroma>/contorch-index.json`: which chromadb last wrote this folder."""
    try:
        return json.loads((Path(chroma_path) / INDEX_STAMP_FILE).read_text())
    except (OSError, ValueError):
        return {}


def _write_index_stamp(chroma_path: Path) -> None:
    """Record the writing chromadb version. Called inside the session lock
    after every write; rewritten only when it changes."""
    stamp = {"schema": "contorch-index/1", "chromadb_version": chromadb_version()}
    if read_index_stamp(chroma_path) == stamp:
        return
    target = Path(chroma_path) / INDEX_STAMP_FILE
    tmp = target.with_name(f".{INDEX_STAMP_FILE}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(stamp, indent=2, sort_keys=True))
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("could not write %s: %s", target, exc)


def _clear_chroma_system_cache() -> None:
    """chromadb keeps one System per path for the life of the process
    (SharedSystemClient._identifier_to_system). Dropping it is what makes the
    next PersistentClient re-read the folder instead of serving a stale (and,
    for writers, divergent) in-memory index."""
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient.clear_system_cache()


_DEFAULT_EF = None


def _shared_default_ef():
    """One MiniLM (ONNX) session per process, however many VectorSearch
    objects exist: the model is ~80 MB and each session owns a thread pool."""
    global _DEFAULT_EF
    if _DEFAULT_EF is None:
        import atexit
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _DEFAULT_EF = DefaultEmbeddingFunction()
        atexit.register(_release_default_ef)
    return _DEFAULT_EF


def _release_default_ef() -> None:
    """Drop the ONNX session before interpreter teardown, so its worker
    threads are joined while the runtime is intact. Left to static
    destructors, a worker sometimes locks a destroyed mutex and the process
    aborts at exit (libc++abi "recursive_mutex lock failed", exit 134)."""
    global _DEFAULT_EF
    _DEFAULT_EF = None
    import gc
    gc.collect()


class VectorSearch:
    """The vector index (Chroma). Two modes:

    - HTTP: talks to the `context-orchestrator-chroma` server (launchd) or
      CO_CHROMA_HOST/PORT. The server serialises access itself, so one
      long-lived client is fine.
    - In-process (a lightweight install with no server, CO_CHROMA_PATH, tests):
      a PersistentClient on a folder that several processes open. chromadb is
      not process-safe there, so EVERY operation runs in a session
      (`with vs.session() as col:`): take the cross-process lock
      (chroma_lock), drop chromadb's cached System, open fresh, operate, drop,
      unlock. No client is held between sessions. Embeddings (the slow, and
      for Gemini networked, part) are computed BEFORE the lock is taken.

    `vs.collection` is only valid inside a session in in-process mode.
    """

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        lock_timeout: Optional[float] = DEFAULT_TIMEOUT_S,
        verify: bool = True,
    ):
        if chroma_path is None and host is None and port is None:
            chroma_path = _embedded_path_if_no_server()
        self.chroma_path = Path(chroma_path) if chroma_path is not None else None
        self.lock_timeout = lock_timeout
        if self.chroma_path is not None:
            self.host = None
            self.port = None
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self.lock_path: Optional[Path] = lock_path_for(self.chroma_path)
        else:
            self.host = host or os.environ.get("CO_CHROMA_HOST", DEFAULT_CHROMA_HOST)
            self.port = int(port if port is not None else os.environ.get("CO_CHROMA_PORT", DEFAULT_CHROMA_PORT))
            self.lock_path = None
        # Lazy BM25 index — built on first hybrid search, refreshed when
        # the caller invalidates explicitly.
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._col = None            # the open collection (HTTP: always; in-process: inside a session)
        self._client = None
        self._name: Optional[str] = None
        self._session_wrote = False
        # CO_EMBEDDING_MODEL=none: no vector index at all. Every method is a
        # no-op / empty; search runs on SQLite full-text search instead.
        self.enabled = embedding_choice() != "none"
        if not self.enabled:
            self.identity = "none"
            self._ef = None
            logger.info("Vector search off (CO_EMBEDDING_MODEL=none) — full-text search only")
            return
        self._ef = _build_embedding_function()
        self.identity = ef_identity(self._ef)
        if self.chroma_path is None:
            self._connect()
            logger.info(f"Vector search initialized at http://{self.host}:{self.port}, collection "
                        f"{self._col.name!r} for {self.identity} ({self._col.count()} docs)")
        else:
            logger.info(f"Vector search in-process at {self.chroma_path} for {self.identity} "
                        f"(session lock {self.lock_path})")
        if verify:
            self._verify_embedding_dim()

    # ---- sessions ------------------------------------------------------------

    @property
    def in_process(self) -> bool:
        return self.chroma_path is not None

    @property
    def collection(self):
        """The Chroma collection. In-process mode: only inside `session()`."""
        if self.in_process and self._col is None and self.enabled:
            raise RuntimeError("VectorSearch.collection used outside `with vs.session():` "
                               "(in-process Chroma must not be touched without the session lock)")
        return self._col

    @property
    def collection_name(self) -> Optional[str]:
        if not self.enabled:
            return None
        if self._name is None:
            self._name = self._collection_name(self.identity)
        return self._name

    @contextlib.contextmanager
    def session(self, write: bool = False, timeout: Optional[float] = None):
        """Yield the collection with exclusive access (in-process mode) or the
        long-lived HTTP collection. `write=True` stamps the chromadb version
        into the folder when the session ends. Raises chroma_lock.LockTimeout
        if the lock can't be had within `timeout` (default: self.lock_timeout)."""
        if not self.enabled:
            yield None
            return
        if not self.in_process:
            yield self._col
            return
        if self._col is not None:          # nested session of this instance
            self._session_wrote = self._session_wrote or write
            yield self._col
            return
        to = self.lock_timeout if timeout is None else timeout
        with session_lock(self.lock_path, timeout=to,
                          on_first_acquire=_clear_chroma_system_cache,
                          on_last_release=_clear_chroma_system_cache):
            self._session_wrote = write
            try:
                self._open_local()
                yield self._col
            finally:
                try:
                    if self._session_wrote:
                        _write_index_stamp(self.chroma_path)
                finally:
                    self._col = None
                    self._client = None
                    self._session_wrote = False

    def _open_local(self) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._col = self._client.get_or_create_collection(**self._collection_kwargs())

    def _collection_kwargs(self) -> dict:
        kwargs: dict = {"name": self.collection_name, "metadata": {"hnsw:space": "cosine"}}
        if self._ef is not None:
            kwargs["embedding_function"] = self._ef
        return kwargs

    def _connect(self) -> None:
        """HTTP mode: one long-lived client (the server serialises access)."""
        self._client = self._http_client_with_retry()
        self._col = self._client.get_or_create_collection(**self._collection_kwargs())

    # ---- embeddings (always computed outside the lock) ------------------------

    @property
    def embedding_function(self):
        """The function that makes this collection's vectors (Chroma's
        built-in MiniLM when no model is configured)."""
        if self._ef is not None:
            return self._ef
        return _shared_default_ef()

    def embed_documents(self, texts: list[str]) -> list:
        """Document vectors, exactly as Chroma would compute them at upsert."""
        if not texts:
            return []
        return [list(map(float, v)) for v in self.embedding_function(input=list(texts))]

    def embed_query(self, text: str) -> list:
        """A query vector, exactly as Chroma would compute it at query time
        (Gemini uses a different task type for queries)."""
        ef = self.embedding_function
        if hasattr(ef, "embed_query"):
            out = ef.embed_query(input=[text])
        else:
            out = ef(input=[text])
        return list(map(float, out[0]))

    def _verify_embedding_dim(self) -> None:
        """Compare the configured EF's output dim against the dim of vectors
        already stored in the collection. Mismatches are silent footguns:
        upserts go through OK but every query returns a 400. Surface it
        loudly at startup with the exact remediation steps.
        """
        if not self.enabled:
            return
        try:
            with self.session() as col:
                if col.count() == 0:
                    return  # Empty collection takes whatever the EF produces.
                stored = col.get(limit=1, include=["embeddings"])
            embs = stored.get("embeddings")
            if embs is None or len(embs) == 0:
                return
            stored_dim = len(embs[0])
        except Exception as e:
            logger.warning(f"Could not read collection dim for verification: {e}")
            return
        try:
            ef_dim = len(self.embed_documents(["dim probe"])[0])   # outside the lock
        except Exception as e:
            logger.warning(f"Could not probe EF dim for verification: {e}")
            return
        if ef_dim != stored_dim:
            ef_name = type(self.embedding_function).__name__
            logger.error(
                f"EMBEDDING DIM MISMATCH: collection has {stored_dim}d vectors "
                f"but the configured embedding function ({ef_name}) produces "
                f"{ef_dim}d. Every query and most upserts will 400. "
                f"Fix one of: (a) set CO_EMBEDDING_MODEL to match the model "
                f"that built the collection ({stored_dim}d), (b) switch with "
                f"`contorch-memory embeddings …` (one collection per model), or "
                f"(c) `contorch-transcripts reindex` after wiping the index."
            )

    def _collection_name(self, identity: str) -> str:
        """One collection per embedding model, so switching models never mixes
        vector spaces and switching back reuses the old vectors. The first
        model seen keeps the historical name "context" (existing installs
        keep their index); later ones get "context-<model>". The mapping lives
        next to the Chroma data."""
        base = self.chroma_path or DEFAULT_CHROMA_PATH
        map_file = Path(base) / "contorch-collections.json"
        try:
            mapping = json.loads(map_file.read_text())
        except (OSError, ValueError):
            mapping = {}
        if identity in mapping:
            return mapping[identity]
        if "context" in mapping.values():
            name = "context-" + re.sub(r"[^a-zA-Z0-9._-]+", "-", identity).strip("-._")[:50]
        else:
            name = "context"   # fresh install, or the pre-0.4 collection built by this model
        mapping[identity] = name
        try:
            map_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = map_file.with_name(f".{map_file.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(mapping, indent=2, sort_keys=True))
            os.replace(tmp, map_file)
        except OSError:
            pass
        return name

    def _http_client_with_retry(self):
        """Retry chromadb.HttpClient with exponential backoff.

        At cold-boot or fresh-Claude-Code launch, the chroma launchd daemon
        is still binding port 8765 (~20-30s after login) while the MCP
        server is already trying to connect. Without a retry we crash with
        "Connection refused" on the very first attempt and the MCP server
        stays dead until the user manually /mcp reconnects.

        Retry sleeps: 0.5, 1, 2, 4, 8, 15 — total ~30s, which covers a
        typical chroma cold-start window. Each individual attempt is fast
        because the failure mode is a TCP refused, not a hang.
        """
        import time as _t
        import chromadb
        last_err: Exception | None = None
        for delay in (0.5, 1.0, 2.0, 4.0, 8.0, 15.0):
            try:
                return chromadb.HttpClient(host=self.host, port=self.port)
            except Exception as e:
                # chromadb raises ValueError for "Could not connect" and
                # httpx propagates ConnectError; either means the daemon
                # isn't ready yet. Anything else (auth, schema, etc.) we
                # want to surface immediately.
                msg = str(e).lower()
                if "could not connect" not in msg and "connection refused" not in msg:
                    raise
                last_err = e
                _t.sleep(delay)
        raise RuntimeError(
            f"chroma daemon at {self.host}:{self.port} did not become reachable "
            f"after ~30s. Is `context-orchestrator-chroma status` healthy?"
        ) from last_err

    def reload(self) -> None:
        """Make other processes' writes visible. In-process mode needs nothing
        (every session opens fresh); the BM25 cache is dropped either way."""
        if not self.enabled:
            return
        if not self.in_process:
            self._connect()
        self.invalidate_bm25()

    def invalidate_bm25(self) -> None:
        """Force the next hybrid search to rebuild the BM25 index. Call this
        after batch writes if you can't wait for the natural reload."""
        self._bm25 = None
        self._bm25_ids = []

    def _ensure_bm25(self, col) -> None:
        """Build (or reuse) an in-memory BM25 index over every document in
        the collection. Lazy: only constructed on first hybrid search.
        Called inside a session.
        """
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError(
                "hybrid search requires rank-bm25 — pip install rank-bm25"
            ) from e
        data = col.get(include=["documents"], limit=100000)
        ids = data["ids"]
        docs = data["documents"]
        if not ids:
            # Empty collection — leave _bm25 as None; hybrid degrades to dense
            return
        tokenized = [_bm25_tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_ids = ids
        self._bm25_tokens = [set(t) for t in tokenized]

    # ---- writes ----------------------------------------------------------------

    def add(self, doc_id: str, text: str, metadata: dict) -> None:
        """Add or update a document in the vector index."""
        self.upsert([doc_id], [text], [metadata])

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict],
               embeddings: Optional[list] = None) -> None:
        """Upsert documents. Vectors are computed before the lock is taken
        (or come from a bundle); the session only writes."""
        if not self.enabled or not ids:
            return
        if embeddings is None:
            embeddings = self.embed_documents(documents)
        with self.session(write=True) as col:
            col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        self.invalidate_bm25()

    def remove(self, doc_id: str) -> None:
        """Remove a document from the vector index. Deleting an id that isn't
        there is not an error in Chroma; anything that does fail is logged."""
        self.remove_many([doc_id])

    def remove_many(self, ids: list[str]) -> None:
        if not self.enabled or not ids:
            return
        try:
            with self.session(write=True) as col:
                col.delete(ids=list(ids))
        except Exception:
            logger.exception("could not remove %d id(s) from the vector index (first: %s)",
                             len(ids), ids[0])
        self.invalidate_bm25()

    def all_ids(self) -> list[str]:
        if not self.enabled:
            return []
        with self.session() as col:
            return col.get(include=[])["ids"]

    # ---- search ----------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[dict] = None,
        mmr: bool = False,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        hybrid: bool = False,
        rerank: bool = False,
        rerank_model: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search across all indexed documents.

        Args:
            query: natural-language query string
            n_results: maximum hits to return
            where: optional ChromaDB metadata filter (e.g. time-window, file_path)
            mmr: when True, re-rank with Maximal Marginal Relevance to spread
                results across distinct documents instead of returning many
                near-duplicates from one source. Default False.
            mmr_lambda: MMR trade-off; 1.0 = pure relevance, 0.0 = pure
                diversity. Default 0.7.
            hybrid: when True, run BM25 keyword search alongside dense vector
                search and fuse via Reciprocal Rank Fusion. Helps with
                proper-noun and exact-term queries that pure embeddings miss.
                Default False. Compatible with `mmr` (MMR is applied AFTER
                fusion). Note: incompatible with `where` filtering — hybrid
                runs across the full corpus.
            rerank: when True, send top-N candidates (default 30) through
                an LLM for relevance scoring and re-rank by the LLM's score.
                Particularly good at recognising "no match exists" — assigns
                low scores when nothing in the corpus actually answers the
                query. Default False.
            rerank_model: override the LLM (default reads CO_RERANK_MODEL
                env var, e.g. "gemini-flash-latest"). Soft-fails to base
                ranking if the model can't be reached or no key is set.

        In-process mode raises chroma_lock.LockTimeout when Chroma stays busy
        longer than the lock timeout; callers fall back to full-text search.
        """
        if not self.enabled:
            return []
        rr_model = (rerank_model or os.environ.get(RERANK_MODEL_ENV)) if rerank else None
        try:
            qvec = self.embed_query(query)               # outside the lock
        except Exception as exc:
            # The query couldn't be embedded (no key, invalid key, quota,
            # offline). Keyword matching over the same documents still answers
            # most proper-noun questions, so degrade instead of failing.
            logger.warning("dense search unavailable (%s) — falling back to keyword search",
                           str(exc)[:160])
            with self.session() as col:
                return self._keyword_search(col, query, n_results, where)
        with self.session() as col:
            try:
                base = self._search(col, qvec, query, n_results, where, mmr, mmr_lambda,
                                    hybrid, rr_model)
            except Exception as exc:
                logger.warning("dense search failed (%s) — falling back to keyword search",
                               str(exc)[:160])
                return self._keyword_search(col, query, n_results, where)
        if rr_model:                                      # LLM call: outside the lock
            return _llm_rerank(query, base, n_results, rr_model)
        return base

    def _keyword_search(self, col, query: str, n_results: int, where: Optional[dict]) -> list[dict]:
        """BM25 over every document, filtered by `where` (equality, $gte, $lte,
        $and — the operators search() builds). Called inside a session."""
        self._ensure_bm25(col)
        if self._bm25 is None:
            return []
        terms = _bm25_tokenize(query)
        scores = self._bm25.get_scores(terms)
        # Keep only documents containing a query term (BM25 idf goes negative
        # for terms in most of a small corpus, so the score alone can't say).
        wanted = set(terms)
        order = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i])
                 if wanted & self._bm25_tokens[i]]
        out: list[dict] = []
        for start in range(0, len(order), 200):
            ids = [self._bm25_ids[i] for i in order[start:start + 200]]
            got = col.get(ids=ids, include=["documents", "metadatas"])
            by_id = {id_: (got["documents"][k], got["metadatas"][k]) for k, id_ in enumerate(got["ids"])}
            for id_ in ids:
                if id_ not in by_id:
                    continue
                doc, meta = by_id[id_]
                if where and not _matches_where(meta or {}, where):
                    continue
                out.append({"id": id_, "text": doc, "metadata": meta or {}, "distance": None})
                if len(out) >= n_results:
                    return out
        return out

    def _search(self, col, qvec, query, n_results, where, mmr, mmr_lambda, hybrid, rr_model):
        """Dense (+ BM25) retrieval inside a session. Returns the base ranking;
        an LLM rerank, if any, happens after the session."""
        total = col.count() or 1

        if hybrid:
            return self._hybrid_search(
                col, qvec, query,
                n_results=(RERANK_FETCH if rr_model else n_results),
                mmr=mmr,
                mmr_lambda=mmr_lambda,
            )

        if mmr:
            # If reranking, fetch enough for the rerank stage
            mmr_target = RERANK_FETCH if rr_model else n_results
            fetch_n = min(total, max(MMR_CANDIDATE_MIN, mmr_target * MMR_CANDIDATE_MULTIPLIER))
            kwargs = {
                "query_embeddings": [qvec],
                "n_results": fetch_n,
                "include": ["documents", "metadatas", "embeddings", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = col.query(**kwargs)
            if not results or not results["ids"] or not results["ids"][0]:
                return []
            candidates = []
            for i, doc_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i]
                # cosine distance is in [0, 2]; convert to similarity in [0, 1]
                sim = max(0.0, 1.0 - dist / 2.0)
                candidates.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": dist,
                    "sim_to_query": sim,
                    "embedding": results["embeddings"][0][i],
                })
            reranked = _mmr_select(candidates, mmr_target, lam=mmr_lambda)
            # Strip the embedding before returning — callers don't need it.
            return [
                {k: v for k, v in c.items() if k not in ("embedding", "sim_to_query")}
                for c in reranked
            ]

        # Standard path: raw cosine top-K
        fetch_for_base = RERANK_FETCH if rr_model else n_results
        kwargs = {
            "query_embeddings": [qvec],
            "n_results": min(fetch_for_base, total),
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        return hits

    def _hybrid_search(
        self,
        col,
        qvec,
        query: str,
        n_results: int,
        mmr: bool,
        mmr_lambda: float,
    ) -> list[dict]:
        """Dense + BM25 → RRF fuse → optional MMR → top-K."""
        total = col.count() or 1
        fetch = min(total, HYBRID_FETCH_PER_RETRIEVER)

        # Dense retrieval — also gives us embeddings for an optional MMR step
        include = ["documents", "metadatas", "distances"]
        if mmr:
            include.append("embeddings")
        dense = col.query(query_embeddings=[qvec], n_results=fetch, include=include)
        if not dense or not dense["ids"] or not dense["ids"][0]:
            return []
        dense_ids = dense["ids"][0]
        doc_by_id = {id_: dense["documents"][0][i] for i, id_ in enumerate(dense_ids)}
        meta_by_id = {id_: dense["metadatas"][0][i] for i, id_ in enumerate(dense_ids)}
        dist_by_id = {id_: dense["distances"][0][i] for i, id_ in enumerate(dense_ids)}
        embed_by_id = (
            {id_: dense["embeddings"][0][i] for i, id_ in enumerate(dense_ids)} if mmr else {}
        )

        # BM25 retrieval over the cached corpus
        self._ensure_bm25(col)
        if self._bm25 is None:
            # Empty collection — degrade to dense only
            return [
                {"id": id_, "text": doc_by_id[id_], "metadata": meta_by_id[id_],
                 "distance": dist_by_id[id_]}
                for id_ in dense_ids[:n_results]
            ]
        bm25_scores = self._bm25.get_scores(_bm25_tokenize(query))
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:fetch]
        bm25_ids = [self._bm25_ids[i] for i in bm25_top_idx]

        # Fuse rankings
        fused = _rrf_fuse([dense_ids, bm25_ids])
        if not fused:
            return []

        # Hydrate any BM25-only ids that aren't in dense's payload
        missing = [id_ for id_, _ in fused if id_ not in doc_by_id]
        if missing:
            extra_include = ["documents", "metadatas"] + (["embeddings"] if mmr else [])
            extra = col.get(ids=missing, include=extra_include)
            for i, id_ in enumerate(extra["ids"]):
                doc_by_id[id_] = extra["documents"][i]
                meta_by_id[id_] = extra["metadatas"][i]
                if mmr and extra.get("embeddings") is not None:
                    embed_by_id[id_] = extra["embeddings"][i]
                # No distance for BM25-only docs — use the RRF score as a
                # rough sim signal for downstream MMR
                dist_by_id.setdefault(id_, None)

        if mmr:
            # Re-rank top-30 of fused candidates with MMR
            top_for_mmr = fused[: max(MMR_CANDIDATE_MIN, n_results * MMR_CANDIDATE_MULTIPLIER)]
            candidates = []
            for id_, rrf_score in top_for_mmr:
                if id_ not in embed_by_id:
                    continue
                d = dist_by_id.get(id_)
                sim = max(0.0, 1.0 - d / 2.0) if d is not None else rrf_score
                candidates.append({
                    "id": id_,
                    "text": doc_by_id[id_],
                    "metadata": meta_by_id[id_],
                    "distance": d,
                    "sim_to_query": sim,
                    "embedding": embed_by_id[id_],
                })
            reranked = _mmr_select(candidates, n_results, lam=mmr_lambda)
            return [
                {k: v for k, v in c.items() if k not in ("embedding", "sim_to_query")}
                for c in reranked
            ]

        # No MMR — just take the top-N fused
        out = []
        for id_, _rrf_score in fused[:n_results]:
            out.append({
                "id": id_,
                "text": doc_by_id.get(id_, ""),
                "metadata": meta_by_id.get(id_, {}),
                "distance": dist_by_id.get(id_),
            })
        return out

    def count(self) -> int:
        if not self.enabled:
            return 0
        with self.session() as col:
            return col.count()
