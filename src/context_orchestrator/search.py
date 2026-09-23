import logging
import os
import re
from pathlib import Path
from typing import Optional

import chromadb

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
    model_name = os.environ.get(EMBEDDING_MODEL_ENV)
    if model_name and model_name.lower() in ("off", "none", "default", "local"):
        logger.info(f"{EMBEDDING_MODEL_ENV}={model_name}: forcing chroma default (local 384d)")
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
    if not api_key:
        raise RuntimeError(
            f"{EMBEDDING_MODEL_ENV}={model_name} needs a Gemini API key. "
            "Set GOOGLE_API_KEY or GEMINI_API_KEY, or write the key to "
            f"{GEMINI_KEY_FILE} (mode 600)."
        )
    client = genai.Client(api_key=api_key)

    import numpy as np

    def _embed(texts, task_type: str):
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


class VectorSearch:
    """Connects to Chroma. Two modes:

    - HTTP (default): connects to a `chroma run` daemon. Multiple processes can
      share one index without SQLite-lock contention. Override host/port via the
      `CO_CHROMA_HOST` / `CO_CHROMA_PORT` env vars.
    - Local persistent (tests, migration): pass `chroma_path` to use a
      `PersistentClient` against an on-disk directory. Same on-disk format as
      the daemon, so data is portable between modes.
    """

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.chroma_path = chroma_path
        if chroma_path is not None:
            self.host = None
            self.port = None
            chroma_path.mkdir(parents=True, exist_ok=True)
        else:
            self.host = host or os.environ.get("CO_CHROMA_HOST", DEFAULT_CHROMA_HOST)
            self.port = int(port if port is not None else os.environ.get("CO_CHROMA_PORT", DEFAULT_CHROMA_PORT))
        self._connect()
        # Lazy BM25 index — built on first hybrid search, refreshed when
        # the caller invalidates explicitly.
        self._bm25 = None
        self._bm25_ids: list[str] = []
        target = str(self.chroma_path) if self.chroma_path else f"http://{self.host}:{self.port}"
        logger.info(f"Vector search initialized at {target} ({self.collection.count()} docs)")
        self._verify_embedding_dim()

    def _verify_embedding_dim(self) -> None:
        """Compare the configured EF's output dim against the dim of vectors
        already stored in the collection. Mismatches are silent footguns:
        upserts go through OK but every query returns a 400. Surface it
        loudly at startup with the exact remediation steps.
        """
        if self.collection.count() == 0:
            return  # Empty collection takes whatever the EF produces.
        try:
            stored = self.collection.get(limit=1, include=["embeddings"])
            embs = stored.get("embeddings")
            if embs is None or len(embs) == 0:
                return
            stored_dim = len(embs[0])
        except Exception as e:
            logger.warning(f"Could not read collection dim for verification: {e}")
            return
        ef = self.collection._embedding_function
        try:
            probe = ef(["dim probe"])
            ef_dim = len(probe[0])
        except Exception as e:
            logger.warning(f"Could not probe EF dim for verification: {e}")
            return
        if ef_dim != stored_dim:
            ef_name = type(ef).__name__
            logger.error(
                f"EMBEDDING DIM MISMATCH: collection has {stored_dim}d vectors "
                f"but the configured embedding function ({ef_name}) produces "
                f"{ef_dim}d. Every query and most upserts will 400. "
                f"Fix one of: (a) set CO_EMBEDDING_MODEL to match the model "
                f"that built the collection ({stored_dim}d), (b) re-run "
                f"enable-gemini-pipeline.sh to wipe + reindex at the new "
                f"dim, or (c) wipe ~/.context-orchestrator/chroma and let "
                f"the watcher rebuild from disk."
            )

    def _connect(self) -> None:
        if self.chroma_path is not None:
            self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        else:
            self.client = self._http_client_with_retry()
        kwargs: dict = {
            "name": "context",
            "metadata": {"hnsw:space": "cosine"},
        }
        ef = _build_embedding_function()
        if ef is not None:
            kwargs["embedding_function"] = ef
        self.collection = self.client.get_or_create_collection(**kwargs)

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
        """Reconnect to disk so writes from other processes (e.g. the watcher) become visible."""
        self._connect()
        # Underlying corpus may have changed — drop cached BM25 index.
        self._bm25 = None
        self._bm25_ids = []

    def invalidate_bm25(self) -> None:
        """Force the next hybrid search to rebuild the BM25 index. Call this
        after batch writes if you can't wait for the natural reload."""
        self._bm25 = None
        self._bm25_ids = []

    def _ensure_bm25(self) -> None:
        """Build (or reuse) an in-memory BM25 index over every document in
        the collection. Lazy: only constructed on first hybrid search.
        """
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError(
                "hybrid search requires rank-bm25 — pip install rank-bm25"
            ) from e
        data = self.collection.get(include=["documents"], limit=100000)
        ids = data["ids"]
        docs = data["documents"]
        if not ids:
            # Empty collection — leave _bm25 as None; hybrid degrades to dense
            return
        tokenized = [_bm25_tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_ids = ids

    def add(self, doc_id: str, text: str, metadata: dict) -> None:
        """Add or update a document in the vector index."""
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
        )

    def remove(self, doc_id: str) -> None:
        """Remove a document from the vector index."""
        try:
            self.collection.delete(ids=[doc_id])
        except Exception:
            pass  # Ignore if not found

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
        """
        total = self.collection.count() or 1

        # Resolve the rerank model up front so misconfig fails loudly only
        # when actually requested.
        rr_model = None
        if rerank:
            rr_model = rerank_model or os.environ.get(RERANK_MODEL_ENV)

        if hybrid:
            base = self._hybrid_search(
                query,
                n_results=(RERANK_FETCH if rr_model else n_results),
                mmr=mmr,
                mmr_lambda=mmr_lambda,
            )
            if rr_model:
                return _llm_rerank(query, base, n_results, rr_model)
            return base

        if mmr:
            # If reranking, fetch enough for the rerank stage
            mmr_target = RERANK_FETCH if rr_model else n_results
            fetch_n = min(total, max(MMR_CANDIDATE_MIN, mmr_target * MMR_CANDIDATE_MULTIPLIER))
            kwargs = {
                "query_texts": [query],
                "n_results": fetch_n,
                "include": ["documents", "metadatas", "embeddings", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = self.collection.query(**kwargs)
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
            stripped = [
                {k: v for k, v in c.items() if k not in ("embedding", "sim_to_query")}
                for c in reranked
            ]
            if rr_model:
                return _llm_rerank(query, stripped, n_results, rr_model)
            return stripped

        # Standard path: raw cosine top-K
        fetch_for_base = RERANK_FETCH if rr_model else n_results
        kwargs = {
            "query_texts": [query],
            "n_results": min(fetch_for_base, total),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        if rr_model:
            return _llm_rerank(query, hits, n_results, rr_model)
        return hits

    def _hybrid_search(
        self,
        query: str,
        n_results: int,
        mmr: bool,
        mmr_lambda: float,
    ) -> list[dict]:
        """Dense + BM25 → RRF fuse → optional MMR → top-K."""
        total = self.collection.count() or 1
        fetch = min(total, HYBRID_FETCH_PER_RETRIEVER)

        # Dense retrieval — also gives us embeddings for an optional MMR step
        include = ["documents", "metadatas", "distances"]
        if mmr:
            include.append("embeddings")
        dense = self.collection.query(query_texts=[query], n_results=fetch, include=include)
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
        self._ensure_bm25()
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
            extra = self.collection.get(ids=missing, include=extra_include)
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
        return self.collection.count()
