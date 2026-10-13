"""Real-corpus check for EmbeddingGemma 2 — LOCAL ONLY, never in CI.

The benchmark corpus is the developer's own meeting transcripts, so nothing
from it is in this repository: point the test at the benchmark folder and the
release files and it reads them in place.

    CO_TEST_EG2_BENCH=~/tasks/emb-bench \\
    CO_TEST_EG2_RELEASE_DIR=~/tasks/emb-bench/release/models-eg2-int8-v1 \\
    pytest -q -s -m slow tests/test_local_model_bench.py

The bench folder holds corpus.json ([{id, meeting_id, text}], chunked with
transcripts.chunks_for), queries.json ([{query, relevant: [chunk ids]}]) and
emb/eg2-st.npz (D, Q: the fp32 sentence-transformers reference vectors).
Everything runs in a scratch Chroma folder; no text leaves this process.
"""
import json
import os
from pathlib import Path

import pytest

from context_orchestrator import local_model as lm
from context_orchestrator import search

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    b, rel = os.environ.get("CO_TEST_EG2_BENCH"), os.environ.get("CO_TEST_EG2_RELEASE_DIR")
    if not b or not rel:
        pytest.skip("local only: set CO_TEST_EG2_BENCH and CO_TEST_EG2_RELEASE_DIR")
    b, rel = Path(b).expanduser(), Path(rel).expanduser()
    root = tmp_path_factory.mktemp("eg2-bench")
    saved = {k: os.environ.get(k) for k in ("CO_MODELS_DIR", "CO_MODEL_BASE_URL", "CO_EMBEDDING_MODEL")}
    os.environ.update(CO_MODELS_DIR=str(root / "models"), CO_MODEL_BASE_URL=rel.as_uri(),
                      CO_EMBEDDING_MODEL="local")
    lm.download()
    yield b, root
    lm.release()
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_bench_corpus_fidelity_and_mrr(bench):
    import numpy as np
    b, root = bench
    corpus = json.loads((b / "corpus.json").read_text())
    queries = json.loads((b / "queries.json").read_text())
    ref = np.load(b / "emb" / "eg2-st.npz")
    vs = search.VectorSearch(chroma_path=root / "chroma")
    assert vs.identity == lm.MODEL_NAME
    ids = [c["id"] for c in corpus]
    vs.upsert(ids, [c["text"] for c in corpus],
              [{"type": "transcript", "meeting_id": c["meeting_id"]} for c in corpus])
    with vs.session() as col:
        got = col.get(ids=ids, include=["embeddings"])
    by_id = dict(zip(got["ids"], got["embeddings"]))
    D = np.stack([np.asarray(by_id[i], dtype=np.float32) for i in ids])
    Q = np.stack([np.asarray(vs.embed_query(q["query"]), dtype=np.float32) for q in queries])
    cos_d, cos_q = (D * ref["D"]).sum(1), (Q * ref["Q"]).sum(1)
    print(f"\ncosine vs fp32 reference: docs min {cos_d.min():.4f} mean {cos_d.mean():.4f}; "
          f"queries min {cos_q.min():.4f} mean {cos_q.mean():.4f}")
    assert cos_d.min() > 0.99 and cos_q.min() > 0.99

    def mrr(**kw):
        rr = []
        for q in queries:
            hits = [h["id"] for h in vs.search(q["query"], n_results=10, **kw)]
            pos = next((i for i, h in enumerate(hits) if h in set(q["relevant"])), None)
            rr.append(0.0 if pos is None else 1.0 / (pos + 1))
        return sum(rr) / len(rr)
    dense = mrr()
    hybrid = mrr(hybrid=True, mmr=True, mmr_lambda=0.7)
    print(f"chunk-level MRR@10: dense {dense:.3f}, hybrid+MMR {hybrid:.3f} ({len(queries)} queries)")
    assert dense >= 0.80
