"""Regenerate tokenizer_parity.json and reference.npz from texts.json.

Independent of context_orchestrator.local_model on purpose: the Hugging Face
`tokenizers` library (tokenizer.json from google/embeddinggemma-2) gives the
reference ids, and the reference vectors come from a minimal onnxruntime loop
(the emb-bench harness's ort130_check.py) on the published int8 model.

    python make_fixtures.py RELEASE_DIR TOKENIZER_JSON
"""
import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from tokenizers import Tokenizer

here = Path(__file__).parent
rel, tok_json = Path(sys.argv[1]), sys.argv[2]
texts = json.loads((here / "texts.json").read_text())
hf = Tokenizer.from_file(tok_json)
hf.no_truncation(); hf.no_padding()
parity = [{"text": t, "ids": hf.encode(t).ids} for t in texts["tokenizer"]]
(here / "tokenizer_parity.json").write_text(json.dumps({"_source": "tokenizers " + __import__("tokenizers").__version__
    + " with google/embeddinggemma-2@914f7f8 tokenizer.json", "cases": parity}, ensure_ascii=False, indent=0))

sp = spm.SentencePieceProcessor(model_file=str(rel / "tokenizer.model"))
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(str(rel / "eg2-mixed.onnx"), so, providers=["CPUExecutionProvider"])
z = np.zeros((0, 512), np.float32)
def emb(t):
    ids = np.array([([2] + sp.encode(t) + [1])[:2048]], np.int64)
    e = s.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids),
              "image_features": z, "video_features": z, "audio_features": z})[0][0]
    return e / np.linalg.norm(e)
D = np.stack([emb("title: none | text: " + d["text"]) for d in texts["corpus"]]).astype(np.float32)
Q = np.stack([emb("task: search result | query: " + q["query"]) for q in texts["queries"]]).astype(np.float32)
np.savez_compressed(here / "reference.npz", D=D, Q=Q, ort=np.array(ort.__version__))
print("ort", ort.__version__, "D", D.shape, "Q", Q.shape)
