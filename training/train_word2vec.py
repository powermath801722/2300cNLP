"""Train a skip-gram Word2Vec model on text8 and export it for the browser.

text8 = first 100M characters of English Wikipedia (cleaned, ~17M tokens).
Download: https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz

Usage:
    pip install gensim numpy
    python training/train_word2vec.py path/to/text8.gz word2vec/model

Writes:
    <out>/vectors.bin  int8 unit-length vectors, row-major (N x DIM)
    <out>/scales.bin   float32 per-row dequantization scale (N)
    <out>/vocab.json   {"dim":..., "words":[...], "counts":[...], "params":{...}}
    word2vec.model     full gensim model (not committed, kept next to the corpus)
"""
import gzip, json, os, sys, time
import numpy as np
from gensim.models import Word2Vec

CORPUS, OUT = sys.argv[1], sys.argv[2]
TOP_N = 30000  # most frequent words shipped to the browser

PARAMS = dict(vector_size=100, window=5, min_count=5, sg=1, negative=10,
              sample=1e-4, epochs=15, workers=os.cpu_count(), seed=42)


def sentences(path, chunk=1000):
    """text8 is one long line; split it into pseudo-sentences of 1000 tokens."""
    with gzip.open(path, "rt") as f:
        toks = f.read().split()
    return [toks[i:i + chunk] for i in range(0, len(toks), chunk)]


t0 = time.time()
sents = sentences(CORPUS)
print(f"{sum(map(len, sents)):,} tokens loaded", flush=True)
model = Word2Vec(sentences=sents, **PARAMS)
print(f"trained in {time.time() - t0:.0f}s, vocab {len(model.wv):,}", flush=True)
model.save(os.path.join(os.path.dirname(CORPUS), "word2vec.model"))

words = model.wv.index_to_key[:TOP_N]  # sorted by frequency
vecs = np.stack([model.wv[w] for w in words]).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
scales = np.abs(vecs).max(axis=1) / 127.0
q = np.round(vecs / scales[:, None]).astype(np.int8)

os.makedirs(OUT, exist_ok=True)
q.tofile(os.path.join(OUT, "vectors.bin"))
scales.astype(np.float32).tofile(os.path.join(OUT, "scales.bin"))
with open(os.path.join(OUT, "vocab.json"), "w") as f:
    json.dump({"dim": PARAMS["vector_size"], "words": words,
               "counts": [int(model.wv.get_vecattr(w, "count")) for w in words],
               "corpus": "text8 (Wikipedia, ~17M tokens)",
               "params": {k: v for k, v in PARAMS.items() if k != "workers"}}, f,
              separators=(",", ":"))

# sanity checks
for w in ["king", "france", "computer", "river"]:
    print(w, [x for x, _ in model.wv.most_similar(w, topn=10)])
print("king - man + woman =",
      model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5))
