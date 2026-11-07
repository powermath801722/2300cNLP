"""Train an LDA topic model on the NLTK Reuters-21578 corpus and export it for the browser.

    pip install nltk gensim numpy
    python -c "import nltk; nltk.download('reuters'); nltk.download('stopwords')"
    python training/topics_train.py lab/topics/data

Sweeps K for c_v coherence, trains the final model with K_FINAL topics, and writes <out>/lda.json:
vocab, topic-word probabilities (phi), alpha, top words, a sample of documents with topic mixtures,
and the coherence sweep. The browser re-tokenizes with TOKEN_RE and the same stopword list.
"""
import json, os, re, sys
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from nltk.corpus import reuters, stopwords

OUT = sys.argv[1]
K_SWEEP = [6, 10, 15, 20, 25]
K_FINAL = 15
TOKEN_RE = r"[a-z][a-z]{2,}"
STOP = set(stopwords.words("english")) | set((
    "said mln pct dlrs reuter vs cts year would also inc corp company net shr one two three may last new "
    "the will could share shares per told billion first april march june could").split())

def tok(text):
    return [w for w in re.findall(TOKEN_RE, text.lower()) if w not in STOP]

fids = reuters.fileids()
raw = [reuters.raw(f) for f in fids]
texts = [tok(t) for t in raw]
keep = [i for i, t in enumerate(texts) if len(t) >= 20]
texts = [texts[i] for i in keep]; raw = [raw[i] for i in keep]; fids = [fids[i] for i in keep]

d = Dictionary(texts)
d.filter_extremes(no_below=10, no_above=0.3, keep_n=4000)
corpus = [d.doc2bow(t) for t in texts]

def train(k):
    return LdaModel(corpus, id2word=d, num_topics=k, passes=10, iterations=200, alpha="auto", eta="auto", random_state=7)

sweep = []
models = {}
for k in K_SWEEP:
    m = train(k); models[k] = m
    cv = CoherenceModel(model=m, texts=texts, dictionary=d, coherence="c_v", topn=10).get_coherence()
    sweep.append({"k": k, "c_v": round(float(cv), 4)}); print(k, cv, flush=True)
m = models[K_FINAL]
phi = m.get_topics()  # K x V
vocab = [d[i] for i in range(len(d))]
top = [[[vocab[i], round(float(phi[k, i]), 5)] for i in np.argsort(-phi[k])[:15]] for k in range(K_FINAL)]
per_topic_cv = CoherenceModel(model=m, texts=texts, dictionary=d, coherence="c_v", topn=10).get_coherence_per_topic()

rng = np.random.RandomState(3)
sample = []
for i in rng.choice(len(texts), 60, replace=False):
    mix = m.get_document_topics(corpus[i], minimum_probability=0)
    title = raw[i].strip().split("\n")[0].strip()
    body = " ".join(raw[i].strip().split("\n")[1:]).strip()
    sample.append({"id": fids[i], "title": title[:90], "snippet": re.sub(r"\s+", " ", body)[:220],
                   "theta": [round(float(p), 4) for _, p in mix]})

out = {"token_re": TOKEN_RE, "stopwords": sorted(STOP), "vocab": vocab, "k": K_FINAL,
       "alpha": [round(float(a), 5) for a in m.alpha],
       "phi": [[round(float(x), 6) for x in row] for row in phi],
       "top": top, "coherence_per_topic": [round(float(c), 4) for c in per_topic_cv],
       "sweep": sweep, "docs": sample,
       "meta": {"corpus": "Reuters-21578 (NLTK), 1987 newswire", "n_docs": len(texts), "vocab_size": len(vocab),
                "passes": 10, "alpha": "auto", "eta": "auto"}}
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, "lda.json"), "w"), separators=(",", ":"))
for k in range(K_FINAL): print(k, round(per_topic_cv[k], 3), " ".join(w for w, _ in top[k][:10]))
