"""Precompute data for the NLP Lab tier-1 modules (tokenize, stemming, ngram).

Needs: pip install nltk, plus NLTK corpora 'gutenberg', 'stopwords' in ~/nltk_data
(download the zips from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/<name>.zip
and unzip into ~/nltk_data/corpora/), and text8.gz (see training/train_word2vec.py).

Usage: python training/lab_tier1_data.py path/to/text8.gz
"""
import gzip, json, math, os, re, sys
from collections import Counter
from nltk.corpus import gutenberg, stopwords
from nltk.stem.porter import PorterStemmer

ROOT = os.path.join(os.path.dirname(__file__), "..", "lab")
TEXT8 = sys.argv[1]
with gzip.open(TEXT8, "rt") as f:
    toks = f.read().split()

# ---------- tokenize: Zipf + Heaps from text8 ----------
cnt = Counter(toks)
ranked = cnt.most_common()
top = ranked[:5000]
# Heaps' law: vocabulary size after n tokens, sampled on a log grid
heaps, seen, marks = [], set(), sorted({int(10 ** (k / 8)) for k in range(8, 8 * 7 + 3)})
mi = 0
for i, w in enumerate(toks, 1):
    seen.add(w)
    while mi < len(marks) and marks[mi] == i:
        heaps.append([i, len(seen)]); mi += 1
    if mi >= len(marks):
        break
heaps.append([len(toks), len(cnt)]) if heaps[-1][0] != len(toks) else None
os.makedirs(f"{ROOT}/tokenize/data", exist_ok=True)
json.dump({"corpus": "text8 (first 100M characters of English Wikipedia, cleaned)",
           "tokens": len(toks), "types": len(cnt),
           "words": [w for w, _ in top], "counts": [c for _, c in top],
           "hapax": sum(1 for _, c in ranked if c == 1),
           "heaps": heaps,
           "stopwords": sorted(stopwords.words("english"))},
          open(f"{ROOT}/tokenize/data/corpus.json", "w"), separators=(",", ":"))

# ---------- stemming: Porter (original algorithm) test vectors ----------
ps = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
vocab = [w for w, c in ranked if c >= 5 and w.isalpha() and len(w) > 2]
step = max(1, len(vocab) // 2000)
sample = vocab[::step][:2000]
classic = ["caresses", "ponies", "ties", "caress", "cats", "feed", "agreed", "plastered", "bled", "motoring", "sing",
           "conflated", "troubled", "sized", "hopping", "tanned", "falling", "hissing", "fizzed", "failing", "filing",
           "happy", "sky", "relational", "conditional", "rational", "valenci", "hesitanci", "digitizer", "conformabli",
           "radicalli", "differentli", "vileli", "analogousli", "vietnamization", "predication", "operator", "feudalism",
           "decisiveness", "hopefulness", "callousness", "formaliti", "sensitiviti", "sensibiliti", "triplicate",
           "formative", "formalize", "electriciti", "electrical", "hopeful", "goodness", "revival", "allowance",
           "inference", "airliner", "gyroscopic", "adjustable", "defensible", "irritant", "replacement", "adjustment",
           "dependent", "adoption", "homologou", "communism", "activate", "angulariti", "homologous", "effective",
           "bowdlerize", "probate", "rate", "cease", "controll", "roll", "generalization", "generalizations",
           "university", "universe", "universal", "organization", "organ", "news", "new", "general", "generous"]
words = list(dict.fromkeys(classic + sample))
vec = {w: ps.stem(w) for w in words}
os.makedirs(f"{ROOT}/stemming/data", exist_ok=True)
json.dump(vec, open(f"{ROOT}/stemming/data/porter-vectors.json", "w"), separators=(",", ":"))

# ---------- ngram: corpora, one sentence per line, lowercase tokens ----------
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,;:!?]")
def prep(fileids):
    out = []
    for fid in fileids:
        for sent in gutenberg.sents(fid):
            s = " ".join(TOK.findall(" ".join(sent).lower().replace(" ' ", "'").replace(" '", "'")))
            if len(s.split()) >= 3:
                out.append(s)
    return out

os.makedirs(f"{ROOT}/ngram/data", exist_ok=True)
corpora = {
    "austen": ("Jane Austen — Emma", ["austen-emma.txt"]),
    "shakespeare": ("Shakespeare — Hamlet, Macbeth, Julius Caesar", ["shakespeare-hamlet.txt", "shakespeare-macbeth.txt", "shakespeare-caesar.txt"]),
    "carroll": ("Lewis Carroll — Alice’s Adventures in Wonderland", ["carroll-alice.txt"]),
    "bible": ("King James Bible — Genesis to Numbers", ["bible-kjv.txt"]),
}
meta = []
for key, (name, fids) in corpora.items():
    lines = prep(fids)
    if key == "bible":  # Genesis to Numbers keeps the file small; drop "12 : 3" verse numbers
        lines = [re.sub(r"^(?:[0-9]+ : [0-9]+ )+", "", l) for l in lines[:4200]]
        lines = [l for l in lines if len(l.split()) >= 3]
    open(f"{ROOT}/ngram/data/{key}.txt", "w").write("\n".join(lines))
    meta.append({"key": key, "name": name, "sentences": len(lines), "tokens": sum(len(l.split()) for l in lines)})
# text8 has no sentence boundaries: chunk into pseudo-sentences of 20 tokens
t8 = [" ".join(toks[i:i + 20]) for i in range(0, 150_000, 20)]
open(f"{ROOT}/ngram/data/wikipedia.txt", "w").write("\n".join(t8))
meta.append({"key": "wikipedia", "name": "Wikipedia (text8, first 150k tokens; no sentence breaks)", "sentences": len(t8), "tokens": 150_000})
json.dump(meta, open(f"{ROOT}/ngram/data/corpora.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
