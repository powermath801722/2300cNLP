"""Train a bigram HMM part-of-speech tagger on the NLTK Penn Treebank sample and export it for the browser.

    pip install nltk numpy
    python -c "import nltk; nltk.download('treebank')"
    python training/pos_train.py lab/pos/data

Unknown words: training words seen once are replaced by a signature class (UNK-ing, UNK-cap, ...)
so the model learns P(class | tag). The browser must use the same unk_class() rules.
Writes <out>/hmm.json.
"""
import json, math, os, re, sys, random
from collections import Counter, defaultdict
from nltk.corpus import treebank

OUT = sys.argv[1]
SMOOTH_T = 0.1   # add-k smoothing for transitions
SMOOTH_E = 0.01  # add-k smoothing for emissions (over the vocabulary of each tag)

def unk_class(w, first=False):
    if re.fullmatch(r"[0-9.,:/-]+", w) and re.search(r"[0-9]", w): return "UNK-num"
    if "-" in w: return "UNK-hyph"
    lw = w.lower()
    cap = w[0].isupper() and not first
    for suf in ["ing", "ed", "ly", "ion", "s", "er", "est", "al", "ive", "able", "ous", "ful", "ity", "ment", "ness"]:
        if lw.endswith(suf) and len(lw) > len(suf) + 2:
            return ("UNK-cap-" if cap else "UNK-") + suf
    return "UNK-cap" if cap else "UNK"

sents = [[(w, t) for w, t in s if t != "-NONE-"] for s in treebank.tagged_sents()]
sents = [s for s in sents if s]
random.Random(42).shuffle(sents)
cut = int(len(sents) * 0.9)
train, test = sents[:cut], sents[cut:]

wc = Counter(w for s in train for w, _ in s)
def norm(w, i): return w if wc[w] >= 2 else unk_class(w, i == 0)

tags = sorted({t for s in train for _, t in s})
T = len(tags); ti = {t: i for i, t in enumerate(tags)}
trans = [[0] * (T + 1) for _ in range(T + 1)]  # row T = <s>, col T = </s>
emit = defaultdict(Counter); tagc = Counter()
for s in train:
    prev = T
    for i, (w, t) in enumerate(s):
        trans[prev][ti[t]] += 1; prev = ti[t]
        emit[t][norm(w, i)] += 1; tagc[t] += 1
    trans[prev][T] += 1

ltrans = []
for r in range(T + 1):
    tot = sum(trans[r]) + SMOOTH_T * (T + 1)
    ltrans.append([round(math.log((c + SMOOTH_T) / tot), 3) for c in trans[r]])
vocab_by_tag = {t: len(emit[t]) for t in tags}
lemit = defaultdict(dict)
for t in tags:
    tot = tagc[t] + SMOOTH_E * (vocab_by_tag[t] + 1)
    for w, c in emit[t].items():
        lemit[w][t] = round(math.log((c + SMOOTH_E) / tot), 3)
floor = {t: round(math.log(SMOOTH_E / (tagc[t] + SMOOTH_E * (vocab_by_tag[t] + 1))), 3) for t in tags}

def word_key(w, i):
    if w in lemit: return w
    if i == 0 and w.lower() in lemit: return w.lower()
    return unk_class(w, i == 0)

def viterbi(words):
    n = len(words); V = [[-1e9] * T for _ in range(n)]; B = [[0] * T for _ in range(n)]
    for i, w in enumerate(words):
        e = lemit.get(word_key(w, i), {})
        for t in range(T):
            em = e.get(tags[t])
            if em is None:
                continue  # tags never seen with this word/class are impossible
            if i == 0: V[0][t] = ltrans[T][t] + em
            else:
                best, arg = max((V[i - 1][p] + ltrans[p][t], p) for p in range(T))
                V[i][t] = best + em; B[i][t] = arg
    last = max(range(T), key=lambda t: V[n - 1][t] + ltrans[t][T])
    path = [last]
    for i in range(n - 1, 0, -1): path.append(B[i][path[-1]])
    return [tags[t] for t in reversed(path)]

# most-frequent-tag baseline
mft = {w: max(c, key=c.get) for w, c in ((w, Counter()) for w in []) }
wt = defaultdict(Counter)
for s in train:
    for w, t in s: wt[w][t] += 1
top_tag = tagc.most_common(1)[0][0]
def baseline(w): return wt[w].most_common(1)[0][0] if w in wt else top_tag

conf = Counter(); tot = cor = bcor = unk_tot = unk_cor = 0
for s in test:
    ws = [w for w, _ in s]; gold = [t for _, t in s]
    pred = viterbi(ws)
    for w, g, p in zip(ws, gold, pred):
        tot += 1; cor += g == p; bcor += baseline(w) == g
        if wc[w] == 0: unk_tot += 1; unk_cor += g == p
        if g != p: conf[(g, p)] += 1

out = {
    "tags": tags, "trans": ltrans, "emit": lemit, "floor": floor,
    "unk_suffixes": ["ing", "ed", "ly", "ion", "s", "er", "est", "al", "ive", "able", "ous", "ful", "ity", "ment", "ness"],
    "tag_counts": {t: tagc[t] for t in tags},
    "metrics": {"train_sents": len(train), "test_sents": len(test), "test_tokens": tot,
                "hmm_acc": round(cor / tot, 4), "baseline_acc": round(bcor / tot, 4),
                "unk_tokens": unk_tot, "unk_acc": round(unk_cor / max(unk_tot, 1), 4),
                "top_confusions": [[g, p, c] for (g, p), c in conf.most_common(12)]},
}
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, "hmm.json"), "w"), separators=(",", ":"))
print(out["metrics"])
for s in ["I can can a can .", "Time flies like an arrow .", "The old man the boat ."]:
    print(s, viterbi(s.split()))
