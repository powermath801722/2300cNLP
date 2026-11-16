"""Train the NLP Lab parsing module's tagger and dependency parser on UD English EWT.

Data: Universal Dependencies English Web Treebank (CC BY-SA 4.0)
  https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-{train,dev,test}.conllu

Models (both greedy, averaged perceptron over hashed features; the browser re-implements the
same feature templates and FNV-1a hash, so weights transfer exactly):
  * UPOS tagger: left-to-right, Collins (2002)-style averaged perceptron.
  * Dependency parser: arc-standard transition system (Nivre 2004) with a static oracle,
    labelled actions (SHIFT, LEFT-l, RIGHT-l), trained on projective training sentences.

Accuracy is reported with the *quantized, pruned weights that are shipped*, on dev and test,
with predicted tags (UAS/LAS over all tokens, CoNLL 2018 style, and excluding punctuation).

Usage:
  pip install numpy
  python training/parsing_train.py path/to/ud_dir lab/parsing/data
"""
import json
import os
import random
import sys
import time

import numpy as np

UD, OUT = sys.argv[1], sys.argv[2]
NB = 1 << 19          # hashed feature rows
EPOCHS_TAG, EPOCHS_PARSE = 6, 8
random.seed(7)


# ---------- data ----------
def read_conllu(path):
    sents, cur = [], []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line:
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith('#'):
            continue
        c = line.split('\t')
        if '-' in c[0] or '.' in c[0]:  # multiword token ranges / empty nodes
            continue
        cur.append({'form': c[1], 'upos': c[3], 'head': int(c[6]), 'rel': c[7].split(':')[0]})
    if cur:
        sents.append(cur)
    return sents


train = read_conllu(os.path.join(UD, 'en_ewt-ud-train.conllu'))
dev = read_conllu(os.path.join(UD, 'en_ewt-ud-dev.conllu'))
test = read_conllu(os.path.join(UD, 'en_ewt-ud-test.conllu'))
print(f'train {len(train)} dev {len(dev)} test {len(test)} sentences', flush=True)

TAGS = sorted({t['upos'] for s in train for t in s})
RELS = sorted({t['rel'] for s in train for t in s})
ACTIONS = ['SHIFT'] + ['L-' + r for r in RELS] + ['R-' + r for r in RELS]
A_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ---------- hashing (must match app.js) ----------
_memo = {}


def fh(s):
    h = _memo.get(s)
    if h is None:
        x = 0x811C9DC5
        for b in s.encode('utf-8'):
            x ^= b
            x = (x * 0x01000193) & 0xFFFFFFFF
        h = _memo[s] = x % NB
    return h


def norm(w):
    if any(ch.isdigit() for ch in w):
        return '!DIGIT' if w.replace(',', '').replace('.', '').isdigit() else '!HASDIGIT'
    return w.lower()


def shape(w):
    out = []
    for ch in w[:6]:
        c = 'X' if ch.isupper() else 'x' if ch.islower() else 'd' if ch.isdigit() else ch
        if not out or out[-1] != c:
            out.append(c)
    return ''.join(out)


# ---------- averaged perceptron ----------
class AP:
    def __init__(self, ncls):
        self.W = np.zeros((NB, ncls), np.float32)
        self.U = np.zeros((NB, ncls), np.float64)
        self.c = 1

    def scores(self, ids):
        return self.W[ids].sum(0)

    def update(self, ids, gold, pred):
        if gold != pred:
            ids = np.asarray(ids)
            np.add.at(self.W, (ids, gold), 1.0)
            np.add.at(self.W, (ids, pred), -1.0)
            np.add.at(self.U, (ids, gold), self.c)
            np.add.at(self.U, (ids, pred), -self.c)
        self.c += 1

    def averaged(self):
        return (self.W - self.U / self.c).astype(np.float32)


# ---------- tagger ----------
def tag_feats(words, i, p1, p2):
    w = norm(words[i])
    ctx = ['<s2>', '<s1>'] + [norm(x) for x in words] + ['</s1>', '</s2>']
    j = i + 2
    f = ['b', 'w=' + w, 's3=' + w[-3:], 's2=' + w[-2:], 'p1=' + w[:1], 'sh=' + shape(words[i]),
         't1=' + p1, 't2=' + p2, 't12=' + p1 + '|' + p2, 't1w=' + p1 + '|' + w,
         'w-1=' + ctx[j - 1], 's3-1=' + ctx[j - 1][-3:], 'w-2=' + ctx[j - 2],
         'w+1=' + ctx[j + 1], 's3+1=' + ctx[j + 1][-3:], 'w+2=' + ctx[j + 2]]
    return [fh(x) for x in f]


def tag_sentence(model_scores, words):
    out, p1, p2 = [], '<s1>', '<s2>'
    for i in range(len(words)):
        t = TAGS[int(np.argmax(model_scores(tag_feats(words, i, p1, p2))))]
        out.append(t)
        p2, p1 = p1, t
    return out


def train_tagger():
    m = AP(len(TAGS))
    T_IDX = {t: i for i, t in enumerate(TAGS)}
    for ep in range(EPOCHS_TAG):
        random.shuffle(train)
        t0, cor, tot = time.time(), 0, 0
        for s in train:
            words = [t['form'] for t in s]
            p1, p2 = '<s1>', '<s2>'
            for i, tok in enumerate(s):
                ids = tag_feats(words, i, p1, p2)
                pred = int(np.argmax(m.scores(ids)))
                gold = T_IDX[tok['upos']]
                m.update(ids, gold, pred)
                cor += pred == gold
                tot += 1
                p2, p1 = p1, TAGS[pred]  # train on own predictions for prev-tag features
        print(f'tagger epoch {ep + 1}: train acc {cor / tot:.4f} ({time.time() - t0:.0f}s)', flush=True)
    return m.averaged()


# ---------- parser ----------
def parse_feats(words, tags, stack, b, heads, rels, lefts, rights):
    n = len(words)

    def W(i):
        return '<root>' if i == 0 else norm(words[i - 1]) if 0 < i <= n else '<none>'

    def T(i):
        return '<root>' if i == 0 else tags[i - 1] if 0 < i <= n else '<none>'

    def L(i):
        return rels[i] if i is not None and i > 0 and rels[i] else '<none>'

    s0 = stack[-1] if len(stack) >= 1 else None
    s1 = stack[-2] if len(stack) >= 2 else None
    s2 = stack[-3] if len(stack) >= 3 else None
    b0 = b if b <= n else None
    b1 = b + 1 if b + 1 <= n else None
    b2 = b + 2 if b + 2 <= n else None

    def ww(i):
        return W(i) if i is not None else '<none>'

    def tt(i):
        return T(i) if i is not None else '<none>'

    def lc(i):
        return lefts[i][0] if i is not None and lefts[i] else None

    def rc(i):
        return rights[i][-1] if i is not None and rights[i] else None

    s0w, s0t, s1w, s1t, s2t = ww(s0), tt(s0), ww(s1), tt(s1), tt(s2)
    b0w, b0t, b1w, b1t, b2t = ww(b0), tt(b0), ww(b1), tt(b1), tt(b2)
    lc0, rc0, lc1, rc1 = lc(s0), rc(s0), lc(s1), rc(s1)
    dist = min(s0 - s1, 5) if s0 is not None and s1 is not None else -1
    nl0 = len(lefts[s0]) if s0 is not None else -1
    nr0 = len(rights[s0]) if s0 is not None else -1
    nl1 = len(lefts[s1]) if s1 is not None else -1
    nr1 = len(rights[s1]) if s1 is not None else -1
    f = ['b',
         's0w=' + s0w, 's0t=' + s0t, 's0wt=' + s0w + '|' + s0t,
         's1w=' + s1w, 's1t=' + s1t, 's1wt=' + s1w + '|' + s1t, 's2t=' + s2t,
         'b0w=' + b0w, 'b0t=' + b0t, 'b0wt=' + b0w + '|' + b0t, 'b1w=' + b1w, 'b1t=' + b1t, 'b2t=' + b2t,
         's0t s1t=' + s0t + '|' + s1t, 's0w s1w=' + s0w + '|' + s1w, 's0w s1t=' + s0w + '|' + s1t, 's0t s1w=' + s0t + '|' + s1w,
         's0t b0t=' + s0t + '|' + b0t, 's1t s0t b0t=' + s1t + '|' + s0t + '|' + b0t, 's2t s1t s0t=' + s2t + '|' + s1t + '|' + s0t,
         's0t b0t b1t=' + s0t + '|' + b0t + '|' + b1t,
         'lc0t=' + tt(lc0), 'rc0t=' + tt(rc0), 'lc1t=' + tt(lc1), 'rc1t=' + tt(rc1),
         'lc0l=' + L(lc0), 'rc0l=' + L(rc0), 'lc1l=' + L(lc1), 'rc1l=' + L(rc1),
         's0t lc0l=' + s0t + '|' + L(lc0), 's1t rc1l=' + s1t + '|' + L(rc1), 's1t lc1l=' + s1t + '|' + L(lc1),
         'd s0t s1t=' + str(dist) + '|' + s0t + '|' + s1t, 'd s0w=' + str(dist) + '|' + s0w, 'd s1w=' + str(dist) + '|' + s1w,
         'v s0=' + str(nl0) + '|' + str(nr0) + '|' + s0t, 'v s1=' + str(nl1) + '|' + str(nr1) + '|' + s1t,
         'bempty=' + str(b > n) + '|' + str(len(stack))]
    return [fh(x) for x in f]


def legal(stack, b, n):
    ok = np.zeros(len(ACTIONS), bool)
    if b <= n:
        ok[0] = True
    if len(stack) >= 2:
        s0, s1 = stack[-1], stack[-2]
        if s1 != 0:
            ok[1:1 + len(RELS)] = True
        if s1 != 0 or b > n:  # attach to root only once everything else is done
            ok[1 + len(RELS):] = True
    return ok


def apply(a, stack, heads, rels, lefts, rights, b):
    if a == 0:
        stack.append(b)
        return b + 1
    rel = ACTIONS[a][2:]
    s0, s1 = stack[-1], stack[-2]
    if ACTIONS[a][0] == 'L':  # s0 -> s1
        heads[s1], rels[s1] = s0, rel
        lefts[s0].insert(0, s1)
        lefts[s0].sort()
        stack.pop(-2)
    else:  # s1 -> s0
        heads[s0], rels[s0] = s1, rel
        rights[s1].append(s0)
        rights[s1].sort()
        stack.pop()
    return b


def oracle(stack, b, n, gh, gr, heads):
    if len(stack) >= 2:
        s0, s1 = stack[-1], stack[-2]
        if s1 != 0 and gh[s1] == s0:
            return A_IDX['L-' + gr[s1]]
        if gh[s0] == s1 and all(heads[k] == gh[k] for k in range(1, n + 1) if gh[k] == s0) and (s1 != 0 or b > n):
            return A_IDX['R-' + gr[s0]]
    if b <= n:
        return 0
    return None


def parse(scores, words, tags):
    n = len(words)
    stack, b = [0], 1
    heads, rels = [None] * (n + 1), [None] * (n + 1)
    lefts, rights = [[] for _ in range(n + 1)], [[] for _ in range(n + 1)]
    while not (b > n and len(stack) == 1):
        sc = scores(parse_feats(words, tags, stack, b, heads, rels, lefts, rights))
        ok = legal(stack, b, n)
        sc = np.where(ok, sc, -np.inf)
        b = apply(int(np.argmax(sc)), stack, heads, rels, lefts, rights, b)
    return heads, rels


def projective_ok(s):
    n = len(s)
    gh = [None] + [t['head'] for t in s]
    gr = [None] + [t['rel'] for t in s]
    stack, b = [0], 1
    heads, rels = [None] * (n + 1), [None] * (n + 1)
    lefts, rights = [[] for _ in range(n + 1)], [[] for _ in range(n + 1)]
    seq = []
    while not (b > n and len(stack) == 1):
        a = oracle(stack, b, n, gh, gr, heads)
        if a is None:
            return None
        seq.append((list(stack), b, [x for x in heads], [x for x in rels], [list(x) for x in lefts], [list(x) for x in rights], a))
        b = apply(a, stack, heads, rels, lefts, rights, b)
    return seq if heads[1:] == gh[1:] else None


def train_parser(tag_scores):
    # Train on predicted tags (jackknifing would be better; we tag train with the trained tagger,
    # which is optimistic, so we mix: gold tags 50% of the time).
    data = []
    skipped = 0
    for s in train:
        if projective_ok(s) is None:
            skipped += 1
            continue
        data.append(s)
    print(f'parser: {len(data)} projective training sentences ({skipped} non-projective skipped)', flush=True)
    pred_tags = {id(s): tag_sentence(tag_scores, [t['form'] for t in s]) for s in data}
    m = AP(len(ACTIONS))
    for ep in range(EPOCHS_PARSE):
        random.shuffle(data)
        t0, cor, tot = time.time(), 0, 0
        for s in data:
            words = [t['form'] for t in s]
            tags = pred_tags[id(s)] if random.random() < 0.5 else [t['upos'] for t in s]
            n = len(s)
            gh = [None] + [t['head'] for t in s]
            gr = [None] + [t['rel'] for t in s]
            stack, b = [0], 1
            heads, rels = [None] * (n + 1), [None] * (n + 1)
            lefts, rights = [[] for _ in range(n + 1)], [[] for _ in range(n + 1)]
            while not (b > n and len(stack) == 1):
                ids = parse_feats(words, tags, stack, b, heads, rels, lefts, rights)
                ok = legal(stack, b, n)
                sc = np.where(ok, m.scores(ids), -np.inf)
                pred = int(np.argmax(sc))
                gold = oracle(stack, b, n, gh, gr, heads)
                m.update(ids, gold, pred)
                cor += pred == gold
                tot += 1
                b = apply(gold, stack, heads, rels, lefts, rights, b)  # static oracle: follow gold
        print(f'parser epoch {ep + 1}: transition acc {cor / tot:.4f} ({time.time() - t0:.0f}s)', flush=True)
    return m.averaged()


# ---------- export: prune + int8 quantize, then evaluate the shipped weights ----------
def quantize(Wavg, keep_frac):
    absw = np.abs(Wavg)
    thr = np.quantile(absw[absw > 0], 1 - keep_frac)
    scale = float(absw.max() / 127.0)
    q = np.round(Wavg / scale)
    q[absw < thr] = 0
    q = np.clip(q, -127, 127).astype(np.int8)
    rows = {}
    nz_rows = np.nonzero(np.any(q != 0, axis=1))[0]
    for r in nz_rows:
        cols = np.nonzero(q[r])[0]
        rows[int(r)] = [v for c in cols for v in (int(c), int(q[r, c]))]
    return q, scale, rows


def dense_scorer(q, scale):
    qf = q.astype(np.float32) * scale
    return lambda ids: qf[ids].sum(0)


PUNCT = 'PUNCT'


def evaluate(tag_scores, parse_scores, sents, gold_tags=False):
    tc = tt = ua = la = tot = ua_np = la_np = tot_np = 0
    for s in sents:
        words = [t['form'] for t in s]
        ptags = tag_sentence(tag_scores, words)
        tc += sum(a == t['upos'] for a, t in zip(ptags, s))
        tt += len(s)
        tags = [t['upos'] for t in s] if gold_tags else ptags
        heads, rels = parse(parse_scores, words, tags)
        for i, t in enumerate(s, 1):
            uh = heads[i] == t['head']
            lb = uh and rels[i] == t['rel']
            ua += uh
            la += lb
            tot += 1
            if t['upos'] != PUNCT:
                ua_np += uh
                la_np += lb
                tot_np += 1
    return {'tag_acc': tc / tt, 'uas': ua / tot, 'las': la / tot, 'uas_nopunct': ua_np / tot_np, 'las_nopunct': la_np / tot_np, 'tokens': tot, 'sentences': len(sents)}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    Wt = train_tagger()
    Wp = train_parser(lambda ids: Wt[ids].sum(0))
    qt, st, rows_t = quantize(Wt, 0.35)
    qp, sp, rows_p = quantize(Wp, 0.25)
    ts, ps = dense_scorer(qt, st), dense_scorer(qp, sp)
    res = {'dev': evaluate(ts, ps, dev), 'test': evaluate(ts, ps, test), 'test_gold_tags': evaluate(ts, ps, test, gold_tags=True)}
    print(json.dumps(res, indent=1), flush=True)
    out = {'nb': NB, 'tags': TAGS, 'rels': RELS, 'actions': ACTIONS,
           'tagger': {'scale': st, 'rows': rows_t}, 'parser': {'scale': sp, 'rows': rows_p},
           'eval': res, 'train_sentences': len(train),
           'source': 'Universal Dependencies English EWT (CC BY-SA 4.0)'}
    with open(os.path.join(OUT, 'model.json'), 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print('wrote', os.path.getsize(os.path.join(OUT, 'model.json')) / 1e6, 'MB', flush=True)
