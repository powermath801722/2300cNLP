"""Build the probing module's sentence set from UD English EWT (CC BY-SA 4.0).

500 dev-set sentences of 5-25 words (fixed random seed), with gold UPOS tags; the page uses the
first 400 for training probes and the last 100 for testing. Embeddings are computed in the browser.

Usage: python training/probing_data.py path/to/en_ewt-ud-dev.conllu lab/probing/data/sentences.json
"""
import json, random, sys

sents, cur = [], []
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.rstrip('\n')
    if not line:
        if cur: sents.append(cur); cur = []
        continue
    if line.startswith('#'): continue
    c = line.split('\t')
    if '-' in c[0] or '.' in c[0]: continue
    cur.append((c[1], c[3]))
if cur: sents.append(cur)
pool = [s for s in sents if 5 <= len(s) <= 25 and all(len(w) < 30 for w, _ in s)]
random.seed(13)
pick = random.sample(pool, 500)
tags = sorted({t for s in pick for _, t in s})
out = {'source': 'Universal Dependencies English EWT, dev split (CC BY-SA 4.0)', 'tags': tags,
       'train': 400, 'sentences': [{'w': [w for w, _ in s], 't': [tags.index(t) for _, t in s]} for s in pick]}
json.dump(out, open(sys.argv[2], 'w'), separators=(',', ':'))
print(len(pool), 'eligible;', sum(len(s) for s in pick), 'tokens;', tags)
