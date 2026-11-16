"""WordNet lexicon for the adversarial module: coarse POS sets, synonyms and antonyms for the Word2Vec vocabulary.

    python -c "import nltk; nltk.download('wordnet')"
    python training/adversarial_build.py word2vec/model/vocab.json lab/adversarial/data/lexicon.json

Used to (1) keep substitutions to the same part of speech and (2) label each Word2Vec-neighbour substitution
as a WordNet synonym, antonym or neither. WordNet 3.0 (Princeton University), via NLTK.
"""
import json, sys
from nltk.corpus import wordnet as wn

vocab = json.load(open(sys.argv[1]))["words"]
V = set(vocab)
POS = {"n": "n", "v": "v", "a": "a", "s": "a", "r": "r"}
pos, syn, ant = {}, {}, {}
for w in vocab:
    ss = wn.synsets(w)
    if not ss:
        continue
    pos[w] = "".join(sorted({POS[s.pos()] for s in ss}))
    sy, an = set(), set()
    for s in ss:
        for l in s.lemmas():
            n = l.name().lower()
            if n != w and n in V:
                sy.add(n)
            for a in l.antonyms():
                m = a.name().lower()
                if m in V:
                    an.add(m)
    if sy:
        syn[w] = sorted(sy)
    if an:
        ant[w] = sorted(an)
json.dump({"source": "WordNet 3.0 (Princeton University) via NLTK, restricted to the 30,000-word Word2Vec vocabulary",
           "pos": pos, "syn": syn, "ant": ant}, open(sys.argv[2], "w"), separators=(",", ":"))
print(len(pos), "words with POS,", len(syn), "with synonyms,", len(ant), "with antonyms")
