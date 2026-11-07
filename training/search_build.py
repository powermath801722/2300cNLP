"""Build the NLP Lab search collection: U.S. presidential inaugural addresses (public domain),
split into paragraphs. Source: NLTK 'inaugural' corpus
(https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/inaugural.zip).

Usage: python training/search_build.py path/to/inaugural_dir lab/search/data/docs.json
"""
import json, os, re, sys

src, out = sys.argv[1], sys.argv[2]
docs = []
for fn in sorted(os.listdir(src)):
    m = re.match(r"(\d{4})-(\w+)\.txt$", fn)
    if not m:
        continue
    year, pres = int(m.group(1)), m.group(2)
    raw = open(os.path.join(src, fn), encoding="utf-8", errors="replace").read()
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
    paras = [p for p in paras if len(p.split()) >= 12]
    for i, p in enumerate(paras):
        docs.append({"y": year, "p": pres, "i": i + 1, "n": len(paras), "t": p})
json.dump({"source": "U.S. presidential inaugural addresses, 1789–2025 (public domain; NLTK inaugural corpus)",
           "docs": docs}, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
print(len(docs), "paragraphs from", len({(d['y'], d['p']) for d in docs}), "addresses")
