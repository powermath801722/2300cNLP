"""Word-frequency list for the NLP Lab spelling corrector, counted from text8 (Wikipedia, ~17M tokens).

Usage: python training/spell_build.py path/to/text8.gz lab/spell/data/freq.txt [top_n]
Output: one "word count" pair per line, most frequent first.
"""
import collections, gzip, sys

src, out = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
with gzip.open(src, "rt") as f:
    counts = collections.Counter(f.read().split())
with open(out, "w") as f:
    for w, c in counts.most_common(top):
        f.write(f"{w} {c}\n")
print(f"{min(top, len(counts))} words written; corpus {sum(counts.values()):,} tokens")
