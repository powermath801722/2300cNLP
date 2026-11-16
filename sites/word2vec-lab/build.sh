#!/usr/bin/env bash
# Assembles a standalone Word2Vec Lab site in sites/word2vec-lab/dist from the shared repo files.
# Netlify runs this with the base directory set to sites/word2vec-lab.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/dist"
rm -rf "$OUT"
mkdir -p "$OUT/assets/fonts" "$OUT/assets/vendor"
cp -R "$ROOT/word2vec" "$OUT/word2vec"
cp "$ROOT/favicon.svg" "$OUT/"
cp "$ROOT"/assets/fonts/barlow-*-latin.woff2 "$OUT/assets/fonts/"
cp "$ROOT"/assets/vendor/plotly-*.min.js "$OUT/assets/vendor/"
# Standalone site: the lab is the home page here, so links that belong to the main Braidwater site
# (NLP Lab, Home, the footer cross-link) are rewritten to absolute URLs on that site.
MAIN="${MAIN_SITE_URL:-https://2300c-braidwater.netlify.app}"
python3 - "$OUT/word2vec/index.html" "$MAIN" <<'PY'
import re, sys
p, main = sys.argv[1], sys.argv[2].rstrip('/')
t = open(p, encoding='utf-8').read()
subs = [
    ('<a class="l opt" href="/lab/">NLP Lab</a>', f'<a class="l opt" href="{main}/lab/">NLP Lab</a>'),
    ('<a class="l" href="/">← Home</a>', f'<a class="l" href="{main}/">← Braidwater</a>'),
]
for a, b in subs:
    assert t.count(a) == 1, f'expected one {a!r} in {p}'
    t = t.replace(a, b)
t, n = re.subn(r'<span id="xsite">.*?</span>', f'<span id="xsite">Part of the <a href="{main}/lab/">Braidwater NLP Lab</a></span>', t, count=1, flags=re.S)
assert n == 1, 'footer cross-link (#xsite) not found'
open(p, 'w', encoding='utf-8').write(t)
PY
# Serve the lab at / (rewrite, so its absolute /word2vec/... asset paths keep working).
printf '/  /word2vec/  200\n' > "$OUT/_redirects"
echo "Built $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
