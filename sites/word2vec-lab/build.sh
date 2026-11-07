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
# Standalone site: the lab is the home page, so drop the link back to the landing page.
sed -i.bak 's#<a class="l" href="/">← Home</a>##' "$OUT/word2vec/index.html" && rm "$OUT/word2vec/index.html.bak"
# Serve the lab at / (rewrite, so its absolute /word2vec/... asset paths keep working).
printf '/  /word2vec/  200\n' > "$OUT/_redirects"
echo "Built $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
