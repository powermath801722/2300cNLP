# Braidwater — Landing Page

Static landing page for Braidwater ("Clearer Minds for the Planetary Transition"), ready to deploy on Netlify.

## Structure

- `index.html` — the page (markup, styles, and component logic)
- `assets/js/` — page runtime (`dc-runtime.js`), image placeholder component, design-system bundle
- `assets/vendor/` — React 18.3.1 + ReactDOM (self-hosted, no CDN dependency)
- `assets/fonts/` — Barlow / Barlow Condensed (self-hosted woff2)
- `word2vec/` — Word2Vec Lab page (`index.html`, `app.js`) and the exported model in `word2vec/model/`
- `training/train_word2vec.py` — script that trains the model and exports it for the browser
- `tests/e2e.mjs` — Playwright end-to-end check for both pages
- `assets/vendor/plotly-2.35.2.min.js` — Plotly, used for the 2D/3D plots
- `netlify.toml` — Netlify config (no build step; publishes repo root)

## Deploy

On Netlify: **Add new site → Import from Git**, pick this repo, branch `main`.
Leave the build command empty and the publish directory as `.` (already set in `netlify.toml`).

## Local preview

```bash
python3 -m http.server 8080
```

Then open http://localhost:8080.

## Word2Vec Lab (`/word2vec/`)

An in-browser explorer for a Word2Vec model trained for this NLP class. Linked from the landing page nav, mobile menu and footer.

- **Top-10 nearest neighbors** for any word, ranked by cosine similarity.
- **Compare words**: enter up to 6 comma-separated words (e.g. `king, paris, computer`) to see each neighborhood side by side.
- **2D and 3D views**: the query words and their neighbors are projected with PCA onto their top 2 or 3 principal components (interactive Plotly charts; drag to rotate in 3D).
- **Analogies**: *A is to B as C is to ?* via `B − A + C` (3CosAdd).

### Model

| Setting | Value |
|---|---|
| Algorithm | Skip-gram with negative sampling (gensim) |
| Dimensions | 100 |
| Window | 5 |
| Negative samples | 10 |
| Epochs | 15 |
| min_count | 5 |
| Corpus | text8 (cleaned English Wikipedia, ~17M tokens) |
| Shipped vocabulary | 30,000 most frequent words, int8-quantized unit vectors (≈3 MB) |

Files in `word2vec/model/`: `vectors.bin` (int8, N×100), `scales.bin` (float32 per-row scale), `vocab.json` (words, counts, training params).

### Retraining

1. Download the corpus: `text8.gz` from the gensim-data GitHub release (URL in `training/train_word2vec.py`).
2. Install dependencies:

```bash
pip install gensim numpy
```

3. Train and export (a few minutes on 4 cores):

```bash
python training/train_word2vec.py text8.gz word2vec/model
```

### Testing

```bash
python3 -m http.server 8765
```

```bash
node tests/e2e.mjs
```

Screenshots go to `tests/screenshots/` (ignored by git; override with `SHOT_DIR`).

## Notes

- Forms (intake, steward, newsletter, calculator email) are front-end only right now and don't send data anywhere yet.
- Image slots show placeholders until real photography is added.
