# Braidwater — Landing Page

Static landing page for Braidwater ("Clearer Minds for the Planetary Transition"), ready to deploy on Netlify.

## Structure

- `index.html` — the page (markup, styles, and component logic)
- `assets/js/` — page runtime (`dc-runtime.js`), image placeholder component, design-system bundle
- `assets/vendor/` — React 18.3.1 + ReactDOM (self-hosted, no CDN dependency)
- `assets/fonts/` — Barlow / Barlow Condensed (self-hosted woff2)
- `word2vec/` — Word2Vec Lab page (`index.html`, `app.js`) and the exported model in `word2vec/model/`
- `training/train_word2vec.py` — script that trains the model and exports it for the browser
- `tests/e2e.mjs` — Playwright end-to-end check for the landing page and Word2Vec Lab
- `lab/` — NLP Lab: hub and 19 module pages (see below)
- `tests/lab.mjs`, `tests/lab/` — NLP Lab tests
- `assets/vendor/plotly-2.35.2.min.js` — Plotly, used for the 2D/3D plots
- `netlify.toml` — Netlify config for the main site (no build step; publishes repo root)
- `forms.html` — hidden static forms that register the landing page's forms with Netlify Forms
- `sites/word2vec-lab/` — build script and Netlify config for deploying the Word2Vec Lab as its own site

## Deploy

This repo holds two Netlify projects. Create each one in Netlify with **Add new site → Import an existing project → GitHub**, pick this repo and branch `main`. After that, every push to `main` redeploys both.

| Project | Base directory | Build command / publish dir | What it serves |
|---|---|---|---|
| Braidwater site | *(leave empty)* | from `netlify.toml`: none / `.` | Landing page at `/`, Word2Vec Lab at `/word2vec/`, Netlify Forms |
| Word2Vec Lab (standalone) | `sites/word2vec-lab` | from `sites/word2vec-lab/netlify.toml`: `bash build.sh` / `dist` | The lab on its own, at `/` |

### Netlify Forms

The landing page's forms are rendered by JavaScript, so `forms.html` declares them as static hidden forms for Netlify's deploy-time detection. Submissions are POSTed to `/` with a `form-name`:

| Form name | Fields |
|---|---|
| `intake` | name, email, where, role, question, source, plan, consent |
| `steward` | name, email, commons, crew, mind, consent |
| `calculator-results` | name, email, consent, hoursPerDay, lostMinutesPerDay, decisionsPerWeek, monthlyDividend, sharePercent, reclaimedHoursPerYear |
| `signin-request` | email (there is no account system yet; this just records the request) |
| `newsletter` | email |

On the Braidwater site in Netlify, open **Forms** and click **Enable form detection** if it's off, then redeploy once. Submissions appear under **Forms**, and you can add email notifications under **Site configuration → Notifications → Emails and webhooks → Form submission notifications**. Keep `forms.html` in sync if a form's fields change.

## Local preview

```bash
python3 -m http.server 8080
```

Then open http://localhost:8080.

## NLP Lab (`/lab/`)

Twenty interactive NLP modules in six tiers, linked from the landing page nav, menu, footer and section 09. Everything runs in the browser: classic modules use data shipped in `lab/<slug>/data/`; transformer modules download a quantized model from Hugging Face on first use (sizes are shown on each page) and cache it.

| Tier | Modules |
|---|---|
| 1 Beginner | tokenize, stemming, ngram |
| 2 Intermediate | spell, search, sentiment, textrank |
| 3 Advanced | pos, topics, word2vec, ner |
| 4 Expert | tokenizers, semantic-search, qa, zero-shot, translation |
| 5 Graduate | contextual, generation |
| 6 Research | bias, surprisal |

- `lab/shared/modules.js`: module registry (titles, tiers) and the `LIVE` list that controls which modules the hub links.
- `lab/shared/lab.js`, `lab.css`: shared header/hero/pager/footer and styles. `lab/shared/hf.js`: transformers.js loader (self-hosted v3.8.1 in `assets/vendor/transformers-3.8.1/`).
- `lab/_template/index.html`: starting point for a new module.
- `training/`: scripts that build each module's shipped data (NLTK corpora, text8, gensim LDA, scikit-style NB/LR, HMM, embeddings).
- Tests: `node tests/lab.mjs [baseUrl] [slug ...]` runs every module (or the named ones) at 1440px and 390px, plus `tests/lab/<slug>.mjs` checks. Transformer tests download models, so a full run takes several minutes.

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

- Image slots show placeholders until real photography is added.
