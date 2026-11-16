// Shared sentiment models for the attribution, calibration and adversarial modules.
// All scorers return the log-odds of the POSITIVE class, log P(pos) − log P(neg), for a batch of texts.
//
//   const lex = await lexModels();            // Naive Bayes + logistic regression from /lab/sentiment (NLTK movie_reviews)
//   lex.nb(['great film'])  -> [z]           // additive in token counts
//   lex.lr(['great film'])  -> [z]           // linear in binary word-presence features
//   const bert = await distilbert(ui);        // Xenova/distilbert-base-uncased-finetuned-sst-2-english (q8, ~67 MB)
//   await bert.score(texts, { onProgress, signal }) -> [z]
import { getJSON } from '/lab/shared/lab.js';

export const BERT_ID = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';
export const BERT_MB = 67;
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

let lexP;
/** Naive Bayes and logistic regression trained in training/sentiment_train.py. */
export function lexModels() {
  lexP ||= getJSON('/lab/sentiment/data/models.json').then((M) => {
    const idx = new Map(M.vocab.map((w, i) => [w, i]));
    const re = new RegExp(M.token_re, 'g');
    const toks = (text) => { re.lastIndex = 0; return text.toLowerCase().match(re) || []; };
    const dNB = M.nb.logp_pos.map((p, i) => p - M.nb.logp_neg[i]);
    const nb1 = (text) => { let z = M.nb.prior[1] - M.nb.prior[0]; for (const t of toks(text)) { const i = idx.get(t); if (i != null) z += dNB[i]; } return z; };
    const lrWith = (w) => (text) => { let z = M.lr.bias; const seen = new Set(); for (const t of toks(text)) { const i = idx.get(t); if (i != null && !seen.has(i)) { seen.add(i); z += w[i]; } } return z; };
    const lr1 = lrWith(M.lr.w);
    return {
      M, idx, toks,
      nbWord: (w) => { const i = idx.get(w.toLowerCase()); return i == null ? 0 : dNB[i]; },
      lrWord: (w) => { const i = idx.get(w.toLowerCase()); return i == null ? 0 : M.lr.w[i]; },
      nb: async (texts) => texts.map(nb1),
      lr: async (texts) => texts.map(lr1),
      nbSync: nb1, lrSync: lr1,
      /** Logistic regression with its weights permuted (model-randomization sanity check, Adebayo et al. 2018). */
      lrRandomized(seed = 1) {
        const w = M.lr.w.slice(); let s = seed >>> 0;
        const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
        for (let i = w.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [w[i], w[j]] = [w[j], w[i]]; }
        const f = lrWith(w);
        return { score: async (texts) => texts.map(f), word: (x) => { const i = idx.get(x.toLowerCase()); return i == null ? 0 : w[i]; } };
      },
      metrics: M.metrics,
    };
  });
  return lexP;
}

let bertP;
/** DistilBERT fine-tuned on SST-2, scored in batches with raw logits (label 1 = POSITIVE). */
export function distilbert(ui) {
  bertP ||= (async () => {
    const { transformers } = await import('/lab/shared/hf.js');
    const { AutoTokenizer, AutoModelForSequenceClassification } = transformers;
    const cb = ui ? (x) => ui.onProgress(x) : undefined;
    ui && ui.set(`Downloading DistilBERT (≈${BERT_MB} MB, once)…`, 0);
    const [tok, model] = await Promise.all([
      AutoTokenizer.from_pretrained(BERT_ID, { progress_callback: cb }),
      AutoModelForSequenceClassification.from_pretrained(BERT_ID, { dtype: 'q8', device: 'wasm', progress_callback: cb }),
    ]);
    const pos = Number(Object.entries(model.config.id2label).find(([, v]) => /pos/i.test(v))[0]);
    ui && ui.done('DistilBERT ready · runs locally in your browser');
    let calls = 0;
    async function score(texts, { batch = 16, onProgress, signal } = {}) {
      const out = new Array(texts.length);
      for (let i = 0; i < texts.length; i += batch) {
        if (signal && signal.aborted) throw new DOMException('Cancelled', 'AbortError');
        const chunk = texts.slice(i, i + batch).map((t) => t.trim() || '.');
        const enc = tok(chunk, { padding: true, truncation: true, max_length: 128 });
        const { logits } = await model(enc);
        const d = logits.data, C = logits.dims[1];
        for (let j = 0; j < chunk.length; j++) out[i + j] = d[j * C + pos] - d[j * C + (1 - pos)];
        calls += chunk.length;
        onProgress && onProgress(Math.min(i + batch, texts.length), texts.length);
        await new Promise((r) => setTimeout(r, 0)); // keep the page responsive between batches
      }
      return out;
    }
    return { score, get calls() { return calls; } };
  })().catch((e) => { bertP = null; ui && ui.error(e); throw e; });
  return bertP;
}

/** Word-level segmentation shared by the modules: words are features; punctuation stays fixed. */
export function segment(text) {
  const parts = text.match(/[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]/g) || [];
  return parts.map((t) => ({ t, word: /[A-Za-z0-9]/.test(t) }));
}
/** Rebuild text keeping parts where keep(i) is true (punctuation always kept). */
export function rebuild(parts, keep) {
  let s = '';
  parts.forEach((p, i) => {
    if (p.word && !keep(i)) return;
    s += (s && !/^[.,!?;:)'"]$/.test(p.t) ? ' ' : '') + p.t;
  });
  return s;
}

/** Small seeded RNG (mulberry32). */
export function rng(seed = 42) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
