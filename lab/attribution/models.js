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
    // Default batch size 1: the q8 model quantizes activations with per-batch scales, so batching padded sentences together
    // shifts a sentence's log-odds by up to ~3 depending on its batch-mates. One sentence at a time is exactly reproducible
    // and, without padding, also faster in single-threaded WASM (≈67 vs ≈94 ms per SST-2 sentence measured here).
    async function score(texts, { batch = 1, onProgress, signal } = {}) {
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

export { segment, rebuild, rng } from './text.js';
