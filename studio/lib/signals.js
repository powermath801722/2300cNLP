/**
 * Braidwater Text Studio · signals.js — per-token surprisal and GLTR rank buckets from distilgpt2
 * (same computation as /lab/surprisal/app.js: an <|endoftext|> token is prepended so the first token is scored,
 * surprisal = −log₂ P(token | prefix), rank = position of the actual token in the model's distribution).
 * Side-effect free; transformers.js and the ≈85 MB merged decoder load only when loadSignals() is called.
 *
 * These are statistical signals, not proof of authorship: detectors built on them misfire on formulaic,
 * edited or non-native writing (see the lab's Surprisal module for the caveats).
 *
 * Exports:
 *   SIGNALS_MODEL = 'Xenova/distilgpt2', SIGNALS_SIZE_MB = 85, BUCKETS (GLTR: top-10/100/1000/beyond)
 *   loadSignals({ ui }) → Promise<{ analyze(text, { maxTokens=600, window=300, onProgress }) → Promise<report> }>
 *       report: { tokens: [{ text, bits, p, rank, bucket }], analyzed, totalTokens, meanBits, ppl, buckets: [counts], top10Share }
 */
export const SIGNALS_MODEL = 'Xenova/distilgpt2';
export const SIGNALS_SIZE_MB = 85;
export const BUCKETS = [
  { name: 'top-10', max: 10, color: '#8fbf9a' },
  { name: 'top-100', max: 100, color: '#d8b66a' },
  { name: 'top-1000', max: 1000, color: '#d98c7a' },
  { name: 'beyond 1000', max: Infinity, color: '#b49bd6' },
];
const bucketOf = (rank) => BUCKETS.findIndex((b) => rank <= b.max);
const EOS = 50256, V = 50257, LN2 = Math.LN2;

async function scoreWindow(T, tokenizer, model, used) {
  const input = [EOS].concat(used), n = input.length;
  const o = await model({
    input_ids: new T.Tensor('int64', BigInt64Array.from(input.map(BigInt)), [1, n]),
    attention_mask: new T.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
  });
  const d = o.logits.data, out = [];
  for (let i = 0; i < used.length; i++) {
    const row = d.subarray(i * V, (i + 1) * V), tok = used[i];
    let m = -Infinity; for (let j = 0; j < V; j++) if (row[j] > m) m = row[j];
    let s = 0, rank = 1; const zt = row[tok];
    for (let j = 0; j < V; j++) { s += Math.exp(row[j] - m); if (row[j] > zt) rank++; }
    const lse = m + Math.log(s);
    out.push({ id: tok, text: tokenizer.decode([tok]), bits: (lse - zt) / LN2, p: Math.exp(zt - lse), rank, bucket: bucketOf(rank) });
  }
  return out;
}

let _p = null;
export function loadSignals({ ui } = {}) {
  return (_p ||= import('/lab/shared/hf.js').then(async ({ loadModel, transformers: T }) => {
    const { tokenizer, model } = await loadModel(SIGNALS_MODEL, 'causal', { ui, readyText: 'distilgpt2 ready · cached for next time' });
    return {
      async analyze(text, { maxTokens = 600, window = 300, onProgress } = {}) {
        const ids = tokenizer.encode(text);
        if (!ids.length) throw new Error('No text to analyze.');
        const used = ids.slice(0, maxTokens), tokens = [];
        for (let a = 0; a < used.length; a += window) {
          tokens.push(...await scoreWindow(T, tokenizer, model, used.slice(a, a + window)));
          onProgress && onProgress(Math.min(1, (a + window) / used.length));
          await new Promise((r) => setTimeout(r, 0));
        }
        const meanBits = tokens.reduce((s, t) => s + t.bits, 0) / tokens.length;
        const buckets = BUCKETS.map((_, b) => tokens.filter((t) => t.bucket === b).length);
        return { tokens, analyzed: tokens.length, totalTokens: ids.length, meanBits, ppl: Math.pow(2, meanBits), buckets, top10Share: buckets[0] / tokens.length, model: SIGNALS_MODEL };
      },
    };
  }).catch((e) => { _p = null; throw e; }));
}
