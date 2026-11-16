/**
 * Braidwater Text Studio · topics.js — topic inference for new text against the lab's fixed LDA model
 * (K = 15 topics trained with gensim on Reuters-21578 newswire, 1987; training/topics_train.py).
 * Inference is collapsed Gibbs sampling with the topic–word matrix held fixed, ported verbatim from
 * /lab/topics/app.js. Side-effect free; /lab/topics/data/lda.json loads on the first loadTopics() call.
 *
 * Exports:
 *   loadTopics(url='/lab/topics/data/lda.json') → Promise<{ infer(text, sweeps=300) → { theta, known, total, words }, labels, palette, top, meta }>
 *       theta = topic proportions (sums to 1) or null when no word is in the 4,000-word vocabulary
 *   LABELS, PALETTE — hand-written topic names and colors (same as the lab module)
 */
export const LABELS = [
  'Currencies & FX', 'Banks & interest rates', 'Quarterly results tables', 'Executives & steel', 'Analysts & debt markets',
  'Earnings outlook', 'Divestitures & units', 'Trade balance & economy', 'Mergers & tender offers', 'UK money markets & labor',
  'Sugar & monthly exports', 'Oil & energy', 'Grain & agriculture', 'Trade diplomacy', 'Gold, mining & cocoa',
];
export const PALETTE = ['#7ea3c8', '#d8b66a', '#8fbf9a', '#d98c7a', '#b49bd6', '#6cc3c3', '#e0a35c', '#9cb4d8', '#c3d07a', '#e08fb0', '#7fc7a5', '#cfa0e0', '#e6c87a', '#8ab8c9', '#c79a86'];

let D, K, vidx, stop, re;
// ---- collapsed Gibbs inference with fixed phi ----
function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function tokenize(text) {
  re.lastIndex = 0;
  return (text.toLowerCase().match(re) || []).filter((w) => !stop.has(w));
}
function infer(text, sweeps = 300) {
  const toks = tokenize(text), ids = toks.map((w) => vidx.get(w)).filter((i) => i != null);
  if (!ids.length) return { theta: null, known: 0, total: toks.length };
  const r = rng(12345), n = new Float64Array(K), z = new Int32Array(ids.length), p = new Float64Array(K), acc = new Float64Array(K), zc = ids.map(() => new Float64Array(K));
  ids.forEach((w, i) => { z[i] = Math.floor(r() * K); n[z[i]]++; });
  const burn = Math.floor(sweeps / 2);
  for (let it = 0; it < sweeps; it++) {
    for (let i = 0; i < ids.length; i++) {
      n[z[i]]--;
      let s = 0;
      for (let k = 0; k < K; k++) { p[k] = (n[k] + D.alpha[k]) * D.phi[k][ids[i]]; s += p[k]; }
      let u = r() * s, k = 0; while ((u -= p[k]) > 0 && k < K - 1) k++;
      z[i] = k; n[k]++;
      if (it >= burn) zc[i][k]++;
    }
    if (it >= burn) { const tot = ids.length + D.alpha.reduce((a, b) => a + b, 0); for (let k = 0; k < K; k++) acc[k] += (n[k] + D.alpha[k]) / tot; }
  }
  const theta = Array.from(acc, (x) => x / (sweeps - burn));
  const zbest = zc.map((c) => c.indexOf(Math.max(...c)));
  return { theta, known: ids.length, total: toks.length, words: ids.map((w, i) => [D.vocab[w], zbest[i]]) };
}

let _p = null;
export function loadTopics(url = '/lab/topics/data/lda.json') {
  return (_p ||= fetch(url).then((r) => { if (!r.ok) throw new Error(`Couldn’t load ${url} (${r.status})`); return r.json(); }).then((data) => {
    D = data; K = D.k; vidx = new Map(D.vocab.map((w, i) => [w, i])); stop = new Set(D.stopwords); re = new RegExp(D.token_re, 'g');
    return { infer, labels: LABELS, palette: PALETTE, top: D.top, meta: D.meta, k: K };
  }).catch((e) => { _p = null; throw e; }));
}
