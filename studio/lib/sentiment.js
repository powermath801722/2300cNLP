/**
 * Braidwater Text Studio · sentiment.js — three sentiment scorers, ported verbatim from /lab/sentiment/app.js:
 *   VADER-style lexicon + rules (Hutto & Gilbert 2014), multinomial Naive Bayes and logistic regression
 *   (trained on NLTK movie_reviews by training/sentiment_train.py; held-out accuracy 83.0% / 85.8%).
 * Side-effect free; data (/lab/sentiment/data/vader.json, models.json) loads on the first loadSentiment() call.
 *
 * Exports:
 *   loadSentiment(base='/lab/sentiment/data/') → Promise<{ vader(text), nb(text), lr(text), metrics }>
 *       each scorer → { label: 'pos'|'neg'|'neu', prob (P(positive)), score, detail, ... }
 *   vaderScore(text, lexicon) → same shape as vader(text), for callers holding the lexicon
 *   DISTILBERT = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english' (≈67 MB, load via /lab/shared/hf.js pipe)
 */
// ---------- VADER-style lexicon + rules (after Hutto & Gilbert 2014) ----------
const B_INCR = 0.293, B_DECR = -0.293, C_INCR = 0.733, N_SCALAR = -0.74;
const BOOST = Object.fromEntries([
  ...'absolutely amazingly awfully completely considerably decidedly deeply enormously entirely especially exceptionally extremely fabulously flipping flippin fricking frickin frigging friggin fully fucking greatly hella highly hugely incredibly intensely majorly more most particularly purely quite really remarkably so substantially thoroughly totally tremendously uber unbelievably unusually utterly very'.split(' ').map((w) => [w, B_INCR]),
  ...'almost barely hardly just enough kind of kinda kindof less little marginally occasionally partly scarcely slightly somewhat sort of sorta sortof'.split(' ').map((w) => [w, B_DECR]),
]);
const NEGATE = new Set("aint arent cannot cant couldnt darent didnt doesnt ain't aren't can't couldn't daren't didn't doesn't dont hadnt hasnt havent isnt mightnt mustnt neither don't hadn't hasn't haven't isn't mightn't mustn't neednt needn't never none nope nor not nothing nowhere oughtnt shant shouldnt uhuh wasnt werent oughtn't shan't shouldn't uh-uh wasn't weren't without wont wouldnt won't wouldn't rarely seldom despite".split(' '));

function vader(text, LEX) {
  const raw = text.split(/\s+/).map((t) => t.replace(/^[^\w:;=<>()'-]+|[^\w:;=<>()'-]+$/g, '')).filter((t) => t.length > 1 || /\w/.test(t));
  const isCap = (w) => /[A-Za-z]/.test(w) && w === w.toUpperCase();
  const mixed = raw.some(isCap) && !raw.every(isCap);
  const lower = raw.map((w) => w.toLowerCase().replace(/^'+|'+$/g, ''));
  const val = raw.map((w, i) => {
    const lw = lower[i];
    let v = LEX[lw] ?? LEX[raw[i]];
    if (v == null || BOOST[lw] != null) return 0;
    if (mixed && isCap(w)) v += v > 0 ? C_INCR : -C_INCR;
    for (let k = 1; k <= 3 && i - k >= 0; k++) {
      const p = lower[i - k], b = BOOST[p];
      if (b != null) { let s = (v > 0 ? b : -b) * [1, 0.95, 0.9][k - 1]; if (mixed && isCap(raw[i - k])) s += v > 0 ? C_INCR : -C_INCR; v += s; }
      if (NEGATE.has(p) || p.endsWith("n't")) v *= N_SCALAR;
    }
    return v;
  });
  const bi = lower.indexOf('but');
  if (bi >= 0) for (let i = 0; i < val.length; i++) { if (i < bi) val[i] *= 0.5; else if (i > bi) val[i] *= 1.5; }
  let s = val.reduce((a, b) => a + b, 0);
  const ex = Math.min((text.match(/!/g) || []).length, 4) * 0.292;
  const q = (text.match(/\?/g) || []).length, qa = q > 1 ? (q <= 3 ? q * 0.18 : 0.96) : 0;
  if (s > 0) s += ex + qa; else if (s < 0) s -= ex + qa;
  const compound = s / Math.sqrt(s * s + 15);
  const label = compound >= 0.05 ? 'pos' : compound <= -0.05 ? 'neg' : 'neu';
  return { label, score: compound, prob: (compound + 1) / 2, words: raw.map((w, i) => [w, val[i]]), detail: `compound ${compound.toFixed(3)}` };
}

// ---------- Naive Bayes + logistic regression (trained in training/sentiment_train.py) ----------
function makeBow(M) {
  const re = new RegExp(M.token_re, 'g');
  const idx = new Map(M.vocab.map((w, i) => [w, i]));
  const tokens = (text) => {
    const out = []; let m; const lt = text.toLowerCase(); re.lastIndex = 0;
    // keep original spans for display
    while ((m = re.exec(lt))) out.push({ w: m[0], i: idx.get(m[0]), s: m.index, e: m.index + m[0].length });
    return out;
  };
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const spans = (text, toks, contrib) => {
    const res = []; let last = 0;
    toks.forEach((t, k) => { if (t.s > last) res.push([text.slice(last, t.s), null]); res.push([text.slice(t.s, t.e), contrib[k]]); last = t.e; });
    if (last < text.length) res.push([text.slice(last), null]);
    return res;
  };
  const nb = (text) => {
    const t = tokens(text);
    let z = M.nb.prior[1] - M.nb.prior[0];
    const c = t.map((x) => x.i == null ? 0 : M.nb.logp_pos[x.i] - M.nb.logp_neg[x.i]);
    for (const v of c) z += v;
    const p = sig(z);
    return { label: p >= 0.5 ? 'pos' : 'neg', prob: p, score: z, spans: spans(text, t, t.map((x, k) => x.i == null ? null : c[k])), detail: `log-odds ${z.toFixed(2)} · P(pos) ${p.toFixed(3)}`, known: t.filter((x) => x.i != null).length };
  };
  const lr = (text) => {
    const t = tokens(text), seen = new Set();
    let z = M.lr.bias;
    const c = t.map((x) => { if (x.i == null || seen.has(x.i)) return x.i == null ? null : 0; seen.add(x.i); return M.lr.w[x.i]; });
    for (const v of c) if (v) z += v;
    const p = sig(z);
    return { label: p >= 0.5 ? 'pos' : 'neg', prob: p, score: z, spans: spans(text, t, c), detail: `z = ${z.toFixed(2)} · P(pos) ${p.toFixed(3)}` };
  };
  return { nb, lr };
}

export const vaderScore = vader;
export const DISTILBERT = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';

let _p = null;
export function loadSentiment(base = '/lab/sentiment/data/') {
  return (_p ||= Promise.all(['vader.json', 'models.json'].map((f) => fetch(base + f).then((r) => {
    if (!r.ok) throw new Error(`Couldn’t load ${base + f} (${r.status})`);
    return r.json();
  }))).then(([LEX, M]) => {
    const B = makeBow(M);
    return { vader: (t) => vader(t, LEX), nb: B.nb, lr: B.lr, metrics: M.metrics };
  }).catch((e) => { _p = null; throw e; }));
}
