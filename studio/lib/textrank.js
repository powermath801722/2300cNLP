/**
 * Braidwater Text Studio · textrank.js — TextRank extractive summarization and keyword extraction
 * (Mihalcea & Tarau 2004). Ported verbatim from /lab/textrank/app.js. Side-effect free, no data to load.
 *
 * Exports:
 *   splitSentences(text) → [{ text, heading, para }]   paragraph-aware; short unterminated lines are headings
 *   pagerank(W, d=0.85, tol=1e-4, maxIt=200) → { scores, trace }
 *   keywords(text, d=0.85) → [{ text, score, phrase }]  up to 15, phrases merged from adjacent top words
 *   summarize(text, { mode: 'overlap'|'tfidf', d, k, split }) → { all, cand, scores, trace, picked, lead, W, order }
 *        cand = non-heading sentences (first 400); picked = indices into cand (document order);
 *        split defaults to splitSentences (the lab module's behavior); Text Studio passes its abbreviation-aware splitter
 */
const STOP = new Set(('a about above after again against all also am an and any are as at be because been before being below between both but by can could did do does doing down during each either every few for from further had has have having he her here hers herself him himself his how however i if in into is it its itself just let may me might more most must my myself neither no nor not now of off on once one only or other our ours ourselves out over own same shall she should so some such than that the their theirs them themselves then there these they this those through thus to too under until up upon us very was we were what when where whether which while who whom whose why will with within without would yet you your yours yourself yourselves hath thee thou thy unto many much cannot make makes made like'.split(' ')));

const ABBR = /\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|No|vs|etc|e\.g|i\.e|U\.S|Gen|Gov|Sen|Rep|Prof)\.$/;

/** Split text into paragraphs of {text, heading} sentences. */
export function splitSentences(text) {
  const paras = text.replace(/\r/g, '').split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, ' ').trim()).filter(Boolean);
  const out = [];
  paras.forEach((p, pi) => {
    if (!/[.!?]["”’')\]]*$/.test(p) && p.split(/\s+/).length < 12) { out.push({ text: p, heading: true, para: pi }); return; }
    const parts = p.split(/(?<=[.!?]["”’')\]]*)\s+(?=["“‘'(\[]?[A-Z0-9])/);
    let buf = '';
    for (const s of parts) {
      buf = buf ? buf + ' ' + s : s;
      if (ABBR.test(buf) || /\b[A-Z]\.$/.test(buf)) continue; // abbreviation or initial: keep going
      out.push({ text: buf, heading: false, para: pi }); buf = '';
    }
    if (buf) out.push({ text: buf, heading: false, para: pi });
  });
  return out;
}

const norm = (w) => (w.length > 4 && w.endsWith('ies') ? w.slice(0, -3) + 'y' : w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w);
const words = (s) => s.toLowerCase().replace(/[’']s\b/g, '').match(/[a-z]+/g) || [];
const content = (s) => words(s).filter((w) => w.length > 2 && !STOP.has(w)).map(norm);

/** Weighted PageRank on a symmetric weight matrix W (array of Float64Array). Returns { scores, trace }. */
export function pagerank(W, d = 0.85, tol = 1e-4, maxIt = 200) {
  const n = W.length, out = W.map((r) => r.reduce((a, b) => a + b, 0));
  let ws = new Float64Array(n).fill(1); const trace = [];
  for (let it = 0; it < maxIt; it++) {
    const nx = new Float64Array(n).fill(1 - d);
    for (let j = 0; j < n; j++) { if (!out[j]) continue; const f = (d * ws[j]) / out[j]; const r = W[j]; for (let i = 0; i < n; i++) if (r[i]) nx[i] += r[i] * f; }
    let delta = 0; for (let i = 0; i < n; i++) delta = Math.max(delta, Math.abs(nx[i] - ws[i]));
    ws = nx; trace.push(delta);
    if (delta < tol) break;
  }
  return { scores: ws, trace };
}

function simMatrix(sents, mode) {
  const n = sents.length, W = Array.from({ length: n }, () => new Float64Array(n));
  const toks = sents.map((s) => content(s.text));
  if (mode === 'overlap') {
    const sets = toks.map((t) => new Set(t)), lens = sents.map((s) => words(s.text).length);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let c = 0; for (const w of sets[i]) if (sets[j].has(w)) c++;
      const den = Math.log(lens[i]) + Math.log(lens[j]);
      if (c && den > 0) W[i][j] = W[j][i] = c / den;
    }
  } else {
    const df = new Map(); toks.forEach((t) => new Set(t).forEach((w) => df.set(w, (df.get(w) || 0) + 1)));
    const vecs = toks.map((t) => { const tf = new Map(); t.forEach((w) => tf.set(w, (tf.get(w) || 0) + 1)); const v = new Map(); let s = 0; for (const [w, c] of tf) { const x = c * Math.log(n / df.get(w) + 1e-9); if (x > 0) { v.set(w, x); s += x * x; } } return { v, norm: Math.sqrt(s) }; });
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const a = vecs[i], b = vecs[j]; if (!a.norm || !b.norm) continue;
      let dot = 0; for (const [w, x] of a.v) { const y = b.v.get(w); if (y) dot += x * y; }
      if (dot) W[i][j] = W[j][i] = dot / (a.norm * b.norm);
    }
  }
  return W;
}

/** TextRank keyword extraction: co-occurrence graph (window 2) over the stopword-filtered word sequence. */
export function keywords(text, d = 0.85) {
  // tokens: {key, surface} for candidate words, null for stopwords/punctuation (these break phrases)
  const toks = (text.replace(/[’']s\b/g, '').match(/[A-Za-z]+|[^A-Za-z\s]+/g) || []).map((t) => {
    const w = t.toLowerCase();
    return /^[a-z]+$/.test(w) && w.length > 2 && !STOP.has(w) ? { key: norm(w), surface: w } : null;
  });
  const filtered = toks.filter(Boolean);
  const vocab = new Map(); filtered.forEach((t) => { if (!vocab.has(t.key)) vocab.set(t.key, vocab.size); });
  const n = vocab.size; if (!n) return [];
  const W = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i + 1 < filtered.length; i++) { const x = vocab.get(filtered[i].key), y = vocab.get(filtered[i + 1].key); if (x !== y) W[x][y] = W[y][x] = 1; }
  const { scores } = pagerank(W, d);
  const keys = [...vocab.keys()], sc = (k) => scores[vocab.get(k)];
  const T = Math.max(5, Math.min(20, Math.round(n / 3)));
  const top = new Set(keys.map((k) => [k, sc(k)]).sort((p, q) => q[1] - p[1]).slice(0, T).map(([k]) => k));
  const surface = new Map(); filtered.forEach((t) => { if (!surface.has(t.key)) surface.set(t.key, t.surface); });
  const phrases = new Map(), used = new Set();
  for (let i = 0; i < toks.length; i++) {
    if (!toks[i] || !top.has(toks[i].key)) continue;
    let j = i; while (j + 1 < toks.length && toks[j + 1] && top.has(toks[j + 1].key)) j++;
    if (j > i) {
      const ph = toks.slice(i, j + 1), key = ph.map((t) => t.key).join(' ');
      if (!phrases.has(key)) phrases.set(key, { text: ph.map((t) => t.surface).join(' '), score: ph.reduce((a, t) => a + sc(t.key), 0) / ph.length * Math.sqrt(ph.length), phrase: true });
      ph.forEach((t) => used.add(t.key)); i = j;
    }
  }
  const singles = [...top].filter((k) => !used.has(k)).map((k) => ({ text: surface.get(k), score: sc(k), phrase: false }));
  return [...phrases.values(), ...singles].sort((p, q) => q.score - p.score).slice(0, 15);
}

/** Full pipeline. */
export function summarize(text, { mode = 'overlap', d = 0.85, k = 3, split = splitSentences } = {}) {
  const all = split(text).slice(0, 400);
  const cand = all.filter((s) => !s.heading);
  if (!cand.length) return { all, cand, scores: [], trace: [], picked: [], lead: [], W: [] };
  const W = simMatrix(cand, mode);
  const { scores, trace } = pagerank(W, d);
  const order = [...scores.keys()].sort((a, b) => scores[b] - scores[a]);
  const picked = order.slice(0, Math.min(k, cand.length)).sort((a, b) => a - b);
  const lead = [...cand.keys()].slice(0, Math.min(k, cand.length));
  return { all, cand, scores, trace, picked, lead, W, order };
}
