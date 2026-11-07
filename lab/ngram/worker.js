// N-gram language model worker: counting, smoothing, sampling and evaluation off the main thread.
// Messages: {id, cmd, ...args} → {id, ok, result | error}

const BOS = '<s>', EOS = '</s>', UNK = '<UNK>';
const TOK = /[a-z]+(?:'[a-z]+)?|[0-9]+|[.,;:!?]/g;
export const tokenize = (s) => (s.toLowerCase().replace(/[’‘]/g, "'").match(TOK) || []);

let M = null; // trained model

function build(lines, minCount) {
  const sents = lines.map((l) => l.split(' ').filter(Boolean)).filter((s) => s.length);
  // deterministic train/held-out split: every 10th sentence is held out
  const train = [], test = [];
  sents.forEach((s, i) => (i % 10 === 9 ? test : train).push(s));
  const raw = new Map();
  for (const s of train) for (const w of s) raw.set(w, (raw.get(w) || 0) + 1);
  const vocab = [BOS, EOS, UNK, ...[...raw].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1]).map(([w]) => w)];
  const id = new Map(vocab.map((w, i) => [w, i]));
  const V = vocab.length;
  const enc = (s) => [0, 0, ...s.map((w) => (id.has(w) ? id.get(w) : 2)), 1];
  const uni = new Float64Array(V), bi = new Map(), tri = new Map(), ctx1 = new Float64Array(V), ctx2 = new Map();
  let N = 0;
  const inc = (m, k) => m.set(k, (m.get(k) || 0) + 1);
  for (const s of train) {
    const t = enc(s);
    for (let i = 2; i < t.length; i++) {
      const u = t[i - 2], v = t[i - 1], w = t[i];
      uni[w]++; N++;
      inc(bi, v * V + w); ctx1[v]++;
      inc(tri, (u * V + v) * V + w); inc(ctx2, u * V + v);
    }
  }
  // distinct followers N1+(h•)
  const fol1 = new Float64Array(V), fol2 = new Map();
  for (const k of bi.keys()) fol1[Math.floor(k / V)]++;
  for (const k of tri.keys()) inc(fol2, Math.floor(k / V));
  // Kneser–Ney continuation counts
  const contUni = new Float64Array(V); // N1+(•w) over bigram types
  for (const k of bi.keys()) contUni[k % V]++;
  const biTypes = bi.size;
  const contBi = new Map();           // N1+(•vw) over trigram types, key v*V+w
  const contMid = new Float64Array(V); // N1+(•v•)
  for (const k of tri.keys()) { const vw = k % (V * V); inc(contBi, vw); contMid[Math.floor(vw / V)]++; }
  const contMidFol = new Float64Array(V); // |{w : N1+(•vw) > 0}|
  for (const k of contBi.keys()) contMidFol[Math.floor(k / V)]++;
  M = { vocab, id, V, uni, bi, tri, ctx1, ctx2, fol1, fol2, contUni, biTypes, contBi, contMid, contMidFol, N, train, test, enc };
  const testTokens = test.reduce((a, s) => a + s.length + 1, 0);
  return { sentences: sents.length, trainSentences: train.length, testSentences: test.length, trainTokens: N, testTokens, V, unigrams: V - 1, bigrams: bi.size, trigrams: tri.size, rawTypes: raw.size };
}

// ---------- probability of w given (u, v) under a method; order = 1 | 2 | 3
function prob(w, u, v, order, method, p) {
  const { V, uni, bi, tri, ctx1, ctx2, N } = M;
  const Vp = V - 1; // predictable types (everything but <s>)
  const pUni = () => uni[w] / N;
  const pBi = () => (ctx1[v] ? (bi.get(v * V + w) || 0) / ctx1[v] : 0);
  const pTri = () => { const c = ctx2.get(u * V + v) || 0; return c ? (tri.get((u * V + v) * V + w) || 0) / c : 0; };
  switch (method) {
    case 'mle': return order === 1 ? pUni() : order === 2 ? pBi() : pTri();
    case 'addk': {
      const k = p.k;
      if (order === 1) return (uni[w] + k) / (N + k * Vp);
      if (order === 2) return ((bi.get(v * V + w) || 0) + k) / (ctx1[v] + k * Vp);
      return ((tri.get((u * V + v) * V + w) || 0) + k) / ((ctx2.get(u * V + v) || 0) + k * Vp);
    }
    case 'interp': {
      const [l1, l2, l3] = p.lambdas[order];
      const base = 0.999 * pUni() + 0.001 / Vp; // tiny uniform floor keeps P > 0
      if (order === 1) return base;
      if (order === 2) return l1 * base + l2 * pBi();
      return l1 * base + l2 * pBi() + l3 * pTri();
    }
    case 'kn': {
      if (order === 1) return pUni();
      const D = p.D;
      const pCont = M.contUni[w] / M.biTypes;
      if (order === 2) {
        const c = ctx1[v];
        if (!c) return pCont;
        return Math.max((bi.get(v * V + w) || 0) - D, 0) / c + (D * M.fol1[v] / c) * pCont;
      }
      // middle order uses continuation counts
      const cm = M.contMid[v];
      const pKN2 = cm ? Math.max((M.contBi.get(v * V + w) || 0) - D, 0) / cm + (D * M.contMidFol[v] / cm) * pCont : pCont;
      const c = ctx2.get(u * V + v) || 0;
      if (!c) return pKN2;
      return Math.max((tri.get((u * V + v) * V + w) || 0) - D, 0) / c + (D * (M.fol2.get(u * V + v) || 0) / c) * pKN2;
    }
    case 'backoff': { // stupid backoff score (not a probability)
      const a = 0.4;
      if (order === 3) { const c = ctx2.get(u * V + v) || 0, t = tri.get((u * V + v) * V + w) || 0; if (c && t) return t / c; }
      if (order >= 2) { const b = bi.get(v * V + w) || 0; if (ctx1[v] && b) return (order === 3 ? a : 1) * b / ctx1[v]; }
      return (order === 3 ? a * a : order === 2 ? a : 1) * pUni();
    }
  }
  return 0;
}

function dist(u, v, order, method, p) {
  const out = new Float64Array(M.V);
  let s = 0;
  for (let w = 1; w < M.V; w++) { out[w] = prob(w, u, v, order, method, p); s += out[w]; }
  return { d: out, sum: s };
}

function ctxIds(words) {
  const ids = words.map((w) => (M.id.has(w) ? M.id.get(w) : 2));
  const t = [0, 0, ...ids];
  return { u: t[t.length - 2], v: t[t.length - 1], mapped: ids.map((i) => M.vocab[i]) };
}

const handlers = {
  async train({ url, minCount }) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Couldn’t load corpus (${r.status})`);
    const lines = (await r.text()).split('\n');
    const t0 = performance.now();
    const stats = build(lines, minCount);
    stats.ms = Math.round(performance.now() - t0);
    return stats;
  },
  next({ context, order, method, p, k = 12 }) {
    const words = tokenize(context);
    const { u, v, mapped } = ctxIds(words);
    const { d, sum } = dist(u, v, order, method, p);
    const idx = [...d.keys()].filter((i) => i > 0 && d[i] > 0).sort((a, b) => d[b] - d[a]).slice(0, k);
    const V = M.V;
    return {
      context: order === 1 ? [] : order === 2 ? [M.vocab[v]] : [M.vocab[u], M.vocab[v]],
      contextCount: order === 1 ? M.N : order === 2 ? M.ctx1[v] : (M.ctx2.get(u * V + v) || 0),
      mass: sum,
      mapped,
      top: idx.map((w) => ({ w: M.vocab[w], p: d[w] / (method === 'backoff' ? sum : 1), count: order === 1 ? M.uni[w] : order === 2 ? (M.bi.get(v * V + w) || 0) : (M.tri.get((u * V + v) * V + w) || 0) })),
    };
  },
  generate({ seed, order, method, p, temperature, maxLen = 40, rngSeed }) {
    let rnd = mulberry32(rngSeed ?? (Math.random() * 2 ** 32) >>> 0);
    const words = tokenize(seed);
    const ids = words.map((w) => (M.id.has(w) ? M.id.get(w) : 2));
    const t = [0, 0, ...ids];
    const out = [];
    for (let i = 0; i < maxLen; i++) {
      const u = t[t.length - 2], v = t[t.length - 1];
      const { d } = dist(u, v, order, method, p);
      d[2] = 0; // never emit <UNK>
      if (i === 0 && !ids.length) d[1] = 0; // don't end before starting
      let s = 0; const T = 1 / Math.max(temperature, 0.05);
      for (let w = 1; w < d.length; w++) { d[w] = d[w] > 0 ? Math.pow(d[w], T) : 0; s += d[w]; }
      if (!s) break;
      let r = rnd() * s, w = 1;
      for (; w < d.length - 1; w++) { r -= d[w]; if (r <= 0) break; }
      if (w === 1) break;
      t.push(w); out.push(M.vocab[w]);
    }
    return { seed: words.map((w, i) => (ids[i] === 2 ? UNK : w)), words: out };
  },
  score({ sentence, order, method, p }) {
    const words = tokenize(sentence);
    const ids = words.map((w) => (M.id.has(w) ? M.id.get(w) : 2));
    const t = [0, 0, ...ids, 1];
    const V = M.V, rows = [];
    let lp = 0;
    for (let i = 2; i < t.length; i++) {
      const u = t[i - 2], v = t[i - 1], w = t[i];
      const pr = method === 'backoff' ? prob(w, u, v, order, method, p) / dist(u, v, order, method, p).sum : prob(w, u, v, order, method, p);
      lp += Math.log2(pr);
      rows.push({ w: i - 2 < words.length ? words[i - 2] : EOS, mapped: M.vocab[w], p: pr, tri: M.tri.get((u * V + v) * V + w) || 0, bi: M.bi.get(v * V + w) || 0, uni: M.uni[w] });
    }
    return { rows, log2: lp, ppl: Math.pow(2, -lp / rows.length) };
  },
  perplexity({ p }) {
    const methods = ['mle', 'addk', 'interp', 'kn'];
    const V = M.V, res = {};
    const test = M.test.map(M.enc);
    for (const method of methods) {
      res[method] = [];
      for (const order of [1, 2, 3]) {
        let lp = 0, n = 0, zeros = 0;
        for (const t of test) for (let i = 2; i < t.length; i++) {
          const pr = prob(t[i], t[i - 2], t[i - 1], order, method, p);
          n++;
          if (pr <= 0) zeros++; else lp += Math.log2(pr);
        }
        res[method].push({ order, ppl: zeros ? Infinity : Math.pow(2, -lp / n), zeros, n });
      }
    }
    return res;
  },
};

function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

self.onmessage = async (e) => {
  const { id, cmd, ...args } = e.data;
  try {
    const result = await handlers[cmd](args);
    self.postMessage({ id, ok: true, result: JSON.parse(JSON.stringify(result, (k, v) => (v === Infinity ? 'Infinity' : v))) });
  } catch (err) { self.postMessage({ id, ok: false, error: String(err && err.message || err) }); }
};
