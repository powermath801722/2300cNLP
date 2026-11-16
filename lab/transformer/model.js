// A tiny decoder-only transformer with a hand-written reverse-mode autograd over Float32Arrays.
// Runs in the browser (inside worker.js) and in Node (training/transformer_check.mjs gradient check).
// Everything is 2-D: a tensor is a row-major [rows, cols] matrix; a batch of B sequences of length T
// is stored as B*T rows.

// Array type for all tensors. Float32 in the browser; the gradient check switches to Float64.
let FA = Float32Array;
export function useFloat64(on = true) { FA = on ? Float64Array : Float32Array; }

// ---------- seeded randomness ----------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- tensors + tape ----------
export class Tensor {
  constructor(rows, cols, data) {
    this.rows = rows; this.cols = cols;
    this.data = data || new FA(rows * cols);
    this.grad = null;
  }
  g() { return this.grad || (this.grad = new FA(this.data.length)); }
}

export class Tape {
  constructor() { this.ops = []; }
  push(fn) { this.ops.push(fn); }
  backward() { for (let i = this.ops.length - 1; i >= 0; i--) this.ops[i](); this.ops = []; }
}

// Y = X W  (X: n×k, W: k×m)
export function matmul(tape, X, W) {
  const n = X.rows, k = X.cols, m = W.cols;
  if (W.rows !== k) throw new Error(`matmul shape ${n}x${k} · ${W.rows}x${m}`);
  const Y = new Tensor(n, m), x = X.data, w = W.data, y = Y.data;
  for (let i = 0; i < n; i++) {
    const yi = i * m, xi = i * k;
    for (let p = 0; p < k; p++) {
      const a = x[xi + p]; if (a === 0) continue;
      const wp = p * m;
      for (let j = 0; j < m; j++) y[yi + j] += a * w[wp + j];
    }
  }
  tape && tape.push(() => {
    if (!Y.grad) return;
    const dy = Y.grad, dx = X.g(), dw = W.g();
    for (let i = 0; i < n; i++) {
      const yi = i * m, xi = i * k;
      for (let p = 0; p < k; p++) {
        const wp = p * m; let s = 0; const a = x[xi + p];
        for (let j = 0; j < m; j++) { const d = dy[yi + j]; s += d * w[wp + j]; dw[wp + j] += a * d; }
        dx[xi + p] += s;
      }
    }
  });
  return Y;
}

export function add(tape, A, B) {
  const Y = new Tensor(A.rows, A.cols), a = A.data, b = B.data, y = Y.data;
  for (let i = 0; i < y.length; i++) y[i] = a[i] + b[i];
  tape && tape.push(() => {
    if (!Y.grad) return; const dy = Y.grad, da = A.g(), db = B.g();
    for (let i = 0; i < dy.length; i++) { da[i] += dy[i]; db[i] += dy[i]; }
  });
  return Y;
}

// Adds a row vector b (1×m) to every row.
export function addRow(tape, A, b) {
  const Y = new Tensor(A.rows, A.cols), m = A.cols, a = A.data, y = Y.data, bb = b.data;
  for (let i = 0; i < y.length; i++) y[i] = a[i] + bb[i % m];
  tape && tape.push(() => {
    if (!Y.grad) return; const dy = Y.grad, da = A.g(), db = b.g();
    for (let i = 0; i < dy.length; i++) { da[i] += dy[i]; db[i % m] += dy[i]; }
  });
  return Y;
}

export function relu(tape, A) {
  const Y = new Tensor(A.rows, A.cols), a = A.data, y = Y.data;
  for (let i = 0; i < y.length; i++) y[i] = a[i] > 0 ? a[i] : 0;
  tape && tape.push(() => {
    if (!Y.grad) return; const dy = Y.grad, da = A.g();
    for (let i = 0; i < dy.length; i++) if (a[i] > 0) da[i] += dy[i];
  });
  return Y;
}

// Token embedding lookup + learned positional embedding: X[b*T+t] = E[ids] + P[t]
export function embed(tape, E, P, ids, B, T) {
  const d = E.cols, X = new Tensor(B * T, d), x = X.data, e = E.data, p = P.data;
  for (let r = 0; r < B * T; r++) {
    const t = r % T, id = ids[r];
    for (let j = 0; j < d; j++) x[r * d + j] = e[id * d + j] + p[t * d + j];
  }
  tape && tape.push(() => {
    if (!X.grad) return; const dx = X.grad, de = E.g(), dp = P.g();
    for (let r = 0; r < B * T; r++) {
      const t = r % T, id = ids[r];
      for (let j = 0; j < d; j++) { const g = dx[r * d + j]; de[id * d + j] += g; dp[t * d + j] += g; }
    }
  });
  return X;
}

export function layernorm(tape, X, gain, bias, eps = 1e-5) {
  const n = X.rows, d = X.cols, Y = new Tensor(n, d), x = X.data, y = Y.data, g = gain.data, b = bias.data;
  const xhat = new FA(n * d), rstd = new FA(n);
  for (let i = 0; i < n; i++) {
    let mu = 0; for (let j = 0; j < d; j++) mu += x[i * d + j]; mu /= d;
    let v = 0; for (let j = 0; j < d; j++) { const c = x[i * d + j] - mu; v += c * c; } v /= d;
    const r = 1 / Math.sqrt(v + eps); rstd[i] = r;
    for (let j = 0; j < d; j++) { const h = (x[i * d + j] - mu) * r; xhat[i * d + j] = h; y[i * d + j] = h * g[j] + b[j]; }
  }
  tape && tape.push(() => {
    if (!Y.grad) return; const dy = Y.grad, dx = X.g(), dg = gain.g(), db = bias.g();
    for (let i = 0; i < n; i++) {
      let s1 = 0, s2 = 0;
      for (let j = 0; j < d; j++) {
        const k = i * d + j, dh = dy[k] * g[j];
        dg[j] += dy[k] * xhat[k]; db[j] += dy[k];
        s1 += dh; s2 += dh * xhat[k];
      }
      for (let j = 0; j < d; j++) {
        const k = i * d + j, dh = dy[k] * g[j];
        dx[k] += rstd[i] * (dh - s1 / d - xhat[k] * s2 / d);
      }
    }
  });
  return Y;
}

// Causal multi-head attention on precomputed Q, K, V (each B*T × d). Returns O (B*T × d) and the
// attention probabilities probs[b][h] as Float32Array(T*T) (row t = query position).
export function attention(tape, Q, K, V, B, T, H) {
  const d = Q.cols, dh = d / H, scale = 1 / Math.sqrt(dh);
  const O = new Tensor(B * T, d), q = Q.data, k = K.data, v = V.data, o = O.data;
  const probs = [];
  for (let b = 0; b < B; b++) {
    const ph = [];
    for (let h = 0; h < H; h++) {
      const A = new FA(T * T), off = h * dh;
      for (let t = 0; t < T; t++) {
        const qi = (b * T + t) * d + off; let mx = -Infinity;
        for (let u = 0; u <= t; u++) {
          const ki = (b * T + u) * d + off; let s = 0;
          for (let j = 0; j < dh; j++) s += q[qi + j] * k[ki + j];
          s *= scale; A[t * T + u] = s; if (s > mx) mx = s;
        }
        let z = 0;
        for (let u = 0; u <= t; u++) { const e = Math.exp(A[t * T + u] - mx); A[t * T + u] = e; z += e; }
        for (let u = 0; u <= t; u++) {
          const a = A[t * T + u] / z; A[t * T + u] = a;
          const vi = (b * T + u) * d + off, oi = (b * T + t) * d + off;
          for (let j = 0; j < dh; j++) o[oi + j] += a * v[vi + j];
        }
      }
      ph.push(A);
    }
    probs.push(ph);
  }
  tape && tape.push(() => {
    if (!O.grad) return;
    const dO = O.grad, dQ = Q.g(), dK = K.g(), dV = V.g();
    const dA = new FA(T);
    for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
      const A = probs[b][h], off = h * dh;
      for (let t = 0; t < T; t++) {
        const oi = (b * T + t) * d + off; let dot = 0;
        for (let u = 0; u <= t; u++) {
          const vi = (b * T + u) * d + off; let s = 0; const a = A[t * T + u];
          for (let j = 0; j < dh; j++) { s += dO[oi + j] * v[vi + j]; dV[vi + j] += a * dO[oi + j]; }
          dA[u] = s; dot += a * s;
        }
        const qi = oi;
        for (let u = 0; u <= t; u++) {
          const ds = A[t * T + u] * (dA[u] - dot) * scale; if (ds === 0) continue;
          const ki = (b * T + u) * d + off;
          for (let j = 0; j < dh; j++) { dQ[qi + j] += ds * k[ki + j]; dK[ki + j] += ds * q[qi + j]; }
        }
      }
    }
  });
  return { O, probs };
}

// Mean cross-entropy over rows whose target >= 0. Returns { loss, perRow } and seeds the logits gradient.
export function crossEntropy(tape, logits, targets) {
  const n = logits.rows, V = logits.cols, z = logits.data;
  const p = new FA(n * V), perRow = new FA(n).fill(NaN);
  let loss = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    let mx = -Infinity; for (let j = 0; j < V; j++) if (z[i * V + j] > mx) mx = z[i * V + j];
    let s = 0; for (let j = 0; j < V; j++) { const e = Math.exp(z[i * V + j] - mx); p[i * V + j] = e; s += e; }
    for (let j = 0; j < V; j++) p[i * V + j] /= s;
    const y = targets[i];
    if (y >= 0) { const l = -Math.log(Math.max(p[i * V + y], 1e-30)); perRow[i] = l; loss += l; cnt++; }
  }
  loss /= Math.max(cnt, 1);
  tape && tape.push(() => {
    const dz = logits.g();
    for (let i = 0; i < n; i++) {
      const y = targets[i]; if (y < 0) continue;
      for (let j = 0; j < V; j++) dz[i * V + j] += (p[i * V + j] - (j === y ? 1 : 0)) / cnt;
    }
  });
  return { loss, perRow, probs: p };
}

// ---------- the model ----------
export class TinyTransformer {
  /** cfg: { vocab, ctx, d, heads, layers, mlp:boolean, ln:boolean, seed } */
  constructor(cfg) {
    this.cfg = cfg = { mlp: false, ln: false, seed: 1, ...cfg };
    const { vocab: V, ctx: T, d, layers: L } = cfg;
    if (d % cfg.heads) throw new Error('d must be divisible by heads');
    const rng = mulberry32(cfg.seed);
    const P = (r, c, std) => { const t = new Tensor(r, c); for (let i = 0; i < t.data.length; i++) t.data[i] = gaussian(rng) * std; return t; };
    const ones = (c) => { const t = new Tensor(1, c); t.data.fill(1); return t; };
    this.params = {};
    const reg = (name, t) => (this.params[name] = t);
    reg('E', P(V, d, 1 / Math.sqrt(d)));
    reg('Pos', P(T, d, 1 / Math.sqrt(d)));
    for (let l = 0; l < L; l++) {
      if (cfg.ln) { reg(`ln1g${l}`, ones(d)); reg(`ln1b${l}`, new Tensor(1, d)); }
      for (const w of ['q', 'k', 'v', 'o']) reg(`W${w}${l}`, P(d, d, 1 / Math.sqrt(d)));
      if (cfg.mlp) {
        if (cfg.ln) { reg(`ln2g${l}`, ones(d)); reg(`ln2b${l}`, new Tensor(1, d)); }
        reg(`W1${l}`, P(d, 4 * d, 1 / Math.sqrt(d))); reg(`b1${l}`, new Tensor(1, 4 * d));
        reg(`W2${l}`, P(4 * d, d, 1 / Math.sqrt(4 * d))); reg(`b2${l}`, new Tensor(1, d));
      }
    }
    if (cfg.ln) { reg('lnfg', ones(d)); reg('lnfb', new Tensor(1, d)); }
    reg('U', P(d, V, 1 / Math.sqrt(d)));
    this.adam = { t: 0, m: {}, v: {} };
    for (const [k, t] of Object.entries(this.params)) { this.adam.m[k] = new FA(t.data.length); this.adam.v[k] = new FA(t.data.length); }
  }

  nParams() { return Object.values(this.params).reduce((s, t) => s + t.data.length, 0); }

  /** ids: Int32Array(B*T); targets: Int32Array(B*T) with -1 = ignore. */
  forward(ids, targets, B, tape = null) {
    const { ctx: T, heads: H, layers: L, mlp, ln } = this.cfg, p = this.params;
    let x = embed(tape, p.E, p.Pos, ids, B, T);
    const attn = [];
    for (let l = 0; l < L; l++) {
      const h = ln ? layernorm(tape, x, p[`ln1g${l}`], p[`ln1b${l}`]) : x;
      const { O, probs } = attention(tape, matmul(tape, h, p[`Wq${l}`]), matmul(tape, h, p[`Wk${l}`]), matmul(tape, h, p[`Wv${l}`]), B, T, H);
      attn.push(probs);
      x = add(tape, x, matmul(tape, O, p[`Wo${l}`]));
      if (mlp) {
        const h2 = ln ? layernorm(tape, x, p[`ln2g${l}`], p[`ln2b${l}`]) : x;
        const a = relu(tape, addRow(tape, matmul(tape, h2, p[`W1${l}`]), p[`b1${l}`]));
        x = add(tape, x, addRow(tape, matmul(tape, a, p[`W2${l}`]), p[`b2${l}`]));
      }
    }
    const hf = ln ? layernorm(tape, x, p.lnfg, p.lnfb) : x;
    const logits = matmul(tape, hf, p.U);
    const ce = crossEntropy(tape, logits, targets);
    return { ...ce, logits, attn };
  }

  zeroGrad() { for (const t of Object.values(this.params)) t.grad = null; }

  /** One Adam step on a batch. Returns the loss. */
  step(ids, targets, B, { lr = 3e-3, beta1 = 0.9, beta2 = 0.98, eps = 1e-8, clip = 1, wd = 0 } = {}) {
    const tape = new Tape();
    this.zeroGrad();
    const out = this.forward(ids, targets, B, tape);
    tape.backward();
    let norm = 0;
    for (const t of Object.values(this.params)) if (t.grad) for (const g of t.grad) norm += g * g;
    norm = Math.sqrt(norm);
    const sc = clip && norm > clip ? clip / norm : 1;
    const a = this.adam; a.t++;
    const b1c = 1 - Math.pow(beta1, a.t), b2c = 1 - Math.pow(beta2, a.t);
    for (const [k, t] of Object.entries(this.params)) {
      if (!t.grad) continue;
      const m = a.m[k], v = a.v[k], w = t.data, g = t.grad;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * sc;
        m[i] = beta1 * m[i] + (1 - beta1) * gi;
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        w[i] -= lr * ((m[i] / b1c) / (Math.sqrt(v[i] / b2c) + eps) + wd * w[i]);
      }
    }
    return { loss: out.loss, gradNorm: norm };
  }
}

// ---------- tasks ----------
// Every task returns { ids, targets } for B sequences of length T (targets[r] = next token, -1 = not scored).
export const TASKS = {
  induction: {
    label: 'Repeated random segments (induction)',
    // Random tokens with one segment of length n repeated later, at a random start and a random gap.
    // Because the offset between the copies changes every sequence, attending by absolute position
    // can't solve it: the model has to find the previous occurrence of the current token by content.
    make(cfg) { return { vocab: cfg.vocab, ctx: cfg.ctx }; },
    batch(rng, B, cfg) {
      const T = cfg.ctx, n = cfg.seg, slack = T - 2 * n;
      if (slack < 0) throw new Error('ctx must be at least 2 × segment length');
      const ids = new Int32Array(B * T), tg = new Int32Array(B * T), rep = new Int8Array(B * T), meta = [];
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < T; t++) ids[b * T + t] = (rng() * cfg.vocab) | 0;
        const s1 = (rng() * (slack + 1)) | 0, gap = (rng() * (slack - s1 + 1)) | 0, s2 = s1 + n + gap;
        for (let i = 0; i < n; i++) ids[b * T + s2 + i] = ids[b * T + s1 + i];
        for (let t = 0; t < T; t++) tg[b * T + t] = t + 1 < T ? ids[b * T + t + 1] : -1;
        // targets predictable by induction: next token inside the second copy
        for (let i = 0; i < n - 1; i++) rep[b * T + s2 + i] = 1;
        meta.push({ s1, s2, n });
      }
      return { ids, targets: tg, rep, meta };
    },
  },
  copy: {
    label: 'Copy a sequence',
    // digits x1..xk, SEP, x1..xk ; only the part after SEP is scored.
    make(cfg) { return { vocab: cfg.digits + 1, ctx: 2 * cfg.len + 1 }; },
    batch(rng, B, cfg) { return seqTask(rng, B, cfg, false); },
  },
  reverse: {
    label: 'Reverse a sequence',
    make(cfg) { return { vocab: cfg.digits + 1, ctx: 2 * cfg.len + 1 }; },
    batch(rng, B, cfg) { return seqTask(rng, B, cfg, true); },
  },
};

function seqTask(rng, B, cfg, reverse) {
  const k = cfg.len, SEP = cfg.digits, T = 2 * k + 1, ids = new Int32Array(B * T), tg = new Int32Array(B * T).fill(-1);
  for (let b = 0; b < B; b++) {
    const xs = []; for (let i = 0; i < k; i++) xs.push((rng() * cfg.digits) | 0);
    const ys = reverse ? xs.slice().reverse() : xs;
    const seq = [...xs, SEP, ...ys];
    for (let t = 0; t < T; t++) ids[b * T + t] = seq[t];
    for (let t = k; t < T - 1; t++) tg[b * T + t] = seq[t + 1];
  }
  return { ids, targets: tg };
}

// ---------- diagnostics ----------
/** Induction and previous-token scores per head (layer-major). meta[b] = { s1, s2, n } from TASKS.induction.
 *  Induction score: mean attention from position s2+i to s1+i+1 (the token after the previous occurrence
 *  of the current token). Previous-token score: mean attention from t to t−1. */
export function headScores(attn, meta, T) {
  const L = attn.length, H = attn[0][0].length, ind = [], prev = [];
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
    let si = 0, sp = 0, ci = 0, cp = 0;
    meta.forEach((m, b) => {
      const A = attn[l][b][h];
      for (let i = 0; i < m.n - 1; i++) { const t = m.s2 + i; si += A[t * T + m.s1 + i + 1]; ci++; }
      for (let t = 1; t < T; t++) { sp += A[t * T + t - 1]; cp++; }
    });
    ind.push(si / ci); prev.push(sp / cp);
  }
  return { induction: ind, prevToken: prev };
}
